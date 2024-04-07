from collections import defaultdict
import time
import numpy as np
import torch as th
import os
import _pickle as cPickle
from pathlib import Path

from wacky import CfgNode as CN
from torch.distributions import Categorical

import coref.dataloader as dataloader
from coref.dataloader import SynthCollator, SynthShapesCollator, shapes_collator
import coref.model as Models
from coref.utils.beam_utils import batch_beam_decode
from coref.utils.logger import Logger, profileit
from coref.utils.evaluator import EvalStatEstimator
import coref.utils.optim as optim


class PretrainTrainer():

    def __init__(self, config, resume=False):

        self.device = config.device
        self.dtype = config.dtype
        self.n_dims = config.language_conf.n_dims
        self.resolution = config.resolution
        self.dl_specs = config.data_loader.clone()
        self.lang_conf = config.language_conf.clone()

        self.max_epochs = config.trainer.max_epochs
        self.score_tolerance = config.trainer.score_tolerance
        self.save_frequency = config.trainer.save_frequency
        self.entropy_loss_weight = config.trainer.entropy_loss_weight
        self.log_interval = config.logger.log_interval

        self.eval_specs = config.eval_specs.clone()

        items = self.load_datasets(config, self.dl_specs, self.lang_conf)
        self.train_dataset, self.val_dataset, self.shapes_dataset = items[:3]
        self.train_epoch_iters = items[3]
        items = self.create_model_and_opt(
            config.model, config.optimizer, config.scheduler)
        self.model, self.optimizer, self.scheduler = items
        self.model_save_dir = config.model_save_dir

        self.logger = Logger(config)

        if resume:
            self._resume_model()
        else:
            self.training_initialize()

        self.nll_loss = th.nn.NLLLoss(reduce=False)

    def load_datasets(self, config, dl_specs, lang_conf):
        train_dataset_class = getattr(dataloader, dl_specs.train.name)
        train_dataset = train_dataset_class(dl_specs, dl_specs.train, lang_conf,
                                            device=config.device, dtype=config.dtype, seed=config.seed)
        # First dataset is for Synthetic data
        val_dataset_class = getattr(dataloader, dl_specs.validation_synth.name)
        val_dataset = val_dataset_class(dl_specs, dl_specs.validation_synth, lang_conf,
                                        device=config.device, dtype=config.dtype, seed=config.seed)
        # Second is for evaluating on GT data
        val_dataset_class = getattr(
            dataloader, dl_specs.validation_shapes.name)
        shapes_dataset = val_dataset_class(dl_specs, dl_specs.validation_shapes, lang_conf,
                                           device=config.device, dtype=config.dtype, seed=config.seed)

        train_epoch_iters = len(train_dataset) / \
            config.data_loader.train.batch_size

        return train_dataset, val_dataset, shapes_dataset, train_epoch_iters

    def create_model_and_opt(self, model_config, opt_config, scheduler_config):
        model = getattr(Models, model_config.name)(model_config)
        opt_class = getattr(th.optim, opt_config.name)
        optimizer = opt_class(model.parameters(), **opt_config.kwargs)
        if hasattr(th.optim.lr_scheduler, scheduler_config.name):
            sched_class = getattr(th.optim.lr_scheduler, scheduler_config.name)
        else:
            sched_class = getattr(optim, scheduler_config.name)

        scheduler = sched_class(optimizer, **scheduler_config.kwargs)

        return model, optimizer, scheduler

    def _resume_model(self, path=None):
        load_state = False
        if path is None:
            # load the latest checkpoint
            if os.path.exists(self.model_save_dir):
                files = os.listdir(self.model_save_dir)
                # select files which have _%d.pt format
                selected_files = []
                def epoch_lambda(x): return int(x.split(".")[0].split("_")[-1])
                for cur_file in files:
                    try:
                        epoch_lambda(cur_file)
                        selected_files.append(cur_file)
                    except:
                        pass
                latest_file = max(selected_files, key=epoch_lambda)
                path = os.path.join(self.model_save_dir, latest_file)
                load_state = True
            else:
                load_state = False
        else:
            load_state = True

        if not load_state:
            print("No checkpoint found. Initializing from scratch.")
            self.training_initialize()
        else:
            load_obj = th.load(open(path, "rb"), map_location=self.device)
            self.train_state = load_obj["train_state"]
            self.model.load_state_dict(load_obj["model_state"])
            self.optimizer.load_state_dict(load_obj["optimizer_state"])
            self.scheduler.load_state_dict(load_obj["scheduler_state"])

            print(f"Checkpoint loaded from {path} successfully.")
            print(f"Starting epoch set as {self.train_state.cur_epoch}")
            print(f"Performance: {self.train_state.best_val_metric}")

    def _save_model(self, epoch, tag=None):
        Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
        device = self.model.device
        self.model = self.model.cpu()
        save_obj = {
            "train_state": self.train_state.clone(),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        if tag is None:
            file_name = f"model_{epoch}.pt"
        else:
            file_name = tag
        file_path = os.path.join(self.model_save_dir, file_name)
        print(f"saving model at epoch {epoch} at {file_path}")
        th.save(save_obj, file_path)
        self.model = self.model.to(device)

    def training_initialize(self):

        # make Train state.
        self.train_state = CN()
        self.train_state.best_val_metric = -np.inf
        self.train_state.cur_val_metric = -np.inf
        self.train_state.best_iter = 0
        self.train_state.cur_iter = 0
        self.train_state.best_epoch = 0
        self.train_state.cur_epoch = 0

        # Model weights / optimizer already initialized

    def start_experiment(self):

        dl_specs = self.dl_specs
        lang_conf = self.lang_conf
        train_collator = SynthCollator(dl_specs.collator_eval_size, self.device,
                                       self.dtype, self.resolution, self.n_dims, dl_specs.train, lang_conf)

        train_loader = th.utils.data.DataLoader(self.train_dataset, batch_size=dl_specs.train.batch_size, pin_memory=False,
                                                num_workers=dl_specs.train.workers, shuffle=False, collate_fn=train_collator,
                                                persistent_workers=dl_specs.train.workers > 0)

        val_dataloaders = self.get_val_dataloaders(dl_specs, lang_conf)

        # shift model:
        # Train model on the dataset
        start_epoch = self.train_state.cur_epoch
        for epoch in range(start_epoch, self.max_epochs + 1):
            self.train_state.cur_epoch = epoch
            train_epoch_start = time.time()
            for iter_ind, batch_dict in enumerate(train_loader):
                if iter_ind == 0:
                    self.model = self.model.cuda()
                    self.model.train()
                cur_iter = int(self.train_state.cur_epoch *
                               self.train_epoch_iters + iter_ind)
                self.train_state.cur_iter = cur_iter
                # model forward:
                output = self.model.forward_train(batch_dict)
                actions = batch_dict["actions"]
                action_validity = batch_dict["action_validity"]
                loss, loss_statistics = self._calculate_loss(
                    output, actions, action_validity)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                if iter_ind % self.log_interval == 0:
                    n_actions = batch_dict["n_actions"]
                    all_stats = self.get_gt_statistics(
                        output, actions, action_validity, n_actions)
                    all_stats.update({x: y.item()
                                     for x, y in loss_statistics.items()})
                    self.logger.log_statistics(
                        all_stats, epoch, cur_iter, prefix="train_synth")
            # Train Epoch over
            self.optimizer.zero_grad(set_to_none=True)
            train_epoch_end = time.time()
            train_epoch_stats = {
                "epoch_time": train_epoch_end - train_epoch_start}
            self.logger.log_statistics(
                train_epoch_stats, epoch, cur_iter, prefix="train_synth")

            self.model.eval()
            all_stats = self._evaluate_val_loss(val_dataloaders)
            self.logger.log_statistics(
                all_stats, epoch, cur_iter, prefix="val_synth")
            self.model.train()

            if epoch % self.eval_specs.bs_eval_frequency == 0:
                self.model.eval()
                stat_estimator = self._evaluate_bs(val_dataloaders)
                final_metrics = stat_estimator.get_final_metrics()
                self.model.train()
                # inner saturation check:
                val_metric = final_metrics["val_metric"]
                best_val_metric = self.train_state.best_val_metric
                if val_metric > best_val_metric + self.score_tolerance:
                    print("New best Val Metric: ", val_metric)
                    self.train_state.best_val_metric = val_metric
                    self.train_state.best_epoch = epoch
                    self.train_state.best_iter = cur_iter

                    self._save_model(epoch, tag="best_model.pt")
            # Save model checkpoint?
            if epoch % self.save_frequency == 0:
                self._save_model(epoch)
            self.scheduler.step()
            # For cache gen:
            self.model.cpu()
        # No return for pretraining.
        self._save_model(epoch, tag="final_model.pt")
        return None

    def _calculate_loss(self, cmd_logsf, cmd_target, action_validity):
        # calculate loss
        # return loss, loss_statistics
        # Ignore the start token.
        cmd_target_flat = cmd_target[:, 1:].reshape(-1)
        action_validity_flat = action_validity[:, 1:].reshape(-1)
        cmd_loss = self.nll_loss(cmd_logsf, cmd_target_flat)
        cmd_loss = th.where(action_validity_flat, cmd_loss, 0)
        # Shoud do it batch wise?
        cmd_loss = cmd_loss.reshape(cmd_target.shape[0], -1)
        # action_validity_flat = action_validity_flat.reshape(cmd_target.shape[0], -1)
        cmd_loss = cmd_loss.sum(dim=-1) / action_validity[:, 1:].sum(dim=-1)
        # cmd_loss = th.sum(cmd_loss)/th.sum(action_validity_flat)
        cmd_loss = th.mean(cmd_loss)

        categorical = Categorical(probs=th.exp(cmd_logsf))  # actually not.
        entropy_loss = -categorical.entropy().mean()

        total_loss = th.mean(cmd_loss) + \
            self.entropy_loss_weight * entropy_loss
        stat_obj = {
            "cmd_loss": cmd_loss,
            "total_loss": total_loss,
            "entropy_loss": entropy_loss,
        }

        return total_loss, stat_obj

    def get_gt_statistics(self, cmd_logsf, actions, action_validity, n_actions):

        cmd_validity_flat = action_validity[:, 1:].reshape(-1)
        actions_flat = actions[:, 1:].reshape(-1)
        cmd_action = th.argmax(cmd_logsf, dim=-1)
        cmd_match = (cmd_action == actions_flat).float()
        cmd_acc = th.sum(th.where(cmd_validity_flat, cmd_match, 0)
                         ) / th.sum(cmd_validity_flat)

        mean_expr_len = th.mean(n_actions.float())

        statistics = {
            "cmd_acc": cmd_acc.item(),
            "expr_len": mean_expr_len.item(),
        }

        return statistics

    def get_val_dataloaders(self, dl_specs, lang_conf):
        synth_collator = SynthCollator(dl_specs.collator_eval_size,
                                       self.device, self.dtype, self.resolution, self.n_dims, dl_specs.validation_synth, lang_conf)
        synth_shapes_collator = SynthShapesCollator(dl_specs.collator_eval_size,
                                                    self.device, self.dtype, self.resolution, self.n_dims, dl_specs.validation_shapes, lang_conf)

        synth_val_gt = th.utils.data.DataLoader(self.val_dataset, batch_size=dl_specs.train.batch_size,
                                                num_workers=dl_specs.validation_synth.workers,
                                                persistent_workers=dl_specs.validation_synth.workers > 0, collate_fn=synth_collator)
        synth_val_bs = th.utils.data.DataLoader(self.val_dataset, batch_size=dl_specs.validation_synth.batch_size,
                                                num_workers=dl_specs.validation_synth.workers,
                                                persistent_workers=dl_specs.validation_synth.workers > 0, collate_fn=synth_shapes_collator)
        shapes_val_bs = th.utils.data.DataLoader(self.shapes_dataset, batch_size=dl_specs.validation_shapes.batch_size,
                                                 num_workers=dl_specs.validation_shapes.workers, shuffle=False,
                                                 persistent_workers=dl_specs.validation_shapes.workers > 0, collate_fn=shapes_collator)

        return [synth_val_gt, synth_val_bs, shapes_val_bs]

    def _evaluate_val_loss(self, val_dataloaders):

        synth_val_gt, synth_val_bs, shapes_val_bs = val_dataloaders
        synth_val_gt.dataset.set_train_fetch_mode()

        n_evals = 0
        st = time.time()
        all_stats = defaultdict(list)
        for iter_ind, batch_dict in enumerate(synth_val_gt):
            with th.no_grad():
                actions = batch_dict["actions"]
                action_validity = batch_dict["action_validity"]
                n_actions = batch_dict["n_actions"]
                batch_size = actions.shape[0]
                output = self.model.forward_train(batch_dict)
                loss, loss_statistics = self._calculate_loss(
                    output, actions, action_validity)
                stats = self.get_gt_statistics(
                    output, actions, action_validity, n_actions)
                stats.update({x: y.item() for x, y in loss_statistics.items()})
                for stat, value in stats.items():
                    all_stats[stat].append(value)
            n_evals += batch_size
            # print(f"Evaluated {iter_ind} batches and {n_evals} samples in {time.time() - st}")
            if n_evals >= self.eval_specs.n_samples:
                break

        for key, value in all_stats.items():
            all_stats[f"{key}"] = np.mean(value).item()

        et = time.time()
        all_stats['eval_time'] = et - st
        return all_stats

    def _evaluate_bs(self, val_dataloaders):
        # perform loss calculation on the synthetic val set
        # perform beam search on synthetic
        # perform beam search on GT val set
        epoch = self.train_state.cur_epoch
        cur_iter = self.train_state.cur_iter
        synth_val_gt, synth_val_bs, shapes_val_bs = val_dataloaders

        expression_stats, stat_estimator = self.validation_with_beam_search(
            self.model, synth_val_bs, self.eval_specs.beam_size)
        self.logger.log_statistics(
            expression_stats, epoch, cur_iter, prefix="val_synth_bs")
        expression_stats, stat_estimator = self.validation_with_beam_search(
            self.model, shapes_val_bs, self.eval_specs.beam_size)
        expression_stats['previous_best_val_metric'] = self.train_state.best_val_metric
        self.logger.log_statistics(
            expression_stats, epoch, cur_iter, prefix="val_shapes_bs")
        programs_str = stat_estimator.get_best_programs(
            to_string=True, return_score=True)
        self.logger.save_programs(programs_str, cur_iter, tag=None)
        return stat_estimator

    def validation_with_beam_search(self, model, dataloader, beam_size):
        n_evals = 0
        st = time.time()
        stat_estimator = EvalStatEstimator(self.eval_specs, self.lang_conf,
                                           self.dtype, self.device)

        dataloader.dataset.set_val_fetch_mode()

        for iter_ind, batch_dict in enumerate(dataloader):
            # model forward:
            indices = batch_dict["indices"]
            occs = batch_dict["occs"]
            batch_size = occs.shape[0]
            with th.no_grad():
                pred_actions = batch_beam_decode(model, batch_dict, self.lang_conf,
                                                 batch_size=batch_size,
                                                 beam_size=beam_size, device=self.device)
                # add the actual actions into pred actions
                start_time = time.time()
                stat_estimator.update_from_actions(indices, occs, pred_actions)
                end_time = time.time()
                print("Time for stat update:", end_time - start_time)
            n_evals += batch_size
            print(
                f"Evaluated {iter_ind} batches and {n_evals} samples in {time.time() - st}")
            if n_evals >= self.eval_specs.n_samples:
                break

        et = time.time()
        final_metrics = stat_estimator.get_final_metrics()
        expression_stats = stat_estimator.get_stats()
        expression_stats['eval_time'] = et - st
        expression_stats.update(final_metrics)

        return expression_stats, stat_estimator

    def eval_loop(self, starting_weights):

        self._resume_model(starting_weights)
        self.model.cuda()
        self.model.eval()

        dl_specs = self.dl_specs
        lang_conf = self.lang_conf
        val_dataloaders = self.get_val_dataloaders(dl_specs, lang_conf)

        epoch = self.train_state.cur_epoch
        cur_iter = self.train_state.cur_iter
        synth_val_gt, synth_val_bs, shapes_val_bs = val_dataloaders

        expression_stats, stat_estimator = self.validation_with_beam_search(
            self.model, shapes_val_bs, self.eval_specs.beam_size)
        expression_stats['previous_best_val_metric'] = self.train_state.best_val_metric
        self.logger.log_statistics(
            expression_stats, epoch, cur_iter, prefix="val_shapes_bs")
        programs_str = stat_estimator.get_best_programs(
            to_string=True, return_score=True)
        self.logger.save_programs(programs_str, cur_iter, tag=None)
