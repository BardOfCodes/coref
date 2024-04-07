from collections import defaultdict
import time
import numpy as np
import torch as th
import _pickle as cPickle
import os
from pathlib import Path

from wacky import CfgNode as CN

import coref.dataloader as dataloader
from coref.dataloader import PLADCollator, shapes_collator
from coref.utils.bpd import BestProgramDataStruct
from coref.utils.beam_utils import batch_beam_decode
from coref.utils.evaluator import EvalStatEstimator
from coref.utils.optim import optimizer_to

from .pretrain import PretrainTrainer


class PLADTrainer(PretrainTrainer):

    def __init__(self, config, resume=False):

        # load plad specs
        self.inner_patience = config.plad.inner_patience
        self.outer_patience = config.plad.outer_patience
        self.max_inner_iter = config.plad.max_inner_iter
        self.max_outer_iter = config.plad.max_outer_iter
        self.starting_weights = config.plad.starting_weights
        self.bpds_config = config.bpds.clone()
        self.search_config = config.plad.search_config.clone()

        # for reloading weights
        self.model_config = config.model.clone()
        self.ws_model_config = config.ws_config.model.clone()
        self.ws_config = config.ws_config
        self.opt_config = config.optimizer.clone()
        self.sched_config = config.scheduler.clone()

        self.ws_max_inner_iter = config.ws_config.max_inner_iter
        self.ws_inner_patience = config.ws_config.inner_patience
        self.ws_starting_weights = config.ws_config.starting_weights
        self.ws_kld_loss_weight = config.ws_config.kld_loss_weight
        self.ws_eval_specs = config.ws_eval_specs.clone()
        super(PLADTrainer, self).__init__(config, resume=False)

        del self.model, self.optimizer, self.scheduler, self.train_state

        # reinit
        # TEMP HACK
        # self.load_checkpoint(self.starting_weights)
        self.training_initialize()
        # if load checkpoint?

    def training_initialize(self):
        super(PLADTrainer, self).training_initialize()
        # PLAD specific
        print(f"{self.__class__.__name__} specific initialization.")
        self.train_state.best_outer_iter = 0
        self.train_state.outer_iter = 0
        self.train_state.inner_iter = 0
        self.train_state.best_inner_iter = 0
        self.train_state.best_weights = self.starting_weights
        self.train_state.ws = CN()
        self.train_state.ws.best_weights = self.ws_starting_weights
        self.train_state.ws.best_val_metric = np.inf  # just loss of val set.
        self.train_state.ws.best_iter = 0
        self.train_state.ws.best_epoch = 0
        self.train_state.ws.cur_epoch = 0
        self.train_state.ws.cur_iter = 0
        self.train_state.ws.cur_val_metric = np.inf

    def start_experiment(self):

        dl_specs = self.dl_specs
        bpds = BestProgramDataStruct(self.bpds_config)
        val_dataloaders = self.get_val_dataloaders(self.dl_specs)
        # for search
        search_loader = th.utils.data.DataLoader(self.train_dataset, batch_size=dl_specs.search.batch_size,
                                                 num_workers=dl_specs.search.workers, shuffle=False,
                                                 persistent_workers=dl_specs.search.workers > 0, collate_fn=shapes_collator)
        # for training
        plad_collator = PLADCollator(dl_specs.collator_eval_size,
                                     self.device, self.dtype, self.resolution, self.n_dims)
        train_loader = th.utils.data.DataLoader(self.train_dataset, batch_size=dl_specs.train.batch_size,
                                                num_workers=dl_specs.train.workers, shuffle=False,
                                                persistent_workers=dl_specs.train.workers > 0, collate_fn=plad_collator)
        outer_loop_saturation = False
        outer_iter = 0
        cur_iters = 0

        while (not outer_loop_saturation):
            # Search for good programs:
            self.search_and_generate(search_loader, train_loader, bpds)
            previous_best_score = self.train_state.best_val_metric
            inner_best_score = self.inner_loop(train_loader, val_dataloaders)
            # Load previous best weights?
            print(
                f"Inner loop scores {previous_best_score} to {inner_best_score}")
            # outer saturation check:
            outer_condition_1 = outer_iter - \
                self.train_state.best_outer_iter >= self.outer_patience
            outer_condition_2 = outer_iter > self.max_outer_iter
            if outer_condition_1 or outer_condition_2:
                print("Reached outer saturation.")
                outer_loop_saturation = True
            else:
                print("Outer loop not saturated yet.")
                if inner_best_score > previous_best_score + self.score_tolerance:
                    self.train_state.best_outer_iter = outer_iter
                print("Loading previous best weights...")
            outer_iter += 1

        model, opt, sched = self.create_model_and_opt(
            self.model_config, self.opt_config, self.sched_config)
        model, opt, sched = self._load_model(
            model, opt, sched, self.train_state.best_weights)
        epoch = self.train_state.epoch
        self._save_model(epoch, model, opt, sched, tag="final_model.pt")

    def search_and_generate(self, search_loader, train_loader, bpds):
        # load model
        epoch = self.train_state.cur_epoch
        cur_iter = self.train_state.cur_iter
        expression_stats, stat_estimator = self.search_programs(search_loader)
        self.logger.log_statistics(
            expression_stats, epoch, cur_iter, prefix="train_search_bs")
        new_programs = stat_estimator.get_best_programs(return_score=True)
        bpds.update_programs(new_programs, "BS")
        if self.ws_config.active:
            self.train_dataset.set_programs(
                bpds.get_programs(avoid_keys=["WS"]))
            best_val_metric = self.train_ws_model(train_loader)
            expression_stats, stat_estimator = self.generate_wake_sleep_samples(
                search_loader)
            self.logger.log_statistics(
                expression_stats, epoch, cur_iter, prefix="ws_search_bs")
            new_programs = stat_estimator.get_best_programs(return_score=True)
            bpds.update_programs(new_programs, "WS")
        all_programs = bpds.get_programs()
        self.train_dataset.set_programs(all_programs)
        # delete vae model
        # Log number of programs
        bpd_stats = bpds.get_stats()
        bpd_stats['n_training_samples'] = len(all_programs)
        self.logger.log_statistics(
            bpd_stats, epoch, cur_iter, prefix="bpd_stats")

        return None

    def search_programs(self, search_loader):
        # use the model to search for programs
        model, _, _ = self.create_model_and_opt(
            self.model_config, self.opt_config, self.sched_config)
        model, _, _ = self._load_model(
            model, None, None, self.train_state.best_weights)
        model = model.cuda()
        model.eval()
        search_loader.dataset.to_shapes_mode()

        n_evals = 0
        st = time.time()
        stat_estimator = EvalStatEstimator(self.eval_specs, self.lang_conf,
                                           self.dtype, self.device)

        for iter_ind, batch_dict in enumerate(search_loader):
            # model forward:
            all_occs = batch_dict["occs"]
            indices = batch_dict["indices"]
            batch_size = all_occs.shape[0]
            with th.no_grad():
                pred_actions = batch_beam_decode(model, batch_dict, self.lang_conf,
                                                 batch_size=batch_size,
                                                 beam_size=self.search_config.beam_size,
                                                 #  stochastic_beam_search=True,
                                                 device=self.device)
                stat_estimator.update_from_actions(
                    indices, all_occs, pred_actions)
            n_evals += batch_size
            print(
                f"Evaluated {iter_ind} batches and {n_evals} samples in {time.time() - st}")
        et = time.time()
        final_metrics = stat_estimator.get_final_metrics()
        expression_stats = stat_estimator.get_stats()
        expression_stats['eval_time'] = et - st
        expression_stats.update(final_metrics)
        # cleanup
        del model

        return expression_stats, stat_estimator

    def inner_loop(self, train_loader, val_dataloaders):

        last_improvment_iter = 0
        # load model
        model, opt, sched = self.create_model_and_opt(
            self.model_config, self.opt_config, self.sched_config)
        model, opt, sched = self._load_model(
            model, opt, sched, self.train_state.best_weights)
        model = model.cuda()
        optimizer_to(opt, model.device)
        train_loader.dataset.to_plad_train_mode()
        epochs = self.train_state.cur_epoch
        for inner_iter in range(self.max_inner_iter):
            train_epoch_start = time.time()
            cur_epoch = epochs + inner_iter
            self.train_state.cur_epoch = cur_epoch
            model.train()
            for iter_ind, batch_dict in enumerate(train_loader):
                # model forward:
                actions = batch_dict["actions"]
                action_validity = batch_dict["action_validity"]
                n_actions = batch_dict["n_actions"]

                log_iters = int(cur_epoch * self.train_epoch_iters + iter_ind)

                output = model.forward_train(batch_dict)
                loss, loss_statistics = self._calculate_loss(
                    output, actions, action_validity)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                if iter_ind % self.log_interval == 0:
                    all_stats = self.get_gt_statistics(
                        output, actions, action_validity, n_actions)
                    all_stats.update({x: y.item()
                                     for x, y in loss_statistics.items()})
                    self.logger.log_statistics(
                        all_stats, cur_epoch, log_iters, prefix="train_shapes")

            train_epoch_end = time.time()
            train_epoch_stats = {
                "epoch_time": train_epoch_end - train_epoch_start}
            self.logger.log_statistics(
                train_epoch_stats, cur_epoch, log_iters, prefix="train_shapes")

            stat_estimator = self._evaluate_bs(val_dataloaders, model)
            final_metrics = stat_estimator.get_final_metrics()
            # inner saturation check:
            val_metric = final_metrics["val_metric"]
            best_val_metric = self.train_state.best_val_metric
            if val_metric > best_val_metric + self.score_tolerance:
                print(f"model improved from {best_val_metric} to {val_metric}")
                self.train_state.best_val_metric = float(val_metric)
                self.train_state.best_epoch = cur_epoch
                self.train_state.best_iter = log_iters
                self.train_state.best_weights = os.path.join(
                    self.model_save_dir, f"model_{cur_epoch}.pt")
                self._save_model(cur_epoch, model, opt, sched)
                self._save_model(cur_epoch, model, opt, sched, tag="best_model.pt")
                last_improvment_iter = inner_iter
            else:
                print(
                    f"Model did not improve. Previously {best_val_metric}, current {val_metric}")

            iter_delta = inner_iter - last_improvment_iter
            if iter_delta >= self.inner_patience:
                # hit saturation.
                print("Reached inner saturation with {iter_delta} delta")
                break
            else:
                print(
                    f"Current iter delta {iter_delta} < {self.inner_patience}")
                # Save model checkpoint?
            if cur_epoch % self.save_frequency == 0:
                self._save_model(cur_epoch, model, opt, sched)
            et = time.time()
            print("Epoch Time: ", et - train_epoch_start)
            sched.step()

        del model, opt, sched, stat_estimator

        return best_val_metric

    def train_ws_model(self, train_loader):
        # remove main model
        # load the model
        last_improvment_iter = 0
        # load model
        model, opt, sched = self.create_model_and_opt(
            self.ws_model_config, self.opt_config, self.sched_config)
        model, _, _ = self._load_model(
            model, None, None, self.train_state.ws.best_weights, strict=False)
        model = model.cuda()
        optimizer_to(opt, model.device)
        train_loader.dataset.to_plad_train_mode()
        max_inner_iter = self.ws_max_inner_iter
        epochs = self.train_state.ws.cur_epoch
        for inner_iter in range(max_inner_iter):
            cur_epoch = epochs + inner_iter
            self.train_state.ws.cur_epoch = cur_epoch
            train_epoch_start = time.time()
            model.train()
            for iter_ind, batch_dict in enumerate(train_loader):
                # model forward:
                # model forward:
                actions = batch_dict["actions"]
                action_validity = batch_dict["action_validity"]
                n_actions = batch_dict["n_actions"]
                cur_iter = int(self.train_state.ws.cur_epoch *
                               self.train_epoch_iters + iter_ind)
                self.train_state.ws.cur_iter = cur_iter
                output, mu, logvar = model.forward_train(batch_dict)
                loss, loss_statistics = self._calculate_ws_loss(
                    output, mu, logvar, actions, action_validity)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                if iter_ind % self.log_interval == 0:
                    all_stats = self.get_gt_statistics(
                        output, actions, action_validity, n_actions)
                    all_stats.update({x: y.item()
                                     for x, y in loss_statistics.items()})
                    self.logger.log_statistics(
                        all_stats, cur_epoch, cur_iter, prefix="ws_train_shapes")

            train_epoch_end = time.time()
            train_epoch_stats = {
                "epoch_time": train_epoch_end - train_epoch_start}
            self.logger.log_statistics(
                train_epoch_stats, cur_epoch, cur_iter, prefix="ws_train_shapes")
            # Evaluate loss on data.

            all_stats, val_metric = self._evaluate_ws_train_loss(
                train_loader, model)
            self.logger.log_statistics(
                all_stats, cur_epoch, cur_iter, prefix="ws_train_shapes_val")

            best_val_metric = self.train_state.ws.best_val_metric
            if val_metric < best_val_metric + self.score_tolerance:
                print(
                    f"WS model improved from {best_val_metric} to {val_metric}")
                self.train_state.ws.best_val_metric = float(val_metric)
                self.train_state.ws.best_epoch = cur_epoch
                self.train_state.ws.best_iter = cur_iter
                tag = f"ws_model_{cur_epoch}.pt"
                self.train_state.ws.best_weights = os.path.join(
                    self.model_save_dir, tag)
                self._save_model(cur_epoch, model, opt, sched, tag=tag)
                last_improvment_iter = inner_iter
            else:
                print(
                    f"WS model did not improve. Previously {best_val_metric}, current {val_metric}")

            iter_delta = inner_iter - last_improvment_iter
            if iter_delta >= self.ws_inner_patience:
                # hit saturation.
                print("Reached inner saturation with {iter_delta} delta")
                break
            else:
                print(
                    f"Current iter delta {iter_delta} < {self.ws_inner_patience}")
                # Save model checkpoint?
            if cur_epoch % self.save_frequency == 0:
                tag = f"ws_model_{cur_epoch}.pt"
                self._save_model(cur_epoch, model, opt, sched, tag=tag)
            et = time.time()
            print("Epoch Time: ", et - train_epoch_start)
            sched.step()

        return best_val_metric, model

    def _calculate_ws_loss(self, output, mu, logvar, actions, action_validity):
        nll_loss, loss_statistics = self._calculate_loss(
            output, actions, action_validity)
        # kl loss:
        kld = -0.5 * \
            th.mean(th.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1), 0)
        total_loss = nll_loss + self.ws_kld_loss_weight * kld
        loss_statistics["kld"] = kld
        loss_statistics["total_loss"] = total_loss
        return total_loss, loss_statistics

    def _evaluate_ws_train_loss(self, dataloader, model):
        n_evals = 0
        st = time.time()

        all_stats = defaultdict(list)
        for iter_ind, batch_dict in enumerate(dataloader):
            # create indices:
            actions = batch_dict["actions"]
            action_validity = batch_dict["action_validity"]
            batch_size = actions.shape[0]
            with th.no_grad():
                output, mu, logvar = model.forward_train(batch_dict)
                loss, loss_statistics = self._calculate_ws_loss(
                    output, mu, logvar, actions, action_validity)
                for stat, value in loss_statistics.items():
                    all_stats[stat].append(value.item())
            n_evals += batch_size
            print(
                f"Evaluated {iter_ind} batches and {n_evals} samples in {time.time() - st}")
            if n_evals >= self.ws_eval_specs.n_samples:
                break

        for key, value in all_stats.items():
            all_stats[f"{key}"] = np.mean(value).item()

        et = time.time()
        val_metric = all_stats['total_loss']
        all_stats['eval_time'] = et - st

        return all_stats, val_metric

    def generate_wake_sleep_samples(self, search_loader):

        model, _, _ = self.create_model_and_opt(
            self.ws_model_config, self.opt_config, self.sched_config)
        model, _, _ = self._load_model(
            model, None, None, self.train_state.ws.best_weights)
        model = model.cuda()
        model.eval()
        search_loader.dataset.to_shapes_mode()

        n_evals = 0
        st = time.time()
        stat_estimator = EvalStatEstimator(self.ws_eval_specs, self.lang_conf,
                                           self.dtype, self.device, objective="loglikelihood")

        for iter_ind, batch_dict in enumerate(search_loader):
            # model forward:
            all_occs = batch_dict["occs"]
            indices = batch_dict["indices"]
            batch_size = all_occs.shape[0]
            with th.no_grad():
                pred_actions = batch_beam_decode(model, batch_dict, self.lang_conf,
                                                 batch_size=batch_size,
                                                 beam_size=self.ws_eval_specs.beam_size,
                                                 device=self.device)
                stat_estimator.update_from_actions(
                    indices, all_occs, pred_actions)
            n_evals += batch_size
            print(
                f"Evaluated {iter_ind} batches and {n_evals} samples in {time.time() - st}")

        et = time.time()
        final_metrics = stat_estimator.get_final_metrics()
        expression_stats = stat_estimator.get_stats()
        expression_stats['eval_time'] = et - st
        expression_stats.update(final_metrics)
        # cleanup
        del model

        return expression_stats, stat_estimator

    def _evaluate_bs(self, val_dataloaders, model):
        model.eval()
        synth_train_bs, shapes_val_bs = val_dataloaders
        epoch = self.train_state.cur_epoch
        cur_iter = self.train_state.cur_iter

        expression_stats, stat_estimator = self.train_val_with_beam_search(
            model, synth_train_bs, self.eval_specs.train_val_beam_size)
        self.logger.log_statistics(
            expression_stats, epoch, cur_iter, prefix="train_shapes_bs")
        expression_stats, stat_estimator = self.validation_with_beam_search(
            model, shapes_val_bs, self.eval_specs.beam_size)
        expression_stats['previous_best_val_metric'] = self.train_state.best_val_metric
        self.logger.log_statistics(
            expression_stats, epoch, cur_iter, prefix="val_shapes_bs")
        programs_str = stat_estimator.get_best_programs(to_string=True)
        self.logger.save_programs(programs_str, cur_iter, tag=None)
        return stat_estimator

    def train_val_with_beam_search(self, model, dataloader, beam_size):
        n_evals = 0
        st = time.time()
        stat_estimator = EvalStatEstimator(self.eval_specs, self.lang_conf,
                                           self.dtype, self.device)

        # TODO: Compute some statistics.
        # change the caller:
        dataloader.dataset.to_plad_train_val_mode()
        counter = 0
        for iter_ind, batch_dict in enumerate(dataloader):
            # create indices:
            all_occs = batch_dict["occs"]
            # indices = batch_dict['indices']
            batch_size = all_occs.shape[0]

            indices = th.arange(counter, counter + all_occs.shape[0])
            counter += batch_size
            with th.no_grad():
                pred_actions = batch_beam_decode(model, batch_dict, self.lang_conf,
                                                 batch_size=batch_size,
                                                 beam_size=beam_size,
                                                 device=self.device)
                stat_estimator.update_from_actions(
                    indices, all_occs, pred_actions)
            n_evals += batch_size
            print(
                f"Evaluated {iter_ind} batches and {n_evals} samples in {time.time() - st}")
            if n_evals >= self.eval_specs.n_samples:
                break

        dataloader.dataset.to_plad_train_mode()

        et = time.time()
        final_metrics = stat_estimator.get_final_metrics()
        expression_stats = stat_estimator.get_stats()
        expression_stats['eval_time'] = et - st
        expression_stats.update(final_metrics)

        return expression_stats, stat_estimator

    def load_datasets(self, config, dl_specs, lang_conf):
        train_dataset_class = getattr(dataloader, dl_specs.train.name)
        train_dataset = train_dataset_class(dl_specs, dl_specs.train, lang_conf,
                                            device=config.device, dtype=config.dtype, seed=config.seed)

        # First dataset is for Synthetic data
        val_dataset_class = getattr(
            dataloader, dl_specs.validation_shapes.name)
        shapes_dataset = val_dataset_class(dl_specs, dl_specs.validation_shapes, lang_conf,
                                           device=config.device, dtype=config.dtype, seed=config.seed)

        train_epoch_iters = len(train_dataset) / \
            config.data_loader.train.batch_size
        return train_dataset, None, shapes_dataset, train_epoch_iters

    def get_val_dataloaders(self, dl_specs):
        # train data loader
        plad_collator = PLADCollator(dl_specs.collator_eval_size,
                                     self.device, self.dtype, self.resolution, self.n_dims)
        synth_train_bs = th.utils.data.DataLoader(self.train_dataset, batch_size=dl_specs.train.batch_size,
                                                  num_workers=dl_specs.train.workers,
                                                  persistent_workers=dl_specs.train.workers > 0,
                                                  collate_fn=plad_collator)

        shapes_val_bs = th.utils.data.DataLoader(self.shapes_dataset, batch_size=dl_specs.validation_shapes.batch_size,
                                                 num_workers=dl_specs.validation_shapes.workers,
                                                 persistent_workers=dl_specs.validation_shapes.workers > 0,
                                                 collate_fn=shapes_collator)

        return [synth_train_bs, shapes_val_bs]

    def _load_model(self, model, optimizer, scheduler, path, strict=True):
        load_obj = th.load(open(path, "rb"), map_location=model.device)
        train_state = load_obj["train_state"]
        model.load_state_dict(load_obj["model_state"], strict=strict)
        if optimizer is not None:
            optimizer.load_state_dict(load_obj["optimizer_state"])
        if scheduler is not None:
            scheduler.load_state_dict(load_obj["scheduler_state"])
        print(f"Checkpoint loaded from {path} successfully.")
        print(f"Starting epoch set as {train_state.cur_epoch}")
        print(f"Performance: {train_state.best_val_metric}")
        return model, optimizer, scheduler

    def _save_model(self, epoch, model, opt, sched, tag=None):
        Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
        device = model.device
        model = model.cpu()
        save_obj = {
            "train_state": self.train_state.clone(),
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": sched.state_dict(),
        }
        if tag is None:
            file_name = f"model_{epoch}.pt"
        else:
            file_name = tag
        file_path = os.path.join(self.model_save_dir, file_name)
        print(f"saving model at epoch {epoch} at {file_path}")
        th.save(save_obj, file_path)
        model = model.to(device)

    def parallel_search_programs(self, search_loader):
        # use the model to search for programs
        # Load the model
        # TODO
        ...
