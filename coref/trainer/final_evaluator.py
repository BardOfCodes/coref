import torch as th
import time
import coref.dataloader as dataloader
from coref.utils.bpd import BestProgramDataStruct
from .siri import SIRITrainer
from .plad import PLADTrainer
from coref.utils.evaluator import EvalStatEstimator


class SimpleEvaluator(SIRITrainer):

    def __init__(self, config, resume=False):
        super(SimpleEvaluator, self).__init__(config, resume)
        self.inference_specs = config.inference

    def load_datasets(self, config, dl_specs, lang_conf):
        train_dataset = []

        # if test, load test.

        val_dataset = []
        # Second is for evaluating on GT data
        val_dataset_class = getattr(
            dataloader, dl_specs.validation_shapes.name)
        shapes_dataset = val_dataset_class(dl_specs, dl_specs.validation_shapes, lang_conf,
                                           device=config.device, dtype=config.dtype, seed=config.seed)

        train_epoch_iters = len(train_dataset) / \
            config.data_loader.train.batch_size

        return train_dataset, val_dataset, shapes_dataset, train_epoch_iters

    def _evaluate_bs(self, val_dataloaders):
        # perform loss calculation on the synthetic val set
        # perform beam search on synthetic
        # perform beam search on GT val set
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
        return stat_estimator

    def eval_loop(self, starting_weights):
        # load setup

        self.train_state.best_weights = starting_weights
        dl_specs = self.dl_specs
        lang_conf = self.lang_conf
        cur_iter = self.train_state.cur_iter
        epoch = self.train_state.cur_epoch

        # Load the dataset
        val_dataloaders = super(
            PLADTrainer, self).get_val_dataloaders(dl_specs, lang_conf)
        _, _, shapes_val_bs = val_dataloaders

        # create the loader
        bpds = BestProgramDataStruct(self.bpds_config)

        rewriter_list = self.inference_specs.rewriters
        n_rewrites = self.inference_specs.n_rewrites
        # repeat the rewriter list
        active_rewriters = rewriter_list * n_rewrites
        if "CG" in active_rewriters:
            rewriter = [
                x for x in self.active_rewriters if x.shorthand == "CG"][0]
            rewriter.load_subexpr_cache(bpds)

        process_start = time.time()
        expression_stats, stat_estimator = self.search_programs(shapes_val_bs)
        self.logger.log_statistics(
            expression_stats, epoch, cur_iter, prefix="train_search_bs")
        new_programs = stat_estimator.get_best_programs(return_score=True)
        bpds.update_programs(new_programs, "BS")

        for rewriter_shorthand in active_rewriters:
            rewriter = [
                x for x in self.active_rewriters if x.shorthand == rewriter_shorthand][0]
            new_programs, log_info = rewriter.rewrite_expressions(
                bpds, shapes_val_bs.dataset, eval_mode=True)
            self.logger.log_statistics(
                log_info, epoch, cur_iter, prefix=f"train_rewrite_{rewriter.shorthand}")
            bpds.update_programs(new_programs, f"{rewriter.shorthand}")

        all_programs = bpds.get_best_programs()
        process_end = time.time()

        # Get evaluator and then resend this as input
        stat_estimator = EvalStatEstimator(self.eval_specs, self.lang_conf,
                                           self.dtype, self.device)
        for iter_ind, batch_dict in enumerate(shapes_val_bs):
            indices = batch_dict["indices"]
            predictions = {}
            for inner_count, ind_str in enumerate(indices):
                predictions[inner_count] = [(0, all_programs[ind_str])]

            occs = batch_dict["occs"]

            stat_estimator.update_from_actions(indices, occs, predictions, actions=False)

        final_metrics = stat_estimator.get_final_metrics()
        expression_stats = stat_estimator.get_stats()
        expression_stats.update(final_metrics)
        expression_stats['Total Inference Time'] = process_end - process_start
        expression_stats['Per Shape Inference Time'] = (process_end - process_start) / len(all_programs)
        self.logger.log_statistics(expression_stats, epoch, cur_iter, prefix="final_program_stats")

        programs_str = stat_estimator.get_best_programs(to_string=True, return_score=True, revert=True)
        self.logger.save_programs(programs_str, cur_iter, tag=None)
