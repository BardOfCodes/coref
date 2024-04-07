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
import coref.rewriter as rewriters

from .plad import PLADTrainer


class SIRITrainer(PLADTrainer):

    def __init__(self, config, resume=False):
        super(SIRITrainer, self).__init__(config, resume)
        self.active_rewriter_names = config.siri.rewriters.names
        self.active_rewriters = [getattr(rewriters, name)
                                 for name in self.active_rewriter_names]
        self.active_rewriters = [rewriter(getattr(config.siri.rewriters, name), self.device, self.dtype)
                                 for rewriter, name in zip(self.active_rewriters, self.active_rewriter_names)]

    def search_and_generate(self, search_loader, train_loader, bpds):
        # load model
        epoch = self.train_state.cur_epoch
        cur_iter = self.train_state.cur_iter
        expression_stats, stat_estimator = self.search_programs(search_loader)
        self.logger.log_statistics(
            expression_stats, epoch, cur_iter, prefix="train_search_bs")
        new_programs = stat_estimator.get_best_programs(return_score=True)
        bpds.update_programs(new_programs, "BS")
        for rewriter in self.active_rewriters:
            new_programs, log_info = rewriter.rewrite_expressions(
                bpds, self.train_dataset)
            self.logger.log_statistics(
                log_info, epoch, cur_iter, prefix=f"train_rewrite_{rewriter.shorthand}")
            bpds.update_programs(new_programs, f"{rewriter.shorthand}")

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
