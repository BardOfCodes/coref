import torch as th
from tabulate import tabulate
# from torch.utils.tensorboard import SummaryWriter
import wandb
import cProfile
import functools
import gc
import os
import _pickle as cPickle
from pathlib import Path
import subprocess


SIZE_LIMIT = 75


def return_fmt_str(key, value, key_size, value_size):
    if isinstance(value, float):
        value = f"{value:.4f}"
    value_str = "|" + \
        f" {key}".ljust(key_size, " ") + "|" + \
        f" {value}".ljust(value_size, " ") + "|"
    return value_str


class Logger:

    def __init__(self, config):
        # self.writer = SummaryWriter(config.LOG_DIR)

        self.exp_name = config.name
        self.verbose = config.logger.verbose
        self.use_wandb = config.logger.use_wandb
        self.logdir = config.logger.logdir
        self.manual_axes = True

        if self.use_wandb:
            wandb.init(project=config.logger.project_name,
                       entity=config.logger.user_name,
                       name=config.name,
                       config=config)
            if self.manual_axes:
                wandb.define_metric('log_iters')
                wandb.define_metric("*", step_metric="log_iters")

    def log_statistics(self, statistics, epoch, log_iters, prefix):

        log_dict = {f"{prefix}/{key}": value for key,
                    value in statistics.items()}

        # Print statistics
        if self.verbose:
            self.print_statistics(log_dict, prefix, epoch, log_iters)
        if self.use_wandb:
            if self.manual_axes:
                log_dict["log_iters"] = log_iters
                wandb.log(log_dict)
            else:
                wandb.log(log_dict, step=log_iters)

    def print_statistics(self, statistics, prefix, epoch, log_iter, sort_keys=True):
        header = "|" + "XX".center(SIZE_LIMIT-2, "-") + "|"
        header_stats = f" Experiment: {prefix}_{self.exp_name}, Epoch: {epoch}, Iteration: {log_iter}"
        header_stats = "|" + header_stats.center(SIZE_LIMIT-2, " ") + "|"
        footer = "|" + "XX".center(SIZE_LIMIT-2, "-") + "|"
        header = "\n".join([header, header_stats, footer])

        stat_keys = list(statistics.keys())
        if sort_keys:
            stat_keys = sorted(stat_keys)
        stat_types = set([x.split("/")[0] for x in stat_keys])
        stat_key_size = int((SIZE_LIMIT-3) * 0.75)
        stat_val_size = SIZE_LIMIT - 3 - stat_key_size
        stat_lines = []
        for stat_type in stat_types:
            stat_line = return_fmt_str(
                f" {stat_type}", "", stat_key_size, stat_val_size)
            stat_lines.append(stat_line)
            stat_keys = [x for x in stat_keys if x.startswith(stat_type)]
            stat_dict = {x.split("/")[1]: statistics[x] for x in stat_keys}
            cur_stat_lines = [return_fmt_str(
                f"    {key}", value, stat_key_size, stat_val_size) for key, value in stat_dict.items()]
            stat_lines.extend(cur_stat_lines)

        stat_lines = "\n".join(stat_lines)

        all_stats = "\n".join([header, stat_lines, footer])
        print(all_stats)

    # TODO: Every eval
    def save_programs(self, expression_dict, log_iter, tag=None):
        if tag is None:
            file_name = f"expressions_{log_iter}.pkl"
        else:
            file_name = tag
        # check if directory exists
        Path(self.logdir).mkdir(parents=True, exist_ok=True)
        save_file = os.path.join(self.logdir, file_name)
        gc.disable()
        cPickle.dump(expression_dict, open(save_file, "wb"))
        gc.enable()


def profileit(name=None):
    def inner(func):
        def wrapper(*args, **kwargs):
            prof = cProfile.Profile()
            retval = prof.runcall(func, *args, **kwargs)
            # Note use of name from outer scope
            prof.dump_stats(name)
            return retval
        return wrapper
    return inner

# Reference: https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
