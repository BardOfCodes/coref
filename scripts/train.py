import os
from pathlib import Path
import sys
import torch as th
import argparse
from wacky import load_config_file
from procXD import SketchBuilder
import coref.trainer as trainer
from coref.utils.logger import get_git_revision_hash

from coref.utils.notification_utils import SlackNotifier

if __name__ == '__main__':

    th.backends.cudnn.benchmark = True
    try:
        th.multiprocessing.set_start_method('spawn')
        th.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError:
        pass
    os.system("CUDA_LAUNCH_BLOCKING=1")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str,
                        default="./configs/pretrain.py")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--vis-conf', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-rewrite-check', action='store_false')

    args, reminder_args = parser.parse_known_args()
    if args.debug:
        reminder_args.extend(["--debug"])
    config = load_config_file(args.config_file, reminder_args)
    git_id = get_git_revision_hash()
    config.git_id = git_id

    if not args.no_rewrite_check and not args.debug:
        # check the followings: the log directory, model save file.
        if os.path.exists(config.logger.log_dir):
            print("Log directory already exists. Please check.")
            sys.exit(1)
        if os.path.exists(config.model.save_dir):
            print("Model save file already exists. Please check.")
            sys.exit(1)

    if args.vis_conf:
        G = config.to_graph()
        log_dir = config.logger.log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        save_file = os.path.join(log_dir, "config.excalidraw")
        sketch_builder = SketchBuilder()
        sketch_builder.render_stack_sketch(G, stacking="vertical")
        sketch_builder.export_to_file(save_path=save_file)
        del sketch_builder

    experiment_proc = getattr(trainer, config.trainer.name)
    experiment = experiment_proc(config, resume=args.resume)
    notif = SlackNotifier(config.name, config.notification)
    try:
        notif.start_exp()
        experiment.start_experiment()
    except Exception as ex:
        notif.exp_failed(ex)
        raise ex
