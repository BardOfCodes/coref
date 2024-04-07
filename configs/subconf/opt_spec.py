from wacky import CfgNode as CN


def OptimizerSpec():
    opt_config = CN()
    # opt_config.name = "Adam"
    opt_config.name = "AdamW"
    opt_config.kwargs = CN()
    opt_config.kwargs.lr = 2e-4
    opt_config.kwargs.weight_decay = 0.001

    scheduler_config = CN()
    # scheduler_config.name = "CosineWarmupScheduler"
    scheduler_config.name = "NoScheduler"
    # Note: Scheduler is epoch based
    scheduler_config.kwargs = CN()
    scheduler_config.kwargs.rate = 0.05
    return opt_config, scheduler_config
