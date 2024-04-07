
import os
import numpy as np
import scipy.stats as spyst
from wacky import CfgNode as CN

validation_shapes = CN()
validation_shapes.mode = "VALIDATION"
validation_shapes.name = "GTShapesDataset"
validation_shapes.valid_prim_sizes = []
validation_shapes.sampling_rate_per_size = []
validation_shapes.workers = 0


def PretrainDataLoaderSpec(data_dir, lang_conf, epoch_size, train_batch_size,
                           validation_batch_size, data_gen_mode,
                           train_n_programs, validation_n_programs, target=None):
    config = CN()
    # For Synth
    config.bake_dir = os.path.join(data_dir, f"{lang_conf.name}")

    config.resolution = lang_conf.resolution
    config.epoch_size = epoch_size
    config.train_on_quantized = False
    # TODO: DO speed ablation on this
    config.collator_eval_size = 256  # 128
    if "PCSG" in lang_conf.name:
        valid_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        if "2D" in lang_conf.name:
            eval_sizes = [512, 512, 256, 256, 256, 128, 128, 128, 128, 128]
            n_primitves = [512, 512, 256, 256, 256, 128, 128, 128, 128, 128]
        elif "3D" in lang_conf.name:
            eval_sizes = [128, ] * len(valid_sizes)
            n_primitves = [128, ] * len(valid_sizes)
    elif ("MCSG" in lang_conf.name) or ("HCSG" in lang_conf.name):
        valid_sizes = [2, 3, 4, 5, 6, 7, 8]

        if "2D" in lang_conf.name:
            eval_sizes = [512, 512, 256, 256, 256, 128, 128]
            n_primitves = [512, 512, 256, 256, 256, 128, 128]
        elif "3D" in lang_conf.name:
            eval_sizes = [128, ] * len(valid_sizes)
            n_primitves = [128, ] * len(valid_sizes)
    min_size = min(valid_sizes) - 1
    max_size = max(valid_sizes) + 1

    beta_distr = spyst.beta(2, 3.5)
    complex_values = [beta_distr.pdf(
        (x-min_size)/(max_size-min_size)) for x in valid_sizes]
    sampling_rate_per_size = [x/np.sum(complex_values) for x in complex_values]

    config.train = CN()
    config.train.mode = "TRAIN"
    config.train.name = "SynthDataset"
    config.train.valid_prim_sizes = valid_sizes
    config.train.sampling_rate_per_size = sampling_rate_per_size
    config.train.workers = 0
    config.train.batch_size = train_batch_size
    config.train.data_dir = data_dir
    config.train.n_programs = train_n_programs

    config.validation_synth = CN()
    config.validation_synth.mode = "VALIDATION"
    config.validation_synth.name = "SynthDataset"
    config.validation_synth.valid_prim_sizes = valid_sizes
    config.validation_synth.sampling_rate_per_size = sampling_rate_per_size
    config.validation_synth.workers = 0
    config.validation_synth.batch_size = validation_batch_size
    config.validation_synth.data_dir = data_dir
    config.validation_synth.n_programs = validation_n_programs

    config.validation_shapes = validation_shapes.clone()
    config.validation_shapes.n_programs = validation_n_programs
    config.validation_shapes.batch_size = validation_batch_size
    if "3D" in lang_conf.name:
        if target == "DEFAULT":
            config.validation_shapes.data_dir = os.path.join(
                data_dir, "3d_csg", "data")
            config.validation_shapes.category_labels = [
                '03001627_chair', '04379243_table', '02828884_bench', '04256520_couch']
        else:
            config.validation_shapes.data_dir = os.path.join(
                data_dir, "3DCoMPaT")
            config.validation_shapes.category_labels = ["3DCoMPaT"]
    elif "2D" in lang_conf.name:

        config.validation_shapes.data_dir = os.path.join(data_dir, "2d_csg")
        config.validation_shapes.category_labels = []

    data_gen = CN()
    data_gen.mode = data_gen_mode
    data_gen.cached_form = True
    data_gen.batch_uncached = True
    data_gen.regen_interval = 10
    data_gen.eval_sizes = eval_sizes
    data_gen.n_primitives = n_primitves

    config.train.data_gen = data_gen.clone()
    data_gen.regen_interval = 1000
    config.validation_synth.data_gen = data_gen.clone()
    config.validation_shapes.data_gen = data_gen.clone()

    config.validation_synth.data_gen.mode = "PRECOMPILED"
    config.validation_shapes.data_gen.mode = "PRECOMPILED"

    return config


def PLADDataLoaderSpec(data_dir, lang_conf, epoch_size, train_batch_size, validation_batch_size,
                       train_n_programs=1000, validation_n_programs=1000, target=None):

    config = PretrainDataLoaderSpec(data_dir, lang_conf, epoch_size, train_batch_size,
                                    validation_batch_size, data_gen_mode="PRECOMPILED",
                                    train_n_programs=train_n_programs, validation_n_programs=validation_n_programs,
                                    target=target)
    # For Synth
    config.train.name = "PLADShapesDataset"

    config.search = CN()
    config.search.mode = "TRAIN"
    config.search.batch_size = validation_batch_size
    config.search.sampling_rate_per_size = [
        3746/13998., 1816/13998., 3173/13998., 5263/13998.]
    # This doesn't matter for now.
    config.search.workers = 0
    config.search.data_gen = config.train.data_gen.clone()

    if "3D" in lang_conf.name:
        if target == "DEFAULT":
            labels = ['03001627_chair', '04379243_table',
                      '02828884_bench', '04256520_couch']
            data_dir = os.path.join(data_dir, "3d_csg", "data")
        else:
            labels = ["3DCoMPaT"]
            data_dir = os.path.join(data_dir, "3DCoMPaT")
    elif "2D" in lang_conf.name:
        labels = []
        data_dir = os.path.join(data_dir, "2d_csg")
    config.search.category_labels = labels
    config.search.data_dir = data_dir

    config.train.category_labels = labels
    config.train.data_dir = data_dir

    config.validation_shapes.category_labels = labels
    config.validation_shapes.data_dir = data_dir

    return config
