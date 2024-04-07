import argparse
import _pickle as cPickle
import torch as th
import gc
import os
from geolipi.torch_compute.sketcher import Sketcher
# from coref.language.random_data_generator import sample_random_program_set
from coref.language.streaming_data_generator import generate_programs
from configs.subconf.language_spec import LangSpec
import coref.language as language
from pathlib import Path


def main(args):
    n_programs = args.n_programs
    n_prims_set = args.n_prims
    save_dir = os.path.join(args.save_dir, args.lang)
    # make sure save_dir exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if "3D" in args.lang:
        # Temp hack
        resolution = 32
    else:
        resolution = 64
    # Not important
    unimportant_var = 0
    tokenization = "POSTFIX"

    lang_conf = LangSpec(args.lang, resolution, unimportant_var,
                         unimportant_var, unimportant_var, unimportant_var, tokenization)
    n_dims = lang_conf.n_dims
    device = th.device("cuda")
    lang_specs = getattr(language,  lang_conf.name)(n_float_bins=lang_conf.n_float_bins,
                                                    n_integer_bins=lang_conf.n_integer_bins,
                                                    tokenization=lang_conf.tokenization)

    sketcher = sketcher = Sketcher(
        resolution=resolution, device=device, n_dims=n_dims)
    if args.device == 'cuda':
        device = th.device('cuda')
    program_dict = {}
    for n_prims in n_prims_set:
        program_set = generate_programs(n_prims, n_programs, sketcher, lang_conf, lang_specs,
                                        eval_size=256, n_primitive_set=128)
        if program_set[0].device == th.device('cuda'):
            program_set = [x.cpu() for x in program_set]

        program_dict[n_prims] = [str(x) for x in program_set]

        save_file = os.path.join(
            save_dir, f'{args.lang}_synth_programs_{n_prims}.pkl')
        gc.disable()
        with open(save_file, 'wb') as f:
            cPickle.dump(program_dict[n_prims], f, protocol=-1)
        gc.enable()
        print("Saved to ", save_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-programs', type=int, default=2_000)
    parser.add_argument('--n-prims', type=int, nargs='+',
                        default=[2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--lang', type=str, default='HCSG2D')
    parser.add_argument('--save-dir', type=str, default='../../data/')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    main(args)
