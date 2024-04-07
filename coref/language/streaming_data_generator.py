
import numpy as np
import torch as th
import time
from geolipi.symbolic.combinators import Union, Intersection, Difference
from geolipi.symbolic.primitives_2d import NoParamRectangle2D, NoParamCircle2D, NoParamTriangle2D
from geolipi.symbolic.primitives_3d import NoParamCuboid3D, NoParamSphere3D

from geolipi.symbolic.transforms_2d import Translate2D, Scale2D, EulerRotate2D
from geolipi.symbolic.transforms_3d import Translate3D, Scale3D, EulerRotate3D

from geolipi.torch_compute.evaluate_expression import expr_to_sdf
from geolipi.torch_compute.sdf_operators import sdf_union, sdf_intersection, sdf_difference
from geolipi.torch_compute.sketcher import Sketcher
from geolipi.symbolic.types import TRANSLATE_TYPE, SCALE_TYPE, PRIM_TYPE, ROTATE_TYPE
from geolipi.symbolic.transforms_2d import (
    ReflectX2D, ReflectY2D, RotationSymmetry2D, TranslationSymmetryX2D, TranslationSymmetryY2D)
from geolipi.symbolic.transforms_3d import (ReflectX3D, ReflectY3D, ReflectZ3D,
                                            RotationSymmetryX3D, RotationSymmetryY3D, RotationSymmetryZ3D,
                                            TranslationSymmetryX3D, TranslationSymmetryY3D, TranslationSymmetryZ3D)

from geolipi.torch_compute.compile_expression import create_compiled_expr
from geolipi.torch_compute.batch_evaluate_sdf import create_evaluation_batches, batch_evaluate

from .primal.primal_primitives import PrimalCuboid3D, PrimalSphere3D
from .primal.primal_primitives import PrimalRectangle2D, PrimalCircle2D, PrimalTriangle2D
from .constants import (MIN_SCALE, MAX_SCALE, MIN_TRANSLATE, MAX_TRANSLATE, MIN_ROTATE, MAX_ROTATE,
                        SCALE_PROB_THRES, TRANSLATE_PROB_THRES, ROTATE_PROB_THRES,
                        MACRO_PROB_THRES, HIER_TRANS_PROB_THRES, MAX_TRIES,
                        BETA_DISTR_1, BETA_DISTR_2, SCALE_EPSILON,
                        EXPR_MIN_OCC_THRES, EXPR_MAX_OCC_THRES, EXPR_DIFF_THRES,
                        SCALE_MEAN, SCALE_STD, TRANSLATE_MEAN, TRANSLATE_STD, ROTATE_MEAN, ROTATE_STD,
                        PRIM_MIN_OCC_THRES, PRIM_MAX_OCC_THRES)

from typing import get_args
from .random_data_generator import convert_base_to_primal_prim, insert_macros, insert_transforms, check_modifier_validity, check_combinator_validity

TRANSLATE_TYPE, SCALE_TYPE, ROTATE_TYPE = get_args(
    TRANSLATE_TYPE), get_args(SCALE_TYPE), get_args(ROTATE_TYPE)

COMBINATOR_SET = [Union, Intersection, Difference]

PRIMITIVE_SETS = {
    2: [NoParamRectangle2D, NoParamCircle2D, NoParamTriangle2D],
    3: [NoParamCuboid3D, NoParamSphere3D]
}
TRANSFORM_SET = {
    2: [Translate2D, EulerRotate2D, Scale2D],
    3: [Translate3D, EulerRotate3D, Scale3D]
}
SDF_COMBINATOR_MAP = {
    Union: sdf_union,
    Intersection: sdf_intersection,
    Difference: sdf_difference
}
BASE_TO_PRIMAL_MAPPER = {
    2: {
        NoParamRectangle2D: PrimalRectangle2D,
        NoParamCircle2D: PrimalCircle2D,
        NoParamTriangle2D: PrimalTriangle2D
    },
    3: {
        NoParamCuboid3D: PrimalCuboid3D,
        NoParamSphere3D: PrimalSphere3D
    }
}

MACRO_SET = {
    2: [ReflectX2D, ReflectY2D, RotationSymmetry2D, TranslationSymmetryX2D, TranslationSymmetryY2D],
    3: [ReflectX3D, ReflectY3D, ReflectZ3D, RotationSymmetryX3D, RotationSymmetryY3D, RotationSymmetryZ3D,
        TranslationSymmetryX3D, TranslationSymmetryY3D, TranslationSymmetryZ3D]
}

# prefixed sizes to combine
UPSTEP_DICT = {
    2: ([(1, 1),],),
    3: ([(1, 1), (1, 2),],),
    4: ([(1, 1), (2, 2),],
        [(1, 1), (1, 2), (1, 3)],),
    5: ([(1, 1), (1, 2), (2, 3)],
        [(1, 1), (1, 2), (1, 3), (1, 4)]),
    6: ([(1, 1), (2, 2), (2, 4)],
        [(1, 1), (1, 2), (3, 3)],
        [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]),
    7: ([(1, 1), (1, 2), (1, 3), (3, 4)],
        [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]),
    8: ([(1, 1), (2, 2), (4, 4)],
        [(1, 1), (2, 2), (2, 4), (2, 6)],
        [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)]),
    9: ([(1, 1), (1, 2), (3, 3), (3, 6)],
        [(1, 1), (1, 2), (1, 3), (1, 4), (4, 5)],
        [(1, 1), (1, 2), (2, 3), (2, 5), (2, 7)],
        [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8)]),
    10: ([(1, 1), (1, 2), (2, 3), (5, 5)],
         [(1, 1), (1, 2), (1, 3), (3, 4), (3, 7)],
         [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (5, 5)],
         [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)]),
    11: ([(1, 1), (1, 2), (2, 3), (5, 6)],
         [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (5, 6)],
         [(1, 1), (1, 2), (1, 3), (3, 4), (4, 7)],
         [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10)])
}


def generate_programs(n_prims, n_programs, sketcher, lang_conf, lang_specs, eval_size, n_primitive_set=512,
                      gen_prog_factor=5):
    program_set = []
    start_time = time.time()
    while (len(program_set) < n_programs):
        print(
            f"Generated {len(program_set)} of {n_programs} programs with {n_prims} primitives.")
        programs = sample_random_program(n_prims, n_primitive_set, sketcher, lang_specs, lang_conf,
                                         max_gen_programs=n_primitive_set*gen_prog_factor, eval_size=eval_size)
        program_set.extend(programs)
        end_time = time.time()
        print(
            f"Generated {len(program_set)} programs in {end_time - start_time} seconds.")
    program_set = program_set[:n_programs]
    return program_set


def add_transform_macro_meta(*args, **kwargs):
    return add_transform_macro(*args, **kwargs)


def generate_only_primitives(n_programs, sketcher, lang_conf, lang_specs, eval_size, n_primitive_set=512,
                             gen_prog_factor=5):

    add_rotate = lang_conf.add_rotate
    convert_to_primal = lang_conf.convert_to_primal
    add_macros = lang_conf.has_macros
    n_dims = sketcher.n_dims
    device = sketcher.device

    program_set = []
    start_time = time.time()
    while (len(program_set) < n_programs):
        primitive_set, sdf_set = sample_random_primitives(n_primitive_set, n_dims, device, sketcher, add_rotate,
                                                          add_macros=add_macros, convert_to_primal=convert_to_primal,
                                                          lang_specs=lang_specs, eval_size=eval_size)
        # Batch check primitives
        primitive_set, sdf_set = batch_check_primitive_validity(
            primitive_set, sdf_set)
        for expr_1 in primitive_set:
            expression = expr_1
            # update expression
            expression, _ = add_transform_macro_meta(
                expression, sketcher, lang_conf)
            program_set.append(expression.cpu())
        end_time = time.time()
        print(
            f"Generated {len(program_set)} programs in {end_time - start_time} seconds.")
    return program_set


def sample_random_program(n_prims, n_primitive_set, sketcher, lang_specs, lang_conf,
                          eval_size=512, max_gen_programs=1024):

    add_rotate = lang_conf.add_rotate
    convert_to_primal = lang_conf.convert_to_primal
    add_macros = lang_conf.has_macros
    n_dims = sketcher.n_dims
    device = sketcher.device

    upstep_seqs = UPSTEP_DICT[n_prims]
    upstep_seqs_probs = [1/len(x) for x in upstep_seqs]
    upstep_seqs_probs = np.array(upstep_seqs_probs)
    upstep_seqs_probs = upstep_seqs_probs/np.sum(upstep_seqs_probs)
    upstep_seq_id = np.random.choice(
        range(len(upstep_seqs)), p=upstep_seqs_probs)
    upstep_seq = upstep_seqs[upstep_seq_id]

    all_programs = []
    last_step = len(upstep_seq) - 1
    primitive_set = None
    for cur_step, set_choices in enumerate(upstep_seq):
        if primitive_set is None:
            primitive_set, sdf_set = sample_random_primitives(n_primitive_set, n_dims, device, sketcher, add_rotate,
                                                              add_macros=add_macros, convert_to_primal=convert_to_primal,
                                                              lang_specs=lang_specs, eval_size=eval_size)
            # Batch check primitives
            primitive_set, sdf_set = batch_check_primitive_validity(
                primitive_set, sdf_set)
            expr_length_set = np.ones(len(primitive_set), dtype=np.int32)
            # initially its the same set.
            sdf_set_a = sdf_set
            sdf_set_b = sdf_set
            primitive_set_a = primitive_set
            primitive_set_b = primitive_set
            expr_len_set_a = expr_length_set
            expr_len_set_b = expr_length_set
            prim_id_set_a = [set([i]) for i in range(len(primitive_set))]
            prim_id_set_b = [set([i]) for i in range(len(primitive_set))]
        else:
            # calculate sdfs from the set of new_primitive_set
            # need the SDF set
            if eval_a:
                primitive_set_a = new_primitive_set_a
                expr_len_set_a = np.array(new_expr_len_set_a)
                prim_id_set_a = new_prim_id_set_a
                sdf_set_a = batch_evaluate_expressions(
                    primitive_set_a, sketcher, lang_specs, eval_size=eval_size)

            primitive_set_b = new_primitive_set_b
            expr_len_set_b = np.array(new_expr_len_set_b)
            prim_id_set_b = new_prim_id_set_b
            # eval a only if necessary
            sdf_set_b = batch_evaluate_expressions(
                primitive_set_b, sketcher, lang_specs, eval_size=eval_size)

        # check all combinations
        valid_op_set = get_valid_ops_mask(sdf_set_a, sdf_set_b)
        valid_op_set = valid_op_set.cpu().numpy()
        occ_valid_indices = np.stack(np.where(valid_op_set), -1)
        # create_size_set:
        sizes_1 = expr_len_set_a[occ_valid_indices[:, 0]]
        sizes_2 = expr_len_set_b[occ_valid_indices[:, 1]]
        output_size = sizes_1 + sizes_2
        # calculate intersection
        prim_sets_a = [prim_id_set_a[i].copy()
                       for i in occ_valid_indices[:, 0]]
        prim_sets_b = [prim_id_set_b[i].copy()
                       for i in occ_valid_indices[:, 1]]
        intersection_set = [prim_sets_a[i].intersection(
            prim_sets_b[i]) for i in range(len(prim_sets_a))]
        empty_set_inter = np.array(
            [len(x) == 0 for x in intersection_set], dtype=bool)

        if cur_step < last_step:
            # Just try the next step.
            next_upstep = upstep_seq[cur_step + 1]
            pool_a_size, pool_b_size = next_upstep
            prev_a_size, prev_b_size = set_choices
            if pool_a_size == prev_a_size:
                eval_a = False
            elif pool_a_size == prev_b_size:
                eval_a = True
                new_primitive_set_a = primitive_set_b
                new_expr_len_set_a = expr_len_set_b.tolist()
                new_prim_id_set_a = prim_id_set_b
            else:
                eval_a = True
                new_variants = generate_new_variants(output_size, pool_a_size, empty_set_inter,
                                                     n_primitive_set, occ_valid_indices,
                                                     primitive_set_a, primitive_set_b,
                                                     prim_id_set_a, prim_id_set_b,
                                                     sketcher, lang_conf)
                new_primitive_set_a, new_expr_len_set_a, new_prim_id_set_a = new_variants
                if new_primitive_set_a is None:
                    break

            # For B:
            new_variants = generate_new_variants(output_size, pool_b_size, empty_set_inter,
                                                 n_primitive_set, occ_valid_indices,
                                                 primitive_set_a, primitive_set_b,
                                                 prim_id_set_a, prim_id_set_b,
                                                 sketcher, lang_conf)
            new_primitive_set_b, new_expr_len_set_b, new_prim_id_set_b = new_variants
            if new_primitive_set_b is None:
                break

        elif cur_step == last_step:
            # Extract the valid ones.
            extract_valid = output_size == n_prims
            extract_cand_ids = np.stack(
                np.where(extract_valid & empty_set_inter), -1)[..., 0]
            cur_size = min(len(extract_cand_ids), max_gen_programs)
            if cur_size == 0:
                break
            extract_selected = get_per_op_valid_indices(
                occ_valid_indices, extract_cand_ids, cur_size)

            for selected_ind in extract_selected:
                sel_op = occ_valid_indices[selected_ind]
                expr_1 = primitive_set_a[sel_op[0]]
                expr_2 = primitive_set_b[sel_op[1]]
                op = COMBINATOR_SET[sel_op[2]]
                expression = op(expr_1, expr_2)
                # update expression
                expression, _ = add_transform_macro_meta(
                    expression, sketcher, lang_conf)
                # Due to memory reasons here it is converted to CPU
                all_programs.append(expression.cpu())

    return all_programs


def generate_new_variants(output_size, pool_size, empty_set_inter,
                          n_primitive_set, occ_valid_indices,
                          primitive_set_a, primitive_set_b,
                          prim_id_set_a, prim_id_set_b,
                          sketcher, lang_conf):
    set_b_valid = output_size == pool_size
    set_b_cand_ids = np.stack(
        np.where(set_b_valid & empty_set_inter), -1)[..., 0]
    cur_size = min(len(set_b_cand_ids), n_primitive_set)
    # Try to select cur_size/3 from each channel
    if cur_size == 0:
        return None, None, None
    set_b_selected = get_per_op_valid_indices(
        occ_valid_indices, set_b_cand_ids, cur_size)

    new_primitive_set_b, new_expr_len_set_b, new_prim_id_set_b = [], [], []
    for selected_ind in set_b_selected:
        sel_op = occ_valid_indices[selected_ind]

        expr_1 = primitive_set_a[sel_op[0]]
        expr_2 = primitive_set_b[sel_op[1]]
        op = COMBINATOR_SET[sel_op[2]]
        expression = op(expr_1, expr_2)
        # update expression
        expression, _ = add_transform_macro_meta(
            expression, sketcher, lang_conf)
        expr_size = output_size[selected_ind]
        prim_set = prim_id_set_a[sel_op[0]].copy().union(
            prim_id_set_b[sel_op[1]])
        new_primitive_set_b.append(expression)
        new_expr_len_set_b.append(expr_size)
        new_prim_id_set_b.append(prim_set)
    return new_primitive_set_b, new_expr_len_set_b, new_prim_id_set_b


def get_per_op_valid_indices(occ_valid_indices, set_b_cand_ids, cur_size):
    size_per_op = cur_size // 3
    set_b_selected = []
    for i in range(3):
        cur_set = [x for x in set_b_cand_ids if occ_valid_indices[x, 2] == i]
        cur_size_per_op = min(len(cur_set), size_per_op)
        cur_selected = np.random.choice(
            cur_set, size=cur_size_per_op, replace=False)
        set_b_selected.extend(cur_selected)
    # if its less then sample more?
    remaining = cur_size - len(set_b_selected)
    if remaining > 0:
        cur_set = [x for x in set_b_cand_ids if occ_valid_indices[x, 2] == 0]
        cur_remaining = min(len(cur_set), remaining)
        cur_selected = np.random.choice(
            cur_set, size=cur_remaining, replace=False)
        set_b_selected.extend(cur_selected)
    return set_b_selected


def sample_random_primitives(n_prims, n_dims, device, sketcher, add_rotate,
                             scale_prob_thres=SCALE_PROB_THRES,
                             translate_prob_thres=TRANSLATE_PROB_THRES,
                             rotate_prob_thres=ROTATE_PROB_THRES,
                             add_macros=False, macro_prob_thres=MACRO_PROB_THRES,
                             max_tries=MAX_TRIES, convert_to_primal=True,
                             lang_specs=None, eval_size=2048):
    # sample n_sq and keep only the ones with valid sdfs
    assert not (
        add_macros and convert_to_primal), "Cannot convert macros to primals (without resolution)."

    scale_probability = np.random.uniform(0, 1, size=n_prims)
    scale_params = np.random.beta(
        BETA_DISTR_1, BETA_DISTR_2, size=(n_prims, n_dims))
    scale_params = SCALE_MEAN + SCALE_STD * \
        (scale_params * np.random.choice((1, -1), size=(n_prims, n_dims)))
    scale_params = np.clip(scale_params, MIN_SCALE +
                           SCALE_EPSILON, MAX_SCALE - SCALE_EPSILON)
    scale_params = th.tensor(scale_params, dtype=th.float32, device=device)

    translate_probability = np.random.uniform(0, 1, size=n_prims)
    translate_params = np.random.beta(
        BETA_DISTR_1, BETA_DISTR_2, size=(n_prims, n_dims))
    translate_params = TRANSLATE_MEAN + TRANSLATE_STD * \
        (translate_params * np.random.choice((1, -1), size=(n_prims, n_dims)))
    translate_params = np.clip(translate_params, MIN_TRANSLATE, MAX_TRANSLATE)
    translate_params = th.tensor(
        translate_params, dtype=th.float32, device=device)

    if add_rotate:
        rotate_probability = np.random.uniform(0, 1, size=n_prims)
        rotate_dim = n_dims if n_dims == 3 else 1
        rotate_params = np.random.beta(
            BETA_DISTR_1, BETA_DISTR_2, size=(n_prims, rotate_dim))
        rotate_params = ROTATE_MEAN + ROTATE_STD * \
            (rotate_params * np.random.choice((1, -1), size=(n_prims, rotate_dim)))
        rotate_params = np.clip(rotate_params, MIN_ROTATE, MAX_ROTATE)
        rotate_params = th.tensor(
            rotate_params, dtype=th.float32, device=device)

    # All all and randomly remove a few based on %.
    primitives = np.random.choice(PRIMITIVE_SETS[n_dims], size=n_prims)

    primitive_exprs = []
    for index in range(n_prims):
        translate_func = TRANSFORM_SET[n_dims][0]
        rotate_func = TRANSFORM_SET[n_dims][1]
        scale_func = TRANSFORM_SET[n_dims][2]
        if scale_probability[index] < scale_prob_thres:
            scale_tensor = scale_params[index]
            shape_expr = scale_func(primitives[index](), scale_tensor)
        else:
            shape_expr = primitives[index]()
        if add_rotate:
            if rotate_probability[index] < rotate_prob_thres:
                rotate_tensor = rotate_params[index]
                shape_expr = rotate_func(shape_expr, rotate_tensor)
        if translate_probability[index] < translate_prob_thres:
            translate_tensor = translate_params[index]
            shape_expr = translate_func(shape_expr, translate_tensor)

        if add_macros:
            macro_prob = np.random.uniform(0, 1)
            # given failure rate adjust -> add K different nes.
            if macro_prob < macro_prob_thres:
                sdf = expr_to_sdf(shape_expr, sketcher)
                shape_expr, _ = try_modification(
                    shape_expr, sdf, sketcher, insert_macros, max_tries=max_tries)
        if convert_to_primal:
            shape_expr = convert_base_to_primal_prim(
                shape_expr, n_dims, device, add_rotate)
        primitive_exprs.append(shape_expr)
    # Now batch execute them:
    all_sdfs = batch_evaluate_expressions(
        primitive_exprs, sketcher, lang_specs, eval_size)

    return primitive_exprs, all_sdfs


def batch_evaluate_expressions(primitive_exprs, sketcher, lang_specs, eval_size):

    compiled_exprs = []
    for program in primitive_exprs:
        reverted_program = lang_specs.revert_to_base_expr(program)
        reverted_program = reverted_program.to(sketcher.device)
        compiled_expr = create_compiled_expr(
            reverted_program, sketcher=sketcher)
        compiled_exprs.append(compiled_expr)

    all_sdfs = []
    n_chunks = np.ceil(len(compiled_exprs) / eval_size).astype(np.int32)
    for ind in range(n_chunks):
        cur_batch = compiled_exprs[ind * eval_size: (ind+1) * eval_size]
        with th.no_grad():
            eval_batches = create_evaluation_batches(
                cur_batch, convert_to_cuda=True)
            cur_sdf = batch_evaluate(eval_batches, sketcher)
        all_sdfs.append(cur_sdf)

    all_sdfs = th.cat(all_sdfs, dim=0)
    return all_sdfs


def add_transform_macro(expression, sketcher, lang_conf,
                        macro_prob_thres=MACRO_PROB_THRES,
                        hier_trans_prob_thres=HIER_TRANS_PROB_THRES,
                        max_tries=MAX_TRIES):
    # Create a non checking version.
    sdf = None
    if lang_conf.has_macros:
        sdf = expr_to_sdf(expression, sketcher)
        macro_prob = np.random.uniform(0, 1)
        if macro_prob < macro_prob_thres:
            expression, sdf = try_modification(
                expression, sdf, sketcher, insert_macros, max_tries=max_tries)

    if lang_conf.has_htrans:
        if sdf is None:
            sdf = expr_to_sdf(expression, sketcher)
        transform_prob = np.random.uniform(0, 1)
        if transform_prob < hier_trans_prob_thres:
            expression, sdf = try_modification(
                expression, sdf, sketcher, insert_transforms, max_tries=max_tries)
    return expression, sdf


def add_transform_macro_no_check(expression, sketcher, lang_conf,
                                 macro_prob_thres=MACRO_PROB_THRES,
                                 hier_trans_prob_thres=HIER_TRANS_PROB_THRES,
                                 max_tries=MAX_TRIES):
    # Create a non checking version.
    sdf = None
    if lang_conf.has_macros:
        macro_prob = np.random.uniform(0, 1)
        if macro_prob < macro_prob_thres:
            expression = insert_macros(
                expression, sketcher.n_dims, sketcher.device)

    if lang_conf.has_htrans:
        transform_prob = np.random.uniform(0, 1)
        if transform_prob < hier_trans_prob_thres:
            expression = insert_transforms(
                expression, sketcher.n_dims, sketcher.device)
    return expression, sdf


def get_valid_ops_mask(sdf_set_a, sdf_set_b):

    n_a = sdf_set_a.shape[0]
    n_b = sdf_set_b.shape[0]
    inp_shape_a = (sdf_set_a <= 0).unsqueeze(1).expand(-1, n_b, -1)
    inp_shape_b = (sdf_set_b <= 0).unsqueeze(0).expand(n_a, -1, -1)

    unions = th.logical_or(inp_shape_a, inp_shape_b)
    intersections = th.logical_and(inp_shape_a, inp_shape_b)
    differences = th.logical_and(inp_shape_a, ~inp_shape_b)

    union_valid = batch_validate_single_op(unions, inp_shape_a, inp_shape_b)
    intersection_valid = batch_validate_single_op(
        intersections, inp_shape_a, inp_shape_b)
    difference_valid = batch_validate_single_op(
        differences, inp_shape_a, inp_shape_b)

    valid_ops = th.stack(
        [union_valid, intersection_valid, difference_valid], dim=-1)
    return valid_ops


def batch_validate_single_op(union_shape, inp_shape_a, input_shape_b,  min_occ_thres=EXPR_MIN_OCC_THRES,
                             diff_thres=EXPR_DIFF_THRES, max_occ_thres=EXPR_MAX_OCC_THRES):
    total = union_shape[0, 0].nelement()
    val_1 = th.sum(union_shape, dim=-1) / total
    val_2 = th.sum(union_shape ^ inp_shape_a, dim=-1) / \
        th.sum(union_shape, dim=-1)
    val_3 = th.sum(union_shape ^ input_shape_b, dim=-1) / \
        th.sum(union_shape, dim=-1)

    cond_1 = (min_occ_thres < val_1) & (val_1 < max_occ_thres)
    cond_2 = val_2 > diff_thres
    cond_3 = val_3 > diff_thres
    validity = (cond_1 & cond_2) & (cond_3)
    return validity


def try_modification(expr, sdf, sketcher, mod_func, max_tries):

    n_dims = sketcher.n_dims
    device = sketcher.device
    success = False
    count = 0
    while (not success):
        mod_expr = mod_func(expr, n_dims, device)
        # execute the mod_expr
        mod_sdf = expr_to_sdf(mod_expr, sketcher)
        success = check_modifier_validity(mod_sdf, sdf)
        count += 1
        if count > max_tries:
            break
    if success:
        expr = mod_expr
        sdf = mod_sdf
    return expr, sdf


def insert_macros(expr, n_dims, device):
    macros = MACRO_SET[n_dims]
    macro = np.random.choice(macros)
    if issubclass(macro, (ReflectX2D, ReflectY2D, ReflectX3D, ReflectY3D, ReflectZ3D)):
        mod_expr = macro(expr)
    else:
        # sample the number of repetitions
        count = np.round(2 + 3 * np.random.beta(BETA_DISTR_1, BETA_DISTR_2))
        count = int(count)
        count = th.tensor(count, dtype=th.long, device=device)

        # sample the delta
        delta = np.random.beta(2, 3) / (count.item() - 1)
        delta = np.random.choice((1, -1)) * delta
        if issubclass(macro, (RotationSymmetry2D, RotationSymmetryX3D, RotationSymmetryY3D, RotationSymmetryZ3D)):
            delta = np.pi * delta
        delta = th.tensor(delta, dtype=th.float32, device=device)
        mod_expr = macro(expr, delta, count)

    return mod_expr


def insert_transforms(expr, n_dims, device):
    # simplest version - uniform over all the options:
    transforms = TRANSFORM_SET[n_dims]
    transform = np.random.choice(transforms)
    if issubclass(transform, TRANSLATE_TYPE):
        transform_params = np.random.beta(
            BETA_DISTR_1, BETA_DISTR_2, size=(n_dims))
        transform_params = 0 + transform_params * \
            np.random.choice((1, -1), size=(n_dims))
        transform_params = np.clip(
            transform_params, MIN_TRANSLATE, MAX_TRANSLATE)
    elif issubclass(transform, SCALE_TYPE):
        transform_params = np.random.beta(
            BETA_DISTR_1, BETA_DISTR_2, size=(n_dims))
        transform_params = 0.5 + transform_params * \
            np.random.choice((1, -1), size=(n_dims))
        transform_params = np.clip(
            transform_params, MIN_SCALE + SCALE_EPSILON, MAX_SCALE - SCALE_EPSILON)
    elif issubclass(transform, ROTATE_TYPE):
        rotate_dim = n_dims if n_dims == 3 else 1
        transform_params = np.random.beta(
            BETA_DISTR_1, BETA_DISTR_2, size=(rotate_dim))
        transform_params = 0 + np.pi * transform_params * \
            np.random.choice((1, -1), size=(rotate_dim))
        transform_params = np.clip(transform_params, MIN_ROTATE, MAX_ROTATE)

    transform_tensor = th.tensor(
        transform_params, dtype=th.float32, device=device)
    shape_expr = transform(expr, transform_tensor)
    return shape_expr


def batch_check_primitive_validity(primitive_set, sdf_set,
                                   min_occ_thres=PRIM_MIN_OCC_THRES,
                                   max_occ_thres=PRIM_MAX_OCC_THRES):
    valid_primitives = []
    valid_sdfs = []
    total = sdf_set[0].nelement()
    all_val = th.sum(sdf_set <= 0, dim=-1)/total
    for ind, primitive in enumerate(primitive_set):
        val = all_val[ind]
        sdf = sdf_set[ind]
        if min_occ_thres < val < max_occ_thres:
            valid_primitives.append(primitive)
            valid_sdfs.append(sdf)
    valid_sdfs = th.stack(valid_sdfs, dim=0)
    return valid_primitives, valid_sdfs
