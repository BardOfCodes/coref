import random
import numpy as np
import torch as th
from geolipi.symbolic.base import GLFunction
from geolipi.symbolic.combinators import Union, Intersection, Difference
from geolipi.symbolic.primitives_2d import NoParamRectangle2D, NoParamCircle2D, NoParamTriangle2D
from geolipi.symbolic.primitives_3d import NoParamCuboid3D, NoParamSphere3D

from geolipi.symbolic.transforms_2d import Translate2D, Scale2D, EulerRotate2D
from geolipi.symbolic.transforms_3d import Translate3D, Scale3D, EulerRotate3D

from geolipi.torch_compute.deprecated import expr_to_sdf
from geolipi.torch_compute.sdf_operators import sdf_union, sdf_intersection, sdf_difference
from geolipi.torch_compute.sketcher import Sketcher
from geolipi.symbolic.symbol_types import TRANSLATE_TYPE, SCALE_TYPE, PRIM_TYPE, ROTATE_TYPE
from geolipi.symbolic.transforms_2d import (
    ReflectX2D, ReflectY2D, RotationSymmetry2D, TranslationSymmetryX2D, TranslationSymmetryY2D)
from geolipi.symbolic.transforms_3d import (ReflectX3D, ReflectY3D, ReflectZ3D,
                                            RotationSymmetryX3D, RotationSymmetryY3D, RotationSymmetryZ3D,
                                            TranslationSymmetryX3D, TranslationSymmetryY3D, TranslationSymmetryZ3D)

from .primal.primal_primitives import PrimalCuboid3D, PrimalSphere3D
from .primal.primal_primitives import PrimalRectangle2D, PrimalCircle2D, PrimalTriangle2D
from .primal.primal_primitives import PRIMAL_TYPE
from .constants import (MIN_SCALE, MAX_SCALE, MIN_TRANSLATE, MAX_TRANSLATE, MIN_ROTATE, MAX_ROTATE,
                        SCALE_PROB_THRES, TRANSLATE_PROB_THRES, ROTATE_PROB_THRES,
                        MACRO_PROB_THRES, HIER_TRANS_PROB_THRES, MAX_TRIES,
                        BETA_DISTR_1, BETA_DISTR_2, SCALE_EPSILON,
                        EXPR_MIN_OCC_THRES, EXPR_MAX_OCC_THRES, EXPR_DIFF_THRES,
                        SCALE_MEAN, SCALE_STD, TRANSLATE_MEAN, TRANSLATE_STD, ROTATE_MEAN, ROTATE_STD,
                        PRIM_MIN_OCC_THRES, PRIM_MAX_OCC_THRES)

# TODO: Remove use of random and use np.random only.

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
PRIMAL_MINMAX_MAPPER = {
    0: (MIN_TRANSLATE, MAX_TRANSLATE),
    1: (MIN_SCALE, MAX_SCALE),
    2: (MIN_ROTATE, MAX_ROTATE)
}
# Create a quantize function - slow :(


def quantize(expression, n_bins):
    expr_class = type(expression)
    new_args = []
    for arg in expression.args:
        if isinstance(arg, GLFunction):
            arg = quantize(arg, n_bins)
        else:
            if arg in expression.lookup_table:
                if issubclass(expr_class, TRANSLATE_TYPE):
                    min_val = MIN_TRANSLATE
                    max_val = MAX_TRANSLATE
                elif issubclass(expr_class, SCALE_TYPE):
                    min_val = MIN_SCALE
                    max_val = MAX_SCALE
                elif issubclass(expr_class, ROTATE_TYPE):
                    min_val = MIN_ROTATE
                    max_val = MAX_ROTATE
                elif issubclass(expr_class, PRIMAL_TYPE):
                    arg_index = expression.args.index(arg)
                    min_val, max_val = PRIMAL_MINMAX_MAPPER[arg_index]
                tensor = expression.lookup_table[arg]
                tensor_index = (tensor - min_val) / \
                    (max_val - min_val) * (n_bins - 1)
                tensor_index = tensor_index.round()
                tensor = tensor_index * (max_val - min_val) + min_val
                arg = tensor
        new_args.append(arg)
    new_expression = expr_class(*new_args)
    return new_expression


def convert_base_to_primal_prim(expression, n_dims, device, add_rotate):

    translate_param = th.zeros(n_dims, dtype=th.float32, device=device)
    scale_param = th.ones(n_dims, dtype=th.float32, device=device)
    if n_dims == 3:
        rotate_param = th.zeros(n_dims, dtype=th.float32, device=device)
    elif n_dims == 2:
        rotate_param = th.zeros(1, dtype=th.float32, device=device)
    primal_type = None
    parser_list = [expression]
    while (parser_list):
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, TRANSLATE_TYPE):
            translate_param += cur_expr.lookup_table[cur_expr.args[1]]
        elif isinstance(cur_expr, SCALE_TYPE):
            scale_param *= cur_expr.lookup_table[cur_expr.args[1]]
        elif isinstance(cur_expr, ROTATE_TYPE):
            # Note: This is not the correct way to do this.
            rotate_param += cur_expr.lookup_table[cur_expr.args[1]]
        elif isinstance(cur_expr, PRIM_TYPE):
            primal_type = BASE_TO_PRIMAL_MAPPER[n_dims][type(cur_expr)]
            break
        else:
            raise ValueError(f"Cannot convert {cur_expr} to primal primitive.")
        parser_list.append(cur_expr.args[0])
    if primal_type is None:
        raise ValueError(f"Cannot convert {expression} to primal primitive.")
    if add_rotate:
        args = (translate_param, scale_param, rotate_param)
    else:
        args = (translate_param, scale_param)
    primal_prim = primal_type(*args)
    return primal_prim


def sample_random_primitives(n_prims, n_dims, device, sketcher, add_rotate,
                             scale_prob_thres=SCALE_PROB_THRES,
                             translate_prob_thres=TRANSLATE_PROB_THRES,
                             rotate_prob_thres=ROTATE_PROB_THRES,
                             add_macros=False, macro_prob_thres=MACRO_PROB_THRES,
                             max_tries=MAX_TRIES, convert_to_primal=True):
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

    transform_probability = np.random.uniform(0, 1, size=n_prims)
    transform_params = np.random.beta(
        BETA_DISTR_1, BETA_DISTR_2, size=(n_prims, n_dims))
    transform_params = TRANSLATE_MEAN + TRANSLATE_STD * \
        (transform_params * np.random.choice((1, -1), size=(n_prims, n_dims)))
    transform_params = np.clip(transform_params, MIN_TRANSLATE, MAX_TRANSLATE)

    if add_rotate:
        rotate_probability = np.random.uniform(0, 1, size=n_prims)
        rotate_dim = n_dims if n_dims == 3 else 1
        rotate_params = np.random.beta(
            BETA_DISTR_1, BETA_DISTR_2, size=(n_prims, rotate_dim))
        rotate_params = ROTATE_MEAN + ROTATE_STD * \
            (rotate_params * np.random.choice((1, -1), size=(n_prims, rotate_dim)))
        rotate_params = np.clip(rotate_params, MIN_ROTATE, MAX_ROTATE)

    primitives = np.random.choice(PRIMITIVE_SETS[n_dims], size=n_prims)

    primitive_set = []
    sdf_set = []
    for index in range(n_prims):
        translate_func = TRANSFORM_SET[n_dims][0]
        rotate_func = TRANSFORM_SET[n_dims][1]
        scale_func = TRANSFORM_SET[n_dims][2]
        if scale_probability[index] < scale_prob_thres:
            scale_tensor = th.tensor(
                scale_params[index], dtype=th.float32, device=device)
            shape_expr = scale_func(primitives[index](), scale_tensor)
        else:
            shape_expr = primitives[index]()
        if add_rotate:
            if rotate_probability[index] < rotate_prob_thres:
                rotate_tensor = th.tensor(
                    rotate_params[index], dtype=th.float32, device=device)
                shape_expr = rotate_func(shape_expr, rotate_tensor)
        if transform_probability[index] < translate_prob_thres:
            transform_tensor = th.tensor(
                transform_params[index], dtype=th.float32, device=device)
            shape_expr = translate_func(shape_expr, transform_tensor)
        sdf = expr_to_sdf(shape_expr, sketcher)

        if add_macros:
            macro_prob = np.random.uniform(0, 1)
            if macro_prob < macro_prob_thres:
                shape_expr = try_modification(
                    shape_expr, sdf, n_dims, sketcher, device, insert_macros, max_tries=max_tries)
        if convert_to_primal:
            shape_expr = convert_base_to_primal_prim(
                shape_expr, n_dims, device, add_rotate)
        sdf_set.append(sdf)
        primitive_set.append(shape_expr)

    return primitive_set, sdf_set


def sample_random_program(n_prims, n_dims, sketcher, device,
                          add_rotate=False, convert_to_primal=False,
                          add_macros=False, add_transforms=False,
                          macro_prob_thres=MACRO_PROB_THRES,
                          hier_trans_prob_thres=HIER_TRANS_PROB_THRES,
                          max_tries=MAX_TRIES):

    n_ops = n_prims - 1
    primitive_set = []
    sdf_set = []
    while (len(primitive_set) < n_prims):
        new_primitive_set, new_sdf_set = sample_random_primitives(n_prims - len(primitive_set), n_dims, device, sketcher, add_rotate,
                                                                  add_macros=add_macros, convert_to_primal=convert_to_primal)
        new_primitive_set, new_sdf_set = check_primitive_validity(
            new_primitive_set, new_sdf_set)
        primitive_set += new_primitive_set
        sdf_set += new_sdf_set
    combinator_set = np.random.choice(
        COMBINATOR_SET, replace=True, size=n_ops * 2)
    combinator_set = combinator_set.tolist()
    n_tries = 0
    while (len(primitive_set) > 1):
        n_tries += 1
        np.random.shuffle(combinator_set)
        op = combinator_set.pop()
        n_count = len(primitive_set)
        rand_indies = np.random.choice(n_count, size=2, replace=False)
        prim_1 = primitive_set.pop(rand_indies.max())
        prim_2 = primitive_set.pop(rand_indies.min())
        sdf_1 = sdf_set.pop(rand_indies.max())
        sdf_2 = sdf_set.pop(rand_indies.min())
        output = SDF_COMBINATOR_MAP[op](sdf_1, sdf_2)
        valid = check_combinator_validity(output, sdf_1, sdf_2)
        if valid:
            new_expr = op(prim_1, prim_2)

            if add_macros:
                macro_prob = np.random.uniform(0, 1)
                if macro_prob < macro_prob_thres:
                    new_expr = try_modification(
                        new_expr, output, n_dims, sketcher, device, insert_macros, max_tries=max_tries)

            if add_transforms:
                transform_prob = np.random.uniform(0, 1)
                if transform_prob < hier_trans_prob_thres:
                    new_expr = try_modification(
                        new_expr, output, n_dims, sketcher, device, insert_transforms, max_tries=max_tries)

            primitive_set.insert(0, new_expr)
            sdf_set.insert(0, output)
        else:
            primitive_set.append(prim_2)
            primitive_set.append(prim_1)
            sdf_set.append(sdf_2)
            sdf_set.append(sdf_1)
            combinator_set += [op]
        if len(primitive_set) == 1:
            break
        if n_tries > max_tries:
            break

    if len(primitive_set) == 1:
        return primitive_set[0]
    else:
        return None


def try_modification(expr, sdf, n_dims, sketcher, device, mod_func, max_tries):

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
    return expr


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


def sample_random_program_set(n_programs, n_prims, n_dims, device, add_rotate=False,
                              convert_to_primal=False, add_macros=False, add_transforms=False, resolution=64):

    program_set = []
    failure_count = 0
    sketcher = Sketcher(resolution=resolution, device=device, n_dims=n_dims)
    index = 0
    while (index < n_programs):
        if index % 100 == 0:
            print(f"Generating program {index} with {n_prims} primitives.")
            print(f"Failure count: {failure_count}. Succes count: {index}")
        program = sample_random_program(n_prims, n_dims, sketcher, device, add_rotate, convert_to_primal,
                                        add_macros, add_transforms)
        if program is not None:
            program_set.append(program)
            index += 1
        else:
            failure_count += 1
    program_set = program_set[:n_programs]
    return program_set


def check_primitive_validity(primitive_set, sdf_set,
                             min_occ_thres=PRIM_MIN_OCC_THRES,
                             max_occ_thres=PRIM_MAX_OCC_THRES):
    valid_primitives = []
    valid_sdfs = []
    for ind, primitive in enumerate(primitive_set):
        sdf = sdf_set[ind]
        total = sdf.nelement()
        sdf_shape = (sdf <= 0)
        val = th.sum(sdf_shape) / total
        if min_occ_thres < val < max_occ_thres:
            valid_primitives.append(primitive)
            valid_sdfs.append(sdf)
    return valid_primitives, valid_sdfs


def check_combinator_validity(output, top, bottom,
                              min_occ_thres=EXPR_MIN_OCC_THRES,
                              diff_thres=EXPR_DIFF_THRES,
                              max_occ_thres=EXPR_MAX_OCC_THRES):
    total = output.nelement()
    output_shape = (output <= 0)
    top_shape = (top <= 0)
    bottom_shape = (bottom <= 0)
    val_1 = th.sum(output_shape) / total
    val_2 = th.sum(output_shape ^ top_shape) / th.sum(output_shape)
    val_3 = th.sum(output_shape ^ bottom_shape) / th.sum(output_shape)
    cond_1 = min_occ_thres < val_1 < max_occ_thres
    cond_2 = val_2 > diff_thres
    cond_3 = val_3 > diff_thres
    output = cond_1 and cond_2 and cond_3
    return output


def check_modifier_validity(output, input,
                            min_occ_thres=EXPR_MIN_OCC_THRES,
                            diff_thres=EXPR_DIFF_THRES,
                            max_occ_thres=EXPR_MAX_OCC_THRES):
    total = output.nelement()
    output_shape = (output <= 0)
    input_shape = (input <= 0)
    val_1 = th.sum(output_shape) / total
    val_2 = th.sum(output_shape ^ input_shape) / th.sum(output_shape)
    cond_1 = min_occ_thres < val_1 < max_occ_thres
    cond_2 = val_2 > diff_thres
    output = cond_1 and cond_2
    return output
