import torch as th
import numpy as np

from collections import defaultdict

from geolipi.symbolic.types import COMBINATOR_TYPE, NULL_EXPR_TYPE, PRIM_TYPE, TRANSFORM_TYPE
from geolipi.symbolic.transforms_3d import Translate3D, Scale3D, EulerRotate3D
from geolipi.symbolic.transforms_2d import Translate2D, Scale2D, EulerRotate2D

from .primal_primitives import PRIMAL_PRIMS, PRIMALPRIM_TO_BASE_MAPPER, PRIMAL_TYPE, BASE_TO_PRIMALPRIM_MAPPER

from ..constants import MIN_SCALE, MAX_SCALE, MIN_TRANSLATE, MAX_TRANSLATE, MIN_ROTATE, MAX_ROTATE, CONVERSION_DELTA

MIN_TRANSLATE = -1
MAX_TRANSLATE = 1
MIN_SCALE = 0
MAX_SCALE = 2
# Use quaternion
MIN_ROTATE = -np.pi
MAX_ROTATE = np.pi
SCALE_ADD = 1


def get_base_expr_converter_func(n_dims, rotation=False):

    if n_dims == 2:
        translate_func = Translate2D
        scale_func = Scale2D
        rotate_func = EulerRotate2D
    elif n_dims == 3:
        translate_func = Translate3D
        scale_func = Scale3D
        rotate_func = EulerRotate3D
    if rotation:
        def get_args(cur_expr):
            args = cur_expr.args
            translate_param = cur_expr.lookup_table[args[0]]
            scale_param = cur_expr.lookup_table[args[1]]
            rotate_param = cur_expr.lookup_table[args[2]]
            return translate_param, scale_param, rotate_param

        def get_base(cur_expr):
            translate_param, scale_param, rotate_param = get_args(cur_expr)
            base_expr = translate_func(rotate_func(scale_func(
                PRIMALPRIM_TO_BASE_MAPPER[cur_expr.__class__](), scale_param), rotate_param), translate_param)
            return base_expr
    else:
        def get_args(cur_expr):
            args = cur_expr.args
            translate_param = cur_expr.lookup_table[args[0]]
            scale_param = cur_expr.lookup_table[args[1]]
            return translate_param, scale_param

        def get_base(cur_expr):
            translate_param, scale_param = get_args(cur_expr)
            base_expr = translate_func(scale_func(
                PRIMALPRIM_TO_BASE_MAPPER[cur_expr.__class__](), scale_param), translate_param)
            return base_expr

    return get_args, get_base


def generate_revert_func(n_dims, rotation=False):

    get_args, get_base = get_base_expr_converter_func(n_dims, rotation)

    def revert_to_base_expr(expr):
        parser_list = [expr]
        expr_stack = []
        operator_stack = []
        operator_nargs_stack = []
        execution_pointer_index = []
        while (parser_list):
            cur_expr = parser_list.pop()
            if isinstance(cur_expr, NULL_EXPR_TYPE):
                new_expr = cur_expr
                expr_stack.append(new_expr)
                # break
            elif isinstance(cur_expr, PRIMAL_TYPE):
                new_expr = get_base(cur_expr)
                expr_stack.append(new_expr)
            elif isinstance(cur_expr, COMBINATOR_TYPE):
                operator_stack.append(type(cur_expr))
                n_args = len(cur_expr.args)
                operator_nargs_stack.append(n_args)
                next_to_parse = cur_expr.args[::-1]
                parser_list.extend(next_to_parse)
                execution_pointer_index.append(len(expr_stack))
            while (operator_stack and len(expr_stack) - execution_pointer_index[-1] >= operator_nargs_stack[-1]):
                n_args = operator_nargs_stack.pop()
                operator = operator_stack.pop()
                _ = execution_pointer_index.pop()
                args_1 = expr_stack.pop()
                args_0 = expr_stack.pop()
                new_expr = operator(args_0, args_1)
                expr_stack.append(new_expr)

        assert len(expr_stack) == 1
        base_expr = expr_stack[0]
        return base_expr

    return revert_to_base_expr


def get_gen_action_seq(expr_to_action_mapper, n_t_pre_float, rotation):

    if rotation:
        def gen_action_seq(n_float_bins, cur_expr):
            args = cur_expr.args
            translate_param = cur_expr.lookup_table[args[0]]
            scale_param = cur_expr.lookup_table[args[1]]
            rotate_param = cur_expr.lookup_table[args[2]]
            translate_index = (translate_param - MIN_TRANSLATE) / \
                (MAX_TRANSLATE - MIN_TRANSLATE) * (n_float_bins) - 0.5
            scale_index = (scale_param - MIN_SCALE) / \
                (MAX_SCALE - MIN_SCALE) * (n_float_bins) - 0.5
            rotate_index = (rotate_param - MIN_ROTATE) / \
                (MAX_ROTATE - MIN_ROTATE) * (n_float_bins) - 0.5
            transform_index = th.cat(
                [translate_index, scale_index, rotate_index], 0)
            transform_index = th.round(
                transform_index) + n_t_pre_float  # the base values
            transform_index = transform_index.cpu().numpy().tolist()
            new_expr = [expr_to_action_mapper[type(
                cur_expr)]] + transform_index
            return new_expr
    else:
        def gen_action_seq(n_float_bins, cur_expr):
            args = cur_expr.args
            translate_param = cur_expr.lookup_table[args[0]]
            scale_param = cur_expr.lookup_table[args[1]]
            translate_index = (translate_param - MIN_TRANSLATE) / \
                (MAX_TRANSLATE - MIN_TRANSLATE) * (n_float_bins) - 0.5
            scale_index = (scale_param - MIN_SCALE) / \
                (MAX_SCALE - MIN_SCALE) * (n_float_bins) - 0.5
            transform_index = th.cat([translate_index, scale_index], 0)
            transform_index = th.round(
                transform_index) + n_t_pre_float  # the base values
            transform_index = transform_index.cpu().numpy().tolist()
            new_expr = [expr_to_action_mapper[type(
                cur_expr)]] + transform_index
            return new_expr
    return gen_action_seq


def generate_expr_to_actions(expr_to_action_mapper, n_float_bins, n_t_pre_float, rotation=False, tokenization="POSTFIX"):
    # TODO: Is a stack based implementation required? I think it can be done in one direct pass.
    gen_action_seq = get_gen_action_seq(
        expr_to_action_mapper, n_t_pre_float, rotation)
# Convert to post-fix

    def expr_to_actions(expr):
        # actions are 0-3 combinators, 4-5 primitives, n_float_bins
        parser_list = [expr]
        action_stack = []
        operator_stack = []
        operator_nargs_stack = []
        execution_pointer_index = []
        while (parser_list):
            cur_expr = parser_list.pop()
            if isinstance(cur_expr, NULL_EXPR_TYPE):
                action_stack = [[],]
                break
            if isinstance(cur_expr, PRIMAL_TYPE):
                new_expr = gen_action_seq(n_float_bins, cur_expr)
                if tokenization == "MIXED":
                    new_expr = new_expr[::-1]
                action_stack.append(new_expr)
            elif isinstance(cur_expr, COMBINATOR_TYPE):
                operator_stack.append(type(cur_expr))
                n_args = len(cur_expr.args)
                operator_nargs_stack.append(n_args)
                next_to_parse = cur_expr.args[::-1]
                parser_list.extend(next_to_parse)
                execution_pointer_index.append(len(action_stack))

            while (operator_stack and len(action_stack) - execution_pointer_index[-1] >= operator_nargs_stack[-1]):
                n_args = operator_nargs_stack.pop()
                operator = operator_stack.pop()
                _ = execution_pointer_index.pop()
                args_1 = action_stack.pop()
                args_0 = action_stack.pop()
                new_expr = [expr_to_action_mapper[operator]] + args_0 + args_1
                action_stack.append(new_expr)

        assert len(action_stack) == 1
        actions = action_stack[0]
        if tokenization in ["POSTFIX", "MIXED"]:
            actions = actions[::-1]
        return actions

    return expr_to_actions


def generate_actions_to_expr(n_dims, n_prim_param_tokens, n_t_pre_float, n_float_bins, rotation=False, base=False, tokenization="POSTFIX"):
    # postfix parsing - direct
    start_id = n_float_bins + n_t_pre_float
    end_id = n_float_bins + n_t_pre_float + 1
    get_expr = None

    if n_dims == 2:
        translate_func = Translate2D
        scale_func = Scale2D
        rotate_func = EulerRotate2D
    else:
        translate_func = Translate3D
        scale_func = Scale3D
        rotate_func = EulerRotate3D

    if tokenization in ["POSTFIX", "PREFIX"]:
        def flip(actions):
            return actions[::-1]
    else:
        def flip(actions):
            return actions

    if rotation:
        def get_actions(operand_stack):
            operands = flip(operand_stack[-n_prim_param_tokens:])
            translate_param = (th.cat(operands[:n_dims]).clone(
            ) * (MAX_TRANSLATE - MIN_TRANSLATE)) + MIN_TRANSLATE
            scale_param = (th.cat(
                operands[n_dims: 2 * n_dims]).clone() * (MAX_SCALE - MIN_SCALE)) + MIN_SCALE
            rotate_param = (
                th.cat(operands[2 * n_dims:]).clone() * (MAX_ROTATE - MIN_ROTATE)) + MIN_ROTATE
            return translate_param, scale_param, rotate_param
    else:
        def get_actions(operand_stack):
            operands = flip(operand_stack[-n_prim_param_tokens:])
            translate_param = (th.cat(operands[:n_dims]).clone(
            ) * (MAX_TRANSLATE - MIN_TRANSLATE)) + MIN_TRANSLATE
            scale_param = (th.cat(
                operands[n_dims: 2 * n_dims]).clone() * (MAX_SCALE - MIN_SCALE)) + MIN_SCALE
            return translate_param, scale_param

    if base:
        if rotation:
            def get_expr(value, params):
                translate_param, scale_param, rotate_param = params
                expr = translate_func(rotate_func(scale_func(
                    PRIMALPRIM_TO_BASE_MAPPER[value](), scale_param), rotate_param), translate_param)
                return expr
        else:
            def get_expr(value, params):
                translate_param, scale_param = params
                expr = translate_func(scale_func(
                    PRIMALPRIM_TO_BASE_MAPPER[value](), scale_param), translate_param)
                return expr
    else:
        def get_expr(value, params):
            expr = value(*params)
            return expr

    def mixed_actions_to_expr(actions, action_mapper):
        operand_stack = []
        expression_stack = []
        draw_stack = []
        canvas_count = 0
        for index, cur_action in enumerate(actions):
            # skip start and stop
            if cur_action == end_id:
                break
            elif cur_action == start_id:
                if index == 0:
                    continue
                else:
                    print(actions)
                    raise ValueError(
                        "Start token found in the middle of the program")
            elif cur_action >= n_t_pre_float:
                value = action_mapper[cur_action]
                operand_stack.append(value)
                if len(operand_stack) == n_prim_param_tokens:
                    draw_type = draw_stack.pop()
                    params = get_actions(operand_stack)
                    expr = get_expr(draw_type, params)
                    expression_stack.append(expr)
                    operand_stack = []
                    canvas_count += 1

            elif cur_action >= 3:
                draw_type = action_mapper[cur_action]
                draw_stack.append(draw_type)
            else:
                value = action_mapper[cur_action]
                expr_0 = expression_stack.pop()
                expr_1 = expression_stack.pop()
                new_expr = value(expr_0, expr_1)
                expression_stack.append(new_expr)

        if len(expression_stack) != 1:
            raise ValueError("Expression stack is not 1")
        else:
            expression = expression_stack[0]
        return expression

    def postfix_actions_to_expr(actions, action_mapper, ):
        operand_stack = []
        expression_stack = []
        for index, cur_action in enumerate(actions):
            # skip start and stop
            if cur_action == end_id:
                break
            elif cur_action == start_id:
                if index == 0:
                    continue
                else:
                    print(actions)
                    raise ValueError(
                        "Start token found in the middle of the program")
            elif cur_action >= n_t_pre_float:
                value = action_mapper[cur_action]
                operand_stack.append(value)
            elif cur_action >= 3:
                value = action_mapper[cur_action]
                params = get_actions(operand_stack)
                expr = get_expr(value, params)
                operand_stack = []
                expression_stack.append(expr)
            else:
                value = action_mapper[cur_action]
                expr_0 = expression_stack.pop()
                expr_1 = expression_stack.pop()
                new_expr = value(expr_0, expr_1)
                expression_stack.append(new_expr)

        if len(expression_stack) != 1:
            raise ValueError("Expression stack is not 1")
        else:
            expression = expression_stack[0]
        return expression

    def prefix_actions_to_expr(actions, action_mapper):
        if isinstance(actions, list):
            actions = actions[0:1] + actions[1:-1][::-1] + actions[-1:]
        else:
            actions = actions[0:1].tolist() + actions[1:-
                                                      1].tolist()[::-1] + actions[-1:].tolist()
            actions = np.array(actions)
        actions = postfix_actions_to_expr(actions, action_mapper)
        return actions

    if tokenization == "PREFIX":
        actions_to_expr = prefix_actions_to_expr
    elif tokenization == "POSTFIX":
        actions_to_expr = postfix_actions_to_expr
    elif tokenization == "MIXED":
        actions_to_expr = mixed_actions_to_expr

    return actions_to_expr


def get_expression_size(expression):
    parser_list = [expression]
    size = 0
    while (parser_list):
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, COMBINATOR_TYPE):
            parser_list.extend(cur_expr.args)
        size += 1
    return size


def get_cmd_counts(expression):
    cmd_usage = defaultdict(int)
    parser_list = [expression]
    while (parser_list):
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, COMBINATOR_TYPE):
            cmd_name = type(cur_expr).__name__
            cmd_usage[cmd_name] += 1
            parser_list.extend(cur_expr.args)
        elif isinstance(cur_expr, (PRIMAL_TYPE, NULL_EXPR_TYPE)):
            cmd_name = type(cur_expr).__name__
            cmd_usage[cmd_name] += 1
        else:
            raise ValueError(f"Unknown expression type {type(cur_expr)}")
    return cmd_usage

# need to invert the params:
# TODO: Improve this.


def revert_to_current_langauge(expression, clip=False, quantize=False, device=None, dtype=None, resolution=33):

    scale_stack = [th.tensor(np.array([1, 1, 1]), dtype=dtype, device=device)]
    translate_stack = [
        th.tensor(np.array([0, 0, 0]), dtype=dtype, device=device)]
    rotate_stack = [th.tensor(np.array([0, 0, 0]), dtype=dtype, device=device)]

    conversion_delta = CONVERSION_DELTA
    if quantize:
        two_scale_delta = (2 - 2 * conversion_delta)/(resolution - 1)

    parser_list = [expression]
    expr_stack = []
    operator_stack = []
    operator_nargs_stack = []
    execution_pointer_index = []
    while (parser_list):
        cur_expr = parser_list.pop()
        # print(f'processing {cur_expr}')
        # print("t, s, r", len(translate_stack), len(scale_stack), len(rotate_stack))

        if isinstance(cur_expr, COMBINATOR_TYPE):
            ...
            cloned_scale = th.clone(scale_stack[-1])
            scale_stack.append(cloned_scale)
            cloned_translate = th.clone(translate_stack[-1])
            translate_stack.append(cloned_translate)
            cloned_rotate = th.clone(rotate_stack[-1])
            rotate_stack.append(cloned_rotate)

            operator_stack.append(type(cur_expr))
            n_args = len(cur_expr.args)
            operator_nargs_stack.append(n_args)
            next_to_parse = cur_expr.args[::-1]
            parser_list.extend(next_to_parse)
            execution_pointer_index.append(len(expr_stack))

        elif isinstance(cur_expr, TRANSFORM_TYPE):
            c_type = type(cur_expr)
            param_sym = cur_expr.args[1]
            param = cur_expr.lookup_table[param_sym]
            if c_type == Translate3D:
                cur_scale = scale_stack[-1]
                cur_translate = translate_stack.pop()
                new_translate = cur_scale * param + cur_translate
                translate_stack.append(new_translate)
            elif c_type == Scale3D:
                cur_scale = scale_stack.pop()
                cur_scale = cur_scale * param
                scale_stack.append(cur_scale)
            elif c_type == EulerRotate3D:
                cur_rotate = rotate_stack.pop()
                new_rotate = cur_rotate + param
                rotate_stack.append(new_rotate)
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, PRIMAL_TYPE):
            raise ValueError("Primal type found in the expression")
        elif isinstance(cur_expr, PRIM_TYPE):
            t_p = translate_stack.pop()
            s_p = scale_stack.pop()
            if clip:
                t_p = th.clip(t_p, MIN_TRANSLATE + conversion_delta,
                              MAX_TRANSLATE - conversion_delta)
                s_p = th.clip(s_p, MIN_SCALE + conversion_delta,
                              MAX_SCALE - conversion_delta)

            param = th.cat([t_p, s_p], 0)
            if quantize:
                param[3:6] -= SCALE_ADD
                param = (param - (-1 + conversion_delta)) / two_scale_delta
                param = th.round(param)
                param = (param * two_scale_delta) + (-1 + conversion_delta)
                param[3:6] += SCALE_ADD
            t_p, s_p = param[:3], param[3:]
            new_expr = BASE_TO_PRIMALPRIM_MAPPER[cur_expr.__class__](t_p, s_p)
            expr_stack.append(new_expr)

        while (operator_stack and len(expr_stack) - execution_pointer_index[-1] >= operator_nargs_stack[-1]):
            # print("popping stack")
            n_args = operator_nargs_stack.pop()
            operator = operator_stack.pop()
            _ = execution_pointer_index.pop()
            args_1 = expr_stack.pop()
            args_0 = expr_stack.pop()
            new_expr = operator(args_0, args_1)
            expr_stack.append(new_expr)

    assert len(expr_stack) == 1
    base_expr = expr_stack[0]
    return base_expr
