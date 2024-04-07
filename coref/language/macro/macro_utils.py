import torch as th
import numpy as np
from typing import Union as type_union

from collections import defaultdict

from .macro_primitives import PRIM_TYPE, TRANSFORM_TYPE, MACRO_TYPE, COMBINATOR_TYPE, NULL_EXPR_TYPE
from .macro_primitives import TRANSLATE_TYPE, SCALE_TYPE, ROTATE_TYPE
from .macro_primitives import REFSYM_TYPE, ROTSYM_TYPE, TRANSSYM_TYPE, PARAM_SYM_TYPE
from .macro_primitives import HAS_PARAM_TYPE, NO_PARAM_TYPE, TREE_PARAM_TYPE
MIN_TRANSLATE = -1
MAX_TRANSLATE = 1
MIN_SCALE = 0
MAX_SCALE = 2
# Use quaternion
MIN_ROTATE = -np.pi
MAX_ROTATE = np.pi


def get_expression_size(expression):
    parser_list = [expression]
    size = 0
    while (parser_list):
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, COMBINATOR_TYPE):
            parser_list.extend(cur_expr.args)
        elif isinstance(cur_expr, (TRANSFORM_TYPE, MACRO_TYPE)):
            parser_list.append(cur_expr.args[0])
        size += 1
    return size


def get_cmd_counts(expression):
    cmd_usage = defaultdict(int)
    parser_list = [expression]
    while (parser_list):
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, COMBINATOR_TYPE):
            parser_list.extend(cur_expr.args)
        elif isinstance(cur_expr, (TRANSFORM_TYPE, MACRO_TYPE)):
            parser_list.append(cur_expr.args[0])
        cmd_name = type(cur_expr).__name__
        cmd_usage[cmd_name] += 1
    return cmd_usage


def generate_expr_to_actions(expr_to_action_mapper, n_float_bins, n_t_pre_float, tokenization):
    # Pre fix notation - direct
    def expr_to_actions(expr):
        # actions are 0-3 combinators, 4-5 primitives, n_float_bins
        parser_list = [expr]
        actions = []
        while (parser_list):
            cur_expr = parser_list.pop()
            if isinstance(cur_expr, COMBINATOR_TYPE):
                next_to_parse = cur_expr.args[::-1]
                parser_list.extend(next_to_parse)
                new_expr = [expr_to_action_mapper[type(cur_expr)]]
                actions.extend(new_expr)
            elif isinstance(cur_expr, TRANSFORM_TYPE):
                transform_param = cur_expr.lookup_table[cur_expr.args[1]]
                if isinstance(cur_expr, TRANSLATE_TYPE):
                    transform_index = (transform_param - MIN_TRANSLATE) / \
                        (MAX_TRANSLATE - MIN_TRANSLATE) * (n_float_bins) - 0.5
                elif isinstance(cur_expr, SCALE_TYPE):
                    transform_index = (transform_param - MIN_SCALE) / \
                        (MAX_SCALE - MIN_SCALE) * (n_float_bins) - 0.5
                elif isinstance(cur_expr, ROTATE_TYPE):
                    transform_index = (transform_param - MIN_ROTATE) / \
                        (MAX_ROTATE - MIN_ROTATE) * (n_float_bins) - 0.5
                transform_index = th.round(
                    transform_index) + n_t_pre_float  # the base values
                transform_index = transform_index.cpu().numpy().tolist()
                new_expr = [expr_to_action_mapper[type(
                    cur_expr)]] + transform_index
                if tokenization == "MIXED":
                    new_expr = new_expr[::-1]
                actions.extend(new_expr)
                next_to_parse = cur_expr.args[0]
                parser_list.append(next_to_parse)
            elif isinstance(cur_expr, NULL_EXPR_TYPE):
                actions = []
                break
            elif isinstance(cur_expr, PRIM_TYPE):
                new_expr = [expr_to_action_mapper[type(cur_expr)]]
                actions.extend(new_expr)
            elif isinstance(cur_expr, REFSYM_TYPE):
                new_expr = [expr_to_action_mapper[type(cur_expr)]]
                actions.extend(new_expr)
                next_to_parse = cur_expr.args[0]
                parser_list.append(next_to_parse)
            elif isinstance(cur_expr, PARAM_SYM_TYPE):
                translate_param = cur_expr.lookup_table[cur_expr.args[1]]
                count_param = cur_expr.lookup_table[cur_expr.args[2]].cpu(
                ).numpy().tolist()
                if isinstance(cur_expr, TRANSSYM_TYPE):
                    translate_index = (translate_param - MIN_TRANSLATE) / \
                        (MAX_TRANSLATE - MIN_TRANSLATE) * (n_float_bins) - 0.5
                elif isinstance(cur_expr, ROTSYM_TYPE):
                    translate_index = (translate_param - MIN_ROTATE) / \
                        (MAX_ROTATE - MIN_ROTATE) * (n_float_bins) - 0.5
                translate_index = th.round(
                    translate_index) + n_t_pre_float  # the base values
                translate_index = [translate_index.cpu().numpy().tolist()]

                count_index = [count_param + n_t_pre_float + n_float_bins - 2]
                new_expr = [expr_to_action_mapper[type(
                    cur_expr)]] + translate_index + count_index
                if tokenization == "MIXED":
                    new_expr = new_expr[::-1]
                actions.extend(new_expr)
                next_to_parse = cur_expr.args[0]
                parser_list.append(next_to_parse)
            else:
                raise ValueError(f'Unknown expression type {type(cur_expr)}')
        if tokenization in ["POSTFIX", "MIXED"]:
            actions = actions[::-1]
        return actions

    return expr_to_actions


def generate_actions_to_expr(n_t_pre_float, n_float_bins, n_integer_bins, tokenization):
    # postfix parsing - reverse
    start_id = n_float_bins + n_t_pre_float + n_integer_bins
    end_id = n_float_bins + n_t_pre_float + n_integer_bins + 1

    if tokenization in ["POSTFIX", "PREFIX"]:
        def flip(actions):
            return actions[::-1]
    else:
        def flip(actions):
            return actions

    def get_actions(operation, operand_stack):
        operands = flip(operand_stack)
        if issubclass(operation, TRANSLATE_TYPE):
            transform_param = (th.cat(operands).clone() *
                               (MAX_TRANSLATE - MIN_TRANSLATE)) + MIN_TRANSLATE
            all_params = [transform_param]
        elif issubclass(operation, SCALE_TYPE):
            transform_param = (th.cat(operands).clone() *
                               (MAX_SCALE - MIN_SCALE)) + MIN_SCALE
            all_params = [transform_param]
        elif issubclass(operation, ROTATE_TYPE):
            transform_param = (th.cat(operands).clone() *
                               (MAX_ROTATE - MIN_ROTATE)) + MIN_ROTATE
            all_params = [transform_param]
        elif issubclass(operation, TRANSSYM_TYPE):
            transform_param = (
                operands[0].clone() * (MAX_TRANSLATE - MIN_TRANSLATE)) + MIN_TRANSLATE
            count_param = operands[1].clone()
            all_params = [transform_param, count_param]
        elif issubclass(operation, ROTSYM_TYPE):
            transform_param = (
                operands[0].clone() * (MAX_ROTATE - MIN_ROTATE)) + MIN_ROTATE
            count_param = operands[1].clone()
            all_params = [transform_param, count_param]

        return all_params

    def mixed_actions_to_expr(actions, action_mapper, ):
        operand_stack = []
        expression_stack = []
        param_req = []
        index = 0
        while (index < len(actions)):
            cur_action = actions[index]
            index += 1
            # skip start and stop
            if cur_action == end_id:
                break
            elif cur_action == start_id:
                if index == 1:
                    continue
                else:
                    print(actions)
                    raise ValueError(
                        "Start token found in the middle of the program")
            elif cur_action >= n_t_pre_float:
                raise ValueError("This should be consumed in an operator")
                operand_stack.append(value)
            elif cur_action >= 3:
                value = action_mapper[cur_action]
                if issubclass(value, NO_PARAM_TYPE):
                    expr = value()
                    expression_stack.append(expr)
                elif issubclass(value, TREE_PARAM_TYPE):
                    expr_0 = expression_stack.pop()
                    expr = value(expr_0)
                    expression_stack.append(expr)
                else:
                    delta = 0
                    operand_stack = []
                    while True:
                        new_token = actions[index + delta]
                        if (new_token < n_t_pre_float) or new_token == end_id:
                            break
                        op_value = action_mapper[new_token]
                        operand_stack.append(op_value)
                        delta += 1
                    params = get_actions(value, operand_stack)
                    expr_0 = expression_stack.pop()
                    expr = value(expr_0, *params)
                    expression_stack.append(expr)
                    index += delta
            else:
                value = action_mapper[cur_action]
                expr_0 = expression_stack.pop()
                expr_1 = expression_stack.pop()
                new_expr = value(expr_0, expr_1)
                expression_stack.append(new_expr)
        if len(expression_stack) != 1:
            raise ValueError("Expression stack not empty")
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
                if issubclass(value, NO_PARAM_TYPE):
                    expr = value()
                    expression_stack.append(expr)
                elif issubclass(value, TREE_PARAM_TYPE):
                    expr_0 = expression_stack.pop()
                    expr = value(expr_0)
                    expression_stack.append(expr)

                else:
                    params = get_actions(value, operand_stack)
                    expr_0 = expression_stack.pop()
                    expr = value(expr_0, *params)
                    expression_stack.append(expr)
                operand_stack = []
            else:
                value = action_mapper[cur_action]
                expr_0 = expression_stack.pop()
                expr_1 = expression_stack.pop()
                new_expr = value(expr_0, expr_1)
                expression_stack.append(new_expr)

        if len(expression_stack) != 1:
            raise ValueError("Expression stack not empty")
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
        expression = postfix_actions_to_expr(actions, action_mapper)
        return expression

    if tokenization == "PREFIX":
        actions_to_expr = prefix_actions_to_expr
    elif tokenization == "POSTFIX":
        actions_to_expr = postfix_actions_to_expr
    elif tokenization == "MIXED":
        actions_to_expr = mixed_actions_to_expr

    return actions_to_expr
