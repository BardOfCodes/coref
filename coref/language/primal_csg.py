
# Action space
# compiler & parser
# graph compiler (CS and CP)
# data generator
# draw
from collections import defaultdict
import numpy as np
import torch as th


from geolipi.symbolic.combinators import Union, Intersection, Difference
from geolipi.symbolic.primitives_2d import NullExpression2D
from geolipi.symbolic.primitives_3d import NullExpression3D

from .primal.primal_primitives import PRIMALPRIM_TO_BASE_MAPPER, BASE_TO_PRIMALPRIM_MAPPER
from .primal.primal_primitives import STR_TO_CMD_MAPPER
from .primal.primal_primitives import PrimalCircle2D, PrimalRectangle2D, PrimalTriangle2D
from .primal.primal_primitives import PrimalCuboid3D, PrimalSphere3D, PrimalCylinder3D, PrimalCone3D, PrimalHalfPlan3D, PrimalSQ3D

from .primal.primal_utils import (get_expression_size, get_cmd_counts, generate_revert_func,
                                  generate_expr_to_actions, generate_actions_to_expr)

from .primal.primal_utils import revert_to_current_langauge
from .primal.primal_state_machine import BatchedStateMachinePostF, BatchedStateMachinePreF, BatchedStateMachineMixed
from .constants import MIN_TRANSLATE, MAX_TRANSLATE, MIN_SCALE, MAX_SCALE


class PrimalCSG3D:

    def __init__(self, n_float_bins, n_integer_bins, tokenization):
        assert n_integer_bins == 0
        self.set_config()
        self.tokenization = tokenization
        self.n_float_bins = n_float_bins
        self.null_expression = NullExpression3D
        self.n_t_pre_float = len(self.VALID_OPS)
        self.start = self.n_float_bins + self.n_t_pre_float
        self.end = self.n_float_bins + self.n_t_pre_float + 1

        self.expr_to_action_mapper = {
            k: ind for ind, k in enumerate(self.VALID_OPS)}
        self.get_expression_size = get_expression_size
        self.get_cmd_counts = get_cmd_counts
        self.revert_to_base_expr = generate_revert_func(
            self.n_dims, rotation=self.rotation)
        self.expr_to_actions = generate_expr_to_actions(
            self.expr_to_action_mapper, self.n_float_bins, self.n_t_pre_float, rotation=self.rotation, tokenization=self.tokenization)
        self.actions_to_expr = generate_actions_to_expr(
            self.n_dims, self.n_prim_param_tokens, self.n_t_pre_float, self.n_float_bins, rotation=self.rotation, base=False, tokenization=self.tokenization)
        self.actions_to_base_expr = generate_actions_to_expr(
            self.n_dims, self.n_prim_param_tokens, self.n_t_pre_float, self.n_float_bins, rotation=self.rotation, base=True, tokenization=self.tokenization)
        self.revert_to_current_langauge = revert_to_current_langauge
        # for PO
        self.clip_dict = {
            0: (MIN_TRANSLATE, MAX_TRANSLATE),
            1: (MIN_SCALE, MAX_SCALE),
        }
        self.variable_transform_param = {
            0: (1.0, 0),
            1: (1.0, 1),
        }

    def set_config(self):

        self.name = "PCSG3D"
        self.STR_TO_CMD_MAPPER = STR_TO_CMD_MAPPER
        self.n_dims = 3
        self.n_prim_param_tokens = 6
        self.rotation = False
        self.VALID_OPS = [Union, Intersection,
                          Difference, PrimalCuboid3D, PrimalSphere3D]

    def create_action_mapper(self, dtype=th.float32, device=th.device("cpu")):
        action_mapper = {}
        for ind, action in enumerate(self.VALID_OPS):
            action_mapper[ind] = action
        n_actions = len(action_mapper.keys())
        values = (np.arange(0, self.n_float_bins)) / float(self.n_float_bins-1)
        for ind in range(self.n_float_bins):
            action_mapper[ind + n_actions] = th.tensor(
                [values[ind]], dtype=dtype, device=device)
        # What about start and stop?
        return action_mapper

    def create_action_spec(self, expr, max_actions, device):

        actions = self.expr_to_actions(expr)
        actions = [self.start] + actions + [self.end]  # start - program - stop
        # post-fix form
        n_actions = len(actions)
        n_pad = max_actions - n_actions
        actions = actions + [0] * n_pad
        action_validity = np.zeros((max_actions), dtype=bool)
        action_validity[:n_actions] = True
        actions = th.tensor(actions, device=device, dtype=th.int64)
        action_validity = th.tensor(
            action_validity, device=device, dtype=th.bool)
        return actions, action_validity, n_actions

    def get_batched_state_machine(self, max_canvas_count, batch_size, device):

        if self.tokenization == "POSTFIX":
            state_machine_class = BatchedStateMachinePostF
        elif self.tokenization == "PREFIX":
            state_machine_class = BatchedStateMachinePreF
        elif self.tokenization == "MIXED":
            state_machine_class = BatchedStateMachineMixed
        state_machine = state_machine_class(self.n_t_pre_float,
                                            self.n_prim_param_tokens,
                                            self.n_float_bins,
                                            max_canvas_count,
                                            batch_size,
                                            device)
        return state_machine

    def invert_variable(self, variable_info_set):
        param, command_symbol, var_type = variable_info_set
        clip_min, clip_max = self.clip_dict[var_type]
        param = th.clip(param, clip_min, clip_max)
        mul, extra = self.variable_transform_param[var_type]
        variable = th.atanh((param - extra) / mul)
        variable = th.autograd.Variable(variable, requires_grad=True)
        param = th.tanh(variable) * mul + extra
        return param, variable

    def revert_variable(self, variable_info_set):
        variable, command_symbol, var_type = variable_info_set
        mul, extra = self.variable_transform_param[var_type]
        param = th.tanh(variable) * mul + extra
        return param


class PrimalCSG2D(PrimalCSG3D):

    def __init__(self,  n_float_bins, n_integer_bins, tokenization):
        super(PrimalCSG2D, self).__init__(
            n_float_bins, n_integer_bins, tokenization)

        self.null_expression = NullExpression2D

    def set_config(self):
        self.name = "PCSG2D"
        self.STR_TO_CMD_MAPPER = STR_TO_CMD_MAPPER
        self.n_dims = 2
        self.n_prim_param_tokens = 5
        self.rotation = True
        self.VALID_OPS = [Union, Intersection, Difference,
                          PrimalRectangle2D, PrimalCircle2D, PrimalTriangle2D]


class LATCSG3D(PrimalCSG3D):

    def __init__(self,  n_float_bins, n_integer_bins, tokenization):
        super(LATCSG3D, self).__init__(
            n_float_bins, n_integer_bins, tokenization)

        self.null_expression = NullExpression2D

    def set_config(self):
        self.name = "FCSG3D"
        self.STR_TO_CMD_MAPPER = STR_TO_CMD_MAPPER
        self.n_dims = 3
        self.n_prim_param_tokens = 9
        self.rotation = False
        self.VALID_OPS = [Union, Intersection, Difference, PrimalCuboid3D, PrimalSphere3D,
                          PrimalCylinder3D, PrimalHalfPlan3D, PrimalCone3D, PrimalSQ3D]
