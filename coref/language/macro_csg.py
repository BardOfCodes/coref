
# Action space
# compiler & parser
# graph compiler (CS and CP)
# data generator
# draw
from collections import defaultdict, OrderedDict
import numpy as np
import torch as th


from geolipi.symbolic.combinators import Union, Intersection, Difference
from geolipi.symbolic.primitives_3d import NoParamCuboid3D, NoParamSphere3D, NoParamCylinder3D, NullExpression3D
from geolipi.symbolic.transforms_3d import Translate3D, Scale3D, EulerRotate3D
from geolipi.symbolic.transforms_3d import (ReflectX3D, ReflectY3D, ReflectZ3D,
                                            TranslationSymmetryX3D, TranslationSymmetryY3D, TranslationSymmetryZ3D,
                                            RotationSymmetryX3D, RotationSymmetryY3D, RotationSymmetryZ3D)

from geolipi.symbolic.primitives_2d import NoParamRectangle2D, NoParamCircle2D, NoParamTriangle2D, NullExpression2D
from geolipi.symbolic.transforms_2d import Translate2D, Scale2D, EulerRotate2D
from geolipi.symbolic.transforms_2d import ReflectX2D, ReflectY2D, RotationSymmetry2D, TranslationSymmetryX2D, TranslationSymmetryY2D

from .macro.macro_primitives import STR_TO_CMD_MAPPER

from .macro.macro_utils import (get_expression_size, get_cmd_counts,
                                generate_expr_to_actions, generate_actions_to_expr)

from .macro.macro_state_machine import MacroBatchedStateMachinePostF, MacroBatchedStateMachinePreF, MacroBatchedStateMachineMixed
from .primal_csg import PrimalCSG3D


class MacroCSG3D(PrimalCSG3D):

    def __init__(self, n_float_bins, n_integer_bins=4, tokenization="POSTFIX"):
        self.set_config()
        self.tokenization = tokenization
        self.n_float_bins = n_float_bins
        self.n_integer_bins = n_integer_bins
        self.null_expression = NullExpression3D
        self.n_t_pre_float = len(self.VALID_OPS)
        self.start = self.n_float_bins + self.n_integer_bins + self.n_t_pre_float
        self.end = self.n_float_bins + self.n_integer_bins + self.n_t_pre_float + 1

        self.expr_to_action_mapper = {
            k: ind for ind, k in enumerate(self.VALID_OPS)}
        self.get_expression_size = get_expression_size
        self.get_cmd_counts = get_cmd_counts
        self.expr_to_actions = generate_expr_to_actions(
            self.expr_to_action_mapper, self.n_float_bins, self.n_t_pre_float, tokenization)
        self.actions_to_expr = generate_actions_to_expr(
            self.n_t_pre_float, self.n_float_bins, self.n_integer_bins, tokenization)
        self.actions_to_base_expr = self.actions_to_expr

    def revert_to_base_expr(self, expr):
        return expr

    def set_config(self):

        self.name = "MCSG3D"
        self.STR_TO_CMD_MAPPER = STR_TO_CMD_MAPPER
        self.n_dims = 3
        self.VALID_OPS = [Union, Intersection, Difference, Translate3D, Scale3D, EulerRotate3D,
                          ReflectX3D, ReflectY3D, ReflectZ3D, TranslationSymmetryX3D, TranslationSymmetryY3D, TranslationSymmetryZ3D,
                          RotationSymmetryX3D, RotationSymmetryY3D, RotationSymmetryZ3D, NoParamCuboid3D, NoParamSphere3D, NoParamCylinder3D]
        self.n_t_pre_float = len(self.VALID_OPS)
        self.op_specs = OrderedDict(
            # OP: (n_float, n_param, n_canvas, type)
            Union=(0, 0, 2, 0),
            Intersection=(0, 0, 2, 0),
            Difference=(0, 0, 2, 0),
            Translate3D=(3, 0, 1, 1),
            Scale3D=(3, 0, 1, 1),
            EulerRotate3D=(3, 0, 1, 1),
            ReflectX3D=(0, 0, 1, 1),
            ReflectY3D=(0, 0, 1, 1),
            ReflectZ3D=(0, 0, 1, 1),
            TranslationSymmetryX3D=(1, 1, 1, 1),
            TranslationSymmetryY3D=(1, 1, 1, 1),
            TranslationSymmetryZ3D=(1, 1, 1, 1),
            RotationSymmetryX3D=(1, 1, 1, 1),
            RotationSymmetryY3D=(1, 1, 1, 1),
            RotationSymmetryZ3D=(1, 1, 1, 1),
            NoParamCuboid3D=(0, 0, 0, 2),
            NoParamSphere3D=(0, 0, 0, 2),
            NoParamCylinder3D=(0, 0, 0, 2),

        )

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
        # Also 2-5 integers.
        int_starter = n_actions + self.n_float_bins
        for ind in range(self.n_integer_bins):
            action_mapper[ind + int_starter] = th.tensor(
                [ind + 2], dtype=th.int64, device=device)

        return action_mapper

    def get_batched_state_machine(self, max_canvas_count, batch_size, device):

        if self.tokenization == "POSTFIX":
            state_machine_class = MacroBatchedStateMachinePostF
        elif self.tokenization == "PREFIX":
            state_machine_class = MacroBatchedStateMachinePreF
        elif self.tokenization == "MIXED":
            state_machine_class = MacroBatchedStateMachineMixed

        return state_machine_class(self.n_float_bins, self.n_integer_bins, self.op_specs,
                                   max_canvas_count, batch_size, device)


class MacroCSG2D(MacroCSG3D):

    def set_config(self):
        self.name = "MCSG2D"
        self.null_expression = NullExpression2D
        self.STR_TO_CMD_MAPPER = STR_TO_CMD_MAPPER
        self.n_dims = 2
        self.VALID_OPS = [Union, Intersection, Difference, Translate2D, Scale2D, EulerRotate2D,
                          ReflectX2D, ReflectY2D, TranslationSymmetryX2D, TranslationSymmetryY2D,
                          RotationSymmetry2D, NoParamRectangle2D, NoParamCircle2D, NoParamTriangle2D]
        self.n_t_pre_float = len(self.VALID_OPS)

        self.op_specs = OrderedDict(
            # OP: (n_float, n_integer, n_canvas, type)
            Union=(0, 0, 2, 0),
            Intersection=(0, 0, 2, 0),
            Difference=(0, 0, 2, 0),
            Translate2D=(2, 0, 1, 1),
            Scale2D=(2, 0, 1, 1),
            EulerRotate2D=(1, 0, 1, 1),
            ReflectX2D=(0, 0, 1, 1),
            ReflectY2D=(0, 0, 1, 1),
            TranslationSymmetryX2D=(1, 1, 1, 1),
            TranslationSymmetryY2D=(1, 1, 1, 1),
            RotationSymmetry2D=(1, 1, 1, 1),
            NoParamRectangle2D=(0, 0, 0, 2),
            NoParamCircle2D=(0, 0, 0, 2),
            NoParamTriangle2D=(0, 0, 0, 2),
        )


class HierarchicalCSG3D(MacroCSG3D):

    def set_config(self):
        self.name = "HCSG3D"
        self.null_expression = NullExpression3D
        self.STR_TO_CMD_MAPPER = STR_TO_CMD_MAPPER
        self.n_dims = 3
        self.VALID_OPS = [Union, Intersection, Difference, Translate3D, Scale3D, EulerRotate3D,
                          NoParamCuboid3D, NoParamSphere3D, NoParamCylinder3D]
        self.n_t_pre_float = len(self.VALID_OPS)

        self.op_specs = OrderedDict(
            # OP: (n_float, n_integer, n_canvas, type)
            Union=(0, 0, 2, 0),
            Intersection=(0, 0, 2, 0),
            Difference=(0, 0, 2, 0),
            Translate3D=(3, 0, 1, 1),
            Scale3D=(3, 0, 1, 1),
            EulerRotate3D=(3, 0, 1, 1),
            NoParamCuboid3D=(0, 0, 0, 2),
            NoParamSphere3D=(0, 0, 0, 2),
            NoParamCylinder3D=(0, 0, 0, 2),
        )


class HierarchicalCSG2D(MacroCSG2D):

    def set_config(self):
        self.name = "HCSG2D"
        self.null_expression = NullExpression2D
        self.STR_TO_CMD_MAPPER = STR_TO_CMD_MAPPER
        self.n_dims = 2
        self.VALID_OPS = [Union, Intersection, Difference, Translate2D, Scale2D, EulerRotate2D,
                          NoParamRectangle2D, NoParamCircle2D, NoParamTriangle2D]
        self.n_t_pre_float = len(self.VALID_OPS)
        # self.rotation = True

        self.op_specs = OrderedDict(
            # OP: (n_float, n_integer, n_canvas, type)
            Union=(0, 0, 2, 0),
            Intersection=(0, 0, 2, 0),
            Difference=(0, 0, 2, 0),
            Translate2D=(2, 0, 1, 1),
            Scale2D=(2, 0, 1, 1),
            EulerRotate2D=(1, 0, 1, 1),
            NoParamRectangle2D=(0, 0, 0, 2),
            NoParamCircle2D=(0, 0, 0, 2),
            NoParamTriangle2D=(0, 0, 0, 2),
        )
