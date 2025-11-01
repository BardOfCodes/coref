
from collections import defaultdict
import numpy as np
from typing import Union as type_union
import torch as th

from geolipi.symbolic.base import GLFunction
from geolipi.symbolic.primitives_3d import Primitive3D
from geolipi.symbolic.primitives_3d import NoParamCuboid3D, NoParamSphere3D, NullExpression3D
from geolipi.symbolic.primitives_2d import Primitive2D
from geolipi.symbolic.primitives_2d import NoParamRectangle2D, NoParamCircle2D, NoParamTriangle2D, NullExpression2D
from geolipi.symbolic.combinators import Union, Intersection, Difference


class Primal3D(Primitive3D):
    ...


class PrimalCuboid3D(Primal3D):
    @classmethod
    def eval(cls, *args):
        return None


class PrimalSphere3D(Primal3D):
    @classmethod
    def eval(cls, *args):
        return None


class PrimalCylinder3D(Primal3D):
    @classmethod
    def eval(cls, *args):
        return None


class PrimalHalfPlan3D(Primal3D):
    @classmethod
    def eval(cls, *args):
        return None


class PrimalCone3D(Primal3D):
    @classmethod
    def eval(cls, *args):
        return None


class PrimalSQ3D(Primal3D):
    @classmethod
    def eval(cls, *args):
        return None


class Primal2D(Primitive2D):
    ...


class PrimalRectangle2D(Primal2D):
    @classmethod
    def eval(cls, *args):
        return None


class PrimalCircle2D(Primal2D):
    @classmethod
    def eval(cls, *args):
        return None


class PrimalTriangle2D(Primal2D):
    @classmethod
    def eval(cls, *args):
        return None


BaseNull3D = Difference(NoParamCuboid3D(), NoParamCuboid3D())
BaseNull2D = Difference(NoParamRectangle2D(), NoParamRectangle2D())


BASE_TO_PRIMALPRIM_MAPPER = {
    NoParamCuboid3D: PrimalCuboid3D,
    NoParamSphere3D: PrimalSphere3D,
    NoParamRectangle2D: PrimalRectangle2D,
    NoParamCircle2D: PrimalCircle2D,
    NoParamTriangle2D: PrimalTriangle2D
}
PRIMALPRIM_TO_BASE_MAPPER = {v: k for k,
                             v in BASE_TO_PRIMALPRIM_MAPPER.items()}

NULL_EXPR_TYPE = tuple([x for x in [NullExpression2D, NullExpression3D]])
PRIMAL_PRIMS = [PrimalCuboid3D, PrimalSphere3D, PrimalCylinder3D, PrimalHalfPlan3D, PrimalCone3D, PrimalSQ3D,
                PrimalRectangle2D, PrimalCircle2D, PrimalTriangle2D]
PRIMAL_TYPE = type_union[Primal3D, Primal2D]

ALL_FUNCTIONS = [Union, Intersection, Difference] + \
    PRIMAL_PRIMS + [NullExpression2D, NullExpression3D]
STR_TO_CMD_MAPPER = {x.__name__: x for x in ALL_FUNCTIONS}
STR_TO_CMD_MAPPER["tensor"] = th.tensor
STR_TO_CMD_MAPPER["torch"] = th
