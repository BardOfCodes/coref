import torch as th
from typing import Union as type_union
from geolipi.symbolic.base import GLFunction
from geolipi.symbolic.combinators import Union, Intersection, Difference
from geolipi.symbolic.primitives_2d import NoParamRectangle2D, NoParamCircle2D, NoParamTriangle2D, NullExpression2D
from geolipi.symbolic.transforms_2d import Translate2D, Scale2D, EulerRotate2D
from geolipi.symbolic.transforms_2d import ReflectX2D, ReflectY2D, RotationSymmetry2D, TranslationSymmetryX2D, TranslationSymmetryY2D
from geolipi.symbolic.primitives_3d import NoParamCuboid3D, NoParamSphere3D, NullExpression3D
from geolipi.symbolic.transforms_3d import Translate3D, Scale3D, EulerRotate3D
from geolipi.symbolic.transforms_3d import (ReflectX3D, ReflectY3D, ReflectZ3D,
                                            RotationSymmetryX3D, RotationSymmetryY3D, RotationSymmetryZ3D,
                                            TranslationSymmetryX3D, TranslationSymmetryY3D, TranslationSymmetryZ3D)


# For 3D
# Ref X Y, TransSym X, Y, Z (n = 2, 3, 4, 5) RotSym Z (3, 4, 5).


BaseNull3D = Difference(NoParamCuboid3D(), NoParamCuboid3D())
BaseNull2D = Difference(NoParamRectangle2D(), NoParamRectangle2D())


PRIMS = [NoParamCuboid3D, NoParamSphere3D,
         NoParamRectangle2D, NoParamCircle2D, NoParamTriangle2D]
TRANSFORMS = [Translate2D, Scale2D, EulerRotate2D,
              Translate3D, Scale3D, EulerRotate3D]
TRANSLATES = [Translate2D, Translate3D]
SCALES = [Scale2D, Scale3D]
ROTATES = [EulerRotate2D, EulerRotate3D]
REFLECT_SYMS = [ReflectX2D, ReflectY2D, ReflectX3D, ReflectY3D, ReflectZ3D]
ROTATION_SYMS = [RotationSymmetry2D, RotationSymmetryX3D,
                 RotationSymmetryY3D, RotationSymmetryZ3D]
TRANSLATION_SYMS = [TranslationSymmetryX2D, TranslationSymmetryY2D,
                    TranslationSymmetryX3D, TranslationSymmetryY3D, TranslationSymmetryZ3D]
MACROS = REFLECT_SYMS + ROTATION_SYMS + TRANSLATION_SYMS
COMBINATORS = [Union, Intersection, Difference]


ALL_FUNCTIONS = COMBINATORS + PRIMS + TRANSFORMS + \
    MACROS + [NullExpression2D, NullExpression3D]
STR_TO_CMD_MAPPER = {x.__name__: x for x in ALL_FUNCTIONS}
STR_TO_CMD_MAPPER["tensor"] = th.tensor
STR_TO_CMD_MAPPER["torch"] = th

PRIM_TYPE = tuple([x for x in PRIMS])
TRANSFORM_TYPE = tuple([x for x in TRANSFORMS])
TRANSLATE_TYPE = tuple([x for x in TRANSLATES])
SCALE_TYPE = tuple([x for x in SCALES])
ROTATE_TYPE = tuple([x for x in ROTATES])
REFSYM_TYPE = tuple([x for x in REFLECT_SYMS])
ROTSYM_TYPE = tuple([x for x in ROTATION_SYMS])
TRANSSYM_TYPE = tuple([x for x in TRANSLATION_SYMS])
MACRO_TYPE = tuple([x for x in MACROS])
COMBINATOR_TYPE = tuple([x for x in COMBINATORS])

PARAM_SYM_TYPE = tuple([x for x in MACROS if x not in REFLECT_SYMS])
HAS_PARAM_TYPE = tuple(
    [x for x in TRANSLATE_TYPE + SCALE_TYPE + ROTATE_TYPE + PARAM_SYM_TYPE])
NO_PARAM_TYPE = tuple([x for x in PRIM_TYPE])
TREE_PARAM_TYPE = tuple([x for x in REFSYM_TYPE])
NULL_EXPR_TYPE = tuple([x for x in [NullExpression2D, NullExpression3D]])
