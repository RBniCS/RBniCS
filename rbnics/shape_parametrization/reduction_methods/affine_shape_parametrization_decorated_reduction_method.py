# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ReductionMethodDecoratorFor
from rbnics.shape_parametrization.problems import AffineShapeParametrization
from rbnics.shape_parametrization.reduction_methods.shape_parametrization_decorated_reduction_method import (
    ShapeParametrizationDecoratedReductionMethod)


@ReductionMethodDecoratorFor(AffineShapeParametrization)
def AffineShapeParametrizationDecoratedReductionMethod(DifferentialProblemReductionMethod_DerivedClass):
    return ShapeParametrizationDecoratedReductionMethod(DifferentialProblemReductionMethod_DerivedClass)
