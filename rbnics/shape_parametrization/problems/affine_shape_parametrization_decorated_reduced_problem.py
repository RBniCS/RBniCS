# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ReducedProblemDecoratorFor
from rbnics.shape_parametrization.problems.affine_shape_parametrization import AffineShapeParametrization
from rbnics.shape_parametrization.problems.shape_parametrization_decorated_reduced_problem import (
    ShapeParametrizationDecoratedReducedProblem)


@ReducedProblemDecoratorFor(AffineShapeParametrization)
def AffineShapeParametrizationDecoratedReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    return ShapeParametrizationDecoratedReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass)
