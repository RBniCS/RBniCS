# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import PreserveClassName, ReducedProblemDecoratorFor
from rbnics.shape_parametrization.problems.shape_parametrization import ShapeParametrization


@ReducedProblemDecoratorFor(ShapeParametrization)
def ShapeParametrizationDecoratedReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    # A decorator class that allows to overload methods related to shape parametrization and mesh motion
    @PreserveClassName
    class ShapeParametrizationDecoratedReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):

        def __init__(self, truth_problem, **kwargs):
            # Call the standard initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)

    # return value (a class) for the decorator
    return ShapeParametrizationDecoratedReducedProblem_Class
