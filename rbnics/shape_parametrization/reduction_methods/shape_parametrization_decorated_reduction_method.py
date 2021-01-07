# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import PreserveClassName, ReductionMethodDecoratorFor
from rbnics.shape_parametrization.problems import ShapeParametrization


@ReductionMethodDecoratorFor(ShapeParametrization)
def ShapeParametrizationDecoratedReductionMethod(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class ShapeParametrizationDecoratedReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)

    # return value (a class) for the decorator
    return ShapeParametrizationDecoratedReductionMethod_Class
