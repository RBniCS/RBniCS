# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ProblemDecoratorFor


def WeightedUncertaintyQuantificationDecoratedProblem(**decorator_kwargs):
    from .weighted_uncertainty_quantification import WeightedUncertaintyQuantification

    @ProblemDecoratorFor(WeightedUncertaintyQuantification)
    def WeightedUncertaintyQuantificationDecoratedProblem_Decorator(EllipticCoerciveProblem_DerivedClass):
        # return value (a class) for the decorator
        return EllipticCoerciveProblem_DerivedClass

    # return the decorator itself
    return WeightedUncertaintyQuantificationDecoratedProblem_Decorator
