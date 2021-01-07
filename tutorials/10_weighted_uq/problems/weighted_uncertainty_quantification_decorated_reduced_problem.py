# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ReducedProblemDecoratorFor
from .weighted_uncertainty_quantification import WeightedUncertaintyQuantification


@ReducedProblemDecoratorFor(WeightedUncertaintyQuantification)
def WeightedUncertaintyQuantificationDecoratedReducedProblem(EllipticCoerciveReducedProblem_DerivedClass):
    # return value (a class) for the decorator
    return EllipticCoerciveReducedProblem_DerivedClass
