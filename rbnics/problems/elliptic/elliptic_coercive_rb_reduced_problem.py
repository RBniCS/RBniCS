# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.elliptic.elliptic_coercive_problem import EllipticCoerciveProblem
from rbnics.problems.elliptic.elliptic_coercive_reduced_problem import EllipticCoerciveReducedProblem
from rbnics.problems.elliptic.elliptic_rb_reduced_problem import EllipticRBReducedProblem
from rbnics.reduction_methods.elliptic import EllipticRBReduction
from rbnics.utils.decorators import ReducedProblemFor

EllipticCoerciveRBReducedProblem_Base = EllipticCoerciveReducedProblem(EllipticRBReducedProblem)


@ReducedProblemFor(EllipticCoerciveProblem, EllipticRBReduction)
class EllipticCoerciveRBReducedProblem(EllipticCoerciveRBReducedProblem_Base):
    pass
