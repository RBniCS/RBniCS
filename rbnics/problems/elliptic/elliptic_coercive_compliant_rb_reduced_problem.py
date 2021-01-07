# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from math import sqrt
from numpy import isclose
from rbnics.problems.elliptic.elliptic_coercive_compliant_problem import EllipticCoerciveCompliantProblem
from rbnics.problems.elliptic.elliptic_coercive_compliant_reduced_problem import EllipticCoerciveCompliantReducedProblem
from rbnics.problems.elliptic.elliptic_coercive_rb_reduced_problem import EllipticCoerciveRBReducedProblem
from rbnics.reduction_methods.elliptic import EllipticRBReduction
from rbnics.utils.decorators import ReducedProblemFor

EllipticCoerciveCompliantRBReducedProblem_Base = EllipticCoerciveCompliantReducedProblem(
    EllipticCoerciveRBReducedProblem)


@ReducedProblemFor(EllipticCoerciveCompliantProblem, EllipticRBReduction)
class EllipticCoerciveCompliantRBReducedProblem(EllipticCoerciveCompliantRBReducedProblem_Base):
    # Return an error bound for the current solution
    def estimate_error(self):
        eps2 = self.get_residual_norm_squared()
        beta = self.truth_problem.get_stability_factor_lower_bound()
        assert eps2 >= 0. or isclose(eps2, 0.)
        assert beta >= 0.
        return sqrt(abs(eps2) / beta)

    # Return an error bound for the current compliant output
    def estimate_error_output(self):
        return self.estimate_error()**2
