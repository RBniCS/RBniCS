# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from math import sqrt
from numpy import isclose
from rbnics.backends import product, sum, transpose
from rbnics.problems.base import LinearRBReducedProblem, ParametrizedReducedDifferentialProblem
from rbnics.problems.elliptic.elliptic_problem import EllipticProblem
from rbnics.problems.elliptic.elliptic_reduced_problem import EllipticReducedProblem
from rbnics.reduction_methods.elliptic import EllipticRBReduction
from rbnics.utils.decorators import ReducedProblemFor

EllipticRBReducedProblem_Base = LinearRBReducedProblem(EllipticReducedProblem(ParametrizedReducedDifferentialProblem))


# Base class containing the interface of a projection based ROM
# for elliptic problems.
# The following implementation will be retained if no output is provided in the "s" term
@ReducedProblemFor(EllipticProblem, EllipticRBReduction)
class EllipticRBReducedProblem(EllipticRBReducedProblem_Base):

    # Default initialization of members.
    def __init__(self, truth_problem, **kwargs):
        # Call to parent
        EllipticRBReducedProblem_Base.__init__(self, truth_problem, **kwargs)

        # Skip useless Riesz products
        self.riesz_terms = ["f", "a"]
        self.error_estimation_terms = [("f", "f"), ("a", "f"), ("a", "a")]

    # Return an error bound for the current solution
    def estimate_error(self):
        eps2 = self.get_residual_norm_squared()
        beta = self.truth_problem.get_stability_factor_lower_bound()
        assert eps2 >= 0. or isclose(eps2, 0.)
        assert beta >= 0.
        return sqrt(abs(eps2)) / beta

    # Return a relative error bound for the current solution
    def estimate_relative_error(self):
        return NotImplemented

    # Return an error bound for the current output
    def estimate_error_output(self):
        return NotImplemented

    # Return a relative error bound for the current output
    def estimate_relative_error_output(self):
        return NotImplemented

    # Return the numerator of the error bound for the current solution
    def get_residual_norm_squared(self):
        N = self._solution.N
        theta_a = self.compute_theta("a")
        theta_f = self.compute_theta("f")
        return (sum(product(theta_f, self.error_estimation_operator["f", "f"], theta_f))
                + 2.0 * (transpose(self._solution)
                         * sum(product(theta_a, self.error_estimation_operator["a", "f"][:N], theta_f)))
                + (transpose(self._solution)
                   * sum(product(theta_a, self.error_estimation_operator["a", "a"][:N, :N], theta_a))
                   * self._solution))
