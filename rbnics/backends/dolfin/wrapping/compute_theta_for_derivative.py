# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later


def compute_theta_for_derivative(jacobian_term_to_residual_term):

    def compute_theta_for_derivative_decorator(compute_theta):

        def compute_theta_for_derivative_decorator_impl(self, term):
            residual_term = jacobian_term_to_residual_term.get(term)
            if residual_term is None:  # term was not a jacobian_term
                return compute_theta(self, term)
            else:
                return compute_theta(self, residual_term)

        return compute_theta_for_derivative_decorator_impl

    return compute_theta_for_derivative_decorator
