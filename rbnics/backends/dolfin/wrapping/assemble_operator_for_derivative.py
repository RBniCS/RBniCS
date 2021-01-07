# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import derivative, TrialFunction


def assemble_operator_for_derivative(jacobian_term_to_residual_term):

    def assemble_operator_for_derivative_decorator(assemble_operator):

        def assemble_operator_for_derivative_decorator_impl(self, term):
            residual_term = jacobian_term_to_residual_term.get(term)
            if residual_term is None:  # term was not a jacobian_term
                return assemble_operator(self, term)
            else:
                trial = TrialFunction(self.V)
                return tuple(derivative(op, self._solution, trial) for op in assemble_operator(self, residual_term))

        return assemble_operator_for_derivative_decorator_impl

    return assemble_operator_for_derivative_decorator
