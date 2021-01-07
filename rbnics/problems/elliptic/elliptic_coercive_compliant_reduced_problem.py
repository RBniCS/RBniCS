# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends import product, sum, transpose


def EllipticCoerciveCompliantReducedProblem(EllipticCoerciveReducedProblem_DerivedClass):

    EllipticCoerciveCompliantReducedProblem_Base = EllipticCoerciveReducedProblem_DerivedClass

    # Base class containing the interface of a projection based ROM
    # for elliptic coercive compliant problems.
    class EllipticCoerciveCompliantReducedProblem_Class(EllipticCoerciveCompliantReducedProblem_Base):
        # Perform an online evaluation of the compliant output
        def _compute_output(self, N):
            self._output = transpose(self._solution) * sum(product(self.compute_theta("f"), self.operator["f"][:N]))

        # Internal method for error computation
        def _compute_error(self, **kwargs):
            inner_product = dict()
            inner_product["u"] = sum(product(
                self.truth_problem.compute_theta("a"), self.truth_problem.operator["a"]))  # use the energy norm
            assert "inner_product" not in kwargs
            kwargs["inner_product"] = inner_product
            return EllipticCoerciveCompliantReducedProblem_Base._compute_error(self, **kwargs)

        # Internal method for relative error computation
        def _compute_relative_error(self, absolute_error, **kwargs):
            inner_product = dict()
            inner_product["u"] = sum(product(
                self.truth_problem.compute_theta("a"), self.truth_problem.operator["a"]))  # use the energy norm
            assert "inner_product" not in kwargs
            kwargs["inner_product"] = inner_product
            return EllipticCoerciveCompliantReducedProblem_Base._compute_relative_error(self, absolute_error, **kwargs)

    # return value (a class) for the decorator
    return EllipticCoerciveCompliantReducedProblem_Class
