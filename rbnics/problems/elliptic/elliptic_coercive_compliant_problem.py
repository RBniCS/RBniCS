# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends import product, sum, transpose
from rbnics.problems.elliptic.elliptic_coercive_problem import EllipticCoerciveProblem


# Base class containing the definition of elliptic coercive compliant problems
class EllipticCoerciveCompliantProblem(EllipticCoerciveProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call to parent
        EllipticCoerciveProblem.__init__(self, V, **kwargs)

        # Remove "s" from both terms and terms_order
        self.terms.remove("s")
        del self.terms_order["s"]

    # Perform a truth evaluation of the compliant output
    def _compute_output(self):
        self._output = transpose(self._solution) * sum(product(self.compute_theta("f"), self.operator["f"]))
