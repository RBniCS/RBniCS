# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

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
        self._output = transpose(self._solution)*sum(product(self.compute_theta("f"), self.operator["f"]))
