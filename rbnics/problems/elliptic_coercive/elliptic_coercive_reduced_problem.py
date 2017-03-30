# Copyright (C) 2015-2017 by the RBniCS authors
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

from rbnics.problems.base import ParametrizedReducedDifferentialProblem
from rbnics.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from rbnics.backends import LinearSolver, product, sum, transpose
from rbnics.backends.online import OnlineFunction
from rbnics.utils.decorators import Extends, override, ReducedProblemFor, MultiLevelReducedProblem
from rbnics.reduction_methods.elliptic_coercive import EllipticCoerciveReductionMethod

# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
@Extends(ParametrizedReducedDifferentialProblem) # needs to be first in order to override for last the methods.
@ReducedProblemFor(EllipticCoerciveProblem, EllipticCoerciveReductionMethod)
@MultiLevelReducedProblem
class EllipticCoerciveReducedProblem(ParametrizedReducedDifferentialProblem):
    
    ## Default initialization of members.
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call to parent
        ParametrizedReducedDifferentialProblem.__init__(self, truth_problem, **kwargs)
    
    # Perform an online solve (internal)
    def _solve(self, N, **kwargs):
        assembled_operator = dict()
        assembled_operator["a"] = sum(product(self.compute_theta("a"), self.operator["a"][:N, :N]))
        assembled_operator["f"] = sum(product(self.compute_theta("f"), self.operator["f"][:N]))
        if self.dirichlet_bc and not self.dirichlet_bc_are_homogeneous:
            theta_bc = self.compute_theta("dirichlet_bc")
        else:
            theta_bc = None
        solver = LinearSolver(assembled_operator["a"], self._solution, assembled_operator["f"], theta_bc)
        solver.solve()
        
    # Perform an online evaluation of the (compliant) output
    @override
    def _compute_output(self, N):
        assembled_output_operator = sum(product(self.compute_theta("f"), self.operator["f"][:N]))
        self._output = transpose(assembled_output_operator)*self._solution
    
    # Internal method for error computation
    @override
    def _compute_error(self, **kwargs):
        inner_product = dict()
        inner_product["u"] = sum(product(self.truth_problem.compute_theta("a"), self.truth_problem.operator["a"])) # use the energy norm
        assert "inner_product" not in kwargs
        kwargs["inner_product"] = inner_product
        return ParametrizedReducedDifferentialProblem._compute_error(self, **kwargs)
        
    # Internal method for relative error computation
    @override
    def _compute_relative_error(self, absolute_error, **kwargs):
        inner_product = dict()
        inner_product["u"] = sum(product(self.truth_problem.compute_theta("a"), self.truth_problem.operator["a"])) # use the energy norm
        assert "inner_product" not in kwargs
        kwargs["inner_product"] = inner_product
        return ParametrizedReducedDifferentialProblem._compute_relative_error(self, absolute_error, **kwargs)
        
