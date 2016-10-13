# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file elliptic_coercive_problem.py
#  @brief Base class for elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.problems.base import ParametrizedDifferentialProblem
from RBniCS.backends import AffineExpansionStorage, Function, LinearSolver, product, sum, transpose
from RBniCS.utils.decorators import Extends, override

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE PROBLEM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveProblem
#
# Base class containing the definition of saddle point problems
@Extends(ParametrizedDifferentialProblem)
class SaddlePointProblem(ParametrizedDifferentialProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the elliptic problem
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, V, **kwargs):
        # Call to parent
        ParametrizedDifferentialProblem.__init__(self, V, **kwargs)
        
        # Form names for saddle point problems
        self.terms = [
            "a", "b", "bt", "f", "g",
            # Auxiliary terms for supremizer enrichment
            "inner_product_s_restricted", "bt_restricted"
        ]
        self.terms_order = {
            "a": 2, "b": 2, "bt": 2, "f": 1, "g": 1,
            # Auxiliary terms for supremizer enrichment
            "inner_product_s_restricted": 2, "bt_restricted": 2
        }
        self.components_name = ["u", "s", "p"]
        
        # Saddle point problems have three components: two variables and one supremizer: must
        # declare mapping from components to basis.
        # The first component of each snapshot will be mapped to the first component of the basis.
        # The first component of each auxiliary supremizer snapshot will be mapped to the second component of the basis.
        # The second component of each snapshot will be mapped to the third component of the basis.
        self.component_name_to_basis_component_index = {
            "u": 0,
            "s": 1,
            "p": 2
        }
        self.component_name_to_function_component = {
            "u": 0,
            "s": 0,
            "p": 1
        }
        
        # Auxiliary storage for supremizer enrichment, using a subspace of V
        self._supremizer = Function(V, self.component_name_to_function_component["s"])
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
        
    ## Perform a truth solve
    @override
    def solve(self, **kwargs):
        assembled_operator = dict()
        for term in ("a", "b", "bt", "f", "g"):
            assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term]))
        assembled_dirichlet_bc = list()
        for component_name in ("u", "p"):
            if self.dirichlet_bc[component_name] is not None:
                assembled_dirichlet_bc.extend(sum(product(self.compute_theta("dirichlet_bc_" + component_name), self.dirichlet_bc[component_name])))
        if len(assembled_dirichlet_bc) == 0:
            assembled_dirichlet_bc = None
        solver = LinearSolver(
            assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"],
            self._solution,
            assembled_operator["f"] + assembled_operator["g"],
            assembled_dirichlet_bc
        )
        solver.solve()
        return self._solution
    
    def solve_supremizer(self):
        assert len(self.operator["inner_product_s_restricted"]) == 1 # the affine expansion storage contains only the inner product matrix
        assembled_operator_lhs = self.operator["inner_product_s_restricted"][0]
        assembled_operator_bt = sum(product(self.compute_theta("bt_restricted"), self.operator["bt_restricted"]))
        assembled_operator_rhs = assembled_operator_bt*self._solution.vector()
        if self.dirichlet_bc["s"] is not None:
            assembled_dirichlet_bc = sum(product(self.compute_theta("dirichlet_bc_s"), self.dirichlet_bc["s"]))
        else:
            assembled_dirichlet_bc = None
        solver = LinearSolver(
            assembled_operator_lhs,
            self._supremizer,
            assembled_operator_rhs,
            assembled_dirichlet_bc
        )
        solver.solve()
        return self._supremizer
        
    ## Perform a truth evaluation of the output
    @override
    def output(self):
        return 0. # TODO
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
