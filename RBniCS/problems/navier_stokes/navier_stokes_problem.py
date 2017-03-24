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
## @file parabolic_coercive_problem.py
#  @brief Base class for parabolic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.problems.base import NonlinearProblem
from RBniCS.problems.stokes import StokesProblem
from RBniCS.backends import assign, Function, LinearSolver, NonlinearSolver, product, sum
from RBniCS.utils.decorators import Extends, override

@Extends(StokesProblem)
@NonlinearProblem
class NavierStokesProblem(StokesProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of Navier Stokes problem
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, V, **kwargs):
        # Call to parent
        StokesProblem.__init__(self, V, **kwargs)
        
        # Form names for Navier-Stokes problems
        self.terms = [
            "a", "b", "bt", "c", "f", "g",
            "da", "db", "dbt", "dc",
            # Auxiliary terms for supremizer enrichment
            "bt_restricted"
        ]
        self.terms_order = {
            "a": 1, "b": 1, "bt": 1, "c": 1, "f": 1, "g": 1,
            "da": 2, "db": 2, "dbt": 2, "dc": 2,
            # Auxiliary terms for supremizer enrichment
            "bt_restricted": 2
        }
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
            
    ## Perform a truth solve.
    @override
    def _solve(self, **kwargs):
        # Functions required by the NonlinearSolver interface
        def residual_eval(solution):
            self._store_solution(solution)
            assembled_operator = dict()
            for term in ("a", "b", "bt", "c", "f", "g"):
                assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term]))
            return (
                  assembled_operator["a"] + assembled_operator["c"]
                + assembled_operator["b"] + assembled_operator["bt"]
                - assembled_operator["f"] - assembled_operator["g"]
            )
        def jacobian_eval(solution):
            self._store_solution(solution)
            assembled_operator = dict()
            for term in ("da", "db", "dbt", "dc"):
                assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term]))
            return (
                  assembled_operator["da"] + assembled_operator["dc"]
                + assembled_operator["db"] + assembled_operator["dbt"]
            )
        def bc_eval():
            assembled_dirichlet_bc = dict()
            for component in ("u", "p"):
                if self.dirichlet_bc[component] is not None:
                    assembled_dirichlet_bc[component] = sum(product(self.compute_theta("dirichlet_bc_" + component), self.dirichlet_bc[component]))
            if len(assembled_dirichlet_bc) == 0:
                assembled_dirichlet_bc = None
            return assembled_dirichlet_bc
        # Solve by NonlinearSolver object
        assign(self._solution, Function(self.V))
        solver = NonlinearSolver(jacobian_eval, self._solution, residual_eval, bc_eval())
        solver.set_parameters(self._nonlinear_solver_parameters)
        solver.solve()
        
    def solve_supremizer(self):
        assert len(self.inner_product["s"]) == 1 # the affine expansion storage contains only the inner product matrix
        assembled_operator_lhs = self.inner_product["s"][0]
        assembled_operator_rhs = sum(product(self.compute_theta("bt_restricted"), self.operator["bt_restricted"]))
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
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
