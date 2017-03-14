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
from RBniCS.problems.elliptic_coercive import EllipticCoerciveProblem
from RBniCS.backends import assign, Function, NonlinearSolver, product, sum
from RBniCS.utils.decorators import Extends, override

@Extends(EllipticCoerciveProblem)
@NonlinearProblem
class NonlinearEllipticProblem(EllipticCoerciveProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the elliptic problem
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, V, **kwargs):
        # Call to parent
        EllipticCoerciveProblem.__init__(self, V, **kwargs)
        
        # Form names for nonlinear problems
        self.terms = ["a", "da", "f"]
        self.terms_order = {"a": 1, "da": 2, "f": 1}
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
            
    ## Perform a truth solve.
    @override
    def solve(self, **kwargs):
        # Functions required by the NonlinearSolver interface
        def residual_eval(solution):
            self._store_solution(solution)
            assembled_operator = dict()
            assembled_operator["a"] = sum(product(self.compute_theta("a"), self.operator["a"]))
            assembled_operator["f"] = sum(product(self.compute_theta("f"), self.operator["f"]))
            return assembled_operator["a"] - assembled_operator["f"]
        def jacobian_eval(solution):
            self._store_solution(solution)
            return sum(product(self.compute_theta("da"), self.operator["da"]))
        def bc_eval():
            if self.dirichlet_bc is not None:
                return sum(product(self.compute_theta("dirichlet_bc"), self.dirichlet_bc))
            else:
                return None
        # Solve by NonlinearSolver object
        assign(self._solution, Function(self.V))
        solver = NonlinearSolver(jacobian_eval, self._solution, residual_eval, bc_eval())
        solver.set_parameters(self._nonlinear_solver_parameters)
        solver.solve()
        return self._solution
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
