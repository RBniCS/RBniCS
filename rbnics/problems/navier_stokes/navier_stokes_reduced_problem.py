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
## @file 
#  @brief 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from rbnics.problems.base import NonlinearReducedProblem
from rbnics.problems.base import ParametrizedReducedDifferentialProblem
from rbnics.problems.navier_stokes.navier_stokes_problem import NavierStokesProblem
from rbnics.backends import assign, NonlinearSolver, product, sum
from rbnics.backends.online import OnlineFunction
from rbnics.utils.decorators import Extends, override, MultiLevelReducedProblem

def NavierStokesReducedProblem(StokesReducedProblem_DerivedClass):
    @Extends(StokesReducedProblem_DerivedClass) # needs to be first in order to override for last the methods.
    #@ReducedProblemFor(NavierStokesProblem, NavierStokesReductionMethod) # disabled, since now this is a decorator which depends on a derived (e.g. POD or RB) class
    @MultiLevelReducedProblem
    @NonlinearReducedProblem
    class NavierStokesReducedProblem_Class(StokesReducedProblem_DerivedClass):
        
        ###########################     CONSTRUCTORS     ########################### 
        ## @defgroup Constructors Methods related to the construction of the reduced order model object
        #  @{
        
        ## Default initialization of members.
        @override
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            StokesReducedProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            
        #  @}
        ########################### end - CONSTRUCTORS - end ########################### 
        
        ###########################     ONLINE STAGE     ########################### 
        ## @defgroup OnlineStage Methods related to the online stage
        #  @{
        
        # Perform an online solve (internal)
        def _solve(self, N, **kwargs):
            # Functions required by the NonlinearSolver interface
            def residual_eval(solution):
                self._store_solution(solution)
                assembled_operator = dict()
                for term in ("a", "b", "bt", "c", "f", "g"):
                    assert self.terms_order[term] in (1, 2)
                    if self.terms_order[term] == 2:
                        assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term][:N, :N]))
                    elif self.terms_order[term] == 1:
                        assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term][:N]))
                    else:
                        raise AssertionError("Invalid value for order of term " + term)
                return (
                      assembled_operator["a"] + assembled_operator["c"]
                    + assembled_operator["b"] + assembled_operator["bt"]
                    - assembled_operator["f"] - assembled_operator["g"]
                )
            def jacobian_eval(solution):
                self._store_solution(solution)
                assembled_operator = dict()
                for term in ("da", "db", "dbt", "dc"):
                    assert self.terms_order[term] is 2
                    assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term][:N, :N]))
                return (
                      assembled_operator["da"] + assembled_operator["dc"]
                    + assembled_operator["db"] + assembled_operator["dbt"]
                )
            def bc_eval():
                theta_bc = dict()
                for component in ("u", "p"):
                    if self.dirichlet_bc[component] and not self.dirichlet_bc_are_homogeneous[component]:
                        theta_bc[component] = self.compute_theta("dirichlet_bc_" + component)
                if len(theta_bc) == 0:
                    theta_bc = None
                return theta_bc
            # Solve by NonlinearSolver object
            solver = NonlinearSolver(jacobian_eval, self._solution, residual_eval, bc_eval())
            solver.set_parameters(self._nonlinear_solver_parameters)
            solver.solve()
            
        #  @}
        ########################### end - ONLINE STAGE - end ########################### 
        
    # return value (a class) for the decorator
    return NavierStokesReducedProblem_Class

