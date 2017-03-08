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

from RBniCS.problems.nonlinear_elliptic.nonlinear_elliptic_problem import NonlinearEllipticProblem
from RBniCS.backends import NonlinearSolver, product, sum
from RBniCS.backends.online import OnlineFunction
from RBniCS.utils.decorators import Extends, override, MultiLevelReducedProblem

def NonlinearEllipticReducedProblem(EllipticCoerciveReducedProblem_DerivedClass):
    @Extends(EllipticCoerciveReducedProblem_DerivedClass) # needs to be first in order to override for last the methods.
    #@ReducedProblemFor(NonlinearEllipticProblem, NonlinearEllipticReductionMethod) # disabled, since now this is a decorator which depends on a derived (e.g. POD or RB) class
    @MultiLevelReducedProblem
    class NonlinearEllipticReducedProblem_Class(EllipticCoerciveReducedProblem_DerivedClass):
        
        ###########################     CONSTRUCTORS     ########################### 
        ## @defgroup Constructors Methods related to the construction of the reduced order model object
        #  @{
        
        ## Default initialization of members.
        @override
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            EllipticCoerciveReducedProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # Additional parameters
            self._nonlinear_solver_parameters = dict()
            
        #  @}
        ########################### end - CONSTRUCTORS - end ########################### 
        
        ###########################     ONLINE STAGE     ########################### 
        ## @defgroup OnlineStage Methods related to the online stage
        #  @{
        
        # Perform an online solve (internal)
        def _solve(self, N, **kwargs):
            N += self.N_bc
            # Functions required by the NonlinearSolver interface
            def residual_eval(solution):
                assembled_operator = dict()
                assembled_operator["a"] = sum(product(self.compute_theta("a"), self.operator["a"][:N]))
                assembled_operator["f"] = sum(product(self.compute_theta("f"), self.operator["f"][:N]))
                return assembled_operator["a"] - assembled_operator["f"]
            def jacobian_eval(solution):
                return sum(product(self.compute_theta("da"), self.operator["da"][:N, :N]))
            def bc_eval():
                if self.dirichlet_bc and not self.dirichlet_bc_are_homogeneous:
                    return self.compute_theta("dirichlet_bc")
                else:
                    return None
            # Solve by NonlinearSolver object
            solver = NonlinearSolver(jacobian_eval, self._solution, residual_eval, bc_eval())
            solver.set_parameters(self._nonlinear_solver_parameters)
            return self._solution
            
        #  @}
        ########################### end - ONLINE STAGE - end ########################### 
        
    # return value (a class) for the decorator
    return NonlinearEllipticReducedProblem_Class

