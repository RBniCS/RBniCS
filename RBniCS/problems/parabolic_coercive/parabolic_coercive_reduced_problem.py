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
## @file 
#  @brief 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from math import sqrt
from RBniCS.problems.base import TimeDependentReducedProblem
from RBniCS.problems.elliptic_coercive import EllipticCoerciveReducedProblem
from RBniCS.problems.parabolic_coercive.parabolic_coercive_problem import ParabolicCoerciveProblem
from RBniCS.backends import product, sum, TimeQuadrature, TimeStepping
from RBniCS.backends.online import OnlineFunction
from RBniCS.utils.decorators import Extends, override, ReducedProblemFor, MultiLevelReducedProblem
from RBniCS.reduction_methods.parabolic_coercive import ParabolicCoerciveReductionMethod

#~~~~~~~~~~~~~~~~~~~~~~~~~     PARABOLIC COERCIVE REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ParabolicCoerciveReducedOrderModelBase
#
# Base class containing the interface of a projection based ROM
# for parabolic coercive problems.
@Extends(EllipticCoerciveReducedProblem) # needs to be first in order to override for last the methods.
@ReducedProblemFor(ParabolicCoerciveProblem, ParabolicCoerciveReductionMethod)
@MultiLevelReducedProblem
@TimeDependentReducedProblem
class ParabolicCoerciveReducedProblem(EllipticCoerciveReducedProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members.
    @override
    def __init__(self, truth_problem):
        # Call to parent
        EllipticCoerciveReducedProblem.__init__(self, truth_problem)
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
                
    # Perform an online solve. self.N will be used as matrix dimension if the default value is provided for N.
    @override
    def solve(self, N=None, **kwargs):
        if N is None:
            N = self.N
        uN = self._solve(N, **kwargs)
        return uN
    
    # Perform an online solve (internal)
    def _solve(self, N, **kwargs):
        N += self.N_bc
        # Functions required by the TimeStepping interface
        def residual_eval(t, solution, solution_dot):
            store_time(t)
            store_solution(solution, solution_dot)
            assembled_operator = dict()
            assembled_operator["m"] = sum(product(self.compute_theta("m"), self.operator["m"][:N, :N]))
            assembled_operator["a"] = sum(product(self.compute_theta("a"), self.operator["a"][:N, :N]))
            assembled_operator["f"] = sum(product(self.compute_theta("f"), self.operator["f"][:N]))
            return (
                  assembled_operator["m"]*solution_dot
                + assembled_operator["a"]*solution
                - assembled_operator["f"]
            )
        def jacobian_eval(t, solution, solution_dot, solution_dot_coefficient):
            store_time(t)
            store_solution(solution, solution_dot)
            assembled_operator = dict()
            assembled_operator["m"] = sum(product(self.compute_theta("m"), self.operator["m"][:N, :N]))
            assembled_operator["a"] = sum(product(self.compute_theta("a"), self.operator["a"][:N, :N]))
            return (
                  assembled_operator["m"]*solution_dot_coefficient
                + assembled_operator["a"]
            )
        def bc_eval(t):
            store_time(t)
            if self.dirichlet_bc and not self.dirichlet_bc_are_homogeneous:
                return self.compute_theta("dirichlet_bc")
            else:
                return None
        def store_time(t):
            self.t = t
        def store_solution(solution, solution_dot):
            assign(self._solution, solution)
            assign(self._solution_dot, solution_dot)
        # Solve by TimeStepping object
        self._solution = OnlineFunction(N)
        self._solution_dot = OnlineFunction(N)
        solver = TimeStepping(jacobian_eval, self._solution, residual_eval, bc_eval)
        solver.set_parameters(self._time_stepping_parameters)
        self._solution_over_time = solver.solve()
        assign(self._solution, self._solution_over_time[-1])
        return self._solution_over_time
        
    # Perform an online evaluation of the (compliant) output
    @override
    def output(self):
        return 1. # TODO fill in self._output_over_time and self._output
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    

