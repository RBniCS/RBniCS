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
## @file parabolic_coercive_problem.py
#  @brief Base class for parabolic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.problems.base import TimeDependentProblem
from RBniCS.problems.elliptic_coercive import EllipticCoerciveProblem
from RBniCS.backends import assign, Function, product, sum, TimeStepping
from RBniCS.utils.decorators import Extends, override

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE PROBLEM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveProblem
#
# Base class containing the definition of parabolic coercive problems
@Extends(EllipticCoerciveProblem)
@TimeDependentProblem
class ParabolicCoerciveProblem(EllipticCoerciveProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the elliptic problem
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, V, **kwargs):
        # Call to parent
        EllipticCoerciveProblem.__init__(self, V, **kwargs)
        
        # Form names for parabolic problems
        self.terms = ["m", "a", "f"]
        self.terms_order = {"m": 2, "a": 2, "f": 1}
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Perform a truth solve
    @override
    def solve(self, **kwargs):
        # Functions required by the TimeStepping interface
        def residual_eval(t, solution, solution_dot):
            self.t = t
            assembled_operator = dict()
            assembled_operator["m"] = sum(product(self.compute_theta("m"), self.operator["m"]))
            assembled_operator["a"] = sum(product(self.compute_theta("a"), self.operator["a"]))
            assembled_operator["f"] = sum(product(self.compute_theta("f"), self.operator["f"]))
            return (
                  assembled_operator["m"]*solution_dot
                + assembled_operator["a"]*solution
                - assembled_operator["f"]
            )
        def jacobian_eval(t, solution, solution_dot, solution_dot_coefficient):
            self.t = t
            assembled_operator = dict()
            assembled_operator["m"] = sum(product(self.compute_theta("m"), self.operator["m"]))
            assembled_operator["a"] = sum(product(self.compute_theta("a"), self.operator["a"]))
            return (
                  assembled_operator["m"]*solution_dot_coefficient
                + assembled_operator["a"]
            )
        def bc_eval(t):
            self.t = t
            if self.dirichlet_bc is not None:
                return sum(product(self.compute_theta("dirichlet_bc"), self.dirichlet_bc))
            else:
                return None
        # Setup initial condition
        if self.initial_condition is not None:
            assign(self._solution, sum(product(self.compute_theta("initial_condition"), self.initial_condition)))
        else:
            assign(self._solution, Function(self.V))
        # Solve by TimeStepping object
        solver = TimeStepping(jacobian_eval, self._solution, residual_eval, bc_eval)
        solver.set_parameters(self._time_stepping_parameters)
        (_, self._solution_over_time, self._solution_dot_over_time) = solver.solve()
        assign(self._solution, self._solution_over_time[-1])
        assign(self._solution_dot, self._solution_dot_over_time[-1])
        return self._solution_over_time
        
    ## Perform a truth evaluation of the (compliant) output
    @override
    def output(self):
        return 1. # TODO fill in self._output_over_time and self._output
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
