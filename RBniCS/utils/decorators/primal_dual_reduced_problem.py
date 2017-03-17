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

from RBniCS.utils.decorators.extends import Extends
from RBniCS.utils.decorators.override import override

def PrimalDualReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
            
    @Extends(ParametrizedReducedDifferentialProblem_DerivedClass, preserve_class_name=True)
    class PrimalDualReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        ## Default initialization of members.
        @override
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # Dual reduced problem, which will be attached by reduction method at the end of the offline stage
            self.dual_reduced_problem = None
            self._dual_solve_latest_N = None
            self._dual_solve_latest_kwargs = None

        # Perform an online solve. Overridden to also solve the dual problem for output correction and error estimation.
        @override
        def solve(self, N=None, **kwargs):
            # Solve primal problem
            primal_solution = ParametrizedReducedDifferentialProblem_DerivedClass.solve(self, N, **kwargs)
            # Defer dual problem solve to output() method, since (i) reduced dual problem has not been built yet during the
            # primal offline stage, (ii) in any case dual solution is only required when computing output and its error estimation
            if self.dual_reduced_problem is not None:
                if "dual" in kwargs:
                    self._dual_solve_latest_N = kwargs["dual"]
                else:
                    self._dual_solve_latest_N = min(N, self.dual_reduced_problem.N)
                self._dual_solve_latest_kwargs = kwargs
            # Return primal solution
            return primal_solution
            
        # Perform an online evaluation of the non compliant output. Overridden to add output correction.
        @override
        def compute_output(self):
            # Solve dual problem for output correction and its error estimation
            self.dual_reduced_problem.solve(self._dual_solve_latest_N, **self._dual_solve_latest_kwargs)
            # Compute primal output ...
            primal_output = ParametrizedReducedDifferentialProblem_DerivedClass.compute_output(self)
            # ... and also dual output for output correction ...
            dual_output = self.dual_reduced_problem.compute_output()
            # ... and sum the results
            self._output = primal_output - dual_output
            return self._output
        
        ## Return an error bound for the current non compliant output. Overriden to use dual problem in error estimation
        def estimate_error_output(self):
            return self.estimate_error()*self.dual_reduced_problem.estimate_error()
        
            
    # return value (a class) for the decorator
    return PrimalDualReducedProblem_Class
    
