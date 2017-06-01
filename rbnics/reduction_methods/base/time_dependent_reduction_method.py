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

import types
from numpy import isclose
from rbnics.backends import TimeQuadrature
from rbnics.utils.decorators import Extends, override

def TimeDependentReductionMethod(DifferentialProblemReductionMethod_DerivedClass):
    @Extends(DifferentialProblemReductionMethod_DerivedClass, preserve_class_name=True)
    class TimeDependentReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):
        
        ## Default initialization of members
        @override
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # Time quadrature
            self.time_quadrature = None
            
            # Indices for undersampling snapshots, e.g. after a transient
            self.reduction_first_index = None # keep temporal evolution from the beginning by default
            self.reduction_delta_index = None # keep every time step by default
            self.reduction_last_index = None # keep temporal evolution until the end by default
            
        ## Set reduction initial time
        def set_reduction_initial_time(self, t0):
            assert isinstance(t0, (float, int))
            t0 = float(t0)
            assert t0 >= self.truth_problem.t0
            self.reduction_first_index = int(t0/self.truth_problem.dt)
                    
        ## Set reduction time step size
        def set_reduction_time_step_size(self, dt):
            assert isinstance(dt, (float, int))
            dt = float(dt)
            assert dt >= self.truth_problem.dt
            self.reduction_delta_index = int(dt/self.truth_problem.dt)
            assert isclose(self.reduction_delta_index*self.truth_problem.dt, dt), "Reduction time step size should be a multiple of discretization time step size"
            
        ## Set reduction final time
        def set_reduction_final_time(self, T):
            assert isinstance(T, (float, int))
            T = float(T)
            assert T <= self.truth_problem.T
            self.reduction_last_index = int(T/self.truth_problem.dt)
            
        ## Initialize data structures required for the offline phase
        @override
        def _init_offline(self):
            output = DifferentialProblemReductionMethod_DerivedClass._init_offline(self)
            
            if self.time_quadrature is None:
                self.time_quadrature = TimeQuadrature((0., self.truth_problem.T), self.truth_problem.dt)
                
            return output
            
        ## Initialize data structures required for the error analysis phase
        @override
        def _init_error_analysis(self, **kwargs):
            DifferentialProblemReductionMethod_DerivedClass._init_error_analysis(self, **kwargs)
            
            if self.time_quadrature is None:
                self.time_quadrature = TimeQuadrature((0., self.truth_problem.T), self.truth_problem.dt)
        
        ## Initialize data structures required for the speedup analysis phase
        @override
        def _init_speedup_analysis(self, **kwargs):
            DifferentialProblemReductionMethod_DerivedClass._init_speedup_analysis(self, **kwargs)
            
            # Make sure to clean up problem and reduced problem solution cache to ensure that
            # solution and reduced solution are actually computed
            self.reduced_problem._solution_dot_cache.clear()
            self.reduced_problem._solution_over_time_cache.clear()
            self.reduced_problem._solution_dot_over_time_cache.clear()
            # ... and also disable the capability of importing truth solutions
            self._speedup_analysis__original_import_solution = self.truth_problem.import_solution
            def disabled_import_solution(self_, folder, filename, solution_over_time=None, solution_dot_over_time=None):
                return False
            self.truth_problem.import_solution = types.MethodType(disabled_import_solution, self.truth_problem)
            
        
        ## Finalize data structures required after the speedup analysis phase
        @override
        def _finalize_speedup_analysis(self, **kwargs):
            # Restore the capability to import truth solutions
            self.truth_problem.import_solution = self._speedup_analysis__original_import_solution
            del self._speedup_analysis__original_import_solution
        
    # return value (a class) for the decorator
    return TimeDependentReductionMethod_Class
    
