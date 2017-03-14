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
## @file elliptic_coercive_reduction_method.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.backends import TimeQuadrature
from RBniCS.utils.decorators import Extends, override

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
            
            # Make sure to clean up reduced problem solution cache to ensure that
            # reduced solution are actually computed
            self.reduced_problem._solution_over_time_cache.clear()
            self.reduced_problem._solution_dot_over_time_cache.clear()
        
    # return value (a class) for the decorator
    return TimeDependentReductionMethod_Class
    
