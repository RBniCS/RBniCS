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

from math import sqrt
from RBniCS.utils.decorators import Extends, override
from RBniCS.utils.io import ErrorAnalysisTable

def TimeDependentRBReduction(DifferentialProblemReductionMethod_DerivedClass):
    @Extends(DifferentialProblemReductionMethod_DerivedClass, preserve_class_name=True)
    class TimeDependentRBReduction_Class(DifferentialProblemReductionMethod_DerivedClass):
        
        ## Choose the next parameter in the offline stage in a greedy fashion
        def _greedy(self):
            def solve_and_estimate_error(mu, index):
                self.reduced_problem.set_mu(mu)
                self.reduced_problem.solve()
                error_estimator_over_time = self.reduced_problem.estimate_error()
                error_estimator_squared_over_time = [v**2 for v in error_estimator_over_time]
                return sqrt(self.time_quadrature.integrate(error_estimator_squared_over_time))
                
            return self.training_set.max(solve_and_estimate_error)
            
        # Compute the error of the reduced order approximation with respect to the full order one
        # over the testing set
        @override
        def error_analysis(self, N=None, **kwargs):
            if "components" in kwargs:
                components = kwargs["components"]
            else:
                components = self.truth_problem.components
                
            for component in components:
                def solution_preprocess_setitem(component):
                    def solution_preprocess_setitem__function(list_over_time):
                        list_squared_over_time = [v**2 for v in list_over_time]
                        return sqrt(self.time_quadrature.integrate(list_squared_over_time))
                    return solution_preprocess_setitem__function
                for column_prefix in ("error_", "error_estimator_", "relative_error_", "relative_error_estimator_"):
                    ErrorAnalysisTable.preprocess_setitem(column_prefix + component, solution_preprocess_setitem(component))
                
            def output_preprocess_setitem(list_over_time):
                return self.time_quadrature.integrate(list_over_time)
            for column in ("error_output", "error_estimator_output", "relative_error_output", "relative_error_estimator_output"):
                ErrorAnalysisTable.preprocess_setitem(column, solution_preprocess_setitem)
            
            DifferentialProblemReductionMethod_DerivedClass.error_analysis(self, N, **kwargs)
            
            ErrorAnalysisTable.clear_setitem_preprocessing()
        
    # return value (a class) for the decorator
    return TimeDependentRBReduction_Class
    
