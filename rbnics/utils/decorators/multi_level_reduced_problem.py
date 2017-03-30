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

from rbnics.utils.decorators.extends import Extends
from rbnics.utils.decorators.override import override

def MultiLevelReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
            
    @Extends(ParametrizedReducedDifferentialProblem_DerivedClass, preserve_class_name=True)
    class MultiLevelReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        
        additional_folder_prefix = dict({
            1: "",                  # remember that online data for level i-th is stored under the name of the (i-1)-th truth
            2: "Reduced",           # e.g. for the standard case it is stored in the folder ProblemName
            3: "DoubleReduced",
            4: "TripleReduced",
            5: "QuadrupleReduced"   # ... you can go on if needed ...
        })
        
        @override
        def __init__(self, truth_problem, **kwargs):
            # Get the truth_problem recursion level: indeed a truth problem itself
            # can be a reduced problem! In the standard case (truth_problem is actually
            # a FE approximation) then this is the first reduction, so reduction level is 1
            self._reduction_level = 1
            flattened_truth_problem = truth_problem
            while hasattr(flattened_truth_problem, "truth_problem"):
                flattened_truth_problem = flattened_truth_problem.truth_problem
                self._reduction_level += 1
                
            # In case of multilevel reduction create a fake V attribute to the provided
            # truth problem, because it is been already reduced
            if self._reduction_level > 1:
                truth_problem.V = truth_problem.Z
                
            # Call the parent initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # Change the folder names in Parent
            self.folder_prefix = self.additional_folder_prefix[self._reduction_level] + self.folder_prefix
            for (key, name) in self.folder.iteritems():
                self.folder[key] = self.additional_folder_prefix[self._reduction_level] + name
                
            # Avoid useless computations
            self._error_computation_override__previous_truth_problem = None
            self._error_computation_override__current_with_respect_to = None
            self._error_computation_override__current_with_respect_to_level = None
            self._error_computation_override__current_bak_truth_problem = None
            
            # Default values for kwargs
            self._error_computation_override__default_with_respect_to = None
            self._error_computation_override__default_with_respect_to_level = None
            
        @override_error_computation
        def compute_error(self, N=None, **kwargs):
            return ParametrizedReducedDifferentialProblem_DerivedClass.compute_error(self, N, **kwargs)
            
        @override_error_computation
        def compute_relative_error(self, N=None, **kwargs):
            return ParametrizedReducedDifferentialProblem_DerivedClass.compute_relative_error(self, N, **kwargs)
            
        @override_error_computation
        def compute_error_output(self, N=None, **kwargs):
            return ParametrizedReducedDifferentialProblem_DerivedClass.compute_error_output(self, N, **kwargs)
            
        @override_error_computation
        def compute_relative_error_output(self, N=None, **kwargs):
            return ParametrizedReducedDifferentialProblem_DerivedClass.compute_relative_error_output(self, N, **kwargs)
            
    # return value (a class) for the decorator
    return MultiLevelReducedProblem_Class
    
def override_error_computation(error_computation_method):
    def overridden_error_computation_method(self, N=None, **kwargs):
        if "with_respect_to" not in kwargs and self._error_computation_override__default_with_respect_to is not None:
            kwargs["with_respect_to"] = self._error_computation_override__default_with_respect_to
        if "with_respect_to_level" not in kwargs and self._error_computation_override__default_with_respect_to_level is not None:
            kwargs["with_respect_to_level"] = self._error_computation_override__default_with_respect_to_level
        if "with_respect_to" in kwargs or "with_respect_to_level" in kwargs:
            if "with_respect_to" in kwargs:
                assert "with_respect_to_level" not in kwargs # the two options are mutually exclusive
                                                             # otherwise how should we know to which level in the 
                                                             # hierarchy is this truth problem supposed to be?
                self._error_computation_override__current_bak_truth_problem = self.truth_problem
                self.truth_problem = kwargs["with_respect_to"]
                self._error_computation_override__current_with_respect_to = kwargs["with_respect_to"]
                self._error_computation_override__current_with_respect_to_level = None
            elif "with_respect_to_level" in kwargs:
                raise NotImplementedError # TODO
                with_respect_to_level = kwargs["with_respect_to_level"]
                self._error_computation_override__current_with_respect_to = None
                self._error_computation_override__current_with_respect_to_level = with_respect_to_level
                assert isinstance(with_respect_to_level, int)
                assert with_respect_to_level >= 0
                assert with_respect_to_level <= self._reduction_level - 1
                self._error_computation_override__current_bak_truth_problem = self.truth_problem
                for level in range(self._reduction_level - 1, with_respect_to_level, -1):
                    self.truth_problem = self.truth_problem.truth_problem
            # Check if truth problem has changed
            truth_problem_has_changed = (self._error_computation_override__previous_truth_problem != self.truth_problem)
            if truth_problem_has_changed:
                self._error_computation_override__previous_truth_problem = self.truth_problem
            # Make sure that truth solution is recomputed if truth problem is different from the previous one
            if truth_problem_has_changed:
                self._compute_error__previous_mu = None # of Parent class
            # Make sure that truth output is recomputed if truth problem is different from the previous one
            if truth_problem_has_changed:
                self._compute_error_output__previous_mu = None # of Parent class
            # Call Parent
            error = error_computation_method(self, N, **kwargs)
            # Restore backup
            self.truth_problem = self._error_computation_override__current_bak_truth_problem
            # Return
            return error
        else:
            return error_computation_method(self, N, **kwargs)
    return overridden_error_computation_method
    
