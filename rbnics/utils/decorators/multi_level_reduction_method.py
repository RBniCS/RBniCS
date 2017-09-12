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
import inspect
from rbnics.utils.decorators.extends import Extends

def MultiLevelReductionMethod(DifferentialProblemReductionMethod_DerivedClass):
            
    @Extends(DifferentialProblemReductionMethod_DerivedClass, preserve_class_name=True)
    class MultiLevelReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):
        
        additional_folder_prefix = dict({
            1: "",                  # remember that online data for level i-th is stored under the name of the (i-1)-th truth
            2: "Reduced",           # e.g. for the standard case it is stored in the folder ProblemName
            3: "DoubleReduced",
            4: "TripleReduced",
            5: "QuadrupleReduced"   # ... you can go on if needed ...
        })
        
        def __init__(self, truth_problem, **kwargs):
            # Get the truth_problem recursion level: indeed a truth problem itself
            # can be a reduced problem! In the standard case (truth_problem is actually
            # a FE approximation) then this is the first reduction, so reduction level is 1
            self._reduction_level = 1
            flattened_truth_problem = truth_problem
            while hasattr(flattened_truth_problem, "truth_problem"):
                flattened_truth_problem = flattened_truth_problem.truth_problem
                self._reduction_level += 1
                
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # Change the folder names in Parent
            self.folder_prefix = self.additional_folder_prefix[self._reduction_level] + self.folder_prefix
            for (key, name) in self.folder.items():
                self.folder[key] = self.additional_folder_prefix[self._reduction_level] + name
        
        def _init_error_analysis(self, **kwargs):
            # Replace truth problem, if needed
            self._replace_truth_problem(**kwargs)
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._init_error_analysis(self, **kwargs)
                
        def _finalize_error_analysis(self, **kwargs):
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._finalize_error_analysis(self, **kwargs)
            # Undo replacement of truth problem, if needed
            self._undo_replace_truth_problem(**kwargs)
            
        def _init_speedup_analysis(self, **kwargs):
            # Replace truth problem, if needed
            self._replace_truth_problem(**kwargs)
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._init_speedup_analysis(self, **kwargs)
            
        def _finalize_speedup_analysis(self, **kwargs):
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._finalize_speedup_analysis(self, **kwargs)
            # Undo replacement of truth problem, if needed
            self._undo_replace_truth_problem(**kwargs)
                 
        def _replace_truth_problem(self, **kwargs):
            if "with_respect_to" in kwargs:
                if not hasattr(self, "_replace_truth_problem__bak_truth_problem"):
                    self._replace_truth_problem__bak_truth_problem = self.truth_problem
                    assert inspect.isfunction(kwargs["with_respect_to"])
                    self.truth_problem = kwargs["with_respect_to"](self.truth_problem)
                    self.reduced_problem.truth_problem = self.truth_problem
            
        def _undo_replace_truth_problem(self, **kwargs):
            if "with_respect_to" in kwargs:
                if hasattr(self, "_replace_truth_problem__bak_truth_problem"):
                    self.truth_problem = self._replace_truth_problem__bak_truth_problem
                    self.reduced_problem.truth_problem = self.truth_problem
                    del self._replace_truth_problem__bak_truth_problem
            
    # return value (a class) for the decorator
    return MultiLevelReductionMethod_Class
    
