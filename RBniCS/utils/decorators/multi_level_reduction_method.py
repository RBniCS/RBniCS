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

import types
from RBniCS.utils.decorators.extends import Extends
from RBniCS.utils.decorators.override import override

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
                
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # Change the folder names in Parent
            self.folder_prefix = self.additional_folder_prefix[self._reduction_level] + self.folder_prefix
            for (key, name) in self.folder.iteritems():
                self.folder[key] = self.additional_folder_prefix[self._reduction_level] + name
                
            # Backup truth problem for error analysis
            self._error_analysis__current_bak_truth_problem = None
        
        @override
        def _init_error_analysis(self, **kwargs):
            if "with_respect_to" in kwargs or "with_respect_to_level" in kwargs:
                if "with_respect_to" in kwargs:
                    assert "with_respect_to_level" not in kwargs # the two options are mutually exclusive
                                                                 # otherwise how should we know to which level in the 
                                                                 # hierarchy is this truth problem supposed to be?
                    self.truth_problem = kwargs["with_respect_to"]
                    self._compute_error__current_bak_truth_problem = self.truth_problem
                    
                    # Make sure that stability factors computations at the reduced order level
                    # call the correct problem
                    self._finalize_error_analysis__get_stability_factor__original = self.reduced_problem.get_stability_factor
                    def get_stability_factor__with_respect_to(self):
                        return kwargs["with_respect_to"].get_stability_factor()
                    self.reduced_problem.get_stability_factor = types.MethodType(get_stability_factor__with_respect_to, self.reduced_problem)
                    
                elif "with_respect_to_level" in kwargs:
                    pass # TODO
                else:
                    raise ValueError("Invalid value for kwargs")
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._init_error_analysis(self, **kwargs)
                
        @override
        def _finalize_error_analysis(self, **kwargs):
            if "with_respect_to" in kwargs or "with_respect_to_level" in kwargs:
                if "with_respect_to" in kwargs:
                    assert "with_respect_to_level" not in kwargs # the two options are mutually exclusive
                                                                 # otherwise how should we know to which level in the 
                                                                 # hierarchy is this truth problem supposed to be?
                    self.truth_problem = self._compute_error__current_bak_truth_problem
                    
                    # Make sure that stability factors computations at the reduced order level
                    # are reset to the standard method
                    self.reduced_problem.get_stability_factor = types.MethodType(self._finalize_error_analysis__get_stability_factor__original, self.reduced_problem)
                    del self._finalize_error_analysis__get_stability_factor__original

                elif "with_respect_to_level" in kwargs:
                    pass # TODO
                else:
                    raise ValueError("Invalid value for kwargs")
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._finalize_error_analysis(self, **kwargs)
            
    # return value (a class) for the decorator
    return MultiLevelReductionMethod_Class
    
