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
from rbnics.utils.decorators.extends import Extends
from rbnics.utils.decorators.override import override

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
        
        @override
        def _init_error_analysis(self, **kwargs):
            if "with_respect_to" in kwargs or "with_respect_to_level" in kwargs:
                if "with_respect_to" in kwargs:
                    assert "with_respect_to_level" not in kwargs # the two options are mutually exclusive
                                                                 # otherwise how should we know to which level in the 
                                                                 # hierarchy is this truth problem supposed to be?
                    self._replace_truth_problem(kwargs["with_respect_to"])
                elif "with_respect_to_level" in kwargs:
                    pass # TODO
                else:
                    raise ValueError("Invalid value for kwargs")
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._init_error_analysis(self, **kwargs)
                
        @override
        def _finalize_error_analysis(self, **kwargs):
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._finalize_error_analysis(self, **kwargs)
            
            if "with_respect_to" in kwargs or "with_respect_to_level" in kwargs:
                if "with_respect_to" in kwargs:
                    assert "with_respect_to_level" not in kwargs # the two options are mutually exclusive
                                                                 # otherwise how should we know to which level in the 
                                                                 # hierarchy is this truth problem supposed to be?
                    self._undo_replace_truth_problem()
                elif "with_respect_to_level" in kwargs:
                    pass # TODO
                else:
                    raise ValueError("Invalid value for kwargs")
            
        @override
        def _init_speedup_analysis(self, **kwargs):
            if "with_respect_to" in kwargs or "with_respect_to_level" in kwargs:
                if "with_respect_to" in kwargs:
                    assert "with_respect_to_level" not in kwargs # the two options are mutually exclusive
                                                                 # otherwise how should we know to which level in the 
                                                                 # hierarchy is this truth problem supposed to be?
                    self._replace_truth_problem(kwargs["with_respect_to"])
                elif "with_respect_to_level" in kwargs:
                    pass # TODO
                else:
                    raise ValueError("Invalid value for kwargs")
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._init_speedup_analysis(self, **kwargs)
            
        @override
        def _finalize_speedup_analysis(self, **kwargs):
            # Call Parent
            DifferentialProblemReductionMethod_DerivedClass._finalize_speedup_analysis(self, **kwargs)
            
            if "with_respect_to" in kwargs or "with_respect_to_level" in kwargs:
                if "with_respect_to" in kwargs:
                    assert "with_respect_to_level" not in kwargs # the two options are mutually exclusive
                                                                 # otherwise how should we know to which level in the 
                                                                 # hierarchy is this truth problem supposed to be?
                    self._undo_replace_truth_problem()
                elif "with_respect_to_level" in kwargs:
                    pass # TODO
                else:
                    raise ValueError("Invalid value for kwargs")
                 
        def _replace_truth_problem(self, other_truth_problem):
            # Make sure that mu is in sync, for both the other truth problem ...
            if not hasattr(self, "_replace_truth_problem__other_truth_set_mu__original"):
                self._replace_truth_problem__other_truth_set_mu__original = other_truth_problem.set_mu
                def other_truth_set_mu(self_, mu):
                    self._replace_truth_problem__other_truth_set_mu__original(mu)
                    if self.reduced_problem.mu != mu:
                        self.reduced_problem.set_mu(mu)
                other_truth_problem.set_mu = types.MethodType(other_truth_set_mu, other_truth_problem)
            # ... and the reduced problem
            if not hasattr(self, "_replace_truth_problem__reduced_set_mu__original"):
                self._replace_truth_problem__reduced_set_mu__original = self.reduced_problem.set_mu
                def reduced_set_mu(self_, mu):
                    self._replace_truth_problem__reduced_set_mu__original(mu)
                    if other_truth_problem.mu != mu:
                        other_truth_problem.set_mu(mu)
                self.reduced_problem.set_mu = types.MethodType(reduced_set_mu, self.reduced_problem)
            
            # Make sure that stability factors computations at the reduced order level
            # call the correct problem
            self._replace_truth_problem__get_stability_factor__original = self.reduced_problem.get_stability_factor
            def get_stability_factor__with_respect_to(self_):
                return other_truth_problem.get_stability_factor()
            self.reduced_problem.get_stability_factor = types.MethodType(get_stability_factor__with_respect_to, self.reduced_problem)
            
            # Change truth problem
            if not hasattr(self, "_replace_truth_problem__bak_truth_problem"):
                self._replace_truth_problem__bak_truth_problem = self.truth_problem
                self.truth_problem = other_truth_problem
            else:
                assert self.truth_problem is other_truth_problem
            
        def _undo_replace_truth_problem(self):
            # Restore the original mu sync, for both the other truth problem (currently stored in self.truth_problem)...
            if hasattr(self, "_replace_truth_problem__other_truth_set_mu__original"):
                self.truth_problem.set_mu = self._replace_truth_problem__other_truth_set_mu__original
                del self._replace_truth_problem__other_truth_set_mu__original
            # ... and the reduced problem
            if hasattr(self, "_replace_truth_problem__reduced_set_mu__original"):
                self.reduced_problem.set_mu = self._replace_truth_problem__reduced_set_mu__original
                del self._replace_truth_problem__reduced_set_mu__original
            
            # Make sure that stability factors computations at the reduced order level
            # are reset to the standard method
            self.reduced_problem.get_stability_factor = self._replace_truth_problem__get_stability_factor__original
            del self._replace_truth_problem__get_stability_factor__original
            
            # Reset truth problem
            if hasattr(self, "_replace_truth_problem__bak_truth_problem"):
                self.truth_problem = self._replace_truth_problem__bak_truth_problem
                del self._replace_truth_problem__bak_truth_problem
            
    # return value (a class) for the decorator
    return MultiLevelReductionMethod_Class
    
