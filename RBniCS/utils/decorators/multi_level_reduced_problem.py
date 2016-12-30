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

from RBniCS.utils.decorators.extends import Extends
from RBniCS.utils.decorators.override import override

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
            self._compute_error__previous_truth_problem = None
            self._compute_error__current_with_respect_to = None
            self._compute_error__current_with_respect_to_level = None
            self._compute_error__current_bak_truth_problem = None
        
        @override
        def compute_error(self, N=None, **kwargs):
            if "with_respect_to" in kwargs or "with_respect_to_level" in kwargs:
                if "with_respect_to" in kwargs:
                    assert "with_respect_to_level" not in kwargs # the two options are mutually exclusive
                                                                 # otherwise how should we know to which level in the 
                                                                 # hierarchy is this truth problem supposed to be?
                    self._compute_error__current_bak_truth_problem = self.truth_problem
                    self.truth_problem = kwargs["with_respect_to"]
                    self._compute_error__current_with_respect_to = kwargs["with_respect_to"]
                    self._compute_error__current_with_respect_to_level = None
                elif "with_respect_to_level" in kwargs:
                    with_respect_to_level = kwargs["with_respect_to_level"]
                    self._compute_error__current_with_respect_to = None
                    self._compute_error__current_with_respect_to_level = with_respect_to_level
                    assert isinstance(with_respect_to_level, int)
                    assert with_respect_to_level >= 0
                    assert with_respect_to_level <= self._reduction_level - 1
                    self._compute_error__current_bak_truth_problem = self.truth_problem
                    for level in range(self._reduction_level - 1, with_respect_to_level, -1):
                        self.truth_problem = self.truth_problem.truth_problem
                else:
                    raise ValueError("Invalid value for kwargs")
                # Make sure that truth solution is recomputed is truth problem is different from the previous one
                if self._compute_error__previous_truth_problem != self.truth_problem:
                    self._compute_error__previous_mu = None # of Parent class
                    self._compute_error__previous_truth_problem = self.truth_problem
                # Make sure to update mu in the truth_problem (sync is only guaranteed with the original truth problem)
                self.truth_problem.set_mu(self.mu)
                # Call Parent
                error = ParametrizedReducedDifferentialProblem_DerivedClass.compute_error(self, N, **kwargs)
                # Restore backup
                self.truth_problem = self._compute_error__current_bak_truth_problem
                # Return
                return error
            else:
                return ParametrizedReducedDifferentialProblem_DerivedClass.compute_error(self, N, **kwargs)
                
        @override
        def _compute_error(self):
            if self._compute_error__current_with_respect_to_level is not None:
                raise NotImplementedError # TODO
                with_respect_to_level = self._compute_error__current_with_respect_to_level
                if with_respect_to_level > 0:
                    reduced_solution = reduced_solution_and_output[0]
                    reduced_output = reduced_solution_and_output[1]
                    truth_problem_l = self._compute_error__current_bak_truth_problem
                    for level in range(self._reduction_level - 1, with_respect_to_level, -1):
                        N_l = reduced_solution.N
                        reduced_solution = truth_problem_l.Z[:N_l]*reduced_solution
                        truth_problem_l = truth_problem_l.truth_problem
                    reduced_solution_and_output = (reduced_solution, reduced_output)
            # Call Parent
            return ParametrizedReducedDifferentialProblem_DerivedClass._compute_error(self)
            
    # return value (a class) for the decorator
    return MultiLevelReducedProblem_Class
    
