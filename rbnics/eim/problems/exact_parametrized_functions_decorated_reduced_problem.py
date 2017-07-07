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

from __future__ import print_function
import types
import rbnics.backends
from rbnics.backends import assign, copy
from rbnics.utils.mpi import log, print, PROGRESS
from rbnics.utils.decorators import Extends, override, ReducedProblemDecoratorFor
from rbnics.eim.problems.eim import EIM
from rbnics.eim.problems.deim import DEIM
from rbnics.eim.problems.exact_parametrized_functions import ExactParametrizedFunctions

@ReducedProblemDecoratorFor(ExactParametrizedFunctions, replaces=(DEIM, EIM))
def ExactParametrizedFunctionsDecoratedReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    
    def _AlsoDecorateErrorEstimationOperators(ReducedParametrizedProblem_DecoratedClass):
        if hasattr(ReducedParametrizedProblem_DecoratedClass, "assemble_error_estimation_operators"):
        
            @Extends(ReducedParametrizedProblem_DecoratedClass, preserve_class_name=True)
            class _AlsoDecorateErrorEstimationOperators_Class(ReducedParametrizedProblem_DecoratedClass):
                @override
                def __init__(self, truth_problem, **kwargs):
                    # Call the parent initialization
                    ReducedParametrizedProblem_DecoratedClass.__init__(self, truth_problem, **kwargs)
                    # Precomputation of error estimation operators is disabled
                    self.folder.pop("error_estimation")
                    
                ## Initialize data structures required for the online phase
                @override
                def init(self, current_stage="online"):
                    ReducedParametrizedProblem_DecoratedClass.init(self, current_stage)
                    # The offline/online separation does not hold anymore, so in 
                    # assemble_error_estimation_operators() we need to re-assemble operators. 
                    # Thus, for any value of current_stage, we initialize error estimation
                    # operators of the reduced problem as if we were offline
                    ReducedParametrizedProblem_DecoratedClass._init_error_estimation_operators(self, "offline")
                    for term in self.riesz_product_terms:
                        self._disable_load_and_save_for_online_storage(self.riesz_product[term])
                                                    
                ## Return the error estimator for the current solution
                @override
                def estimate_error(self, **kwargs):
                    self._rebuild_error_estimation_operators()
                    return ReducedParametrizedProblem_DecoratedClass.estimate_error(self, **kwargs)
                    
                ## Return the relative error estimator for the current solution
                @override
                def estimate_relative_error(self, **kwargs):
                    self._rebuild_error_estimation_operators()
                    return ReducedParametrizedProblem_DecoratedClass.estimate_relative_error(self, **kwargs)
                    
                ## Return the error estimator for the current output
                @override
                def estimate_error_output(self, **kwargs):
                    self._rebuild_error_estimation_operators()
                    return ReducedParametrizedProblem_DecoratedClass.estimate_error_output(self, **kwargs)
                    
                ## Return the relative error estimator for the current output
                @override
                def estimate_relative_error_output(self, **kwargs):
                    self._rebuild_error_estimation_operators()
                    return ReducedParametrizedProblem_DecoratedClass.estimate_relative_error_output(self, **kwargs)
                    
                def _rebuild_error_estimation_operators(self):
                    # The offline/online separation does not hold anymore, so, similarly to what we did in
                    # the truth problem, also at the reduced-order level we need to re-assemble operators,
                    # because the assemble_operator() *may* return parameter dependent operators.
                    self.build_error_estimation_operators("online")
                        
                ## Build operators for error estimation
                @override
                def build_error_estimation_operators(self, current_stage="offline"):
                    if current_stage == "online":
                        log(PROGRESS, "build operators for error estimation (due to inefficient evaluation)")
                        for term in self.riesz_terms:
                            for q in range(self.truth_problem.Q[term]):
                                self.riesz[term][q].clear()
                        self.build_error_estimation_operators__initialized = False
                        ReducedParametrizedProblem_DecoratedClass.build_error_estimation_operators(self)
                    else:
                        # The offline/online separation does not hold anymore, so we cannot precompute 
                        # reduced operators.
                        print("... skipped due to inefficient evaluation")
                    
                ## Assemble operators for error estimation
                @override
                def assemble_error_estimation_operators(self, term, current_stage="online"):
                    if current_stage == "online": # *cannot* load from file
                        # The offline/online separation does not hold anymore, so we need to re-assemble operators,
                        # because the assemble_error_estimation_operators() of the truth problem *may* 
                        # return parameter dependent operators.
                        # Thus, call the parent method enforcing current_stage = "offline"
                        output = ReducedParametrizedProblem_DecoratedClass.assemble_error_estimation_operators(self, term, "offline")
                        # Return
                        return output
                    else:
                        # Call parent method
                        return ReducedParametrizedProblem_DecoratedClass.assemble_error_estimation_operators(self, term, current_stage)
                                
            return _AlsoDecorateErrorEstimationOperators_Class
        else:
            return ReducedParametrizedProblem_DecoratedClass
            
    def _AlsoDecorateNonlinearSolutionStorage(ReducedParametrizedProblem_DecoratedClass):
        if hasattr(ReducedParametrizedProblem_DecoratedClass.ProblemSolver, "store_solution"):
            @Extends(ReducedParametrizedProblem_DecoratedClass, preserve_class_name=True)
            class _AlsoDecorateNonlinearSolutionStorage_Class(ReducedParametrizedProblem_DecoratedClass):
            
                class ProblemSolver(ReducedParametrizedProblem_DecoratedClass.ProblemSolver):
                    # Override online assign to make sure that the truth solution is updated,
                    # and that operators are re-assembled
                    @override
                    def store_solution(self, solution):
                        ReducedParametrizedProblem_DecoratedClass.ProblemSolver.store_solution(self, solution)
                        # Update truth solution
                        reduced_problem = self.problem
                        assign(reduced_problem.truth_problem._solution, reduced_problem.Z[:solution.N]*solution)
                        # Re-assemble 
                        reduced_problem.build_reduced_operators("online")
                
                    # Override to make sure that truth_problem solution is backed up and restored
                    # after the reduced solve
                    @override
                    def solve(self):
                        reduced_problem = self.problem
                        bak_truth_solution = copy(reduced_problem.truth_problem._solution)
                        ReducedParametrizedProblem_DecoratedClass.ProblemSolver.solve(self)
                        assign(reduced_problem.truth_problem._solution, bak_truth_solution)
                    
            return _AlsoDecorateNonlinearSolutionStorage_Class
        else:
            return ReducedParametrizedProblem_DecoratedClass
            
    def _AlsoDecorateNonlinearSolutionDotStorage(ReducedParametrizedProblem_DecoratedClass):
        if hasattr(ReducedParametrizedProblem_DecoratedClass.ProblemSolver, "store_solution_dot"):
            @Extends(ReducedParametrizedProblem_DecoratedClass, preserve_class_name=True)
            class _AlsoDecorateNonlinearSolutionDotStorage_Class(ReducedParametrizedProblem_DecoratedClass):
            
                class ProblemSolver(ReducedParametrizedProblem_DecoratedClass.ProblemSolver):
                    # Override online assign to make sure that the truth solution dot is updated,
                    # and that operators are re-assembled
                    @override
                    def store_solution_dot(self, solution_dot):
                        ReducedParametrizedProblem_DecoratedClass.ProblemSolver.store_solution_dot(self, solution_dot)
                        # Update truth solution
                        reduced_problem = self.problem
                        assign(reduced_problem.truth_problem._solution_dot, reduced_problem.Z[:solution.N]*solution_dot)
                        # Re-assemble 
                        reduced_problem.build_reduced_operators("online")
                        
                    # Override to make sure that truth_problem solution_dot is backed up and restored
                    # after the reduced solve
                    @override
                    def solve(self):
                        reduced_problem = self.problem
                        bak_truth_solution_dot = copy(reduced_problem.truth_problem._solution_dot)
                        ReducedParametrizedProblem_DecoratedClass.ProblemSolver.solve(self)
                        assign(reduced_problem.truth_problem._solution_dot, bak_truth_solution_dot)
            
            return _AlsoDecorateNonlinearSolutionDotStorage_Class
        else:
            return ReducedParametrizedProblem_DecoratedClass
           
    @Extends(ParametrizedReducedDifferentialProblem_DerivedClass, preserve_class_name=True) # needs to be first in order to override for last the methods
    @_AlsoDecorateErrorEstimationOperators
    @_AlsoDecorateNonlinearSolutionStorage
    @_AlsoDecorateNonlinearSolutionDotStorage
    class ExactParametrizedFunctionsDecoratedReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        ## Default initialization of members
        @override
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            # Avoid useless assemblies
            self._solve__previous_mu = None
            self._solve__previous_self_N = None
            # Precomputation of operators is disabled
            self.folder.pop("reduced_operators")
        
        ## Initialize data structures required for the online phase
        @override
        def init(self, current_stage="online"):
            self._init_basis_functions(current_stage)
            # The offline/online separation does not hold anymore, so in assemble_operator()
            # we need to re-assemble operators. Thus, for any value of current_stage,
            # we initialize the operators of the reduced problem as if we were offline
            self._init_operators("offline")
            #
            n_components = len(self.components)
            # Inner products
            if n_components > 1:
                for component in self.components:
                    self._disable_load_and_save_for_online_storage(self.inner_product[component])
            else:
                self._disable_load_and_save_for_online_storage(self.inner_product)
            # Projection inner product
            if n_components > 1:
                for component in self.components:
                    self._disable_load_and_save_for_online_storage(self.projection_inner_product[component])
            else:
                self._disable_load_and_save_for_online_storage(self.projection_inner_product)
            # Terms
            for term in self.terms:
                self._disable_load_and_save_for_online_storage(self.operator[term])

        def _disable_load_and_save_for_online_storage(self, online_storage):
            # Make sure to disable the save() method of the operator, which is 
            # called internally by assemble_operator() since it is not possible
            # to precompute operators, and thus they should not be saved
            def disabled_save(self, folder, filename):
                raise AttributeError("Cannot save to file due to inefficient evaluation")
            online_storage.save = types.MethodType(disabled_save, online_storage)
            # Make sure to raise an error if the load() method of the operator,
            # since we have not saved anything and it should never be called
            def error_load(self, folder, filename):
                raise AttributeError("Cannot load from file due to inefficient evaluation")
            online_storage.load = types.MethodType(error_load, online_storage)
            
        # Perform an online solve (internal)
        @override
        def _solve(self, N, **kwargs):
            # The offline/online separation does not hold anymore, so at the reduced-order level 
            # we need to re-assemble operators, because the assemble_operator() *may* return 
            # parameter dependent operators.
            self.build_reduced_operators("online")
            # Then call Parent solve
            ParametrizedReducedDifferentialProblem_DerivedClass._solve(self, N, **kwargs)
    
        ## Assemble the reduced order affine expansion.
        @override
        def build_reduced_operators(self, current_stage="offline"):
            if current_stage == "online":
                log(PROGRESS, "build reduced operators (due to inefficient evaluation)")
                output = ParametrizedReducedDifferentialProblem_DerivedClass.build_reduced_operators(self)
            else:
                # The offline/online separation does not hold anymore, so we cannot precompute 
                # reduced operators.
                print("... skipped due to inefficient evaluation")
            
        ## Assemble the reduced order affine expansion
        @override
        def assemble_operator(self, term, current_stage="online"):
            if current_stage == "online": # *cannot* load from file
                # The offline/online separation does not hold anymore, so we need to re-assemble operators,
                # because the assemble_operator() of the truth problem *may* return parameter dependent operators.
                # Thus, call the parent method enforcing current_stage = "offline"
                return ParametrizedReducedDifferentialProblem_DerivedClass.assemble_operator(self, term, "offline")
            else:
                # Call parent method
                return ParametrizedReducedDifferentialProblem_DerivedClass.assemble_operator(self, term, current_stage)
                
    # return value (a class) for the decorator
    return ExactParametrizedFunctionsDecoratedReducedProblem_Class
