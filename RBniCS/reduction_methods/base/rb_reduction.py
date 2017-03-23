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

from __future__ import print_function
from abc import ABCMeta, abstractmethod
from RBniCS.backends import GramSchmidt
from RBniCS.utils.io import ErrorAnalysisTable, GreedySelectedParametersList, GreedyErrorEstimatorsList, SpeedupAnalysisTable, Timer
from RBniCS.utils.decorators import Extends, override
from RBniCS.utils.mpi import print

def RBReduction(DifferentialProblemReductionMethod_DerivedClass):
    @Extends(DifferentialProblemReductionMethod_DerivedClass, preserve_class_name=True)
    class RBReduction_Class(DifferentialProblemReductionMethod_DerivedClass):
        __metaclass__ = ABCMeta
        
        ###########################     CONSTRUCTORS     ########################### 
        ## @defgroup Constructors Methods related to the construction of the POD-Galerkin ROM object
        #  @{
        
        ## Default initialization of members
        @override
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
                    
            # $$ OFFLINE DATA STRUCTURES $$ #
            # Declare a GS object
            self.GS = None # GramSchmidt (for problems with one component) or dict of GramSchmidt (for problem with several components)
            # I/O
            self.folder["snapshots"] = self.folder_prefix + "/" + "snapshots"
            self.folder["post_processing"] = self.folder_prefix + "/" + "post_processing"
            self.greedy_selected_parameters = GreedySelectedParametersList()
            self.greedy_error_estimators = GreedyErrorEstimatorsList()
            self.label = "RB"
            
        #  @}
        ########################### end - CONSTRUCTORS - end ########################### 
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
        
        ## Initialize data structures required for the offline phase
        @override
        def _init_offline(self):
            # Call parent to initialize inner product and reduced problem
            output = DifferentialProblemReductionMethod_DerivedClass._init_offline(self)
            
            # Declare a new POD for each basis component
            if len(self.truth_problem.components) > 1:
                self.GS = dict()
                for component in self.truth_problem.components:
                    assert len(self.truth_problem.inner_product[component]) == 1
                    inner_product = self.truth_problem.inner_product[component][0]
                    self.GS[component] = GramSchmidt(inner_product)
            else:
                assert len(self.truth_problem.inner_product) == 1
                inner_product = self.truth_problem.inner_product[0]
                self.GS = GramSchmidt(inner_product)                
                
            # Return
            return output
            
        ## Perform the offline phase of the reduced order model
        @override
        def offline(self):
            need_to_do_offline_stage = self._init_offline()
            if not need_to_do_offline_stage:
                return self.reduced_problem
                        
            print("==============================================================")
            print("=" + "{:^60}".format(self.label + " offline phase begins") + "=")
            print("==============================================================")
            print("")
            
            run = 0
            relative_error_estimator_max = 2.*self.tol
            while self.reduced_problem.N < self.Nmax and relative_error_estimator_max >= self.tol:
                print("############################## N =", self.reduced_problem.N, "######################################")
                
                print("truth solve for mu =", self.truth_problem.mu)
                snapshot = self.truth_problem.solve()
                self.truth_problem.export_solution(self.folder["snapshots"], "truth_" + str(run), snapshot)
                snapshot = self.reduced_problem.postprocess_snapshot(snapshot, run)
                
                print("update basis matrix")
                self.update_basis_matrix(snapshot)
                run += 1
                
                print("build reduced operators")
                self.reduced_problem.build_reduced_operators()
                
                print("reduced order solve")
                self.reduced_problem.solve(self.reduced_problem.N)
                
                print("build operators for error estimation")
                self.reduced_problem.build_error_estimation_operators()
                
                print("find next mu")
                (error_estimator_max, relative_error_estimator_max) = self.greedy()
                print("maximum error estimator =", error_estimator_max)
                print("maximum relative error estimator =", relative_error_estimator_max)

                print("")
                
            print("==============================================================")
            print("=" + "{:^60}".format(self.label + " offline phase ends") + "=")
            print("==============================================================")
            print("")
            
            self._finalize_offline()
            return self.reduced_problem
            
        ## Update basis matrix
        def update_basis_matrix(self, snapshot):
            if len(self.truth_problem.components) > 1:
                for component in self.truth_problem.components:
                    self.reduced_problem.Z.enrich(snapshot, component=component)
                    self.GS[component].apply(self.reduced_problem.Z[component], self.reduced_problem.N_bc[component])
                    self.reduced_problem.N[component] += 1
                self.reduced_problem.Z.save(self.reduced_problem.folder["basis"], "basis")
            else:
                self.reduced_problem.Z.enrich(snapshot)
                self.GS.apply(self.reduced_problem.Z, self.reduced_problem.N_bc)
                self.reduced_problem.N += 1
                self.reduced_problem.Z.save(self.reduced_problem.folder["basis"], "basis")
                
        ## Choose the next parameter in the offline stage in a greedy fashion: wrapper with post processing
        ## of the result (in particular, set greedily selected parameter and save to file)
        def greedy(self):
            (error_estimator_max, error_estimator_argmax) = self._greedy()
            self.reduced_problem.set_mu(self.training_set[error_estimator_argmax])
            self.greedy_selected_parameters.append(self.training_set[error_estimator_argmax])
            self.greedy_selected_parameters.save(self.folder["post_processing"], "mu_greedy")
            self.greedy_error_estimators.append(error_estimator_max)
            self.greedy_error_estimators.save(self.folder["post_processing"], "error_estimator_max")
            return (error_estimator_max, error_estimator_max/self.greedy_error_estimators[0])
            
        ## Choose the next parameter in the offline stage in a greedy fashion
        @abstractmethod
        def _greedy(self):
            raise NotImplementedError("The method _greedy() is problem-specific and needs to be overridden.")
            
        #  @}
        ########################### end - OFFLINE STAGE - end ########################### 
        
        ###########################     ERROR ANALYSIS     ########################### 
        ## @defgroup ErrorAnalysis Error analysis
        #  @{
            
        # Compute the error of the reduced order approximation with respect to the full order one
        # over the testing set
        @override
        def error_analysis(self, N=None, **kwargs):
            N, kwargs = self.reduced_problem._online_size_from_kwargs(N, **kwargs)
            if isinstance(N, dict):
                N = min(N.values())
            
            if "components" in kwargs:
                components = kwargs["components"]
            else:
                components = self.truth_problem.components
            
            self._init_error_analysis(**kwargs)
            
            print("==============================================================")
            print("=" + "{:^60}".format(self.label + " error analysis begins") + "=")
            print("==============================================================")
            print("")
            
            error_analysis_table = ErrorAnalysisTable(self.testing_set)
            error_analysis_table.set_Nmax(N)
            for component in components:
                error_analysis_table.add_column("error_" + component, group_name="solution_" + component + "_error", operations=("mean", "max"))
                error_analysis_table.add_column("error_estimator_" + component, group_name="solution_" + component + "_error", operations=("mean", "max"))
                error_analysis_table.add_column("effectivity_" + component, group_name="solution_" + component + "_error", operations=("min", "mean", "max"))
                error_analysis_table.add_column("relative_error_" + component, group_name="solution_" + component + "_relative_error", operations=("mean", "max"))
                error_analysis_table.add_column("relative_error_estimator_" + component, group_name="solution_" + component + "_relative_error", operations=("mean", "max"))
                error_analysis_table.add_column("relative_effectivity_" + component, group_name="solution_" + component + "_relative_error", operations=("min", "mean", "max"))
            error_analysis_table.add_column("error_output", group_name="output_error", operations=("mean", "max"))
            error_analysis_table.add_column("error_estimator_output", group_name="output_error", operations=("mean", "max"))
            error_analysis_table.add_column("effectivity_output", group_name="output_error", operations=("min", "mean", "max"))
            error_analysis_table.add_column("relative_error_output", group_name="output_relative_error", operations=("mean", "max"))
            error_analysis_table.add_column("relative_error_estimator_output", group_name="output_relative_error", operations=("mean", "max"))
            error_analysis_table.add_column("relative_effectivity_output", group_name="output_relative_error", operations=("min", "mean", "max"))
            
            for (run, mu) in enumerate(self.testing_set):
                print("############################## run =", run, "######################################")
                
                self.reduced_problem.set_mu(mu)
                            
                for n in range(1, N + 1): # n = 1, ... N
                    error = self.reduced_problem.compute_error(n, **kwargs)
                    error_estimator = self.reduced_problem.estimate_error()
                    relative_error = self.reduced_problem.compute_relative_error(n, **kwargs)
                    relative_error_estimator = self.reduced_problem.estimate_relative_error()
                    if len(components) > 1:
                        for component in components:
                            error_analysis_table["error_" + component, n, run] = error[component]
                            error_analysis_table["error_estimator_" + component, n, run] = error_estimator[component]
                            error_analysis_table["effectivity_" + component, n, run] = \
                                error_analysis_table["error_estimator_" + component, n, run]/error_analysis_table["error_" + component, n, run]
                            error_analysis_table["relative_error_" + component, n, run] = relative_error[component]
                            error_analysis_table["relative_error_estimator_" + component, n, run] = relative_error_estimator[component]
                            error_analysis_table["relative_effectivity_" + component, n, run] = \
                                error_analysis_table["relative_error_estimator_" + component, n, run]/error_analysis_table["relative_error_" + component, n, run]
                    else:
                        component = components[0]
                        error_analysis_table["error_" + component, n, run] = error
                        error_analysis_table["error_estimator_" + component, n, run] = error_estimator
                        error_analysis_table["effectivity_" + component, n, run] = \
                            error_analysis_table["error_estimator_" + component, n, run]/error_analysis_table["error_" + component, n, run]
                        error_analysis_table["relative_error_" + component, n, run] = relative_error
                        error_analysis_table["relative_error_estimator_" + component, n, run] = relative_error_estimator
                        error_analysis_table["relative_effectivity_" + component, n, run] = \
                            error_analysis_table["relative_error_estimator_" + component, n, run]/error_analysis_table["relative_error_" + component, n, run]
                    
                    error_analysis_table["error_output", n, run] = self.reduced_problem.compute_error_output(n, **kwargs)
                    error_analysis_table["error_estimator_output", n, run] = self.reduced_problem.estimate_error_output()
                    error_analysis_table["effectivity_output", n, run] = \
                        error_analysis_table["error_estimator_output", n, run]/error_analysis_table["error_output", n, run]
                    error_analysis_table["relative_error_output", n, run] = self.reduced_problem.compute_relative_error_output(n, **kwargs)
                    error_analysis_table["relative_error_estimator_output", n, run] = self.reduced_problem.estimate_relative_error_output()
                    error_analysis_table["relative_effectivity_output", n, run] =  \
                        error_analysis_table["relative_error_estimator_output", n, run]/error_analysis_table["relative_error_output", n, run]
            
            # Print
            print("")
            print(error_analysis_table)
            
            print("")
            print("==============================================================")
            print("=" + "{:^60}".format(self.label + " error analysis ends") + "=")
            print("==============================================================")
            print("")
            
            self._finalize_error_analysis(**kwargs)
            
        # Compute the speedup of the reduced order approximation with respect to the full order one
        # over the testing set
        @override
        def speedup_analysis(self, N=None, **kwargs):
            N, kwargs = self.reduced_problem._online_size_from_kwargs(N, **kwargs)
            if isinstance(N, dict):
                N = min(N.values())
            
            self._init_speedup_analysis(**kwargs)
            
            print("==============================================================")
            print("=" + "{:^60}".format(self.label + " speedup analysis begins") + "=")
            print("==============================================================")
            print("")
            
            speedup_analysis_table = SpeedupAnalysisTable(self.testing_set)
            speedup_analysis_table.set_Nmax(N)
            speedup_analysis_table.add_column("speedup_solve", group_name="speedup_solve", operations=("min", "mean", "max"))
            speedup_analysis_table.add_column("speedup_output", group_name="speedup_output", operations=("min", "mean", "max"))
            
            truth_timer = Timer("parallel")
            reduced_timer = Timer("serial")
                        
            for (run, mu) in enumerate(self.testing_set):
                print("############################## run =", run, "######################################")
                
                self.reduced_problem.set_mu(mu)
                
                truth_timer.start()
                self.truth_problem.solve(**kwargs)
                elapsed_truth_solve = truth_timer.stop()
                
                truth_timer.start()
                self.truth_problem.compute_output()
                elapsed_truth_output = truth_timer.stop()
                
                for n in range(1, N + 1): # n = 1, ... N
                    reduced_timer.start()
                    self.reduced_problem.solve(n, **kwargs)
                    self.reduced_problem.estimate_error()
                    self.reduced_problem.estimate_relative_error()
                    elapsed_reduced_solve = reduced_timer.stop()
                    
                    reduced_timer.start()
                    self.reduced_problem.compute_output()
                    self.reduced_problem.estimate_error_output()
                    self.reduced_problem.estimate_relative_error_output()
                    elapsed_reduced_output = reduced_timer.stop()
                    
                    speedup_analysis_table["speedup_solve", n, run] = elapsed_truth_solve/elapsed_reduced_solve
                    speedup_analysis_table["speedup_output", n, run] = (elapsed_truth_solve + elapsed_truth_output)/(elapsed_reduced_solve + elapsed_reduced_output)
            
            # Print
            print("")
            print(speedup_analysis_table)
            
            print("")
            print("==============================================================")
            print("=" + "{:^60}".format(self.label + " speedup analysis ends") + "=")
            print("==============================================================")
            print("")
            
            self._finalize_speedup_analysis(**kwargs)
        
        #  @}
        ########################### end - ERROR ANALYSIS - end ########################### 
        
    # return value (a class) for the decorator
    return RBReduction_Class
    
