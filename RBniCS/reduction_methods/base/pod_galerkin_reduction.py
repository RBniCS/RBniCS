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
from RBniCS.backends import ProperOrthogonalDecomposition
from RBniCS.utils.io import ErrorAnalysisTable, SpeedupAnalysisTable, Timer
from RBniCS.utils.decorators import Extends, override
from RBniCS.utils.mpi import print

def PODGalerkinReduction(DifferentialProblemReductionMethod_DerivedClass):
    @Extends(DifferentialProblemReductionMethod_DerivedClass, preserve_class_name=True)
    class PODGalerkinReduction_Class(DifferentialProblemReductionMethod_DerivedClass):
        
        ###########################     CONSTRUCTORS     ########################### 
        ## @defgroup Constructors Methods related to the construction of the POD-Galerkin ROM object
        #  @{
        
        ## Default initialization of members
        @override
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
                    
            # $$ OFFLINE DATA STRUCTURES $$ #
            # Declare a POD object
            self.POD = None # ProperOrthogonalDecomposition (for problems with one component) or dict of ProperOrthogonalDecomposition (for problem with several components)
            # I/O
            self.folder["snapshots"] = self.folder_prefix + "/" + "snapshots"
            self.folder["post_processing"] = self.folder_prefix + "/" + "post_processing"
            self.label = "POD-Galerkin"
            
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
                self.POD = dict()
                for component in self.truth_problem.components:
                    assert len(self.truth_problem.inner_product[component]) == 1
                    inner_product = self.truth_problem.inner_product[component][0]
                    self.POD[component] = ProperOrthogonalDecomposition(self.truth_problem.V, inner_product)
            else:
                assert len(self.truth_problem.inner_product) == 1
                inner_product = self.truth_problem.inner_product[0]
                self.POD = ProperOrthogonalDecomposition(self.truth_problem.V, inner_product)
                
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
            
            for (run, mu) in enumerate(self.training_set):
                print("############################## run =", run, "######################################")
                
                self.truth_problem.set_mu(mu)
                
                print("truth solve for mu =", self.truth_problem.mu)
                snapshot = self.truth_problem.solve()
                self.truth_problem.export_solution(self.folder["snapshots"], "truth_" + str(run), snapshot)
                snapshot = self.reduced_problem.postprocess_snapshot(snapshot, run)
                
                print("update snapshots matrix")
                self.update_snapshots_matrix(snapshot)

                print("")
                run += 1
                
            print("############################## perform POD ######################################")
            self.compute_basis_functions()
            
            print("")
            print("build reduced operators")
            self.reduced_problem.build_reduced_operators()
            
            print("")
            print("==============================================================")
            print("=" + "{:^60}".format(self.label + " offline phase ends") + "=")
            print("==============================================================")
            print("")
            
            self._finalize_offline()
            return self.reduced_problem

        ## Update the snapshots matrix
        def update_snapshots_matrix(self, snapshot):
            if len(self.truth_problem.components) > 1:
                for component in self.truth_problem.components:
                    self.POD[component].store_snapshot(snapshot, component=component)
            else:
                self.POD.store_snapshot(snapshot)
            
        ## Compute basis functions performing POD
        def compute_basis_functions(self):
            if len(self.truth_problem.components) > 1:
                for component in self.truth_problem.components:
                    print("# POD for component", component)
                    POD = self.POD[component]
                    (_, Z, N) = POD.apply(self.Nmax)
                    self.reduced_problem.Z.enrich(Z, component=component)
                    self.reduced_problem.N[component] += N
                    POD.print_eigenvalues(N)
                    POD.save_eigenvalues_file(self.folder["post_processing"], "eigs_" + component)
                    POD.save_retained_energy_file(self.folder["post_processing"], "retained_energy_" + component)
                self.reduced_problem.Z.save(self.reduced_problem.folder["basis"], "basis")
            else:
                (_, Z, N) = self.POD.apply(self.Nmax)
                self.reduced_problem.Z.enrich(Z)
                self.reduced_problem.N += N
                self.POD.print_eigenvalues(N)
                self.POD.save_eigenvalues_file(self.folder["post_processing"], "eigs")
                self.POD.save_retained_energy_file(self.folder["post_processing"], "retained_energy")
                self.reduced_problem.Z.save(self.reduced_problem.folder["basis"], "basis")
            
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
                error_analysis_table.add_column("error_" + component, group_name="solution_" + component, operations=("mean", "max"))
                error_analysis_table.add_column("relative_error_" + component, group_name="solution_" + component, operations=("mean", "max"))
            error_analysis_table.add_column("error_output", group_name="output", operations=("mean", "max"))
            error_analysis_table.add_column("relative_error_output", group_name="output", operations=("mean", "max"))
            
            for (run, mu) in enumerate(self.testing_set):
                print("############################## run =", run, "######################################")
                
                self.reduced_problem.set_mu(mu)
                            
                for n in range(1, N + 1): # n = 1, ... N
                    error = self.reduced_problem.compute_error(n, **kwargs)
                    relative_error = self.reduced_problem.compute_relative_error(n, **kwargs)
                    if len(components) > 1:
                        for component in components:
                            error_analysis_table["error_" + component, n, run] = error[component]
                            error_analysis_table["relative_error_" + component, n, run] = relative_error[component]
                    else:
                        component = components[0]
                        error_analysis_table["error_" + component, n, run] = error
                        error_analysis_table["relative_error_" + component, n, run] = relative_error
                    
                    error_analysis_table["error_output", n, run] = self.reduced_problem.compute_error_output(n, **kwargs)
                    error_analysis_table["relative_error_output", n, run] = self.reduced_problem.compute_relative_error_output(n, **kwargs)
            
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
                self.truth_problem.output()
                elapsed_truth_output = truth_timer.stop()
                
                for n in range(1, N + 1): # n = 1, ... N
                    reduced_timer.start()
                    self.reduced_problem.solve(n, **kwargs)
                    elapsed_reduced_solve = reduced_timer.stop()
                    
                    reduced_timer.start()
                    self.reduced_problem.output()
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
    return PODGalerkinReduction_Class
    
