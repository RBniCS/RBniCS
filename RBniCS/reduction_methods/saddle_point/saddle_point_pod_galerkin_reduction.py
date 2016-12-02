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
## @file saddle_point_pod_galerkin_reduction.py
#  @brief Implementation of a POD-Galerkin ROM for elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from RBniCS.backends import ProperOrthogonalDecomposition
from RBniCS.utils.io import ErrorAnalysisTable, SpeedupAnalysisTable
from RBniCS.utils.mpi import print
from RBniCS.utils.decorators import Extends, override, ReductionMethodFor
from RBniCS.problems.saddle_point.saddle_point_problem import SaddlePointProblem
from RBniCS.reduction_methods.saddle_point.saddle_point_reduction_method import SaddlePointReductionMethod

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE POD BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class SaddlePointPODGalerkinReduction
#
# Base class containing the interface of a POD-Galerkin ROM
# for elliptic coercive problems
@Extends(SaddlePointReductionMethod) # needs to be first in order to override for last the methods
@ReductionMethodFor(SaddlePointProblem, "PODGalerkin")
class SaddlePointPODGalerkinReduction(SaddlePointReductionMethod):    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the POD-Galerkin ROM object
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, truth_problem):
        # Call the parent initialization
        SaddlePointReductionMethod.__init__(self, truth_problem)
                
        # $$ OFFLINE DATA STRUCTURES $$ #
        # Declare a POD object
        self.POD = dict() # of length equal to the number of basis components
        # I/O
        self.folder["snapshots"] = self.folder_prefix + "/" + "snapshots"
        self.folder["post_processing"] = self.folder_prefix + "/" + "post_processing"
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Initialize data structures required for the offline phase
    @override
    def _init_offline(self):
        # Call parent to initialize inner product and reduced problem
        output = SaddlePointReductionMethod._init_offline(self)
        
        # Declare a new POD for each basis component
        assert len(self.truth_problem.inner_product) == 3 # saddle point problems have two components and one supremizer
        for component_name in ("u", "s", "p"):
            inner_product = self.truth_problem.inner_product[component_name][0]
            self.POD[component_name] = ProperOrthogonalDecomposition(self.truth_problem.V, inner_product)        
        # Return
        return output
            
    ## Perform the offline phase of the reduced order model
    @override
    def offline(self):
        need_to_do_offline_stage = self._init_offline()
        if not need_to_do_offline_stage:
            return self.reduced_problem
        
        print("==============================================================")
        print("=             Offline phase begins                           =")
        print("==============================================================")
        print("")
        
        for (run, mu) in enumerate(self.training_set):
            print("############################## run =", run, "######################################")
            
            self.truth_problem.set_mu(mu)
            
            print("truth solve for mu =", self.truth_problem.mu)
            snapshot = self.truth_problem.solve()
            for component_name in ("u", "p"):
                function_component = self.truth_problem.component_name_to_function_component[component_name]
                self.truth_problem.export_solution(self.folder["snapshots"], "truth_" + component_name + "_" + str(run), snapshot, function_component)
            snapshot = self.reduced_problem.postprocess_snapshot(snapshot)
            
            print("supremizer solve for mu =", self.truth_problem.mu)
            supremizer = self.truth_problem.solve_supremizer()
            for component_name in ("s", ):
                function_component = self.truth_problem.component_name_to_function_component[component_name]
                self.truth_problem.export_solution(self.folder["snapshots"], "truth_" + component_name + "_" + str(run), supremizer, function_component)
            
            print("update snapshots matrix")
            self.update_snapshots_matrix(snapshot, supremizer)

            print("")
            run += 1
            
        print("############################## perform POD ######################################")
        self.compute_basis_functions()
        
        print("")
        print("build reduced operators")
        self.reduced_problem.build_reduced_operators()
        
        print("")
        print("==============================================================")
        print("=             Offline phase ends                             =")
        print("==============================================================")
        print("")
        
        self._finalize_offline()
        return self.reduced_problem
        
    ## Compute basis functions performing POD
    def compute_basis_functions(self):
        for component_name in ("u", "s", "p"):
            print("# POD for component", component_name)
            POD = self.POD[component_name]
            (_, Z, N) = POD.apply(self.Nmax)
            self.reduced_problem.Z.enrich(Z, component_name=component_name)
            self.reduced_problem.N[component_name] += N
            POD.print_eigenvalues(N)
            POD.save_eigenvalues_file(self.folder["post_processing"], "eigs_" + component_name)
            POD.save_retained_energy_file(self.folder["post_processing"], "retained_energy_" + component_name)
        self.reduced_problem.Z.save(self.reduced_problem.folder["basis"], "basis")
        
    ## Update the snapshots matrix
    def update_snapshots_matrix(self, snapshot, supremizer):
        for component_name in ("u", "p"):
            function_component = self.truth_problem.component_name_to_function_component[component_name]
            self.POD[component_name].store_snapshot(snapshot, function_component)
        for component_name in ("s", ):
            function_component = self.truth_problem.component_name_to_function_component[component_name]
            self.POD[component_name].store_snapshot(supremizer, function_component)
        
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
        Nmax = max(N.values())
            
        self._init_error_analysis(**kwargs)
        
        print("==============================================================")
        print("=             Error analysis begins                          =")
        print("==============================================================")
        print("")
        
        error_analysis_table = ErrorAnalysisTable(self.testing_set)
        error_analysis_table.set_Nmax(Nmax)
        error_analysis_table.add_column("error_u", group_name="solution_u", operations="mean")
        error_analysis_table.add_column("relative_error_u", group_name="solution_u", operations="mean")
        error_analysis_table.add_column("error_p", group_name="solution_p", operations="mean")
        error_analysis_table.add_column("relative_error_p", group_name="solution_p", operations="mean")
        error_analysis_table.add_column("error_j", group_name="output", operations="mean")
        error_analysis_table.add_column("relative_error_j", group_name="output", operations="mean")
        
        for (run, mu) in enumerate(self.testing_set):
            print("############################## run =", run, "######################################")
            
            self.reduced_problem.set_mu(mu)
                        
            for n in range(1, Nmax + 1): # n = 1, ... Nmax
                (
                    error_analysis_table["error_u", n, run], error_analysis_table["relative_error_u", n, run], 
                    error_analysis_table["error_p", n, run], error_analysis_table["relative_error_p", n, run], 
                    error_analysis_table["error_j", n, run], error_analysis_table["relative_error_j", n, run]
                ) = self.reduced_problem.compute_error(n, **kwargs)
        
        # Print
        print("")
        print(error_analysis_table)
        
        print("")
        print("==============================================================")
        print("=             Error analysis ends                            =")
        print("==============================================================")
        print("")
        
        self._finalize_error_analysis(**kwargs)
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
