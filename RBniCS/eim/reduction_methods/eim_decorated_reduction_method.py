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
## @file eim.py
#  @brief Implementation of the empirical interpolation method for the interpolation of parametrized functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
import os
from dolfin import Function, project, vertices
from RBniCS.reduction_methods import ReductionMethod
from RBniCS.linear_algebra import SnapshotsMatrix, OnlineMatrix
from RBniCS.utils.io import Folders, ErrorAnalysisTable, SpeedupAnalysisTable, GreedySelectedParametersList, GreedyErrorEstimatorsList
from RBniCS.utils.mpi import print, mpi_comm
from RBniCS.utils.decorators import extends, override

def EIMDecoratedReductionMethod(ReductionMethod_DerivedClass):

    #~~~~~~~~~~~~~~~~~~~~~~~~~     EIM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
    ## @class EIM
    #
    # Empirical interpolation method for the interpolation of parametrized functions
    @extends(ReductionMethod)
    class _EIMReductionMethod(ReductionMethod):
        
        ###########################     CONSTRUCTORS     ########################### 
        ## @defgroup Constructors Methods related to the construction of the EIM object
        #  @{
        
        ## Default initialization of members
        @override
        def __init__(self, EIM_approximation, folder_prefix):
            # Call the parent initialization
            ReductionMethod.__init__(self, folder_prefix, EIM_approximation.mu_range)
            
            # $$ OFFLINE DATA STRUCTURES $$ #
            # High fidelity problem
            self.EIM_approximation = EIM_approximation
            # Declare a new matrix to store the snapshots
            self.snapshot = Function(EIM_approximation.V)
            self.snapshots_matrix = SnapshotsMatrix()
            # I/O
            self.folder["snapshots"] = self.folder_prefix + "/" + "snapshots"
            self.folder["post_processing"] = self.folder_prefix + "/" + "post_processing"
            self.greedy_selected_parameters = GreedySelectedParametersList()
            self.greedy_errors = GreedyErrorEstimatorsList()
            #
            self.offline.__func__.mu_index = 0
            
        #  @}
        ########################### end - CONSTRUCTORS - end ###########################
        
        @override
        def set_xi_train(self, ntrain, enable_import=True, sampling=None):
            import_successful = ReductionMethod.set_xi_train(self, ntrain, enable_import, sampling)
            # Since exact evaluation is required, we cannot use a distributed xi_train
            self.xi_train.distributed_max = False
            return import_successful
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
        
        ## Initialize data structures required for the offline phase
        @override
        def _init_offline(self):
            # Prepare folders and init EIM approximation
            all_folders = Folders()
            all_folders.update(self.folder)
            all_folders.update(self.EIM_approximation.folder)
            all_folders.pop("xi_test") # this is required only in the error analysis
            at_least_one_folder_created = all_folders.create()
            if not at_least_one_folder_created:
                self.EIM_approximation.init("online")
                return False # offline construction should be skipped, since data are already available
            else:
                self.EIM_approximation.init("offline")
                return True # offline construction should be carried out
        
        ## Perform the offline phase of EIM
        @override
        def offline(self):
            need_to_do_offline_stage = self._init_offline()
            if not need_to_do_offline_stage:
                return self.EIM_approximation
            
            # Project the parametrized function on the mesh grid for all parameters in xi_train
            print("==============================================================")
            print("=             EIM preprocessing phase begins                 =")
            print("==============================================================")
            print("")
            
            for run in range(len(self.xi_train)):
                print(":::::::::::::::::::::::::::::: EIM run =", run, "::::::::::::::::::::::::::::::")
                
                print("evaluate parametrized function")
                self.EIM_approximation.set_mu(self.xi_train[run])
                project(self.EIM_approximation.parametrized_expression, V=self.EIM_approximation.V, function=self.snapshot)
                self.EIM_approximation.export_solution(self.snapshot, self.folder["snapshots"], "truth_" + str(run))
                
                print("update snapshot matrix")
                self.update_snapshots_matrix(self.snapshot)

                print("")
            
            print("==============================================================")
            print("=             EIM preprocessing phase ends                   =")
            print("==============================================================")
            print("")
            
            print("==============================================================")
            print("=             EIM offline phase begins                       =")
            print("==============================================================")
            print("")
            
            # Arbitrarily start from the first parameter in the training set
            self.EIM_approximation.set_mu(self.xi_train[0])
            self.offline.__func__.mu_index = 0
            # Resize the interpolation matrix
            self.EIM_approximation.interpolation_matrix[0] = OnlineMatrix(self.Nmax, self.Nmax)
            for run in range(self.Nmax):
                print(":::::::::::::::::::::::::::::: EIM run =", run, "::::::::::::::::::::::::::::::")
                
                print("solve eim for mu =", self.EIM_approximation.mu)
                self.EIM_approximation.solve()
                
                print("compute maximum interpolation error")
                self.snapshot = self.load_snapshot()
                (error, maximum_error, maximum_point) = self.compute_maximum_interpolation_error()
                self.update_interpolation_points(maximum_point)
                
                print("update basis matrix")
                self.update_basis_matrix(error, maximum_error)
                
                print("update interpolation matrix")
                self.update_interpolation_matrix()
                
                if self.EIM_approximation.N < self.Nmax:
                    print("find next mu")
                    
                self.greedy()

                print("")
                
            print("==============================================================")
            print("=             EIM offline phase ends                         =")
            print("==============================================================")
            print("")
            
            # mu_index does not make any sense from now on
            self.offline.__func__.mu_index = None
            
            self.EIM_approximation.init("online")
            return self.EIM_approximation
            
        ## Update the snapshots matrix
        def update_snapshots_matrix(self, snapshot):
            self.snapshots_matrix.enrich(snapshot)
            
        ## Update basis matrix
        def update_basis_matrix(self, error, maximum_error):
            error.vector()[:] /= maximum_error
            self.EIM_approximation.Z.enrich(error)
            self.EIM_approximation.Z.save(self.EIM_approximation.folder["basis"], "basis", self.EIM_approximation.V)
            self.EIM_approximation.N += 1
            
        def update_interpolation_points(self, maximum_point):
            self.EIM_approximation.interpolation_points.append(maximum_point)
            self.EIM_approximation.interpolation_points.save(self.EIM_approximation.folder["reduced_operators"], "interpolation_points")
        
        ## Assemble the interpolation matrix
        def update_interpolation_matrix(self):
            (last_point, last_point_processor_id) = self.EIM_approximation.interpolation_points[self.EIM_approximation.N - 1]
            for j in range(self.EIM_approximation.N):
                Z_j = Function(self.EIM_approximation.V, self.EIM_approximation.Z[j])
                value = None
                if mpi_comm.rank == last_point_processor_id:
                    value = Z_j(last_point)
                value = mpi_comm.bcast(value, root=last_point_processor_id)
                self.EIM_approximation.interpolation_matrix[0][self.EIM_approximation.N - 1, j] = value
            self.EIM_approximation.interpolation_matrix.save(self.EIM_approximation.folder["reduced_operators"], "interpolation_matrix")
                
        ## Load the precomputed snapshot
        def load_snapshot(self):
            mu = self.EIM_approximation.mu
            mu_index = self.offline.__func__.mu_index
            assert mu_index is not None
            assert mu == self.xi_train[mu_index]
            return Function(self.EIM_approximation.V, self.snapshots_matrix[mu_index])
        
        # Compute the interpolation error and/or its maximum location
        def compute_maximum_interpolation_error(self, N=None):
            if N is None:
                N = self.EIM_approximation.N
            
            # Compute the error (difference with the eim approximation)
            error = Function(self.EIM_approximation.V)
            error.vector().add_local(self.snapshot.vector().array())
            if N > 0:
                error.vector().add_local(- (self.EIM_approximation.Z[:N]*self.EIM_approximation._interpolation_coefficients).array())
            error.vector().apply("")
            
            # Locate the vertex of the mesh where the error is maximum
            mesh = self.EIM_approximation.V.mesh()
            maximum_error = None
            maximum_point = None
            for v in vertices(mesh):
                point = mesh.coordinates()[v.index()]
                err = error(point)
                if maximum_error is None or abs(err) > abs(maximum_error):
                    maximum_point = point
                    maximum_error = err
            assert maximum_error is not None
            assert maximum_point is not None
            
            # Communicate the result in parallel
            from mpi4py.MPI import MAX
            local_abs_maximum_error = abs(maximum_error)
            global_abs_maximum_error = mpi_comm.allreduce(local_abs_maximum_error, op=MAX)
            global_abs_maximum_error_processor_argmax = -1
            if global_abs_maximum_error == local_abs_maximum_error:
                global_abs_maximum_error_processor_argmax = mpi_comm.rank
            global_abs_maximum_error_processor_argmax = mpi_comm.allreduce(global_abs_maximum_error_processor_argmax, op=MAX)
            global_maximum_point = mpi_comm.bcast(maximum_point, root=global_abs_maximum_error_processor_argmax)
            global_maximum_error = mpi_comm.bcast(maximum_error, root=global_abs_maximum_error_processor_argmax)
                
            # Return
            return (error, global_maximum_error, global_maximum_point)
            
        ## Choose the next parameter in the offline stage in a greedy fashion
        def greedy(self):
            def solve_and_computer_error(mu, index):
                self.offline.__func__.mu_index = index
                self.EIM_approximation.set_mu(mu)
                
                self.EIM_approximation.solve()
                self.snapshot = self.load_snapshot()
                (_, err, _) = self.compute_maximum_interpolation_error()
                return err
                
            (error_max, error_argmax) = self.xi_train.max(solve_and_computer_error, abs)
            print("maximum EIM error =", abs(error_max))
            self.EIM_approximation.set_mu(self.xi_train[error_argmax])
            self.offline.__func__.mu_index = error_argmax
            self.greedy_selected_parameters.append(self.xi_train[error_argmax])
            self.greedy_selected_parameters.save(self.folder["post_processing"], "mu_greedy")
            self.greedy_errors.append(error_max)
            self.greedy_errors.save(self.folder["post_processing"], "error_max")
            
        #  @}
        ########################### end - OFFLINE STAGE - end ########################### 
        
        ###########################     ERROR ANALYSIS     ########################### 
        ## @defgroup ErrorAnalysis Error analysis
        #  @{
        
        @override
        def _init_error_analysis(self):
            pass
        
        # Compute the error of the empirical interpolation approximation with respect to the
        # exact function over the test set
        @override
        def error_analysis(self, N=None):
            if N is None:
                N = self.EIM_approximation.N
                
            self._init_error_analysis()
            
            print("==============================================================")
            print("=             EIM error analysis begins                      =")
            print("==============================================================")
            print("")
            
            error_analysis_table = ErrorAnalysisTable(self.xi_test)
            error_analysis_table.set_Nmax(N)
            error_analysis_table.add_column("error", group_name="eim", operations="mean")
            
            for run in range(len(self.xi_test)):
                print(":::::::::::::::::::::::::::::: EIM run =", run, "::::::::::::::::::::::::::::::")
                
                self.EIM_approximation.set_mu(self.xi_test[run])
                
                # Evaluate the exact function on the truth grid
                project(self.EIM_approximation.parametrized_expression, V=self.EIM_approximation.V, function=self.snapshot)
                
                for n in range(1, N + 1): # n = 1, ... N
                    self.EIM_approximation.solve(n)
                    (_, error_analysis_table["error", n, run], _) = self.compute_maximum_interpolation_error(n)
                    error_analysis_table["error", n, run] = abs(error_analysis_table["error", n, run])
            
            # Print
            print("")
            print(error_analysis_table)
            
            print("")
            print("==============================================================")
            print("=             EIM error analysis ends                        =")
            print("==============================================================")
            print("")
            
        #  @}
        ########################### end - ERROR ANALYSIS - end ########################### 
    
    @extends(ReductionMethod_DerivedClass, preserve_class_name=True)
    class EIMDecoratedReductionMethod_Class(ReductionMethod_DerivedClass):
        @override
        def __init__(self, truth_problem):
            # Call the parent initialization
            ReductionMethod_DerivedClass.__init__(self, truth_problem)
            # Storage for EIM reduction methods
            self.EIM_reductions = dict() # from coefficients to _EIMReductionMethod
            
            # Preprocess each term in the affine expansions
            for coeff in self.truth_problem.EIM_approximations:
                self.EIM_reductions[coeff] = _EIMReductionMethod(self.truth_problem.EIM_approximations[coeff], type(self.truth_problem).__name__ + "/eim/" + str(coeff.hash_code))
            
        ###########################     SETTERS     ########################### 
        ## @defgroup Setters Set properties of the reduced order approximation
        #  @{
    
        # Propagate the values of all setters also to the EIM object
        
        ## OFFLINE: set maximum reduced space dimension (stopping criterion)
        @override
        def set_Nmax(self, Nmax, **kwargs):
            ReductionMethod_DerivedClass.set_Nmax(self, Nmax, **kwargs)
            assert "EIM" in kwargs
            Nmax_EIM = kwargs["EIM"]
            if isinstance(Nmax_EIM, dict):
                for term in self.separated_forms:
                    for q in range(len(self.separated_forms[term])):
                        for i in range(len(self.separated_forms[term][q].coefficients)):
                            for coeff in self.separated_forms[term][q].coefficients[i]:
                                assert term in Nmax_EIM and q in Nmax_EIM[term]
                                assert coeff in self.EIM_reductions
                                self.EIM_reductions[coeff].set_Nmax(max(self.EIM_reductions[coeff].Nmax, Nmax_EIM[term][q])) # kwargs are not needed
            else:
                assert isinstance(Nmax_EIM, int)
                for coeff in self.EIM_reductions:
                    self.EIM_reductions[coeff].set_Nmax(Nmax_EIM) # kwargs are not needed

            
        ## OFFLINE: set the elements in the training set \xi_train.
        @override
        def set_xi_train(self, ntrain, enable_import=True, sampling=None):
            import_successful = ReductionMethod_DerivedClass.set_xi_train(self, ntrain, enable_import, sampling)
            # Since exact evaluation is required, we cannot use a distributed xi_train
            self.xi_train.distributed_max = False
            for coeff in self.EIM_reductions:
                import_successful_EIM = self.EIM_reductions[coeff].set_xi_train(ntrain, enable_import, sampling)
                import_successful = import_successful and import_successful_EIM
            return import_successful
            
        ## ERROR ANALYSIS: set the elements in the test set \xi_test.
        @override
        def set_xi_test(self, ntest, enable_import=False, sampling=None):
            import_successful = ReductionMethod_DerivedClass.set_xi_test(self, ntest, enable_import, sampling)
            for coeff in self.EIM_reductions:
                import_successful_EIM = self.EIM_reductions[coeff].set_xi_test(ntest, enable_import, sampling)
                import_successful = import_successful and import_successful_EIM
            return import_successful
            
        #  @}
        ########################### end - SETTERS - end ########################### 
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
    
        ## Perform the offline phase of the reduced order model
        @override
        def offline(self):
            # Perform first the EIM offline phase, ...
            bak_first_mu = tuple(list(self.truth_problem.mu))
            for coeff in self.EIM_reductions:
                self.EIM_reductions[coeff].offline()
            # ..., and then call the parent method.
            self.truth_problem.set_mu(bak_first_mu)
            return ReductionMethod_DerivedClass.offline(self)
    
        #  @}
        ########################### end - OFFLINE STAGE - end ###########################
    
        ###########################     ERROR ANALYSIS     ########################### 
        ## @defgroup ErrorAnalysis Error analysis
        #  @{
    
        # Compute the error of the reduced order approximation with respect to the full order one
        # over the test set
        @override
        def error_analysis(self, N=None):
            # Perform first the EIM error analysis, ...
            for coeff in self.EIM_reductions:
                self.EIM_reductions[coeff].error_analysis(N)
            # ..., and then call the parent method.
            ReductionMethod_DerivedClass.error_analysis(self, N)
            
        #  @}
        ########################### end - ERROR ANALYSIS - end ########################### 
        
    # return value (a class) for the decorator
    return EIMDecoratedReductionMethod_Class
    
