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
import random # to randomize selection in case of equal error bound
from dolfin import Function, LagrangeInterpolator, vertices, Point
from RBniCS.reduction_methods import ReductionMethod
from RBniCS.linear_algebra import SnapshotsMatrix, OnlineMatrix
from RBniCS.eim.io_utils import AffineExpansionEIMStorage
from RBniCS.io_utils import NumpyIO

def EIMDecoratedReductionMethod(ReductionMethod_DerivedClass):

    #~~~~~~~~~~~~~~~~~~~~~~~~~     EIM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
    ## @class EIM
    #
    # Empirical interpolation method for the interpolation of parametrized functions
    class _EIMReductionMethod(ReductionMethod):
        
        ###########################     CONSTRUCTORS     ########################### 
        ## @defgroup Constructors Methods related to the construction of the EIM object
        #  @{
        
        ## Default initialization of members
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
            #
            self.offline.__func__.mu_index = 0
            self.interpolator = LagrangeInterpolator()
            
        #  @}
        ########################### end - CONSTRUCTORS - end ###########################
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
        
        ## Initialize data structures required for the offline phase
        def _init_offline(self):
            # Prepare folders and init EIM approximation
            all_folders_exist = True
            all_folders = list()
            all_folders.extend(self.folder.values())
            all_folders.extend(self.EIM_approximation.folder.values())
            for f in all_folders:
                if os.path.exists(f) and len(os.listdir(f)) == 0: # already created, but empty
                    all_folders_exist = False
                if not os.path.exists(f):
                    all_folders_exist = False
                    os.makedirs(f)
            if all_folders_exist:
                self.EIM_approximation.init("online")
                return False # offline construction should be skipped, since data are already available
            else:
                self.EIM_approximation.init("offline")
                return True # offline construction should be carried out
        
        ## Perform the offline phase of EIM
        def offline(self):
            need_to_do_offline_stage = self._init_offline()
            if not need_to_do_offline_stage:
                return self.EIM_approximation
            
            # Interpolate the parametrized function on the mesh grid for all parameters in xi_train
            print("==============================================================")
            print("=             EIM preprocessing phase begins                 =")
            print("==============================================================")
            print("")
            
            for run in range(len(self.xi_train)):
                print(":::::::::::::::::::::::::::::: EIM run = ", run, " ::::::::::::::::::::::::::::::")
                
                print("evaluate parametrized function")
                self.EIM_approximation.set_mu(self.xi_train[run])
                self.interpolator.interpolate(self.snapshot, self.EIM_approximation.parametrized_expression)
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
                print(":::::::::::::::::::::::::::::: EIM run = ", run, " ::::::::::::::::::::::::::::::")
                
                print("solve eim for mu = ", self.EIM_approximation.mu)
                self.EIM_approximation.solve()
                
                print("compute maximum interpolation error")
                (error, maximum_error, maximum_point) = self.compute_maximum_interpolation_error(output_error=True, output_location=True)
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
            for j in range(self.EIM_approximation.N):
                Z_j = Function(self.EIM_approximation.V, self.EIM_approximation.Z[j])
                self.EIM_approximation.interpolation_matrix[0][self.EIM_approximation.N - 1, j] = Z_j(self.EIM_approximation.interpolation_points[self.EIM_approximation.N - 1])
            self.EIM_approximation.interpolation_matrix.save(self.EIM_approximation.folder["reduced_operators"], "interpolation_matrix")
                
        ## Load the precomputed snapshot
        def load_snapshot(self):
            mu = self.EIM_approximation.mu
            mu_index = self.offline.__func__.mu_index
            assert mu == self.xi_train[mu_index]
            return self.snapshots_matrix[mu_index]
        
        # Compute the interpolation error and/or its maximum location
        def compute_maximum_interpolation_error(self, N=None, **output_options):
            if N is None:
                N = self.EIM_approximation.N
            if not "output_error" in output_options:
                output_options["output_error"] = False
            if not "output_location" in output_options:
                output_options["output_location"] = False
            
            # Compute the error (difference with the eim approximation)
            error = Function(self.EIM_approximation.V)
            error.vector().add_local(self.load_snapshot().array())
            if N > 0:
                error.vector().add_local(- (self.EIM_approximation.Z*self.EIM_approximation._interpolation_coefficients).array())
            error.vector().apply("")
            
            # Locate the vertex of the mesh where the error is maximum
            mesh = self.EIM_approximation.V.mesh()
            bounding_box_tree = mesh.bounding_box_tree()
            maximum_error = 0.
            maximum_point = None
            for v in vertices(mesh):
                point = mesh.coordinates()[v.index()]
                assert bounding_box_tree.collides_entity(Point(point)) # TODO: this will fail in parallel
                err = error(point)
                if abs(err) > abs(maximum_error):
                    maximum_point = point
                    maximum_error = err
            assert maximum_error != 0.
            assert maximum_point is not None
                
            # Return
            if output_options["output_error"] and output_options["output_location"]:
                return (error, abs(maximum_error), maximum_point)
            elif output_options["output_error"]:
                return (error, abs(maximum_error))
            elif output_options["output_location"]:
                return (maximum_point,)
            else:
                raise RuntimeError("Invalid output options")
                                
        ## Choose the next parameter in the offline stage in a greedy fashion
        def greedy(self):
            err_max = -1.0
            munew = None
            munew_index = None
            for i in range(len(self.xi_train)):
                self.EIM_approximation.set_mu(self.xi_train[i])
                self.offline.__func__.mu_index = i
                
                # Compute the EIM approximation ...
                self.EIM_approximation.solve()
                
                # ... and compute the maximum error
                (_, err) = self.compute_maximum_interpolation_error(output_error=True)
                
                if (err > err_max):
                    err_max = err
                    munew = self.xi_train[i]
                    munew_index = i
            assert err_max > 0.
            assert munew is not None
            assert munew_index is not None
            print("absolute error max = ", err_max)
            self.EIM_approximation.set_mu(munew)
            self.offline.__func__.mu_index = munew_index
            self.save_greedy_post_processing_file(self.EIM_approximation.N, err_max, munew, self.folder["post_processing"])
            
        #  @}
        ########################### end - OFFLINE STAGE - end ########################### 
        
        ###########################     ERROR ANALYSIS     ########################### 
        ## @defgroup ErrorAnalysis Error analysis
        #  @{
        
        # Compute the error of the empirical interpolation approximation with respect to the
        # exact function over the test set
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
                print(":::::::::::::::::::::::::::::: EIM run = ", run, " ::::::::::::::::::::::::::::::")
                
                self.EIM_approximation.set_mu(self.xi_test[run])
                
                # Evaluate the exact function on the truth grid
                self.interpolator.interpolate(self.snapshot, self.EIM_approximation.parametrized_expression)
                
                for n in range(1, N + 1): # n = 1, ... N
                    self.online_solve(n)
                    error_analysis_table["error", n, run] = self.compute_maximum_interpolation_error(n, output_error=True)
            
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
        
        ###########################     I/O     ########################### 
        ## @defgroup IO Input/output methods
        #  @{
    
        ## Save greedy post processing to file
        @staticmethod
        def save_greedy_post_processing_file(N, err_max, mu_greedy, directory):
            with open(directory + "/error_max.txt", "a") as outfile:
                outfile.write(str(N) + " " + str(err_max) + "\n")
            with open(directory + "/mu_greedy.txt", "a") as outfile:
                outfile.write(str(mu_greedy) + "\n")
            
        #  @}
        ########################### end - I/O - end ########################### 

    class EIMDecoratedReductionMethod_Class(ReductionMethod_DerivedClass):
        def __init__(self, truth_problem):
            # Call the parent initialization
            ReductionMethod_DerivedClass.__init__(self, truth_problem)
            
            # Preprocess each term in the affine expansions
            EIMApproximation = truth_problem.EIMApproximation # class alias
            EIMReduction = _EIMReductionMethod # class alias
            EIM_reductions_unsorted = dict() # from coefficient to EIM reduction, temporary storage to avoid duplicates
            self.EIM_reductions = dict() # from terms to AffineExpansionEIMStorage
            for term in self.truth_problem.terms:
                forms = self.truth_problem.assemble_operator(term, exact_evaluation=True)
                Q = len(forms)
                self.EIM_reductions[term] = AffineExpansionEIMStorage(Q)
                self.truth_problem.EIM_approximations[term] = AffineExpansionEIMStorage(Q)
                for q in range(Q):
                    if len(forms[q].coefficients()) == 0:
                        self.EIM_reductions[term][q] = None
                        self.truth_problem.EIM_approximations[term][q] = None
                    else:
                        assert len(forms[q].coefficients()) == 1
                        coeff = forms[q].coefficients()[0]
                        if hasattr(coeff, "mu_0"): # is parametrized
                            if coeff not in EIM_reductions_unsorted:
                                current_EIM_approximation = EIMApproximation(self.truth_problem.V, self.truth_problem, coeff, self.truth_problem.name() + "/eim/" + str(len(EIM_reductions_unsorted)))
                                current_EIM_reduction = EIMReduction(current_EIM_approximation, self.truth_problem.name() + "/eim/" + str(len(EIM_reductions_unsorted)))
                                EIM_reductions_unsorted[coeff] = current_EIM_reduction
                            self.EIM_reductions[term][q] = EIM_reductions_unsorted[coeff]
                            self.truth_problem.EIM_approximations[term][q] = EIM_reductions_unsorted[coeff].EIM_approximation
                        else:
                            self.EIM_reductions[term][q] = None
                            self.truth_problem.EIM_approximations[term][q] = None
            
        ###########################     SETTERS     ########################### 
        ## @defgroup Setters Set properties of the reduced order approximation
        #  @{
    
        # Propagate the values of all setters also to the EIM object
        
        ## OFFLINE: set maximum reduced space dimension (stopping criterion)
        def set_Nmax(self, Nmax, **kwargs):
            ReductionMethod_DerivedClass.set_Nmax(self, Nmax, **kwargs)
            assert "EIM" in kwargs
            Nmax_EIM = kwargs["EIM"]
            for term in self.EIM_reductions:
                for q in range(len(self.EIM_reductions[term])):
                    if self.EIM_reductions[term][q] is not None:
                        if isinstance(Nmax_EIM, dict):
                            assert term in Nmax_EIM and q in Nmax_EIM[term]
                            self.EIM_reductions[term][q].set_Nmax(Nmax_EIM[term][q]) # kwargs are not needed
                        else:
                            assert isinstance(Nmax_EIM, int)
                            self.EIM_reductions[term][q].set_Nmax(Nmax_EIM) # kwargs are not needed

            
        ## OFFLINE: set the elements in the training set \xi_train.
        def set_xi_train(self, ntrain, enable_import=True, sampling=None):
            import_successful = ReductionMethod_DerivedClass.set_xi_train(self, ntrain, enable_import, sampling)
            for term in self.EIM_reductions:
                for q in range(len(self.EIM_reductions[term])):
                    if self.EIM_reductions[term][q] is not None:
                        import_successful_EIM = self.EIM_reductions[term][q].set_xi_train(ntrain, enable_import, sampling)
                        import_successful = import_successful and import_successful_EIM
            return import_successful
            
        ## ERROR ANALYSIS: set the elements in the test set \xi_test.
        def set_xi_test(self, ntest, enable_import=False, sampling=None):
            import_successful = ReductionMethod_DerivedClass.set_xi_test(self, ntest, enable_import, sampling)
            for term in self.EIM_reductions:
                for q in range(len(self.EIM_reductions[term])):
                    if self.EIM_reductions[term][q] is not None:
                        import_successful_EIM = self.EIM_reductions[term][q].set_xi_test(ntest, enable_import, sampling)
                        import_successful = import_successful and import_successful_EIM
            return import_successful
            
        #  @}
        ########################### end - SETTERS - end ########################### 
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
    
        ## Perform the offline phase of the reduced order model
        def offline(self):
            # Perform first the EIM offline phase, ...
            bak_first_mu = tuple(list(self.truth_problem.mu))
            for term in self.EIM_reductions:
                for q in range(len(self.EIM_reductions[term])):
                    if self.EIM_reductions[term][q] is not None:
                        self.EIM_reductions[term][q].offline()
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
        def error_analysis(self, N=None):
            # Perform first the EIM error analysis, ...
            for term in self.EIM_reductions:
                for q in range(len(self.EIM_reductions[term])):
                    if self.EIM_reductions[term][q] is not None:
                        self.EIM_reductions[term][q].error_analysis(N)
            # ..., and then call the parent method.
            ReductionMethod_DerivedClass.error_analysis(self, N)
            
        #  @}
        ########################### end - ERROR ANALYSIS - end ########################### 
        
    # return value (a class) for the decorator
    return EIMDecoratedReductionMethod_Class
    
