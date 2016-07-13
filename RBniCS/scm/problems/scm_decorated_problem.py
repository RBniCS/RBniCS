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
## @file scm.py
#  @brief Implementation of the successive constraints method for the approximation of the coercivity constant
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
import os # for path and makedir
import shutil # for rm
import glpk # for LB computation
import sys # for sys.float_info.max
import random # to randomize selection in case of equal error bound
import operator # to find closest parameters
from RBniCS.problems import ParametrizedProblem
from RBniCS.scm.io_utils import BoundingBoxSideList

def SCMDecoratedProblem(
    M_e = -1,
    M_p = -1,
    constrain_minimum_eigenvalue = 1.e5,
    constrain_maximum_eigenvalue = 1.e-5,
    bounding_box_minimum_eigensolver_parameters = dict(spectral_transform="shift-and-invert", spectral_shift=1.e-5),
    bounding_box_maximum_eigensolver_parameters = dict(spectral_transform="shift-and-invert", spectral_shift=1.e5),
    coercivity_eigensolver_parameters = dict(spectral_transform="shift-and-invert", spectral_shift=1.e-5)
):
    def SCMDecoratedProblem_Decorator(ParametrizedProblem_DerivedClass):
    
        #~~~~~~~~~~~~~~~~~~~~~~~~~     SCM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
        ## @class SCM
        #
        # Successive constraint method for the approximation of the coercivity constant
        class _SCMApproximation(ParametrizedProblem):

            ###########################     CONSTRUCTORS     ########################### 
            ## @defgroup Constructors Methods related to the construction of the SCM object
            #  @{
        
            ## Default initialization of members
            def __init__(self, truth_problem, folder_prefix):
                # Call the parent initialization
                ParametrizedProblem.__init__(self, folder_prefix)
                # Store the parametrized problem object and the bc list
                self.truth_problem = truth_problem
                                
                # $$ ONLINE DATA STRUCTURES $$ #
                # Define additional storage for SCM
                self.B_min = BoundingBoxSideList() # minimum values of the bounding box mathcal{B}. Vector of size Q
                self.B_max = BoundingBoxSideList() # maximum values of the bounding box mathcal{B}. Vector of size Q
                self.C_J = TrainingSetIndices() # list storing the indices of greedily selected parameters during the training phase
                self.complement_C_J = TrainingSetIndices() # list storing the indices of the complement of greedily selected parameters during the training phase
                self.alpha_J = CoercivityConstantsList() # list storing the truth coercivity constants at the greedy parameters in C_J
                self.alpha_LB_on_xi_train = CoercivityConstantsList() # list storing the approximation of the coercivity constant on the complement of C_J (at the previous iteration, during the offline phase)
                self.eigenvector_J = EigenVectorsList() # list of eigenvectors corresponding to the truth coercivity constants at the greedy parameters in C_J
                self.UB_vectors_J = UpperBoundsList() # list of Q-dimensional vectors storing the infimizing elements at the greedy parameters in C_J
                self.M_e = M_e # integer denoting the number of constraints based on the exact eigenvalues. If < 0, then it is assumed to be len(C_J)
                self.M_p = M_p # integer denoting the number of constraints based on the previous lower bounds. If < 0, then it is assumed to be len(C_J)
                
                # $$ OFFLINE DATA STRUCTURES $$ #
                # Matrices/vectors resulting from the truth discretization
                # I/O
                self.folder["basis"] = self.folder_prefix + "/" + "basis"
                self.folder["reduced_operators"] = self.folder_prefix + "/" + "reduced_operators"
                # 
                self.exact_coercivity_constant_calculator = ParametrizedHermitianEigenProblem(truth_problem, "a", True, constrain_minimum_eigenvalue, "smallest", coercivity_eigensolver_parameters)
                
                # Override truth_problem's set_mu to propogate the value of the parameters to EIM
                standard_set_mu = truth_problem.set_mu
                def overridden_set_mu(self_, mu): # self_ is truth_problem, self is the EIM approximation
                    standard_set_mu(mu)
                    if self.mu is not mu:
                        self.set_mu(mu)
                truth_problem.set_mu = types.MethodType(overridden_set_mu, truth_problem)
                
                # In a similar way, also override truth_problem's set_mu_range, even though it should have been called before this constructor and never called again
                standard_set_mu_range = truth_problem.set_mu_range
                def overridden_set_mu_range(self_, mu_range): # self_ is truth_problem, self is the EIM approximation
                    standard_set_mu_range(mu_range)
                    self.set_mu_range(mu_range)
                truth_problem.set_mu_range = types.MethodType(overridden_set_mu_range, truth_problem)
                # Make sure that in any case that the current mu_range is up to date
                self.set_mu_range(truth_problem.mu_range)
                
            #  @}
            ########################### end - CONSTRUCTORS - end ###########################
            
            ###########################     SETTERS     ########################### 
            ## @defgroup Setters Set properties of the reduced order approximation
            #  @{
            
            ## OFFLINE/ONLINE: set the current value of the parameter. Overridden to propagate to truth problem.
            def set_mu(self, mu):
                self.mu = mu
                if self.truth_problem.mu is not mu:
                    self.truth_problem.set_mu(mu)
                    
            ## OFFLINE/ONLINE: set the current value of the parameter. Overridden to propagate to truth problem.
            def set_mu_range(self, mu_range):
                self.mu_range = mu_range
                if self.truth_problem.mu_range is not mu_range:
                    self.truth_problem.set_mu(mu_range)
            
            #  @}
            ########################### end - SETTERS - end ########################### 
        
            ###########################     ONLINE STAGE     ########################### 
            ## @defgroup OnlineStage Methods related to the online stage
            #  @{
            
            ## Initialize data structures required for the online phase
            def init(self, current_stage="online"):
                self.current_stage = current_stage
                # Read/Initialize reduced order data structures
                if current_stage == "online":
                    self.B_min.load(self.folder["reduced_operators"], "B_min")
                    self.B_max.load(self.folder["reduced_operators"], "B_max")
                    self.C_J.load(self.folder["reduced_operators"], "C_J")
                    self.complement_C_J.load(self.folder["reduced_operators"], "complement_C_J")
                    self.alpha_J.load(self.folder["reduced_operators"], "alpha_J")
                    self.alpha_LB_on_xi_train.load(self.folder["reduced_operators"], "alpha_LB_on_xi_train")
                    self.UB_vectors_J.load(self.folder["reduced_operators"], "UB_vectors_J")
                    # Set the value of N
                    self.N = len(self.C_J)
                elif current_stage == "offline":
                    # Properly resize structures related to operator
                    Q = self.truth_problem.Q["a"]
                    self.Bmin = BoundingBoxSideList(Q)
                    self.Bmax = BoundingBoxSideList(Q)
                    # Properly resize structures related to xi_train
                    ntrain = len(self.xi_train)
                    self.alpha_LB_on_xi_train = CoercivityConstantsList(ntrain)
                    self.complement_C_J = TrainingSetIndices(:ntrain)
                else:
                    raise RuntimeError("Invalid stage in init().")
        
            ## Get a lower bound for alpha
            def get_stability_factor_lower_bound(self, mu, safeguard=True):
                lp = glpk.glp_create_prob()
                glpk.glp_set_obj_dir(lp, glpk.GLP_MIN)
                Q = self.parametrized_problem.Q["a"]
                N = self.N
                M_e = self.M_e
                if M_e < 0:
                    M_e = N
                if M_e > len(self.C_J):
                    M_e = len(self.C_J) # = N
                M_p = self.M_p
                if M_p < 0:
                    M_p = N
                if M_p > len(self.complement_C_J):
                    M_p = len(self.complement_C_J)
                
                # 1. Linear program unknowns: Q variables, y_1, ..., y_{Q_a}
                glpk.glp_add_cols(lp, Q)
                
                # 2. Range: constrain the variables to be in the bounding box (note: GLPK indexing starts from 1)
                for q in range(Q):
                    if self.B_min[q] < self.B_max[q]: # the usual case
                        glpk.glp_set_col_bnds(lp, q + 1, glpk.GLP_DB, self.B_min[q], self.B_max[q])
                    elif self.B_min[q] == self.B_max[q]: # unlikely, but possible
                        glpk.glp_set_col_bnds(lp, q + 1, glpk.GLP_FX, self.B_min[q], self.B_max[q])
                    else: # there is something wrong in the bounding box: set as unconstrained variable
                        print("Warning: wrong bounding box for affine expansion element #", q)
                        glpk.glp_set_col_bnds(lp, q + 1, glpk.GLP_FR, 0., 0.)
                
                # 3. Add two different sets of constraints
                glpk.glp_add_rows(lp, M_e + M_p)
                array_size = (M_e + M_p)*Q
                matrix_row_index = glpk.intArray(array_size + 1) # + 1 since GLPK indexing starts from 1
                matrix_column_index = glpk.intArray(array_size + 1)
                matrix_content = glpk.doubleArray(array_size + 1)
                glpk_container_size = 0
                
                # 3a. Add constraints: a constraint is added for the closest samples to mu in C_J
                closest_C_J_indices = self.closest_parameters(M_e, self.C_J, mu)
                for j in range(M_e):
                    # Overwrite parameter values
                    omega = self.xi_train[ self.C_J[ closest_C_J_indices[j] ] ]
                    self.parametrized_problem.set_mu(omega)
                    current_theta_a = self.parametrized_problem.compute_theta("a")
                    
                    # Assemble the LHS of the constraint
                    for q in range(Q):
                        matrix_row_index[glpk_container_size + 1] = int(j + 1)
                        matrix_column_index[glpk_container_size + 1] = int(q + 1)
                        matrix_content[glpk_container_size + 1] = current_theta_a[q]
                        glpk_container_size += 1
                    
                    # Assemble the RHS of the constraint
                    glpk.glp_set_row_bnds(lp, j + 1, glpk.GLP_LO, self.alpha_J[ closest_C_J_indices[j] ], 0.)
                closest_C_J_indices = None
                
                # 3b. Add constraints: also constrain the closest point in the complement of C_J, 
                #                      with RHS depending on previously computed lower bounds
                closest_complement_C_J_indices = self._closest_parameters(M_p, self.complement_C_J, mu)
                for j in range(M_p):
                    nu = self.xi_train[ self.complement_C_J[ closest_complement_C_J_indices[j] ] ]
                    self.parametrized_problem.set_mu(nu)
                    current_theta_a = self.parametrized_problem.compute_theta("a")
                    # Assemble first the LHS
                    for q in range(Q):
                        matrix_row_index[glpk_container_size + 1] = int(M_e + j + 1)
                        matrix_column_index[glpk_container_size + 1] = int(q + 1)
                        matrix_content[glpk_container_size + 1] = current_theta_a[q]
                        glpk_container_size += 1
                    # ... and then the RHS
                    glpk.glp_set_row_bnds(lp, M_e + j + 1, glpk.GLP_LO, self.alpha_LB_on_xi_train[ self.complement_C_J[ closest_complement_C_J_indices[j] ] ], 0.)
                closest_complement_C_J_indices = None
                
                # Load the assembled LHS
                glpk.glp_load_matrix(lp, array_size, matrix_row_index, matrix_column_index, matrix_content)
                
                # 4. Add cost function coefficients
                self.parametrized_problem.set_mu(mu)
                current_theta_a = self.parametrized_problem.compute_theta("a")
                for q in range(Q):
                    glpk.glp_set_obj_coef(lp, q + 1, current_theta_a[q])
                
                # 5. Solve the linear programming problem
                options = glpk.glp_smcp()
                glpk.glp_init_smcp(options)
                options.msg_lev = glpk.GLP_MSG_ERR
                options.meth = glpk.GLP_DUAL
                glpk.glp_simplex(lp, options)
                alpha_LB = glpk.glp_get_obj_val(lp)
                glpk.glp_delete_prob(lp)
                
                # 6. If a safeguard is requested (when called in the online stage of the RB method),
                #    we check the resulting value of alpha_LB. In order to avoid divisions by zero
                #    or taking the square root of a negative number, we allow an inefficient evaluation.
                if safeguard == True:
                    from numpy import isclose
                    alpha_UB = self.get_alpha_UB(mu)
                    if alpha_LB/alpha_UB < 0 or isclose(alpha_LB/alpha_UB, 0.):
                        print("SCM warning: alpha_LB is <= 0 at mu = " + str(mu) + ".", end=" ")
                        print("Please consider a larger Nmax for SCM. Meanwhile, a truth", end=" ")
                        print("eigensolve is performed.")
                        
                        (alpha_LB, _) = self.exact_coercivity_constant_calculator.solve()
                        
                    assert alpha_LB < alpha_UB, "alpha_LB is > alpha_UB at mu = " + str(mu) + "."
                
                return alpha_LB
        
            ## Get an upper bound for alpha
            def get_stability_factor_upper_bound(self, mu):
                Q = self.parametrized_problem.Q["a"]
                N = self.N
                UB_vectors_J = self.UB_vectors_J
                
                alpha_UB = sys.float_info.max
                self.parametrized_problem.set_mu(mu)
                current_theta_a = self.parametrized_problem.compute_theta("a")
                
                for j in range(N):
                    UB_vector = UB_vectors_J[j]
                    
                    # Compute the cost function for fixed omega
                    obj = 0.
                    for q in range(Q):
                        obj += UB_vector[q]*current_theta_a[q]
                    
                    if obj < alpha_UB:
                        alpha_UB = obj
                
                return alpha_UB

            ## Auxiliary function: M parameters in the set all_mu closest to mu
            def _closest_parameters(self, M, all_mu_indices, mu):
                # Trivial case 1:
                if M == 0:
                    return
                
                # Trivial case 2:
                if M == len(all_mu_indices):
                    return range(len(all_mu_indices))
                
                # Error case: there are not enough elements    
                if M > len(all_mu_indices):
                    raise RuntimeError("SCM error in closest parameters: this should never happen")
                
                indices_and_distances = list()
                for p in range(len(all_mu_indices)):
                    distance = self._parameters_distance(mu, self.xi_train[ all_mu_indices[p] ])
                    indices_and_distances.append((p, distance))
                indices_and_distances.sort(key=operator.itemgetter(1))
                neighbors = list()
                for p in range(M):
                    neighbors += [indices_and_distances[p][0]]
                return neighbors
                
            ## Auxiliary function: distance bewteen two parameters
            def _parameters_distance(self, mu1, mu2):
                P = len(mu1)
                distance = 0.
                for c in range(P):
                    distance += (mu1[c] - mu2[c])*(mu1[c] - mu2[c])
                return np.sqrt(distance)
        
            #  @}
            ########################### end - ONLINE STAGE - end ########################### 
        
            ###########################     I/O     ########################### 
            ## @defgroup IO Input/output methods
            #  @{
        
            ## Export solution in VTK format
            def export_solution(self, solution, folder, filename):
                self._export_vtk(solution, folder, filename, with_mesh_motion=True, with_preprocessing=True)
                
            #  @}
            ########################### end - I/O - end ###########################
        
        class SCMDecoratedProblem_Class(ParametrizedProblem_DerivedClass):
            ## Default initialization of members
            def __init__(self, V, *args):
                # Call the parent initialization
                ParametrizedProblem_DerivedClass.__init__(self, V, **kwargs)
                # Storage for SCM reduced problems
                self.SCM_approximation = _SCMApproximation(self, self.name() + "/scm")
                
                # Store here input parameters provided by the user that are needed by the reduction method
                self._input_storage_for_SCM_reduction.constrain_minimum_eigenvalue = constrain_minimum_eigenvalue
                self._input_storage_for_SCM_reduction.constrain_maximum_eigenvalue = constrain_minimum_eigenvalue
                self._input_storage_for_SCM_reduction.bounding_box_minimum_eigensolver_parameters = bounding_box_minimum_eigensolver_parameters
                self._input_storage_for_SCM_reduction.bounding_box_maximum_eigensolver_parameters = bounding_box_maximum_eigensolver_parameters
                
                # Signal to the factory that this problem has been decorated
                if not hasattr(self, "_problem_decorators"):
                    self._problem_decorators = dict() # string to bool
                self._problem_decorators["SCM"] = True
                
            ## Return the alpha_lower bound.
            def get_stability_factor(self):
                return self.SCM_approximation.solve()
                
            ## Get the name of the problem, to be used as a prefix for output folders.
            # Overridden to use the parent name
            @classmethod
            def name(cls):
                assert len(cls.__bases__) == 1
                return cls.__bases__[0].name()

        # return value (a class) for the decorator
        return SCMDecoratedProblem_Class
    
    # return the decorator itself
    return SCMDecoratedProblem_Decorator
    
# For the sake of the user, since this is the only class that he/she needs to use, rename it to an easier name
SCM = SCMDecoratedProblem
    
