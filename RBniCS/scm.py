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

from dolfin import *
import os # for path and makedir
import shutil # for rm
import glpk # for LB computation
import sys # for sys.float_info.max
import random # to randomize selection in case of equal error bound
import operator # to find closest parameters
from parametrized_problem import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     SCM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class SCM
#
# Successive constraint method for the approximation of the coercivity constant
class SCM(ParametrizedProblem):

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the SCM object
    #  @{
    
    ## Default initialization of members
    def __init__(self, parametrized_problem):
        # Call the parent initialization
        ParametrizedProblem.__init__(self)
        # Store the parametrized problem object and the bc list
        self.parametrized_problem = parametrized_problem
        self.bc_list = parametrized_problem.bc_list
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # Define additional storage for SCM
        self.B_min = np.array([]) # minimum values of the bounding box mathcal{B}. Vector of size Qa
        self.B_max = np.array([]) # maximum values of the bounding box mathcal{B}. Vector of size Qa
        self.C_J = [] # vector storing the indices of greedily selected parameters during the training phase
        self.complement_C_J = [] # vector storing the indices of the complement of greedily selected parameters during the training phase
        self.alpha_J = [] # vector storing the truth coercivity constants at the greedy parameters in C_J
        self.alpha_LB_on_xi_train = np.array([]) # vector storing the approximation of the coercivity constant on the complement of C_J (at the previous iteration, during the offline phase)
        self.eigenvector_J = [] # vector of eigenvectors corresponding to the truth coercivity constants at the greedy parameters in C_J
        self.UB_vectors_J = [] # array of Qa-dimensional vectors storing the infimizing elements at the greedy parameters in C_J
        self.M_e = -1 # integer denoting the number of constraints based on the exact eigenvalues. If < 0, then it is assumed to be len(C_J)
        self.M_p = -1 # integer denoting the number of constraints based on the previous lower bounds. If < 0, then it is assumed to be len(C_J)
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # We need to discard dofs related to bcs in eigenvalue computations. To avoid having to create a PETSc submatrix
        # we simply zero rows and columns and replace the diagonal element with an eigenvalue that for sure
        # will not be the minimum/maximum
        self.invalid_minimum_eigenvalue = 1.e4
        self.invalid_maximum_eigenvalue = 1.e-4
        # 3c. Matrices/vectors resulting from the truth discretization
        self.truth_A__condensed_for_minimum_eigenvalue = ()
        self.truth_A__condensed_for_maximum_eigenvalue = ()
        self.S__condensed = ()
        # 9. I/O
        self.xi_train_folder = "xi_train__scm/"
        self.xi_test_folder = "xi_test__scm/"
        self.snap_folder = "snapshots__scm/"
        self.basis_folder = "basis__scm/"
        self.dual_folder = "dual__scm/" # never used
        self.reduced_matrices_folder = "reduced_matrices__scm/"
        self.post_processing_folder = "post_processing__scm/"
        # 
        self.mu_index = 0 # index of the greedy select parameter at the current iteration
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     SETTERS     ########################### 
    ## @defgroup Setters Set properties of the reduced order approximation
    #  @{
    
    ## OFFLINE: set the elements in the training set \xi_train. Overridden to resize alpha_LB_on_xi_train
    ##          Note that the default value of enable_import has been changed here to True
    def setxi_train(self, ntrain, enable_import=True, sampling="random"):
        if not enable_import:
        	sys.exit("SCM will not work without import.")
        # Save the flag if can import from file
        import_successful = False
        if os.path.exists(self.xi_train_folder + "xi_train.npy"):
            xi_train = np.load(self.xi_train_folder + "xi_train.npy")
            import_successful = (len(np.asarray(xi_train)) == ntrain)
        # Call parent
        ParametrizedProblem.setxi_train(self, ntrain, enable_import, sampling)
        # If xi_train was not imported, be safe and remove the previous folder, so that
        # if the user overwrites the training set but forgets to run the offline
        # phase he/she will get an error
        if import_successful == False:
            if os.path.exists(self.reduced_matrices_folder):
                shutil.rmtree(self.reduced_matrices_folder)
            os.makedirs(self.reduced_matrices_folder)
        # Properly resize related structures
        self.alpha_LB_on_xi_train = np.zeros([ntrain])
        self.complement_C_J = range(ntrain)
        
    #  @}
    ########################### end - SETTERS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    ## Get a lower bound for alpha
    def get_alpha_LB(self, mu, safeguard=True):
        self.load_reduced_data_structures()
        
        lp = glpk.glp_create_prob()
        glpk.glp_set_obj_dir(lp, glpk.GLP_MIN)
        Qa = self.parametrized_problem.Qa
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
        
        # 1. Linear program unknowns: Qa variables, y_1, ..., y_{Q_a}
        glpk.glp_add_cols(lp, Qa)
        
        # 2. Range: constrain the variables to be in the bounding box (note: GLPK indexing starts from 1)
        for qa in range(Qa):
            if self.B_min[qa] < self.B_max[qa]: # the usual case
                glpk.glp_set_col_bnds(lp, qa + 1, glpk.GLP_DB, self.B_min[qa], self.B_max[qa])
            elif self.B_min[qa] == self.B_max[qa]: # unlikely, but possible
                glpk.glp_set_col_bnds(lp, qa + 1, glpk.GLP_FX, self.B_min[qa], self.B_max[qa])
            else: # there is something wrong in the bounding box: set as unconstrained variable
                print "Warning: wrong bounding box for affine expansion element #", qa
                glpk.glp_set_col_bnds(lp, qa + 1, glpk.GLP_FR, 0., 0.)
        
        # 3. Add two different sets of constraints
        glpk.glp_add_rows(lp, M_e + M_p)
        array_size = (M_e + M_p)*Qa
        matrix_row_index = glpk.intArray(array_size + 1) # + 1 since GLPK indexing starts from 1
        matrix_column_index = glpk.intArray(array_size + 1)
        matrix_content = glpk.doubleArray(array_size + 1)
        glpk_container_size = 0
        
        # 3a. Add constraints: a constraint is added for the closest samples to mu in C_J
        closest_C_J_indices = self.closest_parameters(M_e, self.C_J, mu)
        for j in range(M_e):
            # Overwrite parameter values
            omega = self.xi_train[ self.C_J[ closest_C_J_indices[j] ] ]
            self.parametrized_problem.setmu(omega)
            current_theta_a = self.parametrized_problem.compute_theta_a()
            
            # Assemble the LHS of the constraint
            for qa in range(Qa):
                matrix_row_index[glpk_container_size + 1] = int(j + 1)
                matrix_column_index[glpk_container_size + 1] = int(qa + 1)
                matrix_content[glpk_container_size + 1] = current_theta_a[qa]
                glpk_container_size += 1
            
            # Assemble the RHS of the constraint
            glpk.glp_set_row_bnds(lp, j + 1, glpk.GLP_LO, self.alpha_J[ closest_C_J_indices[j] ], 0.)
        closest_C_J_indices = None
        
        # 3b. Add constraints: also constrain the closest point in the complement of C_J, 
        #                      with RHS depending on previously computed lower bounds
        closest_complement_C_J_indices = self.closest_parameters(M_p, self.complement_C_J, mu)
        for j in range(M_p):
            nu = self.xi_train[ self.complement_C_J[ closest_complement_C_J_indices[j] ] ]
            self.parametrized_problem.setmu(nu)
            current_theta_a = self.parametrized_problem.compute_theta_a()
            # Assemble first the LHS
            for qa in range(Qa):
                matrix_row_index[glpk_container_size + 1] = int(M_e + j + 1)
                matrix_column_index[glpk_container_size + 1] = int(qa + 1)
                matrix_content[glpk_container_size + 1] = current_theta_a[qa]
                glpk_container_size += 1
            # ... and then the RHS
            glpk.glp_set_row_bnds(lp, M_e + j + 1, glpk.GLP_LO, self.alpha_LB_on_xi_train[ self.complement_C_J[ closest_complement_C_J_indices[j] ] ], 0.)
        closest_complement_C_J_indices = None
        
        # Load the assembled LHS
        glpk.glp_load_matrix(lp, array_size, matrix_row_index, matrix_column_index, matrix_content)
        
        # 4. Add cost function coefficients
        self.parametrized_problem.setmu(mu)
        current_theta_a = self.parametrized_problem.compute_theta_a()
        for qa in range(Qa):
            glpk.glp_set_obj_coef(lp, qa + 1, current_theta_a[qa])
        
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
            tol = 1e-10
            alpha_UB = self.get_alpha_UB(mu)
            if alpha_LB/alpha_UB < tol:
                print "SCM warning: alpha_LB is <= 0 at mu = " + str(mu) + ".",
                print "Please consider a larger Nmax for SCM. Meanwhile, a truth",
                print "eigensolve is performed."
                
                (alpha_LB, discarded1, discarded2) = self.truth_coercivity_constant()
                
            if alpha_LB/alpha_UB > 1 + tol:
                print "SCM warning: alpha_LB is > alpha_UB at mu = " + str(mu) + ".",
                print "This should never happen!"
        
        return alpha_LB
    
    ## Get an upper bound for alpha
    def get_alpha_UB(self, mu):
        self.load_reduced_data_structures()
        
        Qa = self.parametrized_problem.Qa
        N = self.N
        UB_vectors_J = self.UB_vectors_J
        
        alpha_UB = sys.float_info.max
        self.parametrized_problem.setmu(mu)
        current_theta_a = self.parametrized_problem.compute_theta_a()
        
        for j in range(N):
            UB_vector = UB_vectors_J[j]
            
            # Compute the cost function for fixed omega
            obj = 0.
            for qa in range(Qa):
                obj += UB_vector[qa]*current_theta_a[qa]
            
            if obj < alpha_UB:
                alpha_UB = obj
        
        return alpha_UB

    ## Auxiliary function: M parameters in the set all_mu closest to mu
    def closest_parameters(self, M, all_mu_indices, mu):
        # Trivial case 1:
        if M == 0:
            return
        
        # Trivial case 2:
        if M == len(all_mu_indices):
            return range(len(all_mu_indices))
        
        # Error case: there are not enough elements    
        if M > len(all_mu_indices):
            sys.exit("SCM error in closest parameters: this should never happen")
        
        indices_and_distances = []
        for p in range(len(all_mu_indices)):
            distance = self.parameters_distance(mu, self.xi_train[ all_mu_indices[p] ])
            indices_and_distances.append((p, distance))
        indices_and_distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for p in range(M):
            neighbors += [indices_and_distances[p][0]]
        return neighbors
        
    ## Auxiliary function: distance bewteen two parameters
    def parameters_distance(self, mu1, mu2):
        P = len(mu1)
        distance = 0.
        for c in range(P):
            distance += (mu1[c] - mu2[c])*(mu1[c] - mu2[c])
        return np.sqrt(distance)
    
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Perform the offline phase of SCM
    def offline(self):
        print "=============================================================="
        print "=             SCM offline phase begins                       ="
        print "=============================================================="
        print ""
        if os.path.exists(self.post_processing_folder):
            shutil.rmtree(self.post_processing_folder)
        folders = (self.snap_folder, self.basis_folder, self.dual_folder, self.reduced_matrices_folder, self.post_processing_folder)
        for f in folders:
            if not os.path.exists(f):
                os.makedirs(f)
        
        # Save M_e and M_p
        np.save(self.reduced_matrices_folder + "M_e", self.M_e)
        np.save(self.reduced_matrices_folder + "M_p", self.M_p)
        
        # Assemble the condensed versions of truth_A and S matrices
        self.assemble_condensed_truth_matrices()
        
        # Compute the bounding box \mathcal{B}
        self.compute_bounding_box()
        
        # Arbitrarily start from the first parameter in the training set
        self.setmu(self.xi_train[0])
        self.mu_index = 0
        
        for run in range(self.Nmax):
            print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SCM run = ", run, " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            
            # Store the greedy parameter
            self.update_C_J()
            
            # Evaluate the coercivity constant
            print "evaluate the coercivity constant for mu = ", self.mu
            (alpha, eigenvector, UB_vector) = self.truth_coercivity_constant()
            self.alpha_J += [alpha]; np.save(self.reduced_matrices_folder + "alpha_J", self.alpha_J)
            self.eigenvector_J += [eigenvector]
            self.UB_vectors_J += [UB_vector]; np.save(self.reduced_matrices_folder + "UB_vectors_J", self.UB_vectors_J)
            self.export_solution(eigenvector, self.snap_folder + "eigenvector_" + str(run))
            
            # Prepare for next iteration
            if self.N < self.Nmax:
                print "find next mu"
                self.greedy()
            else:
                self.greedy()
        
        print "=============================================================="
        print "=             SCM offline phase ends                         ="
        print "=============================================================="
        print ""
        
        # mu_index does not make any sense from now on
        self.mu_index = None
    
    # Assemble condensed versions of truth matrices
    def assemble_condensed_truth_matrices(self):
        # Assemble matrices related to the LHS A of the parametrized problem
        if not self.parametrized_problem.truth_A:
            self.parametrized_problem.truth_A = self.parametrized_problem.assemble_truth_a()
        if self.parametrized_problem.Qa == 0:
            self.parametrized_problem.Qa = len(self.parametrized_problem.truth_A)
        
        # Assemble condensed matrices
        if not self.S__condensed:
            self.S__condensed = self.clear_constrained_dofs(self.parametrized_problem.S, 1.)
        if not self.truth_A__condensed_for_minimum_eigenvalue:
            for qa in range(self.parametrized_problem.Qa):
                self.truth_A__condensed_for_minimum_eigenvalue +=(
                    self.clear_constrained_dofs(self.parametrized_problem.truth_A[qa], self.invalid_minimum_eigenvalue), 
                )
        if not self.truth_A__condensed_for_maximum_eigenvalue:
            for qa in range(self.parametrized_problem.Qa):
                self.truth_A__condensed_for_maximum_eigenvalue +=(
                    self.clear_constrained_dofs(self.parametrized_problem.truth_A[qa], self.invalid_maximum_eigenvalue), 
                )
            
    # Compute the bounding box \mathcal{B}
    def compute_bounding_box(self):
        # Resize the bounding box storage
        Qa = self.parametrized_problem.Qa
        self.B_min = np.zeros((Qa))
        self.B_max = np.zeros((Qa))
        
        # RHS matrix
        S = self.S__condensed
        S = as_backend_type(S)
        
        for qa in range(Qa):
            # Compute the minimum eigenvalue
            A = self.truth_A__condensed_for_minimum_eigenvalue[qa]
            A = as_backend_type(A)
            
            eigensolver = SLEPcEigenSolver(A, S)
            eigensolver.parameters["problem_type"] = "gen_hermitian"
            eigensolver.parameters["spectrum"] = "smallest real"
            self.set_additional_eigensolver_options_for_bounding_box_minimum(eigensolver, qa)
            eigensolver.solve(1)
            r, c = eigensolver.get_eigenvalue(0) # real and complex part of the eigenvalue
            self.B_min[qa] = r
            print "B_min[" + str(qa) + "] = " + str(r)
            
            # Compute the maximum eigenvalue
            A = self.truth_A__condensed_for_maximum_eigenvalue[qa]
            A = as_backend_type(A)
            
            eigensolver = SLEPcEigenSolver(A, S)
            eigensolver.parameters["problem_type"] = "gen_hermitian"
            eigensolver.parameters["spectrum"] = "largest real"
            self.set_additional_eigensolver_options_for_bounding_box_maximum(eigensolver, qa)
            eigensolver.solve(1)
            r, c = eigensolver.get_eigenvalue(0) # real and complex part of the eigenvalue
            self.B_max[qa] = r
            print "B_max[" + str(qa) + "] = " + str(r)
        
        # Save to file
        np.save(self.reduced_matrices_folder + "B_min", self.B_min)
        np.save(self.reduced_matrices_folder + "B_max", self.B_max)
    
    # Store the greedy parameter
    def update_C_J(self):
        if self.mu != self.xi_train[self.mu_index]:
            # There is something wrong if we are here...
            sys.exit("Should never arrive here")
        
        self.C_J += [self.mu_index]
        if self.mu_index in self.complement_C_J: # if not SCM selects twice the same parameter
            self.complement_C_J.remove(self.mu_index)
        
        self.N = len(self.C_J)
        
        # Save to file
        np.save(self.reduced_matrices_folder + "C_J", self.C_J)
        np.save(self.reduced_matrices_folder + "complement_C_J", self.complement_C_J)
        
    # Evaluate the coercivity constant
    def truth_coercivity_constant(self):
        self.assemble_condensed_truth_matrices()
        
        self.parametrized_problem.setmu(self.mu)
        current_theta_a = self.parametrized_problem.compute_theta_a()
        A = self.parametrized_problem.affine_assemble_truth_symmetric_part_matrix(self.truth_A__condensed_for_minimum_eigenvalue, current_theta_a)
        A = as_backend_type(A)
        S = self.S__condensed
        S = as_backend_type(S)
        
        eigensolver = SLEPcEigenSolver(A, S)
        eigensolver.parameters["problem_type"] = "gen_hermitian"
        eigensolver.parameters["spectrum"] = "smallest real"
        self.set_additional_eigensolver_options_for_truth_coercivity_constant(eigensolver)
        eigensolver.solve(1)
        
        r, c, rv, cv = eigensolver.get_eigenpair(0) # real and complex part of the (eigenvalue, eigenvectors)
        rv_f = Function(self.parametrized_problem.V, rv)
        UB_vector = self.compute_UB_vector(self.parametrized_problem.truth_A, self.parametrized_problem.S, rv_f)
        
        return (r, rv_f, UB_vector)
        
    ## Compute the ratio between a_q(u,u) and s(u,u), for all q in vec
    def compute_UB_vector(self, vec, S, u):
        UB_vector = np.zeros((len(vec)))
        norm_S_squared = self.parametrized_problem.compute_scalar(u,u,S)
        for qa in range(len(vec)):
            UB_vector[qa] = self.parametrized_problem.compute_scalar(u,u,vec[qa])/norm_S_squared
        return UB_vector
        
    ## Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
        ntrain = len(self.xi_train)
        alpha_LB_on_xi_train = np.zeros([ntrain])
        #
        delta_max = -1.0
        munew = None
        munew_index = None
        for i in range(ntrain):
            mu = self.xi_train[i]
            self.mu_index = i
            self.setmu(mu)
            LB = self.get_alpha_LB(mu, False)
            UB = self.get_alpha_UB(mu)
            delta = (UB - LB)/UB
            tol = 1.e-10
            if LB/UB < -tol:
                print "SCM warning at mu = ", mu , ": LB = ", LB, " < 0"
            if LB/UB > 1 + tol:
                print "SCM warning at mu = ", mu , ": LB = ", LB, " > UB = ", UB
            alpha_LB_on_xi_train[i] = max(0, LB)
            if ((delta > delta_max) or (delta == delta_max and random.random() >= 0.5)):
                delta_max = delta
                munew = mu
                munew_index = i
                
        print "absolute SCM delta max = ", delta_max
        if os.path.isfile(self.post_processing_folder + "delta_max.npy") == True:
            d = np.load(self.post_processing_folder + "delta_max.npy")
            
            np.save(self.post_processing_folder + "delta_max", np.append(d, delta_max))
    
            m = np.load(self.post_processing_folder + "mu_greedy.npy")
            np.save(self.post_processing_folder + "mu_greedy", np.append(m, munew))
        else:
            np.save(self.post_processing_folder + "delta_max", delta_max)
            np.save(self.post_processing_folder + "mu_greedy", np.array(munew))

        self.setmu(munew)
        self.mu_index = munew_index
        
        # Overwrite alpha_LB_on_xi_train
        self.alpha_LB_on_xi_train = alpha_LB_on_xi_train
        np.save(self.reduced_matrices_folder + "alpha_LB_on_xi_train", self.alpha_LB_on_xi_train)
    
    # Clear constrained dofs
    def clear_constrained_dofs(self, M_in, diag_value):
        M = M_in.copy()
        if self.bc_list != None:
            fake_vector = Function(self.parametrized_problem.V)
            FAKE_VECTOR = fake_vector.vector()
            for bc in self.bc_list:
                bc.zero(M)
                bc.zero_columns(M, FAKE_VECTOR, diag_value)
        return M
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{

    ## Load reduced order data structures
    def load_reduced_data_structures(self):
        if len(np.asarray(self.B_min)) == 0: # avoid loading multiple times
            self.B_min = np.load(self.reduced_matrices_folder + "B_min.npy")
        if len(np.asarray(self.B_max)) == 0: # avoid loading multiple times
            self.B_max = np.load(self.reduced_matrices_folder + "B_max.npy")
        if len(np.asarray(self.C_J)) == 0: # avoid loading multiple times
            self.C_J = np.load(self.reduced_matrices_folder + "C_J.npy")
            self.N = len(self.C_J)
        if len(np.asarray(self.complement_C_J)) == 0: # avoid loading multiple times
            self.complement_C_J = np.load(self.reduced_matrices_folder + "complement_C_J.npy")
        if len(np.asarray(self.alpha_J)) == 0: # avoid loading multiple times
            self.alpha_J = np.load(self.reduced_matrices_folder + "alpha_J.npy")
        if len(np.asarray(self.alpha_LB_on_xi_train)) == 0: # avoid loading multiple times
            self.alpha_LB_on_xi_train = np.load(self.reduced_matrices_folder + "alpha_LB_on_xi_train.npy")
        if len(np.asarray(self.UB_vectors_J)) == 0: # avoid loading multiple times
            self.UB_vectors_J = np.load(self.reduced_matrices_folder + "UB_vectors_J.npy")
        if not self.M_e: # avoid loading multiple times
            self.M_e = np.load(self.reduced_matrices_folder + "M_e.npy")
            self.M_e = int(self.M_e)
        if not self.M_p: # avoid loading multiple times
            self.M_p = np.load(self.reduced_matrices_folder + "M_p.npy")
            self.M_p = int(self.M_p)
        if len(np.asarray(self.xi_train)) == 0: # avoid loading multiple times
            self.xi_train = np.load(self.reduced_matrices_folder + "xi_train.npy")
    
    ## Export solution in VTK format
    def export_solution(self, solution, filename):
        self._export_vtk(solution, filename, {"With mesh motion": True, "With preprocessing": True})
        
    #  @}
    ########################### end - I/O - end ###########################
    
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the test set
    def error_analysis(self):
        print "=============================================================="
        print "=             SCM error analysis begins                      ="
        print "=============================================================="
        print ""
        
        normalized_error = np.zeros((len(self.xi_test)))
        
        for run in range(len(self.xi_test)):
            print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SCM run = ", run, " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            
            self.setmu(self.xi_test[run])
            
            # Truth solves
            (alpha, discarded1, discarded2) = self.truth_coercivity_constant()
            
            # Reduced solves
            alpha_LB = self.get_alpha_LB(self.mu, False)
            alpha_UB = self.get_alpha_UB(self.mu)
            tol = 1.e-10
            if alpha_LB/alpha_UB < -tol:
                print "SCM warning at mu = ", self.mu , ": LB = ", alpha_LB, " < 0"
            if alpha_LB/alpha_UB > 1 + tol:
                print "SCM warning at mu = ", self.mu , ": LB = ", alpha_LB, " > UB = ", alpha_UB
            if alpha_LB/alpha > 1 + tol:
                print "SCM warning at mu = ", self.mu , ": LB = ", alpha_LB, " > exact = ", alpha
            
            normalized_error[run] = (alpha - alpha_LB)/alpha_UB
        
        # Print some statistics
        print ""
        print "min(nerr) \t\t mean(nerr) \t\t max(nerr)"
        min_normalized_error = np.min(normalized_error[:]) # it should not be negative!
        mean_normalized_error = np.mean(normalized_error[:])
        max_normalized_error = np.max(normalized_error[:])
        print str(min_normalized_error) + " \t " + str(mean_normalized_error) \
              + " \t " + str(max_normalized_error)
        
        print ""
        print "=============================================================="
        print "=             SCM error analysis ends                        ="
        print "=============================================================="
        print ""
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Set additional options for the eigensolver (bounding box minimum)
    def set_additional_eigensolver_options_for_bounding_box_minimum(self, eigensolver, qa):
        eigensolver.parameters["spectral_transform"] = "shift-and-invert"
        eigensolver.parameters["spectral_shift"] = 1.e-5
        
    ## Set additional options for the eigensolver (bounding box maximimum)
    def set_additional_eigensolver_options_for_bounding_box_maximum(self, eigensolver, qa):
        eigensolver.parameters["spectral_transform"] = "shift-and-invert"
        eigensolver.parameters["spectral_shift"] = 1.e5
        
    ## Set additional options for the eigensolver (truth_coercivity constant)
    def set_additional_eigensolver_options_for_truth_coercivity_constant(self, eigensolver):
        eigensolver.parameters["spectral_transform"] = "shift-and-invert"
        eigensolver.parameters["spectral_shift"] = 1.e-5
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
    
