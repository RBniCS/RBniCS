# Copyright (C) 2015 SISSA mathLab
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
        # Store the parametrized problem object
        self.parametrized_problem = parametrized_problem
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # Define additional storage for SCM
        self.B_min = [] # minimum values of the bounding box mathcal{B}. Vector of size Qa
        self.B_max = [] # maximum values of the bounding box mathcal{B}. Vector of size Qa
        self.C_J = () # vector storing the greedily select parameters during the training phase
        self.alpha_J =() # vector storing the truth coercivity constants at the greedy parameters in C_J
        self.UB_vectors_J = () # array of Qa-dimensional vectors storing the infimizing elements at the greedy parameters in C_J
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 9. I/O
        self.snap_folder = "snapshots__scm/"
        self.basis_folder = "basis__scm/"
        self.dual_folder = "dual__scm/" # never used
        self.red_matrices_folder = "red_matr__scm/"
        self.pp_folder = "pp__scm/" # post processing
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
        
    def get_alpha_LB(self, mu):
        lp = glpk.glp_create_prob()
        glpk.glp_set_obj_dir(lp, glpk.GLP_MIN);
        Qa = self.parametrized_problem.Qa
        N = self.N
        
        # 1. Linear program unknowns: Qa variables, y_1, ..., y_{Q_a}
        glpk.glp_add_cols(lp, Qa);
        
        # 2. Range: constrain the variables to be in the bounding box (note: GLPK indexing starts from 1)
        for qa in range(Qa):
            if self.B_min[qa] < self.B_max[qa]: # the usual case
                glpk.glp_set_col_bnds(lp, qa+1, glpk.GLP_DB, self.B_min[qa], self.B_max[qa]);
            elif self.B_min[qa] == self.B_max[qa]: # unlikely, but possible
                glpk.glp_set_col_bnds(lp, qa+1, glpk.GLP_FX, self.B_min[qa], self.B_max[qa]);
            else: # there is something wrong in the bounding box: set as unconstrained variable
                print "Warning: wrong bounding box for affine expansion element #", qa
                glp_set_col_bnds(lp, qa+1, glpk.GLP_FR, 0., 0.);
                
        # 3. Add constraints: a constraint is added for each sample in C_J
        glpk.glp_add_rows(lp, N);
        print "N*Qa", N*Qa
        matrix_row_index = glpk.intArray(N*Qa)
        matrix_column_index = glpk.intArray(N*Qa)
        matrix_content = glpk.doubleArray(N*Qa)
        glpk_container_index = 1 # glpk starts from 1
        for j in range(N):
            # Overwrite parameter values
            omega = self.C_J[j]
            self.parametrized_problem.setmu(omega)
            current_theta_a = self.parametrized_problem.compute_theta_a()
            
            # Assemble the LHS of the constraint
            for qa in range(Qa):
                matrix_row_index[glpk_container_index] = j + 1
                matrix_column_index[glpk_container_index] = qa + 1
                print matrix_row_index[glpk_container_index]
                print matrix_column_index[glpk_container_index]
                matrix_content[glpk_container_index] = current_theta_a[qa]
                glpk_container_index += 1
            glpk.glp_load_matrix(lp, N*Qa, matrix_row_index, matrix_column_index, matrix_content)
            
            # Assemble the RHS of the constraint
            glpk.glp_set_row_bnds(lp, j+1, glpk.GLP_LO, self.alpha_J[j], 0.);
            
        # 4. Add cost function coefficients
        self.parametrized_problem.setmu(mu)
        current_theta_a = self.parametrized_problem.compute_theta_a()
        for qa in range(Qa):
            glpk.glp_set_obj_coef(lp, qa+1, current_theta_a[qa])
        
        # 5. Solve the linear programming problem
        options = glpk.glp_smcp()
        glpk.glp_init_smcp(options)
        options.msg_lev = glpk.GLP_MSG_ERR
        options.meth = glpk.GLP_DUAL
        glpk.glp_simplex(lp, options)
        alpha_LB = glpk.glp_get_obj_val(lp)
        glpk.glp_delete_prob(lp)
        
        return alpha_LB
    
    def get_alpha_UB(self, mu):
        Qa = self.parametrized_problem.Qa
        N = self.N
        UB_vectors_J = self.UB_vectors_J
        
        alpha_UB = sys.float_info.max;
        
        for j in range(N):
            # Overwrite parameter values
            omega = self.C_J[j]
            self.parametrized_problem.setmu(omega)
            current_theta_a = self.parametrized_problem.compute_theta_a()
            UB_vector = UB_vectors_J[j]
            
            # Compute the cost function for fixed omega
            obj = 0.
            for qa in range(Qa):
                obj += UB_vector[qa]*current_theta_a[qa]
            
            if obj < alpha_UB:
                alpha_UB = obj
        
        return alpha_UB
    
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
        if os.path.exists(self.pp_folder):
            shutil.rmtree(self.pp_folder)
        folders = (self.snap_folder, self.basis_folder, self.dual_folder, self.red_matrices_folder, self.pp_folder)
        for f in folders:
            if not os.path.exists(f):
                os.makedirs(f)
        
        # Assemble matrices related to the LHS A of the parametrized problem
        self.parametrized_problem.truth_A = self.parametrized_problem.assemble_truth_a()
        self.parametrized_problem.Qa = len(self.parametrized_problem.truth_A)
        
        # Compute the bounding box \mathcal{B}
        self.compute_bounding_box();
        
        for run in range(self.Nmax):
            # Store the greedy parameter
            self.update_C_J()
            
            print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SCM run = ", run, " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            
            # Evaluate the coercivity constant
            self.truth_coercivity_constant()
            
            self.N = len(self.C_J)
            
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
        
    # Compute the bounding box \mathcal{B}
    def compute_bounding_box(self):
        # Resize the bounding box storage
        Qa = self.parametrized_problem.Qa
        self.B_min = np.zeros((Qa))
        self.B_max = np.zeros((Qa))
        
        # RHS matrix
        S = self.parametrized_problem.S
        S = as_backend_type(S)
        
        for qa in range(Qa):
            A = self.parametrized_problem.truth_A[qa]
            A = as_backend_type(A)
            
            eigensolver = SLEPcEigenSolver(A, S)
            eigensolver.parameters["problem_type"] = "gen_hermitian"
            eigensolver.parameters["solver"] = "lapack"
            eigensolver.parameters["spectral_transform"] = "shift-and-invert"
            eigensolver.parameters["spectral_shift"] = 1e-5
            
            # Compute the minimum eigenvalue
            eigensolver.parameters["spectrum"] = "smallest real"
            eigensolver.solve(1)
            r, c = eigensolver.get_eigenvalue(0) # real and complex part of the eigenvalue
            self.B_min[qa] = r
            print "B_min[" + str(qa) + "] = " + str(r)
            
            # Compute the maximum eigenvalue
            eigensolver.parameters["spectrum"] = "largest real"
            eigensolver.solve(1)
            r, c = eigensolver.get_eigenvalue(0) # real and complex part of the eigenvalue
            self.B_max[qa] = r
            print "B_max[" + str(qa) + "] = " + str(r)
        
    # Store the greedy parameter
    def update_C_J(self):
        self.C_J += (self.mu,)
        
    # Evaluate the coercivity constant
    def truth_coercivity_constant(self):
        current_theta_a = self.parametrized_problem.compute_theta_a()
        A = self.parametrized_problem.aff_assemble_truth_sym_matrix(self.parametrized_problem.truth_A, current_theta_a)
        A = as_backend_type(A)
        S = self.parametrized_problem.S
        S = as_backend_type(S)
        
        eigensolver = SLEPcEigenSolver(A, S)
        eigensolver.parameters["problem_type"] = "gen_hermitian"
        eigensolver.parameters["spectrum"] = "smallest real"
        eigensolver.parameters["solver"] = "lapack"
        eigensolver.parameters["spectral_transform"] = "shift-and-invert"
        eigensolver.parameters["spectral_shift"] = 1e-5
        eigensolver.solve(1)
        
        r, c, rv, cv = eigensolver.get_eigenpair(0) # real and complex part of the (eigenvalue, eigenvectors)
        self.alpha_J += (r,)
        print "truth_alpha =" + str(r)
        
        rv_f = Function(self.parametrized_problem.V, rv)
        UB_vector = self.compute_UB_vector(self.parametrized_problem.truth_A, S, rv_f)
        self.UB_vectors_J += (UB_vector,)
        
    ## Compute the ratio between a_q(u,u) and s(u,u), for all q in vec
    def compute_UB_vector(self, vec, S, u):
        UB_vector = np.zeros((len(vec)))
        norm_S_squared = self.parametrized_problem.compute_scalar(u,u,S)
        for qa in range(1,len(vec)):
            UB_vector[qa] = self.parametrized_problem.compute_scalar(u,u,vec[qa])/norm_S_squared;
        return UB_vector
        
    ## Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
        delta_max = -1.0
        for mu in self.xi_train:       # similar to the one in ...
            self.setmu(mu)             # ... elliptic_coercive_rb: ...
            LB = self.get_alpha_LB(mu) # ... notice that only ...  [1]
            UB = self.get_alpha_UB(mu) # ... these three lines ... [2]
            delta = (UB - LB)/UB       # ... have been changed     [3]
            if delta > delta_max:
                delta_max = delta
                munew = mu
        print "absolute SCM delta max = ", delta_max
        if os.path.isfile(self.pp_folder + "delta_max.npy") == True:
            d = np.load(self.pp_folder + "delta_max.npy")
            
            np.save(self.pp_folder + "delta_max", np.append(d, delta_max))
    
            m = np.load(self.pp_folder + "mu_greedy.npy")
            np.save(self.pp_folder + "mu_greedy", np.append(m, munew))
        else:
            np.save(self.pp_folder + "delta_max", delta_max)
            np.save(self.pp_folder + "mu_greedy", np.array(munew))

        self.setmu(munew)
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
