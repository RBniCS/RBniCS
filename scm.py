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

import glpk # for LB computation
import sys # for sys.float_info.max
from elliptic_coercive_rb_base import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     SCM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class SCM
#
# Successive constraint method for the approximation of the coercivity constant
class SCM(EllipticCoerciveRBBase):

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced basis object
    #  @{
    
    ## Default initialization of members
    def __init__(self, RB_problem):
    	# Call the parent initialization
        EllipticCoerciveRBBase.__init__(self, RB_problem.V)
        # Store the reduced basis object
        self.RB_problem = RB_problem
        # Define additional storage for SCM
        B_min = [] # minimum values of the bounding box mathcal{B}. Vector of size Qa
        B_max = [] # maximum values of the bounding box mathcal{B}. Vector of size Qa
        C_K = [] # vector storing the greedily select parameters during the training phase
        alpha_K = [] # vector storing the truth coercivity constants at the greedy parameters in C_K
        UB_vectors_K = [] # array of Qa-dimensional vectors storing the infimizing elements at the greedy parameters in C_K
        # Overwrite folder names
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
        lp = glpk.glp_prob()
        glpk.glp_set_obj_dir(lp, glpk.GLP_MIN);
        Qa = self.RB_problem.Qa
        N = self.N
        
        # 1. Linear program unknowns: Qa variables, y_1, ..., y_{Q_a}
        glpk.glp_add_cols(lp, Qa);
        
        # 2. Range: constrain the variables to be in the bounding box (note: GLPK indexing starts from 1)
        for qa in range(Qa):
            if B_min[qa] < B_max[qa]: # the usual case
                glpk.glp_set_col_bnds(lp, q+1, GLP_DB, B_min[q], B_max[q]);
            elif B_min[qa] == B_max[qa]: # unlikely, but possible
                glpk.glp_set_col_bnds(lp, q+1, GLP_FX, B_min[q], B_max[q]);
            else # there is something wrong in the bounding box: set as unconstrained variable
                print "Warning: wrong bounding box for affine expansion element #", qa
                glp_set_col_bnds(lp, q+1, GLP_FR, 0., 0.);
                
        # 3. Add constraints: a constraint is added for each sample in C_K
        glpk.glp_add_rows(lp, N);
        matrix_row_index = np.zeros((N, Qa))
        matrix_column_index = np.zeros((N, Qa))
        matrix_content = np.zeros((N, Qa))
        for k in range(N):
            # Overwrite parameter values
            omega = self.C_K[k]
            self.setmu(omega)
            current_theta_a = self.RB_problem.compute_theta_a()
            
            # Assemble the LHS of the constraint
            for qa in range(Qa):
                matrix_row_index[k, qa] = k + 1
                matrix_column_index[k, qa] = qa + 1
                matrix_content[k, qa] = current_theta_a[qa]
            glp_load_matrix(lp, N*Qa, matrix_row_index, matrix_column_index, matrix_content)
            
            # Assemble the RHS of the constraint
            glpk.glp_set_row_bnds(lp, k+1, GLP_LO, alpha_K[k], 0.);
        
        # 4. Add cost function coefficients
        self.setmu(mu)
        current_theta_a = self.RB_problem.compute_theta_a()
        for qa in range(Qa):
            glpk.glp_set_obj_coef(lp, qa+1, current_theta_a[qa])
        
        # 5. Solve the linear programming problem
        glpk.glp_smcp options;
        glpk.glp_init_smcp(&options);
        options.msg_lev = GLP_MSG_ERR;
        options.meth = GLP_DUAL;
        glp_simplex(lp, &parm);
        alpha_LB = glpk.glp_get_obj_val(lp);
        glpk.glp_delete_prob(lp);
        
        return alpha_LB;
    
    def get_alpha_UB(self, mu):
        Qa = self.RB_problem.Qa
        N = self.N
        UB_vectors_K = self.UB_vectors_K
        
        alpha_UB = sys.float_info.max;
        
        for k in range(N):
            # Overwrite parameter values
            omega = self.C_K[k]
            self.setmu(omega)
            current_theta_a = self.RB_problem.compute_theta_a()
            UB_vector = UB_vectors[k]
            
            # Compute the cost function for fixed omega
            obj = 0.
            for qa in range(Qa):
                obj += UB_vector[q]*current_theta_a[q]
            
            if obj < alpha_UB
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
        
        # Assemble matrices related to the LHS A of the RB problem
        self.truth_A = self.RB_problem.assemble_truth_a()
        self.theta_a = self.RB_problem.compute_theta_a()
        self.Qa = len(self.theta_A)
        
        # Compute the bounding box \mathcal{B}
        self.compute_bounding_box();
        
        for run in range(self.Nmax):
            # Store the greedy parameter
            self.store_C_k()
            
            print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SCM run = ", run, " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            
            # Evaluate the coercivity constant
            self.truth_coercivity_constant()
            
            self.N = len(self.C_K)
            
            # Prepare for next iteration
			if self.N < self.Nmax:
                print "find next mu"
                self.greedy()
		        self.theta_a = self.RB_problem.compute_theta_a()
            else:
                self.greedy()
        
        print "=============================================================="
        print "=             SCM offline phase ends                         ="
        print "=============================================================="
        print ""
        
    # Compute the bounding box \mathcal{B}
    def compute_bounding_box(self):
        # Resize the bounding box storage
        self.B_min = np.zeros((self.Qa))
        self.B_max = np.zeros((self.Qa))
        
        # RHS matrix
        S = self.RB_problem.S
        
        for qa in range(Qa):
            A = self.truth_A[qa]
            
            eigensolver = SLEPcEigenSolver(A, S)
            eigensolver.parameters["problem_type"] = "gen_hermitian"
            
            # Compute the minimum eigenvalue
            eigensolver.parameters["spectrum"] = "smallest real"
            eigensolver.solve(1)
            r, c = eigensolver.get_eigenvalue(0) # real and complex part of the eigenvalue
            self.B_min[qa] = r
            
            # Compute the maximum eigenvalue
            eigensolver.parameters["spectrum"] = "largest real"
            eigensolver.solve(1)
            r, c = eigensolver.get_eigenvalue(0) # real and complex part of the eigenvalue
            self.B_max[qa] = r
        
    # Store the greedy parameter
    def store_C_k(self):
        C_k += (self.mu,)
        
    # Evaluate the coercivity constant
    def truth_coercivity_constant(self):
        A = self.aff_assemble_truth_sym(self.truth_A, self.theta_a)
        S = self.RB_problem.S
        
        eigensolver = SLEPcEigenSolver(A, S)
        eigensolver.parameters["problem_type"] = "gen_hermitian"
		eigensolver.parameters["spectrum"] = "smallest real"
        eigensolver.solve(1)
        
        r, c, rv, cv = eigensolver.get_pair(0) # real and complex part of the (eigenvalue, eigenvectors)
        self.alpha_k += (r,)
        
        rv_f = Function(self.V, rv)
        UB_vector = compute_UB_vector(self.truth_A, S, rv_f)
        self.UB_vectors_K += (UB_vector,)
        
    ## Assemble the symmetric part of the matrix A
    def aff_assemble_truth_sym(self, vec, theta_v):
        A_ = vec[0]*theta_v[0]
        for qa in range(1,len(vec)):
            A_ += (vec[qa]+vec[qa].T)/2.*theta_v[qa]
        return A_
        
    ## Compute the ratio between a_q(u,u) and s(u,u), for all q in vec
    def compute_UB_vector(self, vec, S, u):
        UB_vector = np.zeros((len(vec)))
        norm_S_squared = compute_scalar(u,u,S)
        for qa in range(1,len(vec)):
            UB_vector[qa] = compute_scalar(u,u,vec[qa])/norm_S_squared;
        return UB_vector
        
    ## Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
        delta_max = -1.0
        for mu in self.xi_train:
            self.setmu(mu)
            LB = self.get_alpha_LB(mu) # notice that only ...
            UB = self.get_alpha_UB(mu) # ... these three line ...
            delta = (UB - LB)/UB   # ... have been changed
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
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    def get_alpha_lb(self):
        sys.exit("The method get_alpha_lb(self) should never be used in SCM!")
        
    def compute_theta_a(self):
        sys.exit("The method compute_theta_a(self) should never be used in SCM!")
    
    def compute_theta_f(self):
        sys.exit("The method compute_theta_f(self) should never be used in SCM!")
        
    def assemble_truth_a(self):
        sys.exit("The method assemble_truth_a(self) should never be used in SCM!")

    def assemble_truth_f(self):
        sys.exit("The method assemble_truth_f(self) should never be used in SCM!")
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
