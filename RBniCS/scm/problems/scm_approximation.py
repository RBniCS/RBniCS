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
## @file scm.py
#  @brief Implementation of the successive constraints method for the approximation of the coercivity constant
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
import operator # to find closest parameters
from math import sqrt
from RBniCS.backends import export, LinearProgramSolver
from RBniCS.backends.common.linear_program_solver import Error as LinearProgramSolverError, Matrix, Vector
from RBniCS.problems.base import ParametrizedProblem
from RBniCS.utils.decorators import sync_setters, Extends, override
from RBniCS.utils.mpi import print
from RBniCS.scm.utils.io import BoundingBoxSideList, CoercivityConstantsList, EigenVectorsList, TrainingSetIndices, UpperBoundsList
from RBniCS.scm.problems.parametrized_coercivity_constant_eigenproblem import ParametrizedCoercivityConstantEigenProblem

#~~~~~~~~~~~~~~~~~~~~~~~~~     SCM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class SCM
#
# Successive constraint method for the approximation of the coercivity constant
@Extends(ParametrizedProblem)
class SCMApproximation(ParametrizedProblem):

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the SCM object
    #  @{

    ## Default initialization of members
    @override
    @sync_setters("truth_problem", "set_mu", "mu")
    @sync_setters("truth_problem", "set_mu_range", "mu_range")
    def __init__(self, truth_problem, folder_prefix, **kwargs):
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
        self.alpha_LB_on_training_set = CoercivityConstantsList() # list storing the approximation of the coercivity constant on the complement of C_J (at the previous iteration, during the offline phase)
        self.eigenvector_J = EigenVectorsList() # list of eigenvectors corresponding to the truth coercivity constants at the greedy parameters in C_J
        self.UB_vectors_J = UpperBoundsList() # list of Q-dimensional vectors storing the infimizing elements at the greedy parameters in C_J
        self.M_e = kwargs["M_e"] # integer denoting the number of constraints based on the exact eigenvalues. If < 0, then it is assumed to be len(C_J)
        self.M_p = kwargs["M_p"] # integer denoting the number of constraints based on the previous lower bounds. If < 0, then it is assumed to be len(C_J)
        self.training_set = None # SCM algorithms needs the training set also in the online stage, e.g. to query alpha_LB_on_training_set
        self.N = 0
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # Matrices/vectors resulting from the truth discretization
        # I/O
        self.folder["reduced_operators"] = self.folder_prefix + "/" + "reduced_operators"
        # 
        self.exact_coercivity_constant_calculator = ParametrizedCoercivityConstantEigenProblem(truth_problem, "a", True, "smallest", kwargs["coercivity_eigensolver_parameters"])
        
        # Store here input parameters provided by the user that are needed by the reduction method
        self._input_storage_for_SCM_reduction = dict()
        self._input_storage_for_SCM_reduction["bounding_box_minimum_eigensolver_parameters"] = kwargs["bounding_box_minimum_eigensolver_parameters"]
        self._input_storage_for_SCM_reduction["bounding_box_maximum_eigensolver_parameters"] = kwargs["bounding_box_maximum_eigensolver_parameters"]
        
        # Avoid useless linear programming solves
        self._get_stability_factor_lower_bound__previous_mu = None
        self._get_stability_factor_lower_bound__previous_alpha_LB = None
        self._get_stability_factor_upper_bound__previous_mu = None
        self._get_stability_factor_upper_bound__previous_alpha_UB = None
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    ## Initialize data structures required for the online phase
    def init(self, current_stage="online"):
        assert current_stage in ("online", "offline")
        # Read/Initialize reduced order data structures
        if current_stage == "online":
            self.B_min.load(self.folder["reduced_operators"], "B_min")
            self.B_max.load(self.folder["reduced_operators"], "B_max")
            self.C_J.load(self.folder["reduced_operators"], "C_J")
            self.complement_C_J.load(self.folder["reduced_operators"], "complement_C_J")
            self.alpha_J.load(self.folder["reduced_operators"], "alpha_J")
            self.alpha_LB_on_training_set.load(self.folder["reduced_operators"], "alpha_LB_on_training_set")
            self.UB_vectors_J.load(self.folder["reduced_operators"], "UB_vectors_J")
            self.training_set.load(self.folder["reduced_operators"], "training_set")
            # Set the value of N
            self.N = len(self.C_J)
        elif current_stage == "offline":
            if len(self.truth_problem.Q) == 0:
                self.truth_problem.init()
            # Properly resize structures related to operator
            Q = self.truth_problem.Q["a"]
            self.B_min = BoundingBoxSideList(Q)
            self.B_max = BoundingBoxSideList(Q)
            # Properly resize structures related to training set
            ntrain = len(self.training_set)
            self.alpha_LB_on_training_set = CoercivityConstantsList(ntrain)
            self.complement_C_J = TrainingSetIndices(ntrain)
            # Save the training set, which was passed by the reduction method,
            # in order to use it online
            self.training_set.save(self.folder["reduced_operators"], "training_set")
            # Init exact coercivity constant computations
            self.exact_coercivity_constant_calculator.init()
        else:
            raise AssertionError("Invalid stage in init().")

    ## Get a lower bound for alpha
    def get_stability_factor_lower_bound(self, mu, safeguard=True):
        if self._get_stability_factor_lower_bound__previous_mu != self.mu:
            Q = self.truth_problem.Q["a"]
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
            
            # 1. Constrain the Q variables to be in the bounding box
            bounds = list() # of Q pairs
            for q in range(Q):
                assert self.B_min[q] <= self.B_max[q]
                bounds.append((self.B_min[q], self.B_max[q]))
                
            # 2. Add three different sets of constraints.
            #    Our constrains are of the form
            #       a^T * x >= b
            constraints_matrix = Matrix(M_e + M_p + 1, Q)
            constraints_vector = Vector(M_e + M_p + 1)
            
            # 2a. Add constraints: a constraint is added for the closest samples to mu in C_J
            closest_C_J_indices = self._closest_parameters(M_e, self.C_J, mu)
            for j in range(M_e):
                # Overwrite parameter values
                omega = self.training_set[ self.C_J[ closest_C_J_indices[j] ] ]
                self.truth_problem.set_mu(omega)
                current_theta_a = self.truth_problem.compute_theta("a")
                
                # Assemble the LHS of the constraint
                for q in range(Q):
                    constraints_matrix[j, q] = current_theta_a[q]
                
                # Assemble the RHS of the constraint
                constraints_vector[j] = self.alpha_J[ closest_C_J_indices[j] ]
            closest_C_J_indices = None
            
            # 2b. Add constraints: also constrain the closest point in the complement of C_J, 
            #                      with RHS depending on previously computed lower bounds
            closest_complement_C_J_indices = self._closest_parameters(M_p, self.complement_C_J, mu)
            for j in range(M_p):
                # Overwrite parameter values
                nu = self.training_set[ self.complement_C_J[ closest_complement_C_J_indices[j] ] ]
                self.truth_problem.set_mu(nu)
                current_theta_a = self.truth_problem.compute_theta("a")
                
                # Assemble the LHS of the constraint
                for q in range(Q):
                    constraints_matrix[M_e + j, q] = current_theta_a[q]
                    
                # Assemble the RHS of the constraint
                constraints_vector[M_e + j] = self.alpha_LB_on_training_set[ self.complement_C_J[ closest_complement_C_J_indices[j] ] ]
            closest_complement_C_J_indices = None
            
            # 2c. Add constraints: also constrain the coercivity constant for mu to be positive
            # Overwrite parameter values
            self.truth_problem.set_mu(mu)
            current_theta_a = self.truth_problem.compute_theta("a")
            
            # Assemble the LHS of the constraint
            for q in range(Q):
                constraints_matrix[M_e + M_p, q] = current_theta_a[q]
                
            # Assemble the RHS of the constraint
            constraints_vector[M_e + M_p] = 0.
            
            # 3. Add cost function coefficients
            cost = Vector(Q)
            for q in range(Q):
                cost[q] = current_theta_a[q]
            
            # 4. Solve the linear programming problem
            linear_program = LinearProgramSolver(cost, constraints_matrix, constraints_vector, bounds)
            try:
                alpha_LB = linear_program.solve()
            except LinearProgramSolverError:
                print("SCM warning at mu = " + str(mu) + ": error occured while solving linear program.")
                print("Please consider switching to a different solver. A truth eigensolve will be performed.")
                
                (alpha_LB, _) = self.exact_coercivity_constant_calculator.solve()
            
            # 5. If a safeguard is requested (when called in the online stage of the RB method),
            #    we check the resulting value of alpha_LB. In order to avoid divisions by zero
            #    or taking the square root of a negative number, we allow an inefficient evaluation.
            if safeguard == True:
                from numpy import isclose
                alpha_UB = self.get_stability_factor_upper_bound(mu)
                if alpha_LB/alpha_UB < 0 and not isclose(alpha_LB/alpha_UB, 0.): # if alpha_LB/alpha_UB << 0
                    print("SCM warning at mu = " + str(mu) + ": LB = " + str(alpha_LB) + " < 0.")
                    print("Please consider a larger Nmax for SCM. Meanwhile, a truth eigensolve is performed.")
                    
                    (alpha_LB, _) = self.exact_coercivity_constant_calculator.solve()
                    
                if alpha_LB/alpha_UB > 1 and not isclose(alpha_LB/alpha_UB, 1.): # if alpha_LB/alpha_UB >> 1
                    print("SCM warning at mu = " + str(mu) + ": LB = " + str(alpha_LB) + " > UB = " + str(alpha_UB) + ".")
                    print("Please consider a larger Nmax for SCM. Meanwhile, a truth eigensolve is performed.")
                    
                    (alpha_LB, _) = self.exact_coercivity_constant_calculator.solve()
            
            self._get_stability_factor_lower_bound__previous_mu = self.mu
            self._get_stability_factor_lower_bound__previous_alpha_LB = alpha_LB
            return alpha_LB
        else:
            return self._get_stability_factor_lower_bound__previous_alpha_LB

    ## Get an upper bound for alpha
    def get_stability_factor_upper_bound(self, mu):
        if self._get_stability_factor_upper_bound__previous_mu != self.mu:
            Q = self.truth_problem.Q["a"]
            N = self.N
            UB_vectors_J = self.UB_vectors_J
            
            alpha_UB = None
            self.truth_problem.set_mu(mu)
            current_theta_a = self.truth_problem.compute_theta("a")
            
            for j in range(N):
                UB_vector = UB_vectors_J[j]
                
                # Compute the cost function for fixed omega
                obj = 0.
                for q in range(Q):
                    obj += UB_vector[q]*current_theta_a[q]
                
                if alpha_UB is None or obj < alpha_UB:
                    alpha_UB = obj
            
            assert alpha_UB is not None
            alpha_UB = float(alpha_UB)
            self._get_stability_factor_upper_bound__previous_mu = self.mu
            self._get_stability_factor_upper_bound__previous_alpha_UB = alpha_UB
            return alpha_UB
        else:
            return self._get_stability_factor_upper_bound__previous_alpha_UB
            

    ## Auxiliary function: M parameters in the set all_mu closest to mu
    def _closest_parameters(self, M, all_mu_indices, mu):
        assert M <= len(all_mu_indices)
        
        # Trivial case 1:
        if M == 0:
            return
        
        # Trivial case 2:
        if M == len(all_mu_indices):
            return range(len(all_mu_indices))
        
        indices_and_distances = list()
        for (local_index, training_set_index) in enumerate(all_mu_indices):
            distance = self._parameters_distance(mu, self.training_set[training_set_index])
            indices_and_distances.append((local_index, distance))
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
        return sqrt(distance)

    #  @}
    ########################### end - ONLINE STAGE - end ########################### 

    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{

    ## Export solution to file
    def export_solution(self, folder, filename, solution=None):
        assert solution is not None
        export(solution, folder, filename)
        
    #  @}
    ########################### end - I/O - end ###########################
    
