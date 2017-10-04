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

import os
import hashlib
import operator # to find closest parameters
from math import sqrt
from rbnics.backends import export, import_, LinearProgramSolver, sum
from rbnics.backends.common.linear_program_solver import Error as LinearProgramSolverError, Matrix, Vector
from rbnics.problems.base import ParametrizedProblem
from rbnics.utils.config import config
from rbnics.utils.decorators import sync_setters
from rbnics.utils.io import GreedySelectedParametersList
from rbnics.utils.mpi import log, PROGRESS
from rbnics.scm.utils.io import BoundingBoxSideList, UpperBoundsList
from rbnics.scm.problems.parametrized_coercivity_constant_eigenproblem import ParametrizedCoercivityConstantEigenProblem

# Successive constraint method for the approximation of the coercivity constant
class SCMApproximation(ParametrizedProblem):

    # Default initialization of members
    @sync_setters("truth_problem", "set_mu", "mu")
    @sync_setters("truth_problem", "set_mu_range", "mu_range")
    def __init__(self, truth_problem, folder_prefix, **kwargs):
        # Call the parent initialization
        ParametrizedProblem.__init__(self, folder_prefix)
        # Store the parametrized problem object and the bc list
        self.truth_problem = truth_problem
                        
        # Define additional storage for SCM
        self.B_min = BoundingBoxSideList() # minimum values of the bounding box mathcal{B}. Vector of size Q
        self.B_max = BoundingBoxSideList() # maximum values of the bounding box mathcal{B}. Vector of size Q
        self.training_set = None # SCM algorithm needs the training set also in the online stage
        self.greedy_selected_parameters = GreedySelectedParametersList() # list storing the parameters selected during the training phase
        self.greedy_selected_parameters_complement = dict() # dict, over N, of list storing the complement of parameters selected during the training phase
        self.UB_vectors = UpperBoundsList() # list of Q-dimensional vectors storing the infimizing elements at the greedily selected parameters
        self.N = 0
        self.M_e = kwargs["M_e"] # integer denoting the number of constraints based on the exact eigenvalues, or None
        self.M_p = kwargs["M_p"] # integer denoting the number of constraints based on the previous lower bounds, or None
        
        # I/O
        self.folder["cache"] = os.path.join(self.folder_prefix, "reduced_cache")
        self.cache_config = config.get("SCM", "cache")
        self.folder["reduced_operators"] = os.path.join(self.folder_prefix, "reduced_operators")
        
        # Coercivity constant eigen problem
        self.exact_coercivity_constant_calculator = ParametrizedCoercivityConstantEigenProblem(truth_problem, "a", True, "smallest", kwargs["coercivity_eigensolver_parameters"], self.folder_prefix)
        
        # Store here input parameters provided by the user that are needed by the reduction method
        self._input_storage_for_SCM_reduction = dict()
        self._input_storage_for_SCM_reduction["bounding_box_minimum_eigensolver_parameters"] = kwargs["bounding_box_minimum_eigensolver_parameters"]
        self._input_storage_for_SCM_reduction["bounding_box_maximum_eigensolver_parameters"] = kwargs["bounding_box_maximum_eigensolver_parameters"]
        
        # Avoid useless linear programming solves
        self._alpha_LB = 0.
        self._alpha_LB_cache = dict()
        self._alpha_UB = 0.
        self._alpha_UB_cache = dict()
    
    # Initialize data structures required for the online phase
    def init(self, current_stage="online"):
        assert current_stage in ("online", "offline")
        # Read/Initialize reduced order data structures
        if current_stage == "online":
            self.B_min.load(self.folder["reduced_operators"], "B_min")
            self.B_max.load(self.folder["reduced_operators"], "B_max")
            self.training_set.load(self.folder["reduced_operators"], "training_set")
            self.greedy_selected_parameters.load(self.folder["reduced_operators"], "greedy_selected_parameters")
            self.UB_vectors.load(self.folder["reduced_operators"], "UB_vectors")
            # Set the value of N
            self.N = len(self.greedy_selected_parameters)
        elif current_stage == "offline":
            self.truth_problem.init()
            # Properly resize structures related to operator
            Q = self.truth_problem.Q["a"]
            self.B_min = BoundingBoxSideList(Q)
            self.B_max = BoundingBoxSideList(Q)
            # Save the training set, which was passed by the reduction method,
            # in order to use it online
            assert self.training_set is not None
            self.training_set.save(self.folder["reduced_operators"], "training_set")
            # Properly initialize structures related to greedy selected parameters
            assert len(self.greedy_selected_parameters) is 0
            # Init exact coercivity constant computations
            self.exact_coercivity_constant_calculator.init()
        else:
            raise ValueError("Invalid stage in init().")
    
    def evaluate_stability_factor(self):
        return self.exact_coercivity_constant_calculator.solve()
    
    # Get a lower bound for alpha
    def get_stability_factor_lower_bound(self, N=None):
        if N is None:
            N = self.N
        assert N <= len(self.greedy_selected_parameters)
        (cache_key, cache_file) = self._cache_key_and_file(N)
        if "RAM" in self.cache_config and cache_key in self._alpha_LB_cache:
            log(PROGRESS, "Loading stability factor lower bound from cache")
            self._alpha_LB = self._alpha_LB_cache[cache_key]
        elif "Disk" in self.cache_config and self.import_stability_factor_lower_bound(self.folder["cache"], cache_file):
            log(PROGRESS, "Loading stability factor lower bound from file")
            if "RAM" in self.cache_config:
                self._alpha_LB_cache[cache_key] = self._alpha_LB
        else:
            log(PROGRESS, "Solving stability factor lower bound reduced problem")
            Q = self.truth_problem.Q["a"]
            M_e = min(self.M_e if self.M_e is not None else N, N, len(self.greedy_selected_parameters))
            M_p = min(self.M_p if self.M_p is not None else N, N, len(self.training_set) - len(self.greedy_selected_parameters))
            
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
            
            # 2a. Add constraints: a constraint is added for the closest samples to mu among the selected parameters
            mu_bak = self.mu
            closest_selected_parameters = self._closest_selected_parameters(M_e, N, self.mu)
            for (j, omega) in enumerate(closest_selected_parameters):
                # Overwrite parameter values
                self.set_mu(omega)
                
                # Compute theta
                current_theta_a = self.truth_problem.compute_theta("a")
                
                # Assemble the LHS of the constraint
                for q in range(Q):
                    constraints_matrix[j, q] = current_theta_a[q]
                
                # Assemble the RHS of the constraint
                (constraints_vector[j], _) = self.evaluate_stability_factor() # note that computations for this call may be already cached
            self.set_mu(mu_bak)
            
            # 2b. Add constraints: also constrain the closest point in the complement of selected parameters,
            #                      with RHS depending on previously computed lower bounds
            mu_bak = self.mu
            closest_selected_parameters_complement = self._closest_unselected_parameters(M_p, N, self.mu)
            for (j, nu) in enumerate(closest_selected_parameters_complement):
                # Overwrite parameter values
                self.set_mu(nu)
                
                # Compute theta
                current_theta_a = self.truth_problem.compute_theta("a")
                
                # Assemble the LHS of the constraint
                for q in range(Q):
                    constraints_matrix[M_e + j, q] = current_theta_a[q]
                    
                # Assemble the RHS of the constraint
                if N > 1:
                    constraints_vector[M_e + j] = self.get_stability_factor_lower_bound(N - 1) # note that computations for this call may be already cached
                else:
                    constraints_vector[M_e + j] = 0.
            self.set_mu(mu_bak)
            
            # 2c. Add constraints: also constrain the coercivity constant for mu to be positive
            # Compute theta
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
                print("SCM warning at mu = " + str(self.mu) + ": error occured while solving linear program.")
                print("Please consider switching to a different solver. A truth eigensolve will be performed.")
                
                (alpha_LB, _) = self.evaluate_stability_factor()
            
            self._alpha_LB = alpha_LB
            if "RAM" in self.cache_config:
                self._alpha_LB_cache[cache_key] = alpha_LB
            self.export_stability_factor_lower_bound(self.folder["cache"], cache_file) # Note that we export to file regardless of config options, because they may change across different runs
        return self._alpha_LB

    # Get an upper bound for alpha
    def get_stability_factor_upper_bound(self, N=None):
        if N is None:
            N = self.N
        (cache_key, cache_file) = self._cache_key_and_file(N)
        if "RAM" in self.cache_config and cache_key in self._alpha_UB_cache:
            log(PROGRESS, "Loading stability factor upper bound from cache")
            self._alpha_UB = self._alpha_UB_cache[cache_key]
        elif "Disk" in self.cache_config and self.import_stability_factor_upper_bound(self.folder["cache"], cache_file):
            log(PROGRESS, "Loading stability factor upper bound from file")
            if "RAM" in self.cache_config:
                self._alpha_UB_cache[cache_key] = self._alpha_UB
        else:
            log(PROGRESS, "Solving stability factor upper bound reduced problem")
            Q = self.truth_problem.Q["a"]
            UB_vectors = self.UB_vectors
            
            alpha_UB = None
            current_theta_a = self.truth_problem.compute_theta("a")
            
            for j in range(N):
                UB_vector = UB_vectors[j]
                
                # Compute the cost function for fixed omega
                obj = 0.
                for q in range(Q):
                    obj += UB_vector[q]*current_theta_a[q]
                
                if alpha_UB is None or obj < alpha_UB:
                    alpha_UB = obj
            
            assert alpha_UB is not None
            alpha_UB = float(alpha_UB)
            self._alpha_UB = alpha_UB
            if "RAM" in self.cache_config:
                self._alpha_UB_cache[cache_key] = alpha_UB
            self.export_stability_factor_upper_bound(self.folder["cache"], cache_file) # Note that we export to file regardless of config options, because they may change across different runs
        return self._alpha_UB
            
    def _cache_key_and_file(self, N):
        cache_key = (self.mu, N)
        cache_file = hashlib.sha1(str(cache_key).encode("utf-8")).hexdigest()
        return (cache_key, cache_file)

    # Auxiliary function: M parameters in the set xi closest to mu
    @staticmethod
    def _closest_parameters(M, xi, mu):
        assert M <= len(xi)
        
        # Trivial case 1:
        if M == 0:
            return list()
        
        # Trivial case 2:
        if M == len(xi):
            return xi
        
        parameters_and_distances = list()
        for xi_i in xi:
            distance = sqrt(sum([(x - y)**2 for (x, y) in zip(mu, xi_i)]))
            parameters_and_distances.append((xi_i, distance))
        parameters_and_distances.sort(key=operator.itemgetter(1))
        return [xi_i for (xi_i, _) in parameters_and_distances[:M]]
        
    def _closest_selected_parameters(self, M, N, mu):
        return self._closest_parameters(M, self.greedy_selected_parameters[:N], mu)
        
    def _closest_unselected_parameters(self, M, N, mu):
        if N not in self.greedy_selected_parameters_complement:
            self.greedy_selected_parameters_complement[N] = self.training_set.diff(self.greedy_selected_parameters[:N])
        return self._closest_parameters(M, self.greedy_selected_parameters_complement[N], mu)

    def export_stability_factor_lower_bound(self, folder, filename):
        export([self._alpha_LB], folder, filename + "_LB")
        
    def export_stability_factor_upper_bound(self, folder, filename):
        export([self._alpha_UB], folder, filename + "_UB")
        
    def import_stability_factor_lower_bound(self, folder, filename):
        eigenvalue_storage = [0.]
        import_successful = import_(eigenvalue_storage, folder, filename + "_LB")
        if import_successful:
            assert len(eigenvalue_storage) == 1
            self._alpha_LB = eigenvalue_storage[0]
        return import_successful
        
    def import_stability_factor_upper_bound(self, folder, filename):
        eigenvalue_storage = [0.]
        import_successful = import_(eigenvalue_storage, folder, filename + "_UB")
        if import_successful:
            assert len(eigenvalue_storage) == 1
            self._alpha_UB = eigenvalue_storage[0]
        return import_successful
