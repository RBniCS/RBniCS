# Copyright (C) 2015-2019 by the RBniCS authors
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
from rbnics.backends import export, import_, LinearProgramSolver
from rbnics.backends.common.linear_program_solver import Error as LinearProgramSolverError, Matrix, Vector
from rbnics.problems.base import ParametrizedProblem
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import sync_setters
from rbnics.utils.io import GreedySelectedParametersList
from rbnics.scm.utils.io import BoundingBoxSideList, UpperBoundsList
from rbnics.scm.problems.parametrized_stability_factor_eigenproblem import ParametrizedStabilityFactorEigenProblem

class SCMApproximation(ParametrizedProblem):

    # Default initialization of members
    @sync_setters("truth_problem", "set_mu", "mu")
    @sync_setters("truth_problem", "set_mu_range", "mu_range")
    def __init__(self, truth_problem, folder_prefix):
        # Call the parent initialization
        ParametrizedProblem.__init__(self, folder_prefix)
        # Store the parametrized problem object and the bc list
        self.truth_problem = truth_problem
                        
        # Define additional storage for SCM
        self.bounding_box_min = BoundingBoxSideList() # minimum values of the bounding box. Vector of size Q
        self.bounding_box_max = BoundingBoxSideList() # maximum values of the bounding box. Vector of size Q
        self.training_set = None # SCM algorithm needs the training set also in the online stage
        self.greedy_selected_parameters = GreedySelectedParametersList() # list storing the parameters selected during the training phase
        self.greedy_selected_parameters_complement = dict() # dict, over N, of list storing the complement of parameters selected during the training phase
        self.upper_bound_vectors = UpperBoundsList() # list of Q-dimensional vectors storing the infimizing elements at the greedily selected parameters
        self.N = 0
        
        # Storage for online computations
        self._stability_factor_lower_bound = 0.
        self._stability_factor_upper_bound = 0.
        
        # I/O
        self.folder["cache"] = os.path.join(self.folder_prefix, "reduced_cache")
        self.folder["reduced_operators"] = os.path.join(self.folder_prefix, "reduced_operators")
        def _stability_factor_cache_key_generator(*args, **kwargs):
            assert len(args) == 2
            assert args[0] == self.mu
            assert len(kwargs) == 0
            return self._cache_key(args[1])
        def _stability_factor_cache_filename_generator(*args, **kwargs):
            assert len(args) == 2
            assert args[0] == self.mu
            assert len(kwargs) == 0
            return self._cache_file(args[1])
        def _stability_factor_lower_bound_cache_import(filename):
            self.import_stability_factor_lower_bound(self.folder["cache"], filename)
            return self._stability_factor_lower_bound
        def _stability_factor_lower_bound_cache_export(filename):
            self.export_stability_factor_lower_bound(self.folder["cache"], filename)
        self._stability_factor_lower_bound_cache = Cache(
            "SCM",
            key_generator=_stability_factor_cache_key_generator,
            import_=_stability_factor_lower_bound_cache_import,
            export=_stability_factor_lower_bound_cache_export,
            filename_generator=_stability_factor_cache_filename_generator
        )
        def _stability_factor_upper_bound_cache_import(filename):
            self.import_stability_factor_upper_bound(self.folder["cache"], filename)
            return self._stability_factor_upper_bound
        def _stability_factor_upper_bound_cache_export(filename):
            self.export_stability_factor_upper_bound(self.folder["cache"], filename)
        self._stability_factor_upper_bound_cache = Cache(
            "SCM",
            key_generator=_stability_factor_cache_key_generator,
            import_=_stability_factor_upper_bound_cache_import,
            export=_stability_factor_upper_bound_cache_export,
            filename_generator=_stability_factor_cache_filename_generator
        )
        
        # Stability factor eigen problem
        self.stability_factor_calculator = ParametrizedStabilityFactorEigenProblem(self.truth_problem, "smallest", self.truth_problem._eigen_solver_parameters["stability_factor"], self.folder_prefix)
        
    # Initialize data structures required for the online phase
    def init(self, current_stage="online"):
        assert current_stage in ("online", "offline")
        # Init truth problem, to setup stability_factor_{left,right}_hand_matrix operators
        self.truth_problem.init()
        # Init exact stability factor computations
        self.stability_factor_calculator.init()
        # Read/Initialize reduced order data structures
        if current_stage == "online":
            self.bounding_box_min.load(self.folder["reduced_operators"], "bounding_box_min")
            self.bounding_box_max.load(self.folder["reduced_operators"], "bounding_box_max")
            self.training_set.load(self.folder["reduced_operators"], "training_set")
            self.greedy_selected_parameters.load(self.folder["reduced_operators"], "greedy_selected_parameters")
            self.upper_bound_vectors.load(self.folder["reduced_operators"], "upper_bound_vectors")
            # Set the value of N
            self.N = len(self.greedy_selected_parameters)
        elif current_stage == "offline":
            # Properly resize structures related to operator
            Q = self.truth_problem.Q["stability_factor_left_hand_matrix"]
            self.bounding_box_min = BoundingBoxSideList(Q)
            self.bounding_box_max = BoundingBoxSideList(Q)
            # Save the training set, which was passed by the reduction method,
            # in order to use it online
            assert self.training_set is not None
            self.training_set.save(self.folder["reduced_operators"], "training_set")
            # Properly initialize structures related to greedy selected parameters
            assert len(self.greedy_selected_parameters) == 0
        else:
            raise ValueError("Invalid stage in init().")
    
    def evaluate_stability_factor(self):
        return self.stability_factor_calculator.solve()
    
    # Get a lower bound for the stability factor
    def get_stability_factor_lower_bound(self, N=None):
        if N is None:
            N = self.N
        try:
            self._stability_factor_lower_bound = self._stability_factor_lower_bound_cache[self.mu, N]
        except KeyError:
            self._get_stability_factor_lower_bound(N)
            self._stability_factor_lower_bound_cache[self.mu, N] = self._stability_factor_lower_bound
        return self._stability_factor_lower_bound
        
    def _get_stability_factor_lower_bound(self, N):
        assert N <= len(self.greedy_selected_parameters)
        Q = self.truth_problem.Q["stability_factor_left_hand_matrix"]
        M_e = N
        M_p = min(N, len(self.training_set) - len(self.greedy_selected_parameters))
        
        # 1. Constrain the Q variables to be in the bounding box
        bounds = list() # of Q pairs
        for q in range(Q):
            assert self.bounding_box_min[q] <= self.bounding_box_max[q]
            bounds.append((self.bounding_box_min[q], self.bounding_box_max[q]))
            
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
            current_theta = self.truth_problem.compute_theta("stability_factor_left_hand_matrix")
            
            # Assemble the LHS of the constraint
            for q in range(Q):
                constraints_matrix[j, q] = current_theta[q]
            
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
            current_theta = self.truth_problem.compute_theta("stability_factor_left_hand_matrix")
            
            # Assemble the LHS of the constraint
            for q in range(Q):
                constraints_matrix[M_e + j, q] = current_theta[q]
                
            # Assemble the RHS of the constraint
            if N > 1:
                constraints_vector[M_e + j] = self.get_stability_factor_lower_bound(N - 1) # note that computations for this call may be already cached
            else:
                constraints_vector[M_e + j] = 0.
        self.set_mu(mu_bak)
        
        # 2c. Add constraints: also constrain the stability factor for mu to be positive
        # Compute theta
        current_theta = self.truth_problem.compute_theta("stability_factor_left_hand_matrix")
        
        # Assemble the LHS of the constraint
        for q in range(Q):
            constraints_matrix[M_e + M_p, q] = current_theta[q]
            
        # Assemble the RHS of the constraint
        constraints_vector[M_e + M_p] = 0.
        
        # 3. Add cost function coefficients
        cost = Vector(Q)
        for q in range(Q):
            cost[q] = current_theta[q]
        
        # 4. Solve the linear programming problem
        linear_program = LinearProgramSolver(cost, constraints_matrix, constraints_vector, bounds)
        try:
            stability_factor_lower_bound = linear_program.solve()
        except LinearProgramSolverError:
            print("SCM warning at mu = " + str(self.mu) + ": error occured while solving linear program.")
            print("Please consider switching to a different solver. A truth eigensolve will be performed.")
            
            (stability_factor_lower_bound, _) = self.evaluate_stability_factor()
        
        self._stability_factor_lower_bound = stability_factor_lower_bound
        
    # Get an upper bound for the stability factor
    def get_stability_factor_upper_bound(self, N=None):
        if N is None:
            N = self.N
        try:
            self._stability_factor_upper_bound = self._stability_factor_upper_bound_cache[self.mu, N]
        except KeyError:
            self._get_stability_factor_upper_bound(N)
            self._stability_factor_upper_bound_cache[self.mu, N] = self._stability_factor_upper_bound
        return self._stability_factor_upper_bound
        
    def _get_stability_factor_upper_bound(self, N):
        Q = self.truth_problem.Q["stability_factor_left_hand_matrix"]
        upper_bound_vectors = self.upper_bound_vectors
        
        stability_factor_upper_bound = None
        current_theta = self.truth_problem.compute_theta("stability_factor_left_hand_matrix")
        
        for j in range(N):
            upper_bound_vector = upper_bound_vectors[j]
            
            # Compute the cost function for fixed omega
            obj = 0.
            for q in range(Q):
                obj += upper_bound_vector[q]*current_theta[q]
            
            if stability_factor_upper_bound is None or obj < stability_factor_upper_bound:
                stability_factor_upper_bound = obj
        
        assert stability_factor_upper_bound is not None
        self._stability_factor_upper_bound = stability_factor_upper_bound
                    
    def _cache_key(self, N):
        return (self.mu, N)
        
    def _cache_file(self, N):
        return hashlib.sha1(str(self._cache_key(N)).encode("utf-8")).hexdigest()
        
    def _closest_selected_parameters(self, M, N, mu):
        return self.greedy_selected_parameters[:N].closest(M, mu)
        
    def _closest_unselected_parameters(self, M, N, mu):
        if N not in self.greedy_selected_parameters_complement:
            self.greedy_selected_parameters_complement[N] = self.training_set.diff(self.greedy_selected_parameters[:N])
        return self.greedy_selected_parameters_complement[N].closest(M, mu)

    def export_stability_factor_lower_bound(self, folder=None, filename=None):
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "stability_factor"
        export([self._stability_factor_lower_bound], folder, filename + "_lower_bound")
        
    def export_stability_factor_upper_bound(self, folder=None, filename=None):
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "stability_factor"
        export([self._stability_factor_upper_bound], folder, filename + "_upper_bound")
        
    def import_stability_factor_lower_bound(self, folder=None, filename=None):
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "stability_factor"
        stability_factor_lower_bound_storage = [0.]
        import_(stability_factor_lower_bound_storage, folder, filename + "_lower_bound")
        assert len(stability_factor_lower_bound_storage) == 1
        self._stability_factor_lower_bound = stability_factor_lower_bound_storage[0]
        
    def import_stability_factor_upper_bound(self, folder=None, filename=None):
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "stability_factor"
        stability_factor_upper_bound_storage = [0.]
        import_(stability_factor_upper_bound_storage, folder, filename + "_upper_bound")
        assert len(stability_factor_upper_bound_storage) == 1
        self._stability_factor_upper_bound = stability_factor_upper_bound_storage[0]
