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
from rbnics.problems.base import ParametrizedProblem
from rbnics.backends import abs, copy, evaluate, export, import_, max
from rbnics.backends.online import OnlineAffineExpansionStorage, OnlineFunction, OnlineLinearSolver
from rbnics.utils.config import config
from rbnics.utils.decorators import sync_setters
from rbnics.eim.utils.decorators import StoreMapFromParametrizedExpressionToEIMApproximation

# Empirical interpolation method for the interpolation of parametrized functions
@StoreMapFromParametrizedExpressionToEIMApproximation
class EIMApproximation(ParametrizedProblem):

    # Default initialization of members
    @sync_setters("truth_problem", "set_mu", "mu")
    @sync_setters("truth_problem", "set_mu_range", "mu_range")
    def __init__(self, truth_problem, parametrized_expression, folder_prefix, basis_generation):
        # Call the parent initialization
        ParametrizedProblem.__init__(self, folder_prefix)
        # Store the parametrized expression
        self.parametrized_expression = parametrized_expression
        self.truth_problem = truth_problem
        assert basis_generation in ("Greedy", "POD")
        self.basis_generation = basis_generation
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # Online reduced space dimension
        self.N = 0
        # Define additional storage for EIM
        self.interpolation_locations = parametrized_expression.create_interpolation_locations_container() # interpolation locations selected by the greedy (either a ReducedVertices or ReducedMesh)
        self.interpolation_matrix = OnlineAffineExpansionStorage(1) # interpolation matrix
        # Solution
        self._interpolation_coefficients = None # OnlineFunction
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        self.snapshot = None # will be filled in by Function, Vector or Matrix as appropriate in the EIM preprocessing
        self.snapshot_cache = dict() # of Function, Vector or Matrix
        # Basis functions container
        self.Z = parametrized_expression.create_basis_container()
        # I/O
        self.folder["basis"] = os.path.join(self.folder_prefix, "basis")
        self.folder["cache"] = os.path.join(self.folder_prefix, "cache")
        self.folder["reduced_operators"] = os.path.join(self.folder_prefix, "reduced_operators")
        self.cache_config = config.get("EIM", "cache")
        
    # Initialize data structures required for the online phase
    def init(self, current_stage="online"):
        assert current_stage in ("online", "offline")
        # Read/Initialize reduced order data structures
        if current_stage == "online":
            self.interpolation_locations.load(self.folder["reduced_operators"], "interpolation_locations")
            self.interpolation_matrix.load(self.folder["reduced_operators"], "interpolation_matrix")
            self.Z.load(self.folder["basis"], "basis")
            self.N = len(self.Z)
        elif current_stage == "offline":
            # Nothing to be done
            pass
        else:
            raise ValueError("Invalid stage in init().")

    def evaluate_parametrized_expression(self):
        (cache_key, cache_file) = self._cache_key_and_file()
        if "RAM" in self.cache_config and cache_key in self.snapshot_cache:
            self.snapshot = self.snapshot_cache[cache_key]
        elif "Disk" in self.cache_config and self.import_solution(self.folder["cache"], cache_file):
            if "RAM" in self.cache_config:
                self.snapshot_cache[cache_key] = copy(self.snapshot)
        else:
            self.snapshot = evaluate(self.parametrized_expression)
            if "RAM" in self.cache_config:
                self.snapshot_cache[cache_key] = copy(self.snapshot)
            self.export_solution(self.folder["cache"], cache_file) # Note that we export to file regardless of config options, because they may change across different runs
        
    def _cache_key_and_file(self):
        cache_key = self.mu
        cache_file = hashlib.sha1(str(cache_key).encode("utf-8")).hexdigest()
        return (cache_key, cache_file)
        
    # Perform an online solve.
    def solve(self, N=None):
        if N is None:
            N = self.N
        
        self._solve(self.parametrized_expression, N)
        return self._interpolation_coefficients
        
    def _solve(self, rhs_, N=None):
        if N is None:
            N = self.N
            
        if N > 0:
            self._interpolation_coefficients = OnlineFunction(N)
                
            # Evaluate the parametrized expression at interpolation locations
            rhs = evaluate(rhs_, self.interpolation_locations[:N])
            
            (max_abs_rhs, _) = max(abs(rhs))
            if max_abs_rhs == 0.:
                # If the rhs is zero, then we are interpolating the zero function
                # and the default zero coefficients are enough.
                pass
            else:
                # Extract the interpolation matrix
                lhs = self.interpolation_matrix[0][:N, :N]
                
                # Solve the interpolation problem
                solver = OnlineLinearSolver(lhs, self._interpolation_coefficients, rhs)
                solver.solve()
        else:
            self._interpolation_coefficients = None # OnlineFunction
        
    # Call online_solve and then convert the result of online solve from OnlineVector to a tuple
    def compute_interpolated_theta(self, N=None):
        interpolated_theta = self.solve(N)
        interpolated_theta_list = list()
        for theta in interpolated_theta:
            interpolated_theta_list.append(theta)
        if N is not None:
            # Make sure to append a 0 coefficient for each basis function
            # which has not been requested
            for n in range(N, self.N):
                interpolated_theta_list.append(0.0)
        return tuple(interpolated_theta_list)

    # Compute the interpolation error and/or its maximum location
    def compute_maximum_interpolation_error(self, N=None):
        if N is None:
            N = self.N
        
        # Compute the error (difference with the eim approximation)
        if N > 0:
            error = self.snapshot - self.Z[:N]*self._interpolation_coefficients
        else:
            error = copy(self.snapshot) # need a copy because it will be rescaled
        
        # Get the location of the maximum error
        (maximum_error, maximum_location) = max(abs(error))
        
        # Return
        return (error, maximum_error, maximum_location)

    # Export solution to file
    def export_solution(self, folder, filename, solution=None):
        if solution is None:
            solution = self.snapshot
        export(solution, folder, filename)
        
    # Import solution from file
    def import_solution(self, folder, filename, solution=None):
        if solution is None:
            if self.snapshot is None:
                self.snapshot = self.parametrized_expression.create_empty_snapshot()
            solution = self.snapshot
        return import_(solution, folder, filename)
