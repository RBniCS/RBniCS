# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import hashlib
from rbnics.problems.base import ParametrizedProblem
from rbnics.backends import abs, assign, copy, evaluate, export, import_, max
from rbnics.backends.online import OnlineAffineExpansionStorage, OnlineFunction, OnlineLinearSolver
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import sync_setters
from rbnics.eim.utils.decorators import StoreMapFromParametrizedExpressionToProblem


# Empirical interpolation method for the interpolation of parametrized functions
@StoreMapFromParametrizedExpressionToProblem
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
        # Define additional storage for EIM:
        # Interpolation locations selected by the greedy (either a ReducedVertices or ReducedMesh)_
        self.interpolation_locations = parametrized_expression.create_interpolation_locations_container()
        # Interpolation matrix
        self.interpolation_matrix = OnlineAffineExpansionStorage(1)
        # Solution
        self._interpolation_coefficients = None  # OnlineFunction

        # $$ OFFLINE DATA STRUCTURES $$ #
        self.snapshot = parametrized_expression.create_empty_snapshot()
        # Basis functions container
        self.basis_functions = parametrized_expression.create_basis_container()
        # I/O
        self.folder["basis"] = os.path.join(self.folder_prefix, "basis")
        self.folder["cache"] = os.path.join(self.folder_prefix, "cache")
        self.folder["reduced_operators"] = os.path.join(self.folder_prefix, "reduced_operators")

        def _snapshot_cache_key_generator(*args, **kwargs):
            assert args == self.mu
            assert len(kwargs) == 0
            return self._cache_key()

        def _snapshot_cache_import(filename):
            snapshot = copy(self.snapshot)
            self.import_solution(self.folder["cache"], filename, snapshot)
            return snapshot

        def _snapshot_cache_export(filename):
            self.export_solution(self.folder["cache"], filename)

        def _snapshot_cache_filename_generator(*args, **kwargs):
            assert args == self.mu
            assert len(kwargs) == 0
            return self._cache_file()

        self._snapshot_cache = Cache(
            "EIM",
            key_generator=_snapshot_cache_key_generator,
            import_=_snapshot_cache_import,
            export=_snapshot_cache_export,
            filename_generator=_snapshot_cache_filename_generator
        )

    # Initialize data structures required for the online phase
    def init(self, current_stage="online"):
        assert current_stage in ("online", "offline")
        # Read/Initialize reduced order data structures
        if current_stage == "online":
            self.interpolation_locations.load(self.folder["reduced_operators"], "interpolation_locations")
            self.interpolation_matrix.load(self.folder["reduced_operators"], "interpolation_matrix")
            self.basis_functions.load(self.folder["basis"], "basis")
            self.N = len(self.basis_functions)
        elif current_stage == "offline":
            # Nothing to be done
            pass
        else:
            raise ValueError("Invalid stage in init().")

    def evaluate_parametrized_expression(self):
        try:
            assign(self.snapshot, self._snapshot_cache[self.mu])
        except KeyError:
            self.snapshot = evaluate(self.parametrized_expression)
            self._snapshot_cache[self.mu] = copy(self.snapshot)

    def _cache_key(self):
        return self.mu

    def _cache_file(self):
        return hashlib.sha1(str(self._cache_key()).encode("utf-8")).hexdigest()

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
            self._interpolation_coefficients = None  # OnlineFunction

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
            error = self.snapshot - self.basis_functions[:N] * self._interpolation_coefficients
        else:
            error = copy(self.snapshot)  # need a copy because it will be rescaled

        # Get the location of the maximum error
        (maximum_error, maximum_location) = max(abs(error))

        # Return
        return (error, maximum_error, maximum_location)

    def compute_maximum_interpolation_relative_error(self, N=None):
        (absolute_error, maximum_absolute_error, maximum_location) = self.compute_maximum_interpolation_error(N)
        (maximum_snapshot_value, _) = max(abs(self.snapshot))
        if maximum_snapshot_value != 0.:
            return (absolute_error / maximum_snapshot_value, maximum_absolute_error / maximum_snapshot_value,
                    maximum_location)
        else:
            if maximum_absolute_error == 0.:
                # the first two arguments are a zero expression and zero scalar
                return (absolute_error, maximum_absolute_error, maximum_location)
            else:
                # the first argument should be a NaN expression
                return (None, float("NaN"), maximum_location)

    # Export solution to file
    def export_solution(self, folder=None, filename=None, solution=None):
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "snapshot"
        if solution is None:
            solution = self.snapshot
        export(solution, folder, filename)

    # Import solution from file
    def import_solution(self, folder=None, filename=None, solution=None):
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "snapshot"
        if solution is None:
            solution = self.snapshot
        import_(solution, folder, filename)
