# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import hashlib
from numpy import isclose
from rbnics.problems.base import ParametrizedProblem
from rbnics.backends import AffineExpansionStorage, assign, copy, EigenSolver, export, Function, import_, product, sum
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import sync_setters


class ParametrizedStabilityFactorEigenProblem(ParametrizedProblem):

    # Default initialization of members
    @sync_setters("truth_problem", "set_mu", "mu")
    @sync_setters("truth_problem", "set_mu_range", "mu_range")
    def __init__(self, truth_problem, spectrum, eigensolver_parameters, folder_prefix, expansion_index=None):
        # Call the parent initialization
        ParametrizedProblem.__init__(self, folder_prefix)
        self.truth_problem = truth_problem

        # Matrices/vectors resulting from the truth discretization
        self.expansion_index = expansion_index
        self.operator = {
            "stability_factor_left_hand_matrix": None,
            # AffineExpansionStorage
            "stability_factor_right_hand_matrix": None
            # AffineExpansionStorage, even though it will contain only one matrix
        }
        self.dirichlet_bc = None  # AffineExpansionStorage
        self.spectrum = spectrum
        self.eigensolver_parameters = eigensolver_parameters

        # Solution
        self._eigenvalue = 0.
        self._eigenvector = Function(truth_problem.stability_factor_V)
        # I/O
        self.folder["cache"] = os.path.join(folder_prefix, "cache")

        def _eigenvalue_cache_key_generator(*args, **kwargs):
            return args

        def _eigenvalue_cache_import(filename):
            self.import_eigenvalue(self.folder["cache"], filename)
            return self._eigenvalue

        def _eigenvalue_cache_export(filename):
            self.export_eigenvalue(self.folder["cache"], filename)

        def _eigenvalue_cache_filename_generator(*args, **kwargs):
            return self._cache_file(args)

        self._eigenvalue_cache = Cache(
            "problems",
            key_generator=_eigenvalue_cache_key_generator,
            import_=_eigenvalue_cache_import,
            export=_eigenvalue_cache_export,
            filename_generator=_eigenvalue_cache_filename_generator
        )

        def _eigenvector_cache_key_generator(*args, **kwargs):
            return args

        def _eigenvector_cache_import(filename):
            self.import_eigenvector(self.folder["cache"], filename)
            return self._eigenvector

        def _eigenvector_cache_export(filename):
            self.export_eigenvector(self.folder["cache"], filename)

        def _eigenvector_cache_filename_generator(*args, **kwargs):
            return self._cache_file(args)

        self._eigenvector_cache = Cache(
            "problems",
            key_generator=_eigenvector_cache_key_generator,
            import_=_eigenvector_cache_import,
            export=_eigenvector_cache_export,
            filename_generator=_eigenvector_cache_filename_generator
        )

    def init(self):
        # Store the left and right hand side operators
        if self.operator["stability_factor_left_hand_matrix"] is None:
            # init was not called already
            if self.expansion_index is None:
                self.operator["stability_factor_left_hand_matrix"] = self.truth_problem.operator[
                    "stability_factor_left_hand_matrix"]
            else:
                self.operator["stability_factor_left_hand_matrix"] = AffineExpansionStorage(
                    (self.truth_problem.operator["stability_factor_left_hand_matrix"][self.expansion_index], ))
        if self.operator["stability_factor_right_hand_matrix"] is None:  # init was not called already
            self.operator["stability_factor_right_hand_matrix"] = self.truth_problem.operator[
                "stability_factor_right_hand_matrix"]
            assert len(self.operator["stability_factor_right_hand_matrix"]) == 1

        # Store Dirichlet boundary conditions
        if self.dirichlet_bc is None:  # init was not called already (or raised a trivial error)
            try:
                self.dirichlet_bc = AffineExpansionStorage(self.truth_problem.assemble_operator(
                    "stability_factor_dirichlet_bc"))
                # need to call assemble_operator because this special bc is not stored among the ones
                # in self.truth_problem.dirichlet_bc
            except ValueError:  # there were no Dirichlet BCs
                self.dirichlet_bc = None

        # Also make sure to create folder for cache
        self.folder.create()

    def solve(self):
        cache_key = self._cache_key()
        try:
            self._eigenvalue = self._eigenvalue_cache[cache_key]
            assign(self._eigenvector, self._eigenvector_cache[cache_key])
        except KeyError:
            self._solve()
            self._eigenvalue_cache[cache_key] = self._eigenvalue
            self._eigenvector_cache[cache_key] = copy(self._eigenvector)
        return (self._eigenvalue, self._eigenvector)

    def _solve(self):
        assert self.operator["stability_factor_left_hand_matrix"] is not None
        if self.expansion_index is None:
            A = sum(product(
                self.truth_problem.compute_theta("stability_factor_left_hand_matrix"),
                self.operator["stability_factor_left_hand_matrix"]))
        else:
            assert len(self.operator["stability_factor_left_hand_matrix"]) == 1
            A = self.operator["stability_factor_left_hand_matrix"][0]
        assert self.operator["stability_factor_right_hand_matrix"] is not None
        assert len(self.operator["stability_factor_right_hand_matrix"]) == 1
        B = self.operator["stability_factor_right_hand_matrix"][0]

        if self.dirichlet_bc is not None:
            dirichlet_bcs_sum = sum(product((0., ) * len(self.dirichlet_bc), self.dirichlet_bc))
            eigensolver = EigenSolver(self.truth_problem.stability_factor_V, A, B, dirichlet_bcs_sum)
        else:
            eigensolver = EigenSolver(self.truth_problem.stability_factor_V, A, B)
        eigensolver_parameters = dict()
        assert self.spectrum == "largest" or self.spectrum == "smallest"
        eigensolver_parameters["spectrum"] = self.spectrum + " real"
        eigensolver_parameters.update(self.eigensolver_parameters)
        eigensolver.set_parameters(eigensolver_parameters)
        eigensolver.solve(1)

        r, c = eigensolver.get_eigenvalue(0)  # real and complex part of the eigenvalue
        r_vector, c_vector = eigensolver.get_eigenvector(0)  # real and complex part of the eigenvectors

        assert isclose(c, 0.), "The required eigenvalue is not real"

        self._eigenvalue = r
        assign(self._eigenvector, r_vector)

    def _cache_key(self):
        if self.expansion_index is None:
            return (self.mu, self.spectrum)
        else:
            return (self.expansion_index, self.spectrum)

    def _cache_file(self, cache_key):
        return hashlib.sha1(str(cache_key).encode("utf-8")).hexdigest()

    def export_eigenvalue(self, folder=None, filename=None):
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "stability_factor"
        export([self._eigenvalue], folder, filename + "_eigenvalue")

    def export_eigenvector(self, folder=None, filename=None):
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "stability_factor"
        export(self._eigenvector, folder, filename + "_eigenvector")

    def import_eigenvalue(self, folder=None, filename=None):
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "stability_factor"
        eigenvalue_storage = [0.]
        import_(eigenvalue_storage, folder, filename + "_eigenvalue")
        assert len(eigenvalue_storage) == 1
        self._eigenvalue = eigenvalue_storage[0]

    def import_eigenvector(self, folder=None, filename=None):
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "stability_factor"
        import_(self._eigenvector, folder, filename + "_eigenvector")
