# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import hashlib
from rbnics.problems.base import LinearProblem, ParametrizedDifferentialProblem
from rbnics.backends import assign, copy, Function, LinearSolver, product, sum
from rbnics.utils.cache import Cache

StokesProblem_Base = LinearProblem(ParametrizedDifferentialProblem)


# Base class containing the definition of saddle point problems
class StokesProblem(StokesProblem_Base):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call to parent
        StokesProblem_Base.__init__(self, V, **kwargs)

        # Form names for saddle point problems
        self.terms = [
            "a", "b", "bt", "f", "g",
            # Auxiliary terms for supremizer enrichment
            "bt_restricted"
        ]
        self.terms_order = {
            "a": 2, "b": 2, "bt": 2, "f": 1, "g": 1,
            # Auxiliary terms for supremizer enrichment
            "bt_restricted": 2
        }
        self.components = ["u", "s", "p"]

        # Auxiliary storage for supremizer enrichment, using a subspace of V
        self._supremizer = Function(V, "s")

        # I/O
        def _supremizer_cache_key_generator(*args, **kwargs):
            assert len(args) == 1
            assert args[0] == self.mu
            return self._supremizer_cache_key_from_kwargs(**kwargs)

        def _supremizer_cache_import(filename):
            supremizer = copy(self._supremizer)
            self.import_supremizer(self.folder["cache"], filename, supremizer)
            return supremizer

        def _supremizer_cache_export(filename):
            self.export_supremizer(self.folder["cache"], filename)

        def _supremizer_cache_filename_generator(*args, **kwargs):
            assert len(args) == 1
            assert args[0] == self.mu
            return self._supremizer_cache_file_from_kwargs(**kwargs)

        self._supremizer_cache = Cache(
            "problems",
            key_generator=_supremizer_cache_key_generator,
            import_=_supremizer_cache_import,
            export=_supremizer_cache_export,
            filename_generator=_supremizer_cache_filename_generator
        )

    class ProblemSolver(StokesProblem_Base.ProblemSolver):
        def matrix_eval(self):
            problem = self.problem
            assembled_operator = dict()
            for term in ("a", "b", "bt"):
                assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term]))
            return assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"]

        def vector_eval(self):
            problem = self.problem
            assembled_operator = dict()
            for term in ("f", "g"):
                assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term]))
            return assembled_operator["f"] + assembled_operator["g"]

        # Custom combination of boundary conditions *not* to add BCs of supremizers
        def bc_eval(self):
            problem = self.problem
            # Temporarily change problem.components
            components_bak = problem.components
            problem.components = ["u", "p"]
            # Call Parent
            bcs = StokesProblem_Base.ProblemSolver.bc_eval(self)
            # Restore and return
            problem.components = components_bak
            return bcs

    def solve_supremizer(self, solution):
        kwargs = self._latest_solve_kwargs
        try:
            # **kwargs is not supported by __getitem__
            assign(self._supremizer, self._supremizer_cache[self.mu, kwargs])
        except KeyError:
            self._solve_supremizer(solution)
            self._supremizer_cache[self.mu, kwargs] = copy(self._supremizer)
        return self._supremizer

    def _solve_supremizer(self, solution):
        assert len(self.inner_product["s"]) == 1  # the affine expansion storage contains only the inner product matrix
        assembled_operator_lhs = self.inner_product["s"][0]
        assembled_operator_bt = sum(product(self.compute_theta("bt_restricted"), self.operator["bt_restricted"]))
        assembled_operator_rhs = assembled_operator_bt * solution
        if self.dirichlet_bc["s"] is not None:
            assembled_dirichlet_bc = sum(product(self.compute_theta("dirichlet_bc_s"), self.dirichlet_bc["s"]))
        else:
            assembled_dirichlet_bc = None
        solver = LinearSolver(
            assembled_operator_lhs,
            self._supremizer,
            assembled_operator_rhs,
            assembled_dirichlet_bc
        )
        solver.set_parameters(self._linear_solver_parameters)
        solver.solve()

    def _supremizer_cache_key_from_kwargs(self, **kwargs):
        return self._cache_key_from_kwargs(**kwargs)

    def _supremizer_cache_file_from_kwargs(self, **kwargs):
        return hashlib.sha1(str(self._supremizer_cache_key_from_kwargs(**kwargs)).encode("utf-8")).hexdigest()

    def export_supremizer(self, folder=None, filename=None, supremizer=None, component=None, suffix=None):
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "supremizer"
        if supremizer is None:
            supremizer = self._supremizer
        assert component is None or isinstance(component, str)
        if component is None:
            component = "s"
        self.export_solution(folder, filename, supremizer, component=component, suffix=suffix)

    def import_supremizer(self, folder=None, filename=None, supremizer=None, component=None, suffix=None):
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "supremizer"
        if supremizer is None:
            supremizer = self._supremizer
        assert component is None or isinstance(component, str)
        if component is None:
            component = "s"
        self.import_solution(folder, filename, supremizer, component=component, suffix=suffix)

    # Export solution to file
    def export_solution(self, folder=None, filename=None, solution=None, component=None, suffix=None):
        if component is None:
            component = ["u", "p"]  # but not "s"
        StokesProblem_Base.export_solution(
            self, folder, filename, solution=solution, component=component, suffix=suffix)

    def import_solution(self, folder=None, filename=None, solution=None, component=None, suffix=None):
        if component is None:
            component = ["u", "p"]  # but not "s"
        StokesProblem_Base.import_solution(
            self, folder, filename, solution=solution, component=component, suffix=suffix)

    # Custom combination of inner products *not* to add inner product corresponding to supremizers
    def _combine_all_inner_products(self):
        # Temporarily change self.components
        components_bak = self.components
        self.components = ["u", "p"]
        # Call Parent
        combined_inner_products = StokesProblem_Base._combine_all_inner_products(self)
        # Restore and return
        self.components = components_bak
        return combined_inner_products

    # Custom combination of inner products *not* to add projection inner product corresponding to supremizers
    def _combine_all_projection_inner_products(self):
        # Temporarily change self.components
        components_bak = self.components
        self.components = ["u", "p"]
        # Call Parent
        combined_projection_inner_products = StokesProblem_Base._combine_all_projection_inner_products(self)
        # Restore and return
        self.components = components_bak
        return combined_projection_inner_products

    # Custom combination of Dirichlet BCs *not* to add BCs corresponding to supremizers
    def _combine_and_homogenize_all_dirichlet_bcs(self):
        # Temporarily change self.components
        components_bak = self.components
        self.components = ["u", "p"]
        # Call Parent
        combined_and_homogenized_dirichlet_bcs = StokesProblem_Base._combine_and_homogenize_all_dirichlet_bcs(self)
        # Restore and return
        self.components = components_bak
        return combined_and_homogenized_dirichlet_bcs
