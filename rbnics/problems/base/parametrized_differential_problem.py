# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from abc import ABCMeta, abstractmethod
import os
import hashlib
from numbers import Number
from rbnics.problems.base.parametrized_problem import ParametrizedProblem
from rbnics.backends import AffineExpansionStorage, assign, copy, export, Function, import_, product, sum
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import (StoreMapFromProblemNameToProblem, StoreMapFromProblemToTrainingStatus,
                                     StoreMapFromSolutionToProblem)
from rbnics.utils.test import PatchInstanceMethod


@StoreMapFromProblemNameToProblem
@StoreMapFromProblemToTrainingStatus
@StoreMapFromSolutionToProblem
class ParametrizedDifferentialProblem(ParametrizedProblem, metaclass=ABCMeta):
    """
    Abstract class describing a parametrized differential problem.
    Inizialization of the solution space V, forms terms and their order, number of terms in the affine expansion Q,
    inner products and boundary conditions, truth solution.

    :param V: functional solution space.
    """

    def __init__(self, V, **kwargs):

        # Call to parent
        ParametrizedProblem.__init__(self, self.name())

        # Input arguments
        self.V = V
        # Form names and order (to be filled in by child classes)
        self.terms = list()
        self.terms_order = dict()
        self.components = list()
        # Number of terms in the affine expansion
        self.Q = dict()  # from string to integer
        # Matrices/vectors resulting from the truth discretization
        self.OperatorExpansionStorage = AffineExpansionStorage
        # operator: string to OperatorExpansionStorage
        self.operator = dict()
        # inner_product: AffineExpansionStorage (for problems with one component) or dict of AffineExpansionStorage
        # (for problem with several components), even though it will contain only one matrix
        self.inner_product = None
        self._combined_inner_product = None
        # projection_inner_product: AffineExpansionStorage (for problems with one component) or dict of
        # AffineExpansionStorage (for problem with several components), even though it will contain only one matrix
        self.projection_inner_product = None
        self._combined_projection_inner_product = None
        # dirichlet_bc: AffineExpansionStorage (for problems with one component) or dict of AffineExpansionStorage
        # (for problem with several components)
        self.dirichlet_bc = None
        # dirichlet_bc_are_homogeneous: bool (for problems with one component) or dict of bools (for problem
        # with several components)
        self.dirichlet_bc_are_homogeneous = None
        self._combined_and_homogenized_dirichlet_bc = None
        # Solution
        self._solution = Function(self.V)
        self._output = 0.
        # I/O
        self.folder["cache"] = os.path.join(self.folder_prefix, "cache")

        def _solution_cache_key_generator(*args, **kwargs):
            assert len(args) == 1
            assert args[0] == self.mu
            return self._cache_key_from_kwargs(**kwargs)

        def _solution_cache_import(filename):
            solution = copy(self._solution)
            self.import_solution(self.folder["cache"], filename, solution)
            return solution

        def _solution_cache_export(filename):
            self.export_solution(self.folder["cache"], filename)

        def _solution_cache_filename_generator(*args, **kwargs):
            assert len(args) == 1
            assert args[0] == self.mu
            return self._cache_file_from_kwargs(**kwargs)

        self._solution_cache = Cache(
            "problems",
            key_generator=_solution_cache_key_generator,
            import_=_solution_cache_import,
            export=_solution_cache_export,
            filename_generator=_solution_cache_filename_generator
        )

        def _output_cache_key_generator(*args, **kwargs):
            assert len(args) == 1
            assert args[0] == self.mu
            return self._cache_key_from_kwargs(**kwargs)

        def _output_cache_import(filename):
            output = [0.]
            self.import_output(self.folder["cache"], filename, output)
            assert len(output) == 1
            return output[0]

        def _output_cache_export(filename):
            self.export_output(self.folder["cache"], filename)

        def _output_cache_filename_generator(*args, **kwargs):
            assert len(args) == 1
            assert args[0] == self.mu
            return self._cache_file_from_kwargs(**kwargs)

        self._output_cache = Cache(
            "problems",
            key_generator=_output_cache_key_generator,
            import_=_output_cache_import,
            export=_output_cache_export,
            filename_generator=_output_cache_filename_generator
        )

    def name(self):
        return type(self).__name__

    def init(self):
        """
        Initialize data structures required during the offline phase.
        """
        self._init_operators()
        self._init_inner_products()
        self._init_dirichlet_bc()

    def _init_operators(self):
        """
        Initialize operators required for the offline phase. Internal method.
        """
        # Assemble operators
        for term in self.terms:
            if term not in self.operator:  # init was not called already
                try:
                    self.operator[term] = self.OperatorExpansionStorage(self.assemble_operator(term))
                except ValueError:  # raised by assemble_operator if output computation is optional
                    self.operator[term] = None
                    self.Q[term] = 0
                else:
                    if term not in self.Q:  # init was not called already
                        self.Q[term] = len(self.operator[term])

    def _init_inner_products(self):
        """
        Initialize inner products required for the offline phase. Internal method.
        """
        # Get helper strings depending on the number of basis components
        n_components = len(self.components)
        assert n_components > 0
        if n_components > 1:
            inner_product_string = "inner_product_{c}"
        else:
            inner_product_string = "inner_product"
        # Assemble inner products
        if self.inner_product is None:  # init was not called already
            inner_product = dict()
            for component in self.components:
                inner_product[component] = AffineExpansionStorage(
                    self.assemble_operator(inner_product_string.format(c=component)))
            if n_components == 1:
                self.inner_product = inner_product[self.components[0]]
            else:
                self.inner_product = inner_product
            assert self._combined_inner_product is None
            self._combined_inner_product = self._combine_all_inner_products()
        # Assemble inner product to be used for projection
        if self.projection_inner_product is None:  # init was not called already
            projection_inner_product = dict()
            for component in self.components:
                try:
                    projection_inner_product[component] = AffineExpansionStorage(
                        self.assemble_operator("projection_" + inner_product_string.format(c=component)))
                except ValueError:  # no projection_inner_product specified, revert to inner_product
                    projection_inner_product[component] = AffineExpansionStorage(
                        self.assemble_operator(inner_product_string.format(c=component)))
            if n_components == 1:
                self.projection_inner_product = projection_inner_product[self.components[0]]
            else:
                self.projection_inner_product = projection_inner_product
            assert self._combined_projection_inner_product is None
            self._combined_projection_inner_product = self._combine_all_projection_inner_products()

    def _combine_all_inner_products(self):
        if len(self.components) > 1:
            all_inner_products = list()
            for component in self.components:
                # the affine expansion storage contains only the inner product matrix
                assert len(self.inner_product[component]) == 1
                all_inner_products.append(self.inner_product[component][0])
            all_inner_products = tuple(all_inner_products)
        else:
            # the affine expansion storage contains only the inner product matrix
            assert len(self.inner_product) == 1
            all_inner_products = (self.inner_product[0], )
        all_inner_products = AffineExpansionStorage(all_inner_products)
        all_inner_products_thetas = (1., ) * len(all_inner_products)
        return sum(product(all_inner_products_thetas, all_inner_products))

    def _combine_all_projection_inner_products(self):
        if len(self.components) > 1:
            all_projection_inner_products = list()
            for component in self.components:
                # the affine expansion storage contains only the inner product matrix
                assert len(self.projection_inner_product[component]) == 1
                all_projection_inner_products.append(self.projection_inner_product[component][0])
            all_projection_inner_products = tuple(all_projection_inner_products)
        else:
            # the affine expansion storage contains only the inner product matrix
            assert len(self.projection_inner_product) == 1
            all_projection_inner_products = (self.projection_inner_product[0], )
        all_projection_inner_products = AffineExpansionStorage(all_projection_inner_products)
        all_projection_inner_products_thetas = (1., ) * len(all_projection_inner_products)
        return sum(product(all_projection_inner_products_thetas, all_projection_inner_products))

    def _init_dirichlet_bc(self):
        """
        Initialize boundary conditions required for the offline phase. Internal method.
        """
        # Get helper strings depending on the number of basis components
        n_components = len(self.components)
        assert n_components > 0
        if n_components > 1:
            dirichlet_bc_string = "dirichlet_bc_{c}"
        else:
            dirichlet_bc_string = "dirichlet_bc"
        # Assemble Dirichlet BCs
        # we do not assert for
        # (self.dirichlet_bc is None) == (self.dirichlet_bc_are_homogeneous is None)
        # because self.dirichlet_bc may still be None after initialization, if there
        # were no Dirichlet BCs at all and the problem had only one component
        if self.dirichlet_bc_are_homogeneous is None:  # init was not called already
            dirichlet_bc = dict()
            dirichlet_bc_are_homogeneous = dict()
            for component in self.components:
                try:
                    operator_bc = AffineExpansionStorage(
                        self.assemble_operator(dirichlet_bc_string.format(c=component)))
                except ValueError:  # there were no Dirichlet BCs
                    dirichlet_bc[component] = None
                    dirichlet_bc_are_homogeneous[component] = False
                else:
                    dirichlet_bc[component] = operator_bc
                    try:
                        self.compute_theta(dirichlet_bc_string.format(c=component))
                    except ValueError:  # there were no theta functions
                        # We provide in this case a shortcut for the case of homogeneous Dirichlet BCs,
                        # that do not require an additional lifting functions.
                        # The user needs to implement the dirichlet_bc case for assemble_operator,
                        # but not the one in compute_theta (since theta would not matter, being multiplied by zero)
                        def generate_modified_compute_theta(component, operator_bc):
                            standard_compute_theta = self.compute_theta

                            def modified_compute_theta(self_, term):
                                if term == dirichlet_bc_string.format(c=component):
                                    return (0., ) * len(operator_bc)
                                else:
                                    return standard_compute_theta(term)

                            return modified_compute_theta

                        PatchInstanceMethod(
                            self, "compute_theta", generate_modified_compute_theta(component, operator_bc)).patch()
                        dirichlet_bc_are_homogeneous[component] = True
                    else:
                        dirichlet_bc_are_homogeneous[component] = False
            if n_components == 1:
                self.dirichlet_bc = dirichlet_bc[self.components[0]]
                self.dirichlet_bc_are_homogeneous = dirichlet_bc_are_homogeneous[self.components[0]]
            else:
                self.dirichlet_bc = dirichlet_bc
                self.dirichlet_bc_are_homogeneous = dirichlet_bc_are_homogeneous
            assert self._combined_and_homogenized_dirichlet_bc is None
            self._combined_and_homogenized_dirichlet_bc = self._combine_and_homogenize_all_dirichlet_bcs()

    def _combine_and_homogenize_all_dirichlet_bcs(self):
        if len(self.components) > 1:
            all_dirichlet_bcs = list()
            for component in self.components:
                if self.dirichlet_bc[component] is not None:
                    all_dirichlet_bcs.extend(self.dirichlet_bc[component])
            if len(all_dirichlet_bcs) > 0:
                all_dirichlet_bcs = tuple(all_dirichlet_bcs)
                all_dirichlet_bcs = AffineExpansionStorage(all_dirichlet_bcs)
            else:
                all_dirichlet_bcs = None
        else:
            all_dirichlet_bcs = self.dirichlet_bc
        if all_dirichlet_bcs is not None:
            all_dirichlet_bcs_thetas = (0., ) * len(all_dirichlet_bcs)
            return sum(product(all_dirichlet_bcs_thetas, all_dirichlet_bcs))
        else:
            return None

    def solve(self, **kwargs):
        """
        Perform a truth solve in case no precomputed solution is imported.
        """
        self._latest_solve_kwargs = kwargs
        try:
            assign(self._solution, self._solution_cache[self.mu, kwargs])  # **kwargs is not supported by __getitem__
        except KeyError:
            assert not hasattr(self, "_is_solving")
            self._is_solving = True
            assign(self._solution, Function(self.V))
            self._solve(**kwargs)  # will also add to cache
            delattr(self, "_is_solving")
        return self._solution

    class ProblemSolver(object, metaclass=ABCMeta):
        def __init__(self, problem, **kwargs):
            self.problem = problem
            self.kwargs = kwargs

        def bc_eval(self):
            problem = self.problem
            if len(problem.components) > 1:
                all_dirichlet_bcs = list()
                all_dirichlet_bcs_thetas = list()
                for component in problem.components:
                    if problem.dirichlet_bc[component] is not None:
                        all_dirichlet_bcs.extend(problem.dirichlet_bc[component])
                        all_dirichlet_bcs_thetas.extend(problem.compute_theta("dirichlet_bc_" + component))
                if len(all_dirichlet_bcs) > 0:
                    all_dirichlet_bcs = tuple(all_dirichlet_bcs)
                    all_dirichlet_bcs = AffineExpansionStorage(all_dirichlet_bcs)
                    all_dirichlet_bcs_thetas = tuple(all_dirichlet_bcs_thetas)
                else:
                    all_dirichlet_bcs = None
                    all_dirichlet_bcs_thetas = None
            else:
                if problem.dirichlet_bc is not None:
                    all_dirichlet_bcs = problem.dirichlet_bc
                    all_dirichlet_bcs_thetas = problem.compute_theta("dirichlet_bc")
                else:
                    all_dirichlet_bcs = None
                    all_dirichlet_bcs_thetas = None
            assert (all_dirichlet_bcs is None) == (all_dirichlet_bcs_thetas is None)
            if all_dirichlet_bcs is not None:
                return sum(product(all_dirichlet_bcs_thetas, all_dirichlet_bcs))
            else:
                return None

        def monitor(self, solution):
            problem = self.problem
            problem._solution_cache[problem.mu, self.kwargs] = copy(solution)

        @abstractmethod
        def solve(self):
            pass

    # Perform a truth solve
    def _solve(self, **kwargs):
        problem_solver = self.ProblemSolver(self, **kwargs)
        problem_solver.solve()

    def compute_output(self):
        """

        :return: output evaluation.
        """
        kwargs = self._latest_solve_kwargs
        try:
            self._output = self._output_cache[self.mu, kwargs]  # **kwargs is not supported by __getitem__
        except KeyError:
            try:
                self._compute_output()
            except ValueError:  # raised by compute_theta if output computation is optional
                self._output = NotImplemented
            self._output_cache[self.mu, kwargs] = self._output
        return self._output

    def _compute_output(self):
        """
        Perform a truth evaluation of the output. Internal method.
        """
        self._output = NotImplemented

    def _cache_key_from_kwargs(self, **kwargs):
        for blacklist in ("components", "inner_product"):
            if blacklist in kwargs:
                del kwargs[blacklist]
        return (self.mu, tuple(sorted(kwargs.items())))

    def _cache_file_from_kwargs(self, **kwargs):
        return hashlib.sha1(str(self._cache_key_from_kwargs(**kwargs)).encode("utf-8")).hexdigest()

    def export_solution(self, folder=None, filename=None, solution=None, component=None, suffix=None):
        """
        Export solution to file.
        """
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "solution"
        if solution is None:
            solution = self._solution
        assert component is None or isinstance(component, (str, list))
        if component is None and len(self.components) > 1:
            component = self.components
        if component is None:
            export(solution, folder, filename, suffix)
        elif isinstance(component, str):
            export(solution, folder, filename + "_" + component, suffix, component)
        elif isinstance(component, list):
            for c in component:
                assert isinstance(c, str)
                export(solution, folder, filename + "_" + c, suffix, c)
        else:
            raise TypeError("Invalid component in export_solution()")

    def import_solution(self, folder=None, filename=None, solution=None, component=None, suffix=None):
        """
        Import solution from file.
        """
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "solution"
        if solution is None:
            solution = self._solution
        assert component is None or isinstance(component, (str, list))
        if component is None and len(self.components) > 1:
            component = self.components
        if component is None:
            import_(solution, folder, filename, suffix)
        elif isinstance(component, str):
            import_(solution, folder, filename + "_" + component, suffix, component)
        elif isinstance(component, list):
            for c in component:
                assert isinstance(c, str)
                import_(solution, folder, filename + "_" + c, suffix, c)
        else:
            raise TypeError("Invalid component in import_solution()")

    def export_output(self, folder=None, filename=None, output=None, suffix=None):
        """
        Export solution to file.
        """
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "solution"
        if output is None:
            output = [self._output]
        else:
            assert isinstance(output, list)
            assert len(output) == 1
        export(output, folder, filename + "_output", suffix)

    def import_output(self, folder=None, filename=None, output=None, suffix=None):
        """
        Import solution from file.
        """
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "solution"
        if output is None:
            output = [0.]
            import_(output, folder, filename + "_output", suffix)
            assert len(output) == 1
            assert isinstance(output[0], Number)
            self._output = output[0]
        else:
            assert isinstance(output, list)
            assert len(output) == 1
            assert isinstance(output[0], Number)
            import_(output, folder, filename + "_output", suffix)

    @abstractmethod
    def compute_theta(self, term):
        """
        Return theta multiplicative terms of the affine expansion of the problem.
        Example of implementation for Poisson problem:
           if term == "a":
               theta_a0 = self.mu[0]
               theta_a1 = self.mu[1]
               theta_a2 = self.mu[0] * self.mu[1] + self.mu[2] / 7.0
               return (theta_a0, theta_a1, theta_a2)
           elif term == "f":
               theta_f0 = self.mu[0] * self.mu[2]
               return (theta_f0,)
           elif term == "dirichlet_bc":
               theta_bc0 = 1.
               return (theta_f0,)
           else:
               raise ValueError("Invalid term for compute_theta().")
        """
        raise NotImplementedError("The method compute_theta() is problem-specific and needs to be overridden.")

    @abstractmethod
    def assemble_operator(self, term):
        """
        Return forms resulting from the discretization of the affine expansion of the problem operators.
        Example of implementation for Poisson problem:
           if term == "a":
               a0 = inner(grad(u), grad(v)) * dx
               return (a0,)
           elif term == "f":
               f0 = v * ds(1)
               return (f0,)
           elif term == "dirichlet_bc":
               bc0 = [(V, Constant(0.0), boundaries, 3)]
               return (bc0,)
           elif term == "inner_product":
               x0 = u * v * dx + inner(grad(u), grad(v)) * dx
               return (x0,)
           else:
               raise ValueError("Invalid term for assemble_operator().")
        """
        raise NotImplementedError("The method assemble_operator() is problem-specific and needs to be overridden.")

    def get_stability_factor_lower_bound(self):
        """
        Return a lower bound for the stability factor
        Example of implementation:
            return 1.0
        Note that this method is not needed in POD-Galerkin reduced order models, and this is the reason
        for which it is not marked as @abstractmethod
        """
        raise NotImplementedError("The method get_stability_factor_lower_bound() is problem-specific"
                                  + "and needs to be overridden.")
