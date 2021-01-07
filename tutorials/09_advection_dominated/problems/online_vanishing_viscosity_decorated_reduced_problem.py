# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from collections import OrderedDict
from rbnics.backends import AffineExpansionStorage, LinearSolver, product, sum, transpose
from rbnics.backends.online import OnlineAffineExpansionStorage, OnlineFunction
from rbnics.utils.decorators import PreserveClassName, ReducedProblemDecoratorFor
from backends.dolfin import NonHierarchicalBasisFunctionsMatrix
from backends.online import OnlineMatrix, OnlineNonHierarchicalAffineExpansionStorage, OnlineSolveKwargsGenerator
from .online_vanishing_viscosity import OnlineVanishingViscosity


@ReducedProblemDecoratorFor(OnlineVanishingViscosity)
def OnlineVanishingViscosityDecoratedReducedProblem(EllipticCoerciveReducedProblem_DerivedClass):

    @PreserveClassName
    class OnlineVanishingViscosityDecoratedReducedProblem_Class(EllipticCoerciveReducedProblem_DerivedClass):

        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            EllipticCoerciveReducedProblem_DerivedClass.__init__(self, truth_problem, **kwargs)

            # Store vanishing viscosity data
            self._viscosity = truth_problem._viscosity
            self._N_threshold_min = truth_problem._N_threshold_min
            self._N_threshold_max = truth_problem._N_threshold_max

            # Temporary storage for vanishing viscosity eigenvalues
            self.vanishing_viscosity_eigenvalues = list()

            # Default values for keyword arguments in solve
            self._online_solve_default_kwargs = OrderedDict()
            self._online_solve_default_kwargs["online_stabilization"] = False
            self._online_solve_default_kwargs["online_vanishing_viscosity"] = True
            self.OnlineSolveKwargs = OnlineSolveKwargsGenerator(**self._online_solve_default_kwargs)

            # Flag to disable inner product combination after vanishing viscosity operator has been setup
            self._disable_inner_product_combination = False
            # Flag to disable error estimation after vanishing viscosity operator has been setup
            self._disable_error_estimation = False

        def _init_operators(self, current_stage="online"):
            # The difference between this method and the parent one is that non-hierarchical affine
            # expansion storage is requested during the online stage and during the postprocessing
            # of the offline stage.
            if current_stage == "online":
                # Initialize all terms using a non-hierarchical affine expansion storage, and then loading from file
                for term in self.terms:
                    self.Q[term] = self.truth_problem.Q[term]
                    self.operator[term] = OnlineNonHierarchicalAffineExpansionStorage(self.Q[term])
                    self.assemble_operator(term, "online")
                # Initialize additional reduced operator related to vanishing viscosity
                self.operator["vanishing_viscosity"] = OnlineNonHierarchicalAffineExpansionStorage(1)
                self.assemble_operator("vanishing_viscosity", "online")
            elif current_stage == "offline":
                # Call Parent
                EllipticCoerciveReducedProblem_DerivedClass._init_operators(self, current_stage)
            elif current_stage == "offline_vanishing_viscosity_postprocessing":
                # Initialize additional truth operators
                self.truth_problem.operator["k"] = AffineExpansionStorage(self.truth_problem.assemble_operator("k"))
                self.truth_problem.operator["m"] = AffineExpansionStorage(self.truth_problem.assemble_operator("m"))
                # Initialize all terms using a non-hierarchical affine expansion storage
                for term in self.terms:
                    self.Q[term] = self.truth_problem.Q[term]
                    self.operator[term] = OnlineNonHierarchicalAffineExpansionStorage(self.Q[term])
                # Initialize additional reduced operator related to vanishing viscosity
                self.operator["vanishing_viscosity"] = OnlineNonHierarchicalAffineExpansionStorage(1)
            else:
                # Call Parent, which may eventually raise an error
                EllipticCoerciveReducedProblem_DerivedClass._init_operators(self, current_stage)

        def _init_inner_products(self, current_stage="online"):
            # The difference between this method and the parent one is that non-hierarchical affine
            # expansion storage is requested during the online stage and during the postprocessing
            # of the offline stage.
            if current_stage == "online":
                self.inner_product = OnlineNonHierarchicalAffineExpansionStorage(1)
                self.projection_inner_product = OnlineNonHierarchicalAffineExpansionStorage(1)
                self.assemble_operator("inner_product", "online")
                self.assemble_operator("projection_inner_product", "online")
                self._disable_inner_product_combination = True
            elif current_stage == "offline":
                EllipticCoerciveReducedProblem_DerivedClass._init_inner_products(self, current_stage)
            elif current_stage == "offline_vanishing_viscosity_postprocessing":
                self.inner_product = OnlineNonHierarchicalAffineExpansionStorage(1)
                self.projection_inner_product = OnlineNonHierarchicalAffineExpansionStorage(1)
                self._disable_inner_product_combination = True
            else:
                # Call Parent, which may eventually raise an error
                EllipticCoerciveReducedProblem_DerivedClass._init_inner_products(self, current_stage)

        def _combine_all_inner_products(self):
            if self._disable_inner_product_combination:
                return NotImplemented
            else:
                return EllipticCoerciveReducedProblem_DerivedClass._combine_all_inner_products(self)

        def _combine_all_projection_inner_products(self):
            if self._disable_inner_product_combination:
                return NotImplemented
            else:
                return EllipticCoerciveReducedProblem_DerivedClass._combine_all_projection_inner_products(self)

        def _init_basis_functions(self, current_stage="online"):
            if current_stage == "online":
                if self.basis_functions is None:  # avoid re-initializing basis functions matrix multiple times
                    self.basis_functions = NonHierarchicalBasisFunctionsMatrix(self.truth_problem.V)
                self.basis_functions.init(self.truth_problem.components)
                basis_functions_loaded = self.basis_functions.load(self.folder["basis"], "basis")
                if basis_functions_loaded:
                    self.N = len(self.basis_functions)
                    self.N_bc = 0  # TODO handle inhomogeneous bcs
            elif current_stage == "offline":
                EllipticCoerciveReducedProblem_DerivedClass._init_basis_functions(self, current_stage)
            elif current_stage == "offline_vanishing_viscosity_postprocessing":
                self.basis_functions = NonHierarchicalBasisFunctionsMatrix(self.truth_problem.V)
                self.basis_functions.init(self.truth_problem.components)
            else:
                # Call Parent, which may eventually raise an error
                EllipticCoerciveReducedProblem_DerivedClass._init_basis_functions(self, current_stage)

        def _init_error_estimation_operators(self, current_stage="online"):
            if current_stage in ("online", "offline_vanishing_viscosity_postprocessing"):
                # Disable error estimation, which would not take into account the additional vanishing viscosity
                # operator
                self._disable_error_estimation = True
            elif current_stage == "offline":
                # Call Parent
                EllipticCoerciveReducedProblem_DerivedClass._init_error_estimation_operators(self, current_stage)
            else:
                # Call Parent, which may eventually raise an error
                EllipticCoerciveReducedProblem_DerivedClass._init_error_estimation_operators(self, current_stage)

        def build_reduced_operators(self, current_stage="offline"):
            if current_stage == "offline_vanishing_viscosity_postprocessing":
                EllipticCoerciveReducedProblem_DerivedClass.build_reduced_operators(
                    self, "offline_vanishing_viscosity_postprocessing")
                # Compute vanishing viscosity reduced operator
                print("build vanishing viscosity reduced operator")
                self.operator["vanishing_viscosity"] = self.assemble_operator(
                    "vanishing_viscosity", "offline_vanishing_viscosity_postprocessing")
            else:
                # Call Parent, which may eventually raise an error
                EllipticCoerciveReducedProblem_DerivedClass.build_reduced_operators(self, current_stage)

        def assemble_operator(self, term, current_stage="online"):
            if term == "vanishing_viscosity":
                assert current_stage in ("online", "offline_vanishing_viscosity_postprocessing")
                if current_stage == "online":  # load from file
                    self.operator["vanishing_viscosity"].load(
                        self.folder["reduced_operators"], "operator_vanishing_viscosity")
                    return self.operator["vanishing_viscosity"]
                elif current_stage == "offline_vanishing_viscosity_postprocessing":
                    assert len(self.vanishing_viscosity_eigenvalues) == self.N
                    assert all([len(vanishing_viscosity_eigenvalues_n) == n + 1
                                for (n, vanishing_viscosity_eigenvalues_n) in enumerate(
                                    self.vanishing_viscosity_eigenvalues)])
                    print("build reduced vanishing viscosity operator")
                    for n in range(1, self.N + 1):
                        vanishing_viscosity_expansion = OnlineAffineExpansionStorage(1)
                        vanishing_viscosity_eigenvalues = self.vanishing_viscosity_eigenvalues[n - 1]
                        vanishing_viscosity_operator = OnlineMatrix(n, n)
                        n_min = int(n * self._N_threshold_min)
                        n_max = int(n * self._N_threshold_max)
                        lambda_n_min = vanishing_viscosity_eigenvalues[n_min]
                        lambda_n_max = vanishing_viscosity_eigenvalues[n_max]
                        for i in range(n):
                            lambda_i = vanishing_viscosity_eigenvalues[i]
                            if i < n_min:
                                viscosity_i = 0.
                            elif i < n_max:
                                viscosity_i = (
                                    self._viscosity
                                    * (lambda_i - lambda_n_min)**2 / (lambda_n_max - lambda_n_min)**3
                                    * (2 * lambda_n_max**2 - (lambda_n_min + lambda_n_max) * lambda_i)
                                )
                            else:
                                viscosity_i = self._viscosity * lambda_i
                            vanishing_viscosity_operator[i, i] = viscosity_i * lambda_i
                        vanishing_viscosity_expansion[0] = vanishing_viscosity_operator
                        self.operator["vanishing_viscosity"][:n, :n] = vanishing_viscosity_expansion
                    # Save to file
                    self.operator["vanishing_viscosity"].save(
                        self.folder["reduced_operators"], "operator_vanishing_viscosity")
                    return self.operator["vanishing_viscosity"]
                else:
                    raise ValueError("Invalid stage in assemble_operator().")
            else:
                if current_stage == "offline_vanishing_viscosity_postprocessing":
                    if term in self.terms:
                        for n in range(1, self.N + 1):
                            assert self.Q[term] == self.truth_problem.Q[term]
                            term_expansion = OnlineAffineExpansionStorage(self.Q[term])
                            assert self.terms_order[term] in (1, 2)
                            if self.terms_order[term] == 2:
                                for q in range(self.Q[term]):
                                    term_expansion[q] = (
                                        transpose(self.basis_functions[:n]) * self.truth_problem.operator[term][q]
                                        * self.basis_functions[:n])
                                self.operator[term][:n, :n] = term_expansion
                            elif self.terms_order[term] == 1:
                                for q in range(self.Q[term]):
                                    term_expansion[q] = (
                                        transpose(self.basis_functions[:n]) * self.truth_problem.operator[term][q])
                                self.operator[term][:n] = term_expansion
                            else:
                                raise ValueError("Invalid value for order of term " + term)
                        self.operator[term].save(self.folder["reduced_operators"], "operator_" + term)
                        return self.operator[term]
                    elif term.startswith("inner_product"):
                        assert len(self.inner_product) == 1
                        # the affine expansion storage contains only the inner product matrix
                        assert len(self.truth_problem.inner_product) == 1
                        # the affine expansion storage contains only the inner product matrix
                        for n in range(1, self.N + 1):
                            inner_product_expansion = OnlineAffineExpansionStorage(1)
                            inner_product_expansion[0] = (
                                transpose(self.basis_functions[:n]) * self.truth_problem.inner_product[0]
                                * self.basis_functions[:n])
                            self.inner_product[:n, :n] = inner_product_expansion
                        self.inner_product.save(self.folder["reduced_operators"], term)
                        return self.inner_product
                    elif term.startswith("projection_inner_product"):
                        assert len(self.projection_inner_product) == 1
                        # the affine expansion storage contains only the inner product matrix
                        assert len(self.truth_problem.projection_inner_product) == 1
                        # the affine expansion storage contains only the inner product matrix
                        for n in range(1, self.N + 1):
                            projection_inner_product_expansion = OnlineAffineExpansionStorage(1)
                            projection_inner_product_expansion[0] = (
                                transpose(self.basis_functions[:n]) * self.truth_problem.projection_inner_product[0]
                                * self.basis_functions[:n])
                            self.projection_inner_product[:n, :n] = projection_inner_product_expansion
                        self.projection_inner_product.save(self.folder["reduced_operators"], term)
                        return self.projection_inner_product
                    elif term.startswith("dirichlet_bc"):
                        raise ValueError("There should be no need to assemble Dirichlet BCs when querying"
                                         + " the offline vanishing viscosity postprocessing stage.")
                    else:
                        raise ValueError("Invalid term for assemble_operator().")
                else:
                    return EllipticCoerciveReducedProblem_DerivedClass.assemble_operator(self, term, current_stage)

        def _online_size_from_kwargs(self, N, **kwargs):
            N, kwargs = EllipticCoerciveReducedProblem_DerivedClass._online_size_from_kwargs(self, N, **kwargs)
            kwargs = self.OnlineSolveKwargs(**kwargs)
            return N, kwargs

        def _solve(self, N, **kwargs):
            # Temporarily change value of stabilized attribute in truth problem
            bak_stabilized = self.truth_problem.stabilized
            self.truth_problem.stabilized = kwargs["online_stabilization"]
            # Solve reduced problem
            if kwargs["online_vanishing_viscosity"]:
                assembled_operator = dict()
                assembled_operator["a"] = (
                    sum(product(self.compute_theta("a"), self.operator["a"][:N, :N]))
                    + self.operator["vanishing_viscosity"][:N, :N][0]
                )
                assembled_operator["f"] = sum(product(self.compute_theta("f"), self.operator["f"][:N]))
                self._solution = OnlineFunction(N)
                solver = LinearSolver(assembled_operator["a"], self._solution, assembled_operator["f"])
                solver.set_parameters(self._linear_solver_parameters)
                solver.solve()
            else:
                EllipticCoerciveReducedProblem_DerivedClass._solve(self, N, **kwargs)
            # Restore original value of stabilized attribute in truth problem
            self.truth_problem.stabilized = bak_stabilized

        def estimate_error(self):
            if self._disable_error_estimation:
                return NotImplemented
            else:
                return EllipticCoerciveReducedProblem_DerivedClass.estimate_error(self)

    # return value (a class) for the decorator
    return OnlineVanishingViscosityDecoratedReducedProblem_Class
