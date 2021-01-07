# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from abc import ABCMeta, abstractmethod
from numbers import Number
from rbnics.backends import BasisFunctionsMatrix, Function, FunctionsList, LinearSolver, transpose
from rbnics.backends.online import OnlineAffineExpansionStorage
from rbnics.utils.decorators import overload, PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(None)
def RBReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class RBReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass, metaclass=ABCMeta):
        """
        Abstract class. All the terms for error estimator are initialized.

        :param truth_problem: class of the truth problem to be solved.
        """

        # Default initialization of members.
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)

            # $$ ONLINE DATA STRUCTURES $$ #
            # Residual terms
            self.RieszExpansionStorage = OnlineAffineExpansionStorage
            self.riesz = dict()  # from string to RieszExpansionStorage
            self.riesz_terms = list()
            self.ErrorEstimationOperatorExpansionStorage = OnlineAffineExpansionStorage
            self.error_estimation_operator = dict()  # from string to ErrorEstimationOperatorExpansionStorage
            self.error_estimation_terms = list()  # of tuple

            # $$ OFFLINE DATA STRUCTURES $$ #
            # Residual terms
            self._riesz_solve_storage = Function(self.truth_problem.V)
            self._riesz_solve_inner_product = None  # setup by init()
            self._riesz_solve_homogeneous_dirichlet_bc = None  # setup by init()
            self._error_estimation_inner_product = None  # setup by init()
            # I/O
            self.folder["error_estimation"] = os.path.join(self.folder_prefix, "error_estimation")

            # Provide a default value for Riesz terms and Riesz product terms
            self.riesz_terms = [term for term in self.terms]
            self.error_estimation_terms = [
                (term1, term2)
                for term1 in self.terms
                for term2 in self.terms if self.terms_order[term1] >= self.terms_order[term2]]

        def init(self, current_stage="online"):
            """
            Initialize data structures required for the online phase.
            """
            ParametrizedReducedDifferentialProblem_DerivedClass.init(self, current_stage)
            self._init_error_estimation_operators(current_stage)

        def _init_error_estimation_operators(self, current_stage="online"):
            """
            Initialize data structures related to error estimation.
            """
            # Initialize inner product for Riesz solve
            if self._riesz_solve_inner_product is None:  # init was not called already
                self._riesz_solve_inner_product = self.truth_problem._combined_inner_product
            # Setup homogeneous Dirichlet BCs for Riesz solve, if any (no check if init was already called
            # because this variable can actually be None)
            self._riesz_solve_homogeneous_dirichlet_bc = self.truth_problem._combined_and_homogenized_dirichlet_bc
            # Initialize Riesz representation
            for term in self.riesz_terms:
                if term not in self.riesz:  # init was not called already
                    self.riesz[term] = self.RieszExpansionStorage(self.Q[term])
                    for q in range(self.Q[term]):
                        assert self.terms_order[term] in (1, 2)
                        if self.terms_order[term] > 1:
                            riesz_term_q = BasisFunctionsMatrix(self.truth_problem.V)
                            riesz_term_q.init(self.components)
                        else:
                            riesz_term_q = FunctionsList(self.truth_problem.V)  # will be of size 1
                        self.riesz[term][q] = riesz_term_q
            assert current_stage in ("online", "offline")
            if current_stage == "online":
                for term in self.riesz_terms:
                    self.riesz[term].load(self.folder["error_estimation"], "riesz_" + term)
            elif current_stage == "offline":
                pass  # Nothing else to be done
            else:
                raise ValueError("Invalid stage in _init_error_estimation_operators().")
            # Initialize inner product for Riesz products. This is the same as the inner product for Riesz solves
            # but setting to zero rows & columns associated to boundary conditions
            if self._error_estimation_inner_product is None:  # init was not called already
                if self._riesz_solve_homogeneous_dirichlet_bc is not None:
                    self._error_estimation_inner_product = (
                        self._riesz_solve_inner_product & ~self._riesz_solve_homogeneous_dirichlet_bc)
                else:
                    self._error_estimation_inner_product = self._riesz_solve_inner_product
            # Initialize error estimation operators
            for term in self.error_estimation_terms:
                if term not in self.error_estimation_operator:  # init was not called already
                    self.error_estimation_operator[term] = self.ErrorEstimationOperatorExpansionStorage(
                        self.Q[term[0]], self.Q[term[1]])
            assert current_stage in ("online", "offline")
            if current_stage == "online":
                for term in self.error_estimation_terms:
                    self.assemble_error_estimation_operators(term, "online")
            elif current_stage == "offline":
                pass  # Nothing else to be done
            else:
                raise ValueError("Invalid stage in _init_error_estimation_operators().")

        @abstractmethod
        def estimate_error(self):
            """
            Returns an error bound for the current solution.
            """
            raise NotImplementedError("The method estimate_error() is problem-specific and needs to be overridden.")

        @abstractmethod
        def estimate_relative_error(self):
            """
            It returns a relative error bound for the current solution.
            """
            raise NotImplementedError("The method estimate_relative_error() is problem-specific"
                                      + " and needs to be overridden.")

        def estimate_error_output(self):
            """
            It returns an error bound for the current output.
            """
            return NotImplemented

        def estimate_relative_error_output(self):
            """
            It returns an relative error bound for the current output.
            """
            return NotImplemented

        def build_error_estimation_operators(self, current_stage="offline"):
            self._build_error_estimation_operators(current_stage)

        def _build_error_estimation_operators(self, current_stage="offline"):
            """
            It builds operators for error estimation.
            """

            # Update the Riesz representation with the new basis function(s)
            for term in self.riesz_terms:
                if self.terms_order[term] == 1:
                    lengths = set([len(self.riesz[term][q]) for q in range(self.Q[term])])
                    assert len(lengths) == 1
                    length = lengths.pop()
                    assert length in (0, 1)
                    if length == 0:  # this part does not depend on N, so we compute it only once
                        # Compute the Riesz representation of terms that do not depend on the solution
                        self.compute_riesz_representation(term, current_stage)
                        # Compute the (term, term) Riesz representors product
                        if (term, term) in self.error_estimation_terms:
                            self.assemble_error_estimation_operators((term, term), current_stage)
                else:  # self.terms_order[term] > 1:
                    self.compute_riesz_representation(term, current_stage)

            # Update the (term1, term2) Riesz representors product with the new basis function
            for term in self.error_estimation_terms:
                # the (1, 1) part does not depend on N, and was computed in the previous loop
                if (self.terms_order[term[0]], self.terms_order[term[1]]) != (1, 1):
                    self.assemble_error_estimation_operators(term, current_stage)

        def compute_riesz_representation(self, term, current_stage="offline"):
            """
            It computes the Riesz representation of term.

            :param term: the forms of the truth problem.
            """
            solver = self.RieszSolver(self)
            # Compute the Riesz representor
            assert self.terms_order[term] in (1, 2)
            if self.terms_order[term] == 1:
                for q in range(self.Q[term]):
                    self.riesz[term][q].enrich(
                        solver.solve(self.truth_problem.operator[term][q])
                    )
                self.riesz[term].save(self.folder["error_estimation"], "riesz_" + term)
            elif self.terms_order[term] == 2:
                for q in range(self.Q[term]):
                    if len(self.components) > 1:
                        for component in self.components:
                            for n in range(len(self.riesz[term][q][component]),
                                           self.N[component] + self.N_bc[component]):
                                self.riesz[term][q][component].enrich(
                                    solver.solve(-1., self.truth_problem.operator[term][q],
                                                 self.basis_functions[component][n]))
                    else:
                        for n in range(len(self.riesz[term][q]), self.N + self.N_bc):
                            self.riesz[term][q].enrich(
                                solver.solve(-1., self.truth_problem.operator[term][q], self.basis_functions[n]))
                self.riesz[term].save(self.folder["error_estimation"], "riesz_" + term)
            else:
                raise ValueError("Invalid value for order of term " + term)

        class RieszSolver(object):
            def __init__(self, problem):
                self.problem = problem

            @overload
            def solve(self, rhs: object):
                problem = self.problem
                solver = LinearSolver(problem._riesz_solve_inner_product, problem._riesz_solve_storage, rhs,
                                      problem._riesz_solve_homogeneous_dirichlet_bc)
                solver.set_parameters(problem._linear_solver_parameters)
                solver.solve()
                return problem._riesz_solve_storage

            @overload
            def solve(self, coef: Number, matrix: object, basis_function: object):
                return self.solve(coef * matrix * basis_function)

        def assemble_error_estimation_operators(self, term, current_stage="online"):
            """
            It assembles operators for error estimation.
            """
            assert current_stage in ("online", "offline")
            assert isinstance(term, tuple)
            assert len(term) == 2
            if current_stage == "online":  # load from file
                self.error_estimation_operator[term].load(
                    self.folder["error_estimation"], "error_estimation_operator_" + term[0] + "_" + term[1])
                return self.error_estimation_operator[term]
            elif current_stage == "offline":
                assert self.terms_order[term[0]] in (1, 2)
                assert self.terms_order[term[1]] in (1, 2)
                assert self.terms_order[term[0]] >= self.terms_order[term[1]], (
                    "Please swap the order of " + str(term) + " in self.error_estimation_terms")
                # otherwise for (term1, term2) of orders (1, 2) we would have a row vector, rather than a column one
                if self.terms_order[term[0]] == 2 and self.terms_order[term[1]] == 2:
                    for q0 in range(self.Q[term[0]]):
                        for q1 in range(self.Q[term[1]]):
                            self.error_estimation_operator[term][q0, q1] = (
                                transpose(self.riesz[term[0]][q0]) * self._error_estimation_inner_product
                                * self.riesz[term[1]][q1])
                elif self.terms_order[term[0]] == 2 and self.terms_order[term[1]] == 1:
                    for q0 in range(self.Q[term[0]]):
                        for q1 in range(self.Q[term[1]]):
                            assert len(self.riesz[term[1]][q1]) == 1
                            self.error_estimation_operator[term][q0, q1] = (
                                transpose(self.riesz[term[0]][q0]) * self._error_estimation_inner_product
                                * self.riesz[term[1]][q1][0])
                elif self.terms_order[term[0]] == 1 and self.terms_order[term[1]] == 1:
                    for q0 in range(self.Q[term[0]]):
                        assert len(self.riesz[term[0]][q0]) == 1
                        for q1 in range(self.Q[term[1]]):
                            assert len(self.riesz[term[1]][q1]) == 1
                            self.error_estimation_operator[term][q0, q1] = (
                                transpose(self.riesz[term[0]][q0][0]) * self._error_estimation_inner_product
                                * self.riesz[term[1]][q1][0])
                else:
                    raise ValueError("Invalid term order for assemble_error_estimation_operators().")
                self.error_estimation_operator[term].save(
                    self.folder["error_estimation"], "error_estimation_operator_" + term[0] + "_" + term[1])
                return self.error_estimation_operator[term]
            else:
                raise ValueError("Invalid stage in assemble_error_estimation_operators().")

    # return value (a class) for the decorator
    return RBReducedProblem_Class
