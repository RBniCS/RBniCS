# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.cache import Cache
from rbnics.utils.decorators import PreserveClassName


def StoreMapFromRieszStorageToReducedProblem(ExactParametrizedFunctionsDecoratedReducedProblem_DerivedClass):

    @PreserveClassName
    class StoreMapFromRieszStorageToReducedProblem_Class(
            ExactParametrizedFunctionsDecoratedReducedProblem_DerivedClass):

        def _init_error_estimation_operators(self, current_stage="online"):
            # Initialize error estimation operators as in Parent class
            ExactParametrizedFunctionsDecoratedReducedProblem_DerivedClass._init_error_estimation_operators(
                self, current_stage)

            # Populate Riesz storage to reduced problem maps
            add_to_map_from_riesz_solve_storage_to_reduced_problem(self._riesz_solve_storage, self)
            add_to_map_from_riesz_solve_inner_product_to_reduced_problem(self._riesz_solve_inner_product, self)
            add_to_map_from_riesz_solve_homogeneous_dirichlet_bc_to_reduced_problem(
                self._riesz_solve_homogeneous_dirichlet_bc, self)
            add_to_map_from_error_estimation_inner_product_to_reduced_problem(
                self._error_estimation_inner_product, self)

    # return value (a class) for the decorator
    return StoreMapFromRieszStorageToReducedProblem_Class


def add_to_map_from_riesz_solve_storage_to_reduced_problem(riesz_solve_storage, reduced_problem):
    if riesz_solve_storage not in _riesz_solve_storage_to_reduced_problem_map:
        _riesz_solve_storage_to_reduced_problem_map[riesz_solve_storage] = reduced_problem
    else:
        assert reduced_problem is _riesz_solve_storage_to_reduced_problem_map[riesz_solve_storage]


def add_to_map_from_riesz_solve_inner_product_to_reduced_problem(riesz_solve_inner_product, reduced_problem):
    if riesz_solve_inner_product not in _riesz_solve_inner_product_to_reduced_problem_map:
        _riesz_solve_inner_product_to_reduced_problem_map[riesz_solve_inner_product] = reduced_problem
    else:
        assert reduced_problem is _riesz_solve_inner_product_to_reduced_problem_map[riesz_solve_inner_product]


def add_to_map_from_riesz_solve_homogeneous_dirichlet_bc_to_reduced_problem(
        riesz_solve_homogeneous_dirichlet_bc, reduced_problem):
    if riesz_solve_homogeneous_dirichlet_bc not in _riesz_solve_homogeneous_dirichlet_bc_to_reduced_problem_map:
        _riesz_solve_homogeneous_dirichlet_bc_to_reduced_problem_map[
            riesz_solve_homogeneous_dirichlet_bc] = reduced_problem
    else:
        assert reduced_problem is _riesz_solve_homogeneous_dirichlet_bc_to_reduced_problem_map[
            riesz_solve_homogeneous_dirichlet_bc]


def add_to_map_from_error_estimation_inner_product_to_reduced_problem(error_estimation_inner_product, reduced_problem):
    if error_estimation_inner_product not in _error_estimation_inner_product_to_reduced_problem_map:
        _error_estimation_inner_product_to_reduced_problem_map[error_estimation_inner_product] = reduced_problem
    else:
        assert reduced_problem is _error_estimation_inner_product_to_reduced_problem_map[
            error_estimation_inner_product]


def get_reduced_problem_from_riesz_solve_storage(riesz_solve_storage):
    assert riesz_solve_storage in _riesz_solve_storage_to_reduced_problem_map
    return _riesz_solve_storage_to_reduced_problem_map[riesz_solve_storage]


def get_reduced_problem_from_riesz_solve_inner_product(riesz_solve_inner_product):
    assert riesz_solve_inner_product in _riesz_solve_inner_product_to_reduced_problem_map
    return _riesz_solve_inner_product_to_reduced_problem_map[riesz_solve_inner_product]


def get_reduced_problem_from_riesz_solve_homogeneous_dirichlet_bc(riesz_solve_homogeneous_dirichlet_bc):
    assert riesz_solve_homogeneous_dirichlet_bc in _riesz_solve_homogeneous_dirichlet_bc_to_reduced_problem_map
    return _riesz_solve_homogeneous_dirichlet_bc_to_reduced_problem_map[riesz_solve_homogeneous_dirichlet_bc]


def get_reduced_problem_from_error_estimation_inner_product(error_estimation_inner_product):
    assert error_estimation_inner_product in _error_estimation_inner_product_to_reduced_problem_map
    return _error_estimation_inner_product_to_reduced_problem_map[error_estimation_inner_product]


_riesz_solve_storage_to_reduced_problem_map = Cache()
_riesz_solve_inner_product_to_reduced_problem_map = Cache()
_riesz_solve_homogeneous_dirichlet_bc_to_reduced_problem_map = Cache()
_error_estimation_inner_product_to_reduced_problem_map = Cache()
