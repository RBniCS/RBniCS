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
from numbers import Number
from rbnics.backends.abstract import ParametrizedTensorFactory as AbstractParametrizedTensorFactory
from rbnics.backends.basic.wrapping.delayed_product import DelayedProduct
from rbnics.backends.basic.wrapping.delayed_sum import DelayedSum
from rbnics.eim.utils.decorators import get_component_and_index_from_basis_function, get_reduced_problem_from_riesz_solve_homogeneous_dirichlet_bc, get_reduced_problem_from_riesz_solve_inner_product, get_reduced_problem_from_riesz_solve_storage, get_term_and_index_from_parametrized_operator, get_problem_from_parametrized_operator
from rbnics.utils.decorators import get_problem_from_problem_name, get_reduced_problem_from_problem
from rbnics.utils.io import Folders, TextIO as BCsIO, TextIO as LHSIO, TextIO as ParametersIO, TextIO as RHSIO, TextIO as SolutionIO

class DelayedLinearSolver(object):
    def __init__(self, lhs=None, solution=None, rhs=None, bcs=None):
        self._lhs = lhs
        self._solution = solution
        assert (
            rhs is None
                or
            isinstance(rhs, (AbstractParametrizedTensorFactory, DelayedProduct, DelayedSum))
        )
        self._rhs = rhs
        self._bcs = bcs
        self._parameters = dict()
        
    def set_parameters(self, parameters):
        self._parameters = parameters
        
    def solve(self):
        from rbnics.backends import AffineExpansionStorage, evaluate, LinearSolver, product, sum
        assert self._lhs is not None
        assert self._solution is not None
        assert isinstance(self._rhs, DelayedSum)
        thetas = list()
        operators = list()
        for addend in self._rhs._args:
            assert isinstance(addend, DelayedProduct)
            assert len(addend._args) in (2, 3)
            assert isinstance(addend._args[0], Number)
            assert isinstance(addend._args[1], AbstractParametrizedTensorFactory)
            thetas.append(addend._args[0])
            if len(addend._args) == 2:
                operators.append(addend._args[1])
            elif len(addend._args) == 3:
                operators.append(addend._args[1]*addend._args[2])
            else:
                raise ValueError("Invalid addend")
        thetas = tuple(thetas)
        operators = AffineExpansionStorage(tuple(evaluate(op) for op in operators))
        rhs = sum(product(thetas, operators))
        solver = LinearSolver(self._lhs, self._solution, rhs, self._bcs)
        solver.set_parameters(self._parameters)
        solver.solve()
        
    def save(self, directory, filename):
        # Get full directory name
        full_directory = Folders.Folder(os.path.join(str(directory), filename))
        full_directory.create()
        # Save problem corresponding to self._lhs
        assert self._lhs is not None
        LHSIO.save_file(get_reduced_problem_from_riesz_solve_inner_product(self._lhs).truth_problem.name(), full_directory, "lhs_problem_name")
        # Save problem corresponding to self._solution
        assert self._solution is not None
        SolutionIO.save_file(get_reduced_problem_from_riesz_solve_storage(self._solution).truth_problem.name(), full_directory, "solution_problem_name")
        # Save problem and operator corresponding to self._rhs
        assert self._rhs is not None
        assert isinstance(self._rhs, (AbstractParametrizedTensorFactory, DelayedProduct))
        if isinstance(self._rhs, AbstractParametrizedTensorFactory):
            RHSIO.save_file("ParametrizedTensorFactory", full_directory, "rhs_type")
            rhs_arg_0 = self._rhs
            rhs_problem_name_0 = get_problem_from_parametrized_operator(rhs_arg_0).name()
            (rhs_term_0, rhs_index_0) = get_term_and_index_from_parametrized_operator(rhs_arg_0)
            RHSIO.save_file((rhs_problem_name_0, rhs_term_0, rhs_index_0), full_directory, "rhs_arg_0")
        elif isinstance(self._rhs, DelayedProduct):
            RHSIO.save_file("DelayedProduct", full_directory, "rhs_type")
            assert len(self._rhs._args) == 3
            rhs_arg_0 = self._rhs._args[0]
            assert rhs_arg_0 == -1.0
            RHSIO.save_file(rhs_arg_0, full_directory, "rhs_arg_0")
            assert isinstance(self._rhs._args[1], AbstractParametrizedTensorFactory)
            rhs_arg_1 = self._rhs._args[1]
            rhs_problem_name_1 = get_problem_from_parametrized_operator(rhs_arg_1).name()
            (rhs_term_1, rhs_index_1) = get_term_and_index_from_parametrized_operator(rhs_arg_1)
            RHSIO.save_file((rhs_problem_name_1, rhs_term_1, rhs_index_1), full_directory, "rhs_arg_1")
            rhs_arg_2 = self._rhs._args[2]
            rhs_problem_name_2 = rhs_problem_name_1
            (rhs_component_2, rhs_index_2) = get_component_and_index_from_basis_function(rhs_arg_2)
            RHSIO.save_file((rhs_problem_name_2, rhs_component_2, rhs_index_2), full_directory, "rhs_arg_2")
        else:
            raise TypeError("Invalid rhs")
        # Save problem corresponding to self._bcs
        BCsIO.save_file(get_reduced_problem_from_riesz_solve_homogeneous_dirichlet_bc(self._bcs).truth_problem.name(), full_directory, "bcs_problem_name")
        # Save parameters
        ParametersIO.save_file(self._parameters, full_directory, "parameters")
        
    def load(self, directory, filename):
        # Get full directory name
        full_directory = Folders.Folder(os.path.join(str(directory), filename))
        # Load problem corresponding to self._lhs, and update self._lhs accordingly
        assert self._lhs is None
        assert LHSIO.exists_file(full_directory, "lhs_problem_name")
        lhs_problem_name = LHSIO.load_file(full_directory, "lhs_problem_name")
        lhs_problem = get_problem_from_problem_name(lhs_problem_name)
        lhs_reduced_problem = get_reduced_problem_from_problem(lhs_problem)
        self._lhs = lhs_reduced_problem._riesz_solve_inner_product
        # Load problem corresponding to self._solution, and update self._solution accordingly
        assert self._solution is None
        assert SolutionIO.exists_file(full_directory, "solution_problem_name")
        solution_problem_name = SolutionIO.load_file(full_directory, "solution_problem_name")
        solution_problem = get_problem_from_problem_name(solution_problem_name)
        solution_reduced_problem = get_reduced_problem_from_problem(solution_problem)
        self._solution = solution_reduced_problem._riesz_solve_storage
        # Load problem and operator corresponding to self._rhs, and update self._solution accordingly
        assert self._rhs is None
        assert RHSIO.exists_file(full_directory, "rhs_type")
        rhs_type = RHSIO.load_file(full_directory, "rhs_type")
        assert rhs_type in ("ParametrizedTensorFactory", "DelayedProduct")
        if rhs_type == "ParametrizedTensorFactory":
            assert RHSIO.exists_file(full_directory, "rhs_arg_0")
            (rhs_problem_name_0, rhs_term_0, rhs_index_0) = RHSIO.load_file(full_directory, "rhs_arg_0")
            rhs_problem_0 = get_problem_from_problem_name(rhs_problem_name_0)
            rhs_arg_0 = rhs_problem_0.operator[rhs_term_0][rhs_index_0]
            assert isinstance(rhs_arg_0, AbstractParametrizedTensorFactory)
            self._rhs = rhs_arg_0
        elif rhs_type == "DelayedProduct":
            assert RHSIO.exists_file(full_directory, "rhs_arg_0")
            rhs_arg_0 = RHSIO.load_file(full_directory, "rhs_arg_0")
            assert rhs_arg_0 == -1.0
            assert RHSIO.exists_file(full_directory, "rhs_arg_1")
            (rhs_problem_name_1, rhs_term_1, rhs_index_1) = RHSIO.load_file(full_directory, "rhs_arg_1")
            rhs_problem_1 = get_problem_from_problem_name(rhs_problem_name_1)
            rhs_arg_1 = rhs_problem_1.operator[rhs_term_1][rhs_index_1]
            assert isinstance(rhs_arg_1, AbstractParametrizedTensorFactory)
            assert RHSIO.exists_file(full_directory, "rhs_arg_2")
            (rhs_problem_name_2, rhs_component_2, rhs_index_2) = RHSIO.load_file(full_directory, "rhs_arg_2")
            rhs_problem_2 = get_problem_from_problem_name(rhs_problem_name_2)
            rhs_reduced_problem_2 = get_reduced_problem_from_problem(rhs_problem_2)
            if rhs_component_2 is not None:
                rhs_arg_2 = rhs_reduced_problem_2.basis_functions[rhs_component_2][rhs_index_2]
            else:
                rhs_arg_2 = rhs_reduced_problem_2.basis_functions[rhs_index_2]
            rhs = DelayedProduct(rhs_arg_0)
            rhs *= rhs_arg_1
            rhs *= rhs_arg_2
            self._rhs = rhs
        else:
            raise TypeError("Invalid rhs")
        # Load problem corresponding to self._bcs, and update self._bcs accordingly
        assert self._bcs is None
        assert BCsIO.exists_file(full_directory, "bcs_problem_name")
        bcs_problem_name = BCsIO.load_file(full_directory, "bcs_problem_name")
        bcs_problem = get_problem_from_problem_name(bcs_problem_name)
        bcs_reduced_problem = get_reduced_problem_from_problem(bcs_problem)
        self._bcs = bcs_reduced_problem._riesz_solve_homogeneous_dirichlet_bc
        # Load parameters
        assert len(self._parameters) == 0
        assert ParametersIO.exists_file(full_directory, "parameters")
        self._parameters = ParametersIO.load_file(full_directory, "parameters")
        # Return
        return True
        
    def get_problem_name(self):
        return get_reduced_problem_from_riesz_solve_storage(self._solution).truth_problem.name()
