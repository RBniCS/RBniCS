# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.abstract import LinearSolver as AbstractLinearSolver, LinearProblemWrapper
from rbnics.backends.online.basic.wrapping import DirichletBC, preserve_solution_attributes
from rbnics.utils.decorators import DictOfThetaType, overload, ThetaType


def LinearSolver(backend, wrapping):

    class LinearSolver_Class(AbstractLinearSolver):
        @overload((backend.Matrix.Type(), wrapping.DelayedTransposeWithArithmetic),
                  backend.Function.Type(),
                  (backend.Vector.Type(), wrapping.DelayedTransposeWithArithmetic),
                  ThetaType + DictOfThetaType + (None,))
        def __init__(self, lhs, solution, rhs, bcs=None):
            self.solution = solution
            self._init_lhs(lhs)
            self._init_rhs(rhs)
            self._apply_bcs(bcs)
            preserve_solution_attributes(self.lhs, self.solution, self.rhs)
            self.monitor = None

        @overload(LinearProblemWrapper, backend.Function.Type())
        def __init__(self, problem_wrapper, solution):
            lhs = problem_wrapper.matrix_eval()
            rhs = problem_wrapper.vector_eval()
            bcs = problem_wrapper.bc_eval()
            self.__init__(lhs, solution, rhs, bcs)
            self.monitor = problem_wrapper.monitor

        @overload
        def _init_lhs(self, lhs: backend.Matrix.Type()):
            self.lhs = lhs

        @overload
        def _init_lhs(self, lhs: wrapping.DelayedTransposeWithArithmetic):
            self.lhs = lhs.evaluate()

        @overload
        def _init_rhs(self, rhs: backend.Vector.Type()):
            self.rhs = rhs

        @overload
        def _init_rhs(self, rhs: wrapping.DelayedTransposeWithArithmetic):
            self.rhs = rhs.evaluate()

        @overload
        def _apply_bcs(self, bcs: None):
            pass

        @overload
        def _apply_bcs(self, bcs: ThetaType):
            bcs = DirichletBC(bcs)
            bcs.apply_to_vector(self.rhs)
            bcs.apply_to_matrix(self.lhs)

        @overload
        def _apply_bcs(self, bcs: DictOfThetaType):
            # Auxiliary dicts should have been stored in lhs and rhs, and should be consistent
            assert (self.rhs._component_name_to_basis_component_index
                    == self.lhs._component_name_to_basis_component_index[0])
            assert (self.rhs._component_name_to_basis_component_length
                    == self.lhs._component_name_to_basis_component_length[0])
            # Provide auxiliary dicts to DirichletBC constructor, and apply
            bcs = DirichletBC(bcs, self.rhs._component_name_to_basis_component_index, self.rhs.N)
            bcs.apply_to_vector(self.rhs)
            bcs.apply_to_matrix(self.lhs)

    return LinearSolver_Class
