# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl import Form
from dolfin import assemble, DirichletBC, PETScLUSolver
from rbnics.backends.abstract import LinearSolver as AbstractLinearSolver, LinearProblemWrapper
from rbnics.backends.dolfin.evaluate import evaluate
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.parametrized_tensor_factory import ParametrizedTensorFactory
from rbnics.backends.dolfin.vector import Vector
from rbnics.backends.dolfin.wrapping.dirichlet_bc import ProductOutputDirichletBC
from rbnics.utils.decorators import BackendFor, dict_of, list_of, overload


@BackendFor("dolfin", inputs=((Form, Matrix.Type(), ParametrizedTensorFactory, LinearProblemWrapper),
                              Function.Type(), (Form, ParametrizedTensorFactory, Vector.Type(), None),
                              (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)),
                               dict_of(str, ProductOutputDirichletBC), None)))
class LinearSolver(AbstractLinearSolver):
    @overload((Form, Matrix.Type(), ParametrizedTensorFactory),
              Function.Type(), (Form, ParametrizedTensorFactory, Vector.Type()),
              (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)),
               dict_of(str, ProductOutputDirichletBC), None))
    def __init__(self, lhs, solution, rhs, bcs=None):
        self.solution = solution
        self._init_lhs(lhs, bcs)
        self._init_rhs(rhs, bcs)
        self._apply_bcs(bcs)
        self._linear_solver = "default"
        self.monitor = None

    @overload(LinearProblemWrapper, Function.Type())
    def __init__(self, problem_wrapper, solution):
        self.__init__(problem_wrapper.matrix_eval(), solution, problem_wrapper.vector_eval(), problem_wrapper.bc_eval())
        self.monitor = problem_wrapper.monitor

    @overload(Form, (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)),
                     dict_of(str, ProductOutputDirichletBC), None))
    def _init_lhs(self, lhs, bcs):
        self.lhs = assemble(lhs, keep_diagonal=True)

    @overload(ParametrizedTensorFactory, (list_of(DirichletBC), ProductOutputDirichletBC,
                                          dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC),
                                          None))
    def _init_lhs(self, lhs, bcs):
        self.lhs = evaluate(lhs)

    @overload(Matrix.Type(), None)
    def _init_lhs(self, lhs, bcs):
        self.lhs = lhs

    @overload(Matrix.Type(), (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)),
                              dict_of(str, ProductOutputDirichletBC)))
    def _init_lhs(self, lhs, bcs):
        # Create a copy of lhs, in order not to change
        # the original references when applying bcs
        self.lhs = lhs.copy()

    @overload(Form, (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)),
                     dict_of(str, ProductOutputDirichletBC), None))
    def _init_rhs(self, rhs, bcs):
        self.rhs = assemble(rhs)

    @overload(ParametrizedTensorFactory, (list_of(DirichletBC), ProductOutputDirichletBC,
                                          dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC),
                                          None))
    def _init_rhs(self, rhs, bcs):
        self.rhs = evaluate(rhs)

    @overload(Vector.Type(), None)
    def _init_rhs(self, rhs, bcs):
        self.rhs = rhs

    @overload(Vector.Type(), (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)),
                              dict_of(str, ProductOutputDirichletBC)))
    def _init_rhs(self, rhs, bcs):
        # Create a copy of rhs, in order not to change
        # the original references when applying bcs
        self.rhs = rhs.copy()

    @overload(None)
    def _apply_bcs(self, bcs):
        pass

    @overload((list_of(DirichletBC), ProductOutputDirichletBC))
    def _apply_bcs(self, bcs):
        for bc in bcs:
            bc.apply(self.lhs, self.rhs)

    @overload((dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC)))
    def _apply_bcs(self, bcs):
        for key in bcs:
            for bc in bcs[key]:
                bc.apply(self.lhs, self.rhs)

    def set_parameters(self, parameters):
        assert len(parameters) in (0, 1)
        if len(parameters) == 1:
            assert "linear_solver" in parameters
        self._linear_solver = parameters.get("linear_solver", "default")

    def solve(self):
        solver = PETScLUSolver(self._linear_solver)
        solver.solve(self.lhs, self.solution.vector(), self.rhs)
        if self.monitor is not None:
            self.monitor(self.solution)
