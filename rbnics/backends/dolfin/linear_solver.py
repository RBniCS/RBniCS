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

@BackendFor("dolfin", inputs=((Form, Matrix.Type(), ParametrizedTensorFactory, LinearProblemWrapper), Function.Type(), (Form, ParametrizedTensorFactory, Vector.Type(), None), (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC), None)))
class LinearSolver(AbstractLinearSolver):
    @overload((Form, Matrix.Type(), ParametrizedTensorFactory), Function.Type(), (Form, ParametrizedTensorFactory, Vector.Type()), (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC), None))
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
    
    @overload
    def _init_lhs(self, lhs: Form, bcs: (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC), None)):
        self.lhs = assemble(lhs, keep_diagonal=True)
        
    @overload
    def _init_lhs(self, lhs: ParametrizedTensorFactory, bcs: (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC), None)):
        self.lhs = evaluate(lhs)
        
    @overload
    def _init_lhs(self, lhs: Matrix.Type(), bcs: None):
        self.lhs = lhs
        
    @overload
    def _init_lhs(self, lhs: Matrix.Type(), bcs: (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC))):
        # Create a copy of lhs, in order not to change
        # the original references when applying bcs
        self.lhs = lhs.copy()
        
    @overload
    def _init_rhs(self, rhs: Form, bcs: (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC), None)):
        self.rhs = assemble(rhs)
        
    @overload
    def _init_rhs(self, rhs: ParametrizedTensorFactory, bcs: (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC), None)):
        self.rhs = evaluate(rhs)
        
    @overload
    def _init_rhs(self, rhs: Vector.Type(), bcs: None):
        self.rhs = rhs
        
    @overload
    def _init_rhs(self, rhs: Vector.Type(), bcs: (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC))):
        # Create a copy of rhs, in order not to change
        # the original references when applying bcs
        self.rhs = rhs.copy()
        
    @overload
    def _apply_bcs(self, bcs: None):
        pass
        
    @overload
    def _apply_bcs(self, bcs: (list_of(DirichletBC), ProductOutputDirichletBC)):
        for bc in bcs:
            bc.apply(self.lhs, self.rhs)
            
    @overload
    def _apply_bcs(self, bcs: (dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC))):
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
