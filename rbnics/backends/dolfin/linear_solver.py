# Copyright (C) 2015-2017 by the RBniCS authors
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
from rbnics.backends.abstract import LinearSolver as AbstractLinearSolver
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.vector import Vector
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.wrapping.dirichlet_bc import ProductOutputDirichletBC
from rbnics.utils.decorators import BackendFor, dict_of, Extends, list_of, override

@Extends(AbstractLinearSolver)
@BackendFor("dolfin", inputs=((Matrix.Type(), Form), Function.Type(), (Vector.Type(), Form), (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)), dict_of(str, ProductOutputDirichletBC), None)))
class LinearSolver(AbstractLinearSolver):
    @override
    def __init__(self, lhs, solution, rhs, bcs=None):
        self.solution = solution
        # Store lhs
        assert isinstance(lhs, (Matrix.Type(), Form))
        if isinstance(lhs, Matrix.Type()):
            if bcs is not None:
                # Create a copy of lhs, in order not to change
                # the original references when applying bcs
                self.lhs = lhs.copy()
            else:
                self.lhs = lhs
        elif isinstance(lhs, Form):
            self.lhs = assemble(lhs, keep_diagonal=True)
        else:
            raise AssertionError("Invalid lhs provided to dolfin LinearSolver")
        # Store rhs
        assert isinstance(rhs, (Vector.Type(), Form))
        if isinstance(rhs, Vector.Type()):
            if bcs is not None:
                # Create a copy of rhs, in order not to change
                # the original references when applying bcs
                self.rhs = rhs.copy()
            else:
                self.rhs = rhs
        elif isinstance(rhs, Form):
            self.rhs = assemble(rhs)
        else:
            raise AssertionError("Invalid rhs provided to dolfin LinearSolver")
        # Store and apply BCs
        self.bcs = bcs
        if bcs is not None:
            # Apply BCs
            assert isinstance(self.bcs, (dict, list))
            if isinstance(self.bcs, list):
                for bc in self.bcs:
                    assert isinstance(bc, DirichletBC)
                    bc.apply(self.lhs, self.rhs)
            elif isinstance(self.bcs, dict):
                for key in self.bcs:
                    for bc in self.bcs[key]:
                        assert isinstance(bc, DirichletBC)
                        bc.apply(self.lhs, self.rhs)
            else:
                raise AssertionError("Invalid type for bcs.")
            
    @override
    def set_parameters(self, parameters):
        assert len(parameters) == 0, "dolfin linear solver does not accept parameters yet"
        
    @override
    def solve(self):
        solver = PETScLUSolver("mumps")
        solver.solve(self.lhs, self.solution.vector(), self.rhs)
        return self.solution
        
