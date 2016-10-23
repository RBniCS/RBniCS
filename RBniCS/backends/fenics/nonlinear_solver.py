# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file solve.py
#  @brief solve function for the solution of a linear system, similar to FEniCS' solve
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import types
from dolfin import DirichletBC, NonlinearVariationalProblem, NonlinearVariationalSolver
from RBniCS.backends.abstract import NonlinearSolver as AbstractNonlinearSolver
from RBniCS.backends.fenics.function import Function
from RBniCS.utils.decorators import BackendFor, Extends, list_of, override

@Extends(AbstractNonlinearSolver)
@BackendFor("FEniCS", inputs=(types.FunctionType, Function.Type(), types.FunctionType, (list_of(DirichletBC), None)))
class NonlinearSolver(AbstractNonlinearSolver):
    @override
    def __init__(self, jacobian_form_eval, solution, residual_form_eval, bcs=None):
        """
            Signatures:
                def jacobian_form_eval(solution):
                    return grad(u)*grad(v)
                
                def residual_form_eval(solution):
                    return grad(solution)*grad(v)*dx - f*v*dx
        """
        jacobian_form = jacobian_form_eval(solution)
        residual_form = residual_form_eval(solution)
        problem = NonlinearVariationalProblem(residual_form, solution, bcs, jacobian_form)
        self.solver  = NonlinearVariationalSolver(problem)
        self.solution = solution
            
    @override
    def set_parameters(self, parameters):
        self.solver.parameters.update(parameters)
        
    @override
    def solve(self):
        self.solver.solve()
        return self.solution
    
