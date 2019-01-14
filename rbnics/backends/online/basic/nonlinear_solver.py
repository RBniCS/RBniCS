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

from rbnics.backends.online.basic.wrapping import DirichletBC, preserve_solution_attributes
from rbnics.utils.decorators import DictOfThetaType, overload, ThetaType

def _NonlinearProblem(backend, wrapping):
    class _NonlinearProblem_Class(object):
        def __init__(self, residual_eval, solution, bcs, jacobian_eval):
            self.residual_eval_callback = residual_eval
            self.solution = solution
            self.jacobian_eval_callback = jacobian_eval
            # Preserve solution auxiliary attributes
            self.residual_vector = self.residual_eval(solution)
            self.jacobian_matrix = self.jacobian_eval(solution)
            preserve_solution_attributes(self.jacobian_matrix, self.solution, self.residual_vector)
            # Initialize BCs
            self._init_bcs(bcs)
            if self.bcs is not None:
                self.bcs.apply_to_vector(self.solution.vector())
        
        @overload
        def _init_bcs(self, bcs: None):
            self.bcs = None
            
        @overload
        def _init_bcs(self, bcs: ThetaType):
            self.bcs = DirichletBC(bcs)
            
        @overload
        def _init_bcs(self, bcs: DictOfThetaType):
            self.bcs = DirichletBC(bcs, self.residual_vector._component_name_to_basis_component_index, self.solution.vector().N)
            
        def residual_eval(self, solution):
            output = self.residual_eval_callback(solution)
            assert isinstance(output, (backend.Vector.Type(), wrapping.DelayedTransposeWithArithmetic))
            if isinstance(output, backend.Vector.Type()):
                return output
            elif isinstance(output, wrapping.DelayedTransposeWithArithmetic):
                return output.evaluate()
            else:
                raise TypeError("Invalid residual")
                
        def jacobian_eval(self, solution):
            output = self.jacobian_eval_callback(solution)
            assert isinstance(output, (backend.Matrix.Type(), wrapping.DelayedTransposeWithArithmetic))
            if isinstance(output, backend.Matrix.Type()):
                return output
            elif isinstance(output, wrapping.DelayedTransposeWithArithmetic):
                return output.evaluate()
            else:
                raise TypeError("Invalid residual")
                
    return _NonlinearProblem_Class
