# Copyright (C) 2015-2018 by the RBniCS authors
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

class _NonlinearProblem(object):
    def __init__(self, residual_eval, solution, bcs, jacobian_eval):
        self.residual_eval = residual_eval
        self.solution = solution
        self.jacobian_eval = jacobian_eval
        # Preserve solution auxiliary attributes
        self.residual_vector = residual_eval(solution)
        self.jacobian_matrix = jacobian_eval(solution)
        preserve_solution_attributes(self.jacobian_matrix, self.solution, self.residual_vector)
        # Initialize BCs
        self._init_bcs(bcs)
    
    @overload
    def _init_bcs(self, bcs: None):
        self.bcs = None
        
    @overload
    def _init_bcs(self, bcs: ThetaType):
        self.bcs = DirichletBC(bcs)
        
    @overload
    def _init_bcs(self, bcs: DictOfThetaType):
        self.bcs = DirichletBC(bcs, self.residual_vector._basis_component_index_to_component_name, self.solution.vector().N)
