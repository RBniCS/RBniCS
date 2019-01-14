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
from ufl.core.operator import Operator
from dolfin import FunctionSpace
from rbnics.backends.basic import GramSchmidt as BasicGramSchmidt
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.transpose import transpose
from rbnics.backends.dolfin.wrapping import function_extend_or_restrict, function_from_ufl_operators, get_function_subspace, gram_schmidt_projection_step
from rbnics.utils.decorators import BackendFor, dict_of, ModuleWrapper, overload

backend = ModuleWrapper(Function, transpose)
wrapping = ModuleWrapper(function_extend_or_restrict, get_function_subspace, gram_schmidt_projection_step)
GramSchmidt_Base = BasicGramSchmidt(backend, wrapping)

@BackendFor("dolfin", inputs=(FunctionSpace, (Form, Matrix.Type()), (str, None)))
class GramSchmidt(GramSchmidt_Base):
    @overload(Operator, (None, str, dict_of(str, str)))
    def _extend_or_restrict_if_needed(self, function, component):
        function = function_from_ufl_operators(function)
        return GramSchmidt_Base._extend_or_restrict_if_needed(self, function, component)
