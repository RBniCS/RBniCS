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

from ufl.core.multiindex import MultiIndex
from ufl.indexed import Indexed
from ufl.tensors import ListTensor
from dolfin import Function

def function_from_subfunction_if_any(node):
    if isinstance(node, Function):
        return node
    elif isinstance(node, Indexed):
        if len(node.ufl_operands) == 2 and isinstance(node.ufl_operands[0], Function) and isinstance(node.ufl_operands[1], MultiIndex):
            return node.ufl_operands[0]
    elif isinstance(node, ListTensor):
        if (
            all(isinstance(component, Indexed) for component in node.ufl_operands)
                and
            all(
              (len(component.ufl_operands) == 2 and isinstance(component.ufl_operands[0], Function) and isinstance(component.ufl_operands[1], MultiIndex))
              for component in node.ufl_operands
            )
                and
            all(
              component.ufl_operands[0] == node.ufl_operands[-1].ufl_operands[0]
              for component in node.ufl_operands
            )
        ):
            return node.ufl_operands[-1].ufl_operands[0]
        else:
            return node
    else:
        return node
        
