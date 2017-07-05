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

from ufl.core.operator import Operator
from dolfin import FunctionSpace
from rbnics.backends.basic import FunctionsList as BasicFunctionsList
import rbnics.backends.fenics
from rbnics.backends.fenics.wrapping import function_from_ufl_operators
import rbnics.backends.numpy
from rbnics.utils.decorators import BackendFor, Extends, override

@Extends(BasicFunctionsList)
@BackendFor("fenics", online_backend="numpy", inputs=(FunctionSpace, (str, None)))
class FunctionsList(BasicFunctionsList):
    @override
    def __init__(self, V, component=None):
        BasicFunctionsList.__init__(self, V, component, rbnics.backends.fenics, rbnics.backends.fenics.wrapping, rbnics.backends.numpy, AdditionalFunctionTypes=(Operator, ))
        
    @override
    def _enrich(self, function, component=None, weight=None, copy=True):
        function = function_from_ufl_operators(function)
        BasicFunctionsList._enrich(self, function, component, weight, copy)
        
    @override
    def __setitem__(self, key, item):
        item = function_from_ufl_operators(item)
        BasicFunctionsList.__setitem__(self, key, item)
        
