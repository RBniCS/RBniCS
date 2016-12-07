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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from ufl.core.operator import Operator
from dolfin import FunctionSpace
from RBniCS.backends.basic import FunctionsList as BasicFunctionsList
import RBniCS.backends.fenics
import RBniCS.backends.fenics.wrapping
from RBniCS.backends.fenics.wrapping_utils import function_from_ufl_operators
import RBniCS.backends.numpy
from RBniCS.utils.decorators import BackendFor, Extends, override

@Extends(BasicFunctionsList)
@BackendFor("fenics", online_backend="numpy", inputs=(FunctionSpace, ))
class FunctionsList(BasicFunctionsList):
    @override
    def __init__(self, V_or_Z):
        BasicFunctionsList.__init__(self, V_or_Z, RBniCS.backends.fenics, RBniCS.backends.fenics.wrapping, RBniCS.backends.numpy, AdditionalFunctionTypes=(Operator, ))
        
    @override
    def _enrich(self, function, component=None, weight=None, copy=True):
        function = function_from_ufl_operators(function)
        BasicFunctionsList._enrich(self, function, component, weight, copy)
        
    @override
    def __setitem__(self, key, item):
        item = function_from_ufl_operators(item)
        BasicFunctionsList.__setitem__(self, key, item)
        
