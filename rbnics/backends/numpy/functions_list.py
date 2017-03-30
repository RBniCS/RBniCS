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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from rbnics.backends.abstract import FunctionsList as AbstractFunctionsList
from rbnics.backends.basic import FunctionsList as BasicFunctionsList
import rbnics.backends.numpy
import rbnics.backends.numpy.wrapping
from rbnics.utils.decorators import BackendFor, Extends, override

@Extends(BasicFunctionsList)
@BackendFor("numpy", online_backend="numpy", inputs=(AbstractFunctionsList, (str, None)))
class FunctionsList(BasicFunctionsList):
    @override
    def __init__(self, Z, component=None):
        BasicFunctionsList.__init__(self, Z, component, rbnics.backends.numpy, rbnics.backends.numpy.wrapping, rbnics.backends.numpy)
        
