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

from RBniCS.backends.abstract import FunctionsList as AbstractFunctionsList
from RBniCS.backends.basic import ProperOrthogonalDecomposition as BasicProperOrthogonalDecomposition
import RBniCS.backends.numpy
from RBniCS.backends.numpy.matrix import Matrix
from RBniCS.utils.decorators import BackendFor, Extends, override

@Extends(BasicProperOrthogonalDecomposition)
@BackendFor("NumPy", inputs=(Matrix.Type(), AbstractFunctionsList))
class ProperOrthogonalDecomposition(BasicProperOrthogonalDecomposition):
    @override
    def __init__(self, X, V_or_Z):
        BasicProperOrthogonalDecomposition.__init__(self, X, V_or_Z, RBniCS.backends.numpy)
        
