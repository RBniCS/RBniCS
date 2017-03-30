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
from rbnics.backends.abstract import HighOrderProperOrthogonalDecomposition as AbstractHighOrderProperOrthogonalDecomposition
from rbnics.backends.basic import ProperOrthogonalDecompositionBase as BasicHighOrderProperOrthogonalDecomposition
import rbnics.backends.numpy
import rbnics.backends.numpy.wrapping
from rbnics.utils.decorators import BackendFor, Extends, override

HighOrderProperOrthogonalDecompositionBase = BasicHighOrderProperOrthogonalDecomposition(AbstractHighOrderProperOrthogonalDecomposition)

@Extends(HighOrderProperOrthogonalDecompositionBase)
@BackendFor("numpy", inputs=(AbstractFunctionsList, ))
class HighOrderProperOrthogonalDecomposition(HighOrderProperOrthogonalDecompositionBase):
    @override
    def __init__(self, Z, empty_tensor):
        HighOrderProperOrthogonalDecompositionBase.__init__(self, Z, None, empty_tensor, rbnics.backends.numpy, rbnics.backends.numpy.wrapping, rbnics.backends.numpy.TensorSnapshotsList, rbnics.backends.numpy.TensorBasisList)
        
    @override
    def store_snapshot(self, snapshot):
        self.snapshots_matrix.enrich(snapshot)
        
        
