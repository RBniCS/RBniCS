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

from rbnics.backends.abstract import FunctionsList as AbstractFunctionsList
from rbnics.backends.abstract import HighOrderProperOrthogonalDecomposition as AbstractHighOrderProperOrthogonalDecomposition
from rbnics.backends.basic import ProperOrthogonalDecompositionBase as BasicHighOrderProperOrthogonalDecomposition
import rbnics.backends.online.numpy
from rbnics.utils.decorators import BackendFor

HighOrderProperOrthogonalDecompositionBase = BasicHighOrderProperOrthogonalDecomposition(AbstractHighOrderProperOrthogonalDecomposition)

@BackendFor("numpy", inputs=(AbstractFunctionsList, ))
class HighOrderProperOrthogonalDecomposition(HighOrderProperOrthogonalDecompositionBase):
    def __init__(self, Z, empty_tensor):
        HighOrderProperOrthogonalDecompositionBase.__init__(self, Z, None, empty_tensor, rbnics.backends.online.numpy, rbnics.backends.online.numpy.wrapping, rbnics.backends.online.numpy.TensorSnapshotsList, rbnics.backends.online.numpy.TensorBasisList)
        
    def store_snapshot(self, snapshot):
        self.snapshots_matrix.enrich(snapshot)
        
        
