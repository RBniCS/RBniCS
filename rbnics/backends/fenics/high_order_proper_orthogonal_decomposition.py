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

from dolfin import FunctionSpace
from rbnics.backends.abstract import HighOrderProperOrthogonalDecomposition as AbstractHighOrderProperOrthogonalDecomposition
from rbnics.backends.basic import ProperOrthogonalDecompositionBase as BasicHighOrderProperOrthogonalDecomposition
import rbnics.backends.fenics
from rbnics.utils.decorators import BackendFor, Extends, override

HighOrderProperOrthogonalDecompositionBase = BasicHighOrderProperOrthogonalDecomposition(AbstractHighOrderProperOrthogonalDecomposition)

@Extends(HighOrderProperOrthogonalDecompositionBase)
@BackendFor("fenics", inputs=(FunctionSpace, ))
class HighOrderProperOrthogonalDecomposition(HighOrderProperOrthogonalDecompositionBase):
    @override
    def __init__(self, V, empty_tensor):
        HighOrderProperOrthogonalDecompositionBase.__init__(self, V, None, empty_tensor, rbnics.backends.fenics, rbnics.backends.fenics.wrapping, rbnics.backends.fenics.TensorSnapshotsList, rbnics.backends.fenics.TensorBasisList)
        
    @override
    def store_snapshot(self, snapshot):
        self.snapshots_matrix.enrich(snapshot)
        
