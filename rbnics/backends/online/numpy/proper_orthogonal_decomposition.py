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
from rbnics.backends.abstract import ProperOrthogonalDecomposition as AbstractProperOrthogonalDecomposition
from rbnics.backends.basic import ProperOrthogonalDecompositionBase as BasicProperOrthogonalDecomposition
import rbnics.backends.online.numpy
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.utils.decorators import BackendFor, Extends

ProperOrthogonalDecompositionBase = BasicProperOrthogonalDecomposition(AbstractProperOrthogonalDecomposition)

@Extends(ProperOrthogonalDecompositionBase)
@BackendFor("numpy", inputs=(AbstractFunctionsList, Matrix.Type(), (str, None)))
class ProperOrthogonalDecomposition(ProperOrthogonalDecompositionBase):
    def __init__(self, Z, X, component=None):
        ProperOrthogonalDecompositionBase.__init__(self, Z, X, component, rbnics.backends.online.numpy, rbnics.backends.online.numpy.wrapping, rbnics.backends.online.numpy.SnapshotsMatrix, rbnics.backends.online.numpy.FunctionsList)
        
    def store_snapshot(self, snapshot, component=None, weight=None):
        self.snapshots_matrix.enrich(snapshot, component, weight)
        
