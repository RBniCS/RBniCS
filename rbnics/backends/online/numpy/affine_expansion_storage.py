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

from numpy import asmatrix as AffineExpansionStorageContent_AsMatrix
from rbnics.backends.abstract import AffineExpansionStorage as AbstractAffineExpansionStorage
from rbnics.backends.online.basic import AffineExpansionStorage as BasicAffineExpansionStorage
import rbnics.backends.online.numpy
import rbnics.backends.online.numpy.wrapping
from rbnics.backends.online.numpy.matrix import Matrix as OnlineMatrix
from rbnics.backends.online.numpy.vector import Vector as OnlineVector
from rbnics.utils.decorators import BackendFor, Extends, list_of, override, tuple_of

@Extends(BasicAffineExpansionStorage)
@BackendFor("numpy", inputs=((int, tuple_of(OnlineMatrix.Type()), tuple_of(OnlineVector.Type()), AbstractAffineExpansionStorage), (int, None)))
class AffineExpansionStorage(BasicAffineExpansionStorage):
    @override
    def __init__(self, arg1, arg2=None):
        BasicAffineExpansionStorage.__init__(self, arg1, arg2, rbnics.backends.online.numpy, rbnics.backends.online.numpy.wrapping)
        # Additional storage
        self._content_as_matrix = None
        if isinstance(arg1, AbstractAffineExpansionStorage):
            self._content_as_matrix = arg1._content_as_matrix
            
    @override
    def __setitem__(self, key, item):
        BasicAffineExpansionStorage.__setitem__(self, key, item)
        # Reset internal copies
        self._content_as_matrix = None
            
    @override
    def load(self, directory, filename):
        BasicAffineExpansionStorage.load(self, directory, filename)
        # Create internal copy as matrix
        self._content_as_matrix = None
        self.as_matrix()
        
    def as_matrix(self):
        if self._content_as_matrix is None:
            self._content_as_matrix = AffineExpansionStorageContent_AsMatrix(self._content)
        return self._content_as_matrix
