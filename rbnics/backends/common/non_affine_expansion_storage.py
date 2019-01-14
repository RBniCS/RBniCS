# Copyright (C) 2015-2019 by the RBniCS authors
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

from numbers import Number
from rbnics.backends.abstract import NonAffineExpansionStorage as AbstractNonAffineExpansionStorage
from rbnics.utils.decorators import BackendFor, tuple_of

@BackendFor("common", inputs=(tuple_of(Number),))
class NonAffineExpansionStorage(AbstractNonAffineExpansionStorage):
    def __init__(self, args):
        self._content = args
        
    def __getitem__(self, key):
        return self._content[key]
        
    def __iter__(self):
        return self._content.__iter__()
        
    def __len__(self):
        assert self._content is not None
        return len(self._content)
