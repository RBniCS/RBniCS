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

from rbnics.backends.abstract import NonAffineExpansionStorage as AbstractNonAffineExpansionStorage

def NonAffineExpansionStorage(backend, wrapping):
    class _NonAffineExpansionStorage(AbstractNonAffineExpansionStorage):
        def __init__(self, content):
            self._content = tuple(backend.ParametrizedTensorFactory(op) for op in content)
            
        def __getitem__(self, key):
            return self._content[key]
            
        def __iter__(self):
            return iter(self._content)
            
        def __len__(self):
            return len(self._content)
    
    return _NonAffineExpansionStorage
