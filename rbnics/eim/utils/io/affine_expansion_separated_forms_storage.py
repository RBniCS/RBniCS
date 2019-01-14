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

from numpy import empty as AffineExpansionSeparatedFormsStorageContent_Base

class AffineExpansionSeparatedFormsStorage(object):
    def __init__(self, Q):
        self._content = AffineExpansionSeparatedFormsStorageContent_Base((Q,), dtype=object)
        
    def __getitem__(self, key):
        return self._content[key]
        
    def __setitem__(self, key, item):
        self._content[key] = item
        
    def __len__(self):
        return self._content.size
