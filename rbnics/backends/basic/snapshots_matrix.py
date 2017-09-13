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

from numbers import Number
from rbnics.utils.decorators import dict_of, list_of, overload

def SnapshotsMatrix(FunctionsList):
    class _SnapshotsMatrix(FunctionsList):
        
        @overload(FunctionsList, (None, str, dict_of(str, str)), (None, list_of(Number)), bool)
        def _enrich(self, functions, component, weights, copy):
            if weights is not None:
                assert len(weights) == len(functions)
                for (index, function) in enumerate(functions):
                    self._add_to_list(function, component, weights[index], copy)
            else:
                for function in functions:
                    self._add_to_list(function, component, None, copy)
                    
    return _SnapshotsMatrix
