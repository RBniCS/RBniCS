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
from rbnics.backends.common.time_series import TimeSeries
from rbnics.utils.decorators import backend_for, list_of, overload

@backend_for("common", inputs=((list_of(Number), TimeSeries), (list_of(Number), TimeSeries)))
def assign(object_to, object_from):
    _assign(object_to, object_from)
    
@overload
def _assign(object_to: TimeSeries, object_from: TimeSeries):
    if object_from is not object_to:
        from rbnics.backends import assign
        assign(object_to._list, object_from._list)
        
@overload
def _assign(object_to: list_of(Number), object_from: list_of(Number)):
    if object_from is not object_to:
        del object_to[:]
        object_to.extend(object_from)
