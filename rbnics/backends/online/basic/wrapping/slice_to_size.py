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

from rbnics.backends.online.basic.wrapping.slice_to_array import _check_key, _check_length_dict
from rbnics.utils.io import OnlineSizeDict

def slice_to_size(obj, key, length_dict):
    key = _check_key(obj, key)
    length_dict = _check_length_dict(key, length_dict)
    
    size = list()
    for (slice_index, slice_) in enumerate(key):
        assert isinstance(slice_.start, (int, OnlineSizeDict))
        assert slice_.step is None
        assert isinstance(slice_.stop, (int, OnlineSizeDict))
        assert isinstance(slice_.start, int) == isinstance(slice_.stop, int)
        if isinstance(slice_.stop, int):
            assert isinstance(length_dict[slice_index], OnlineSizeDict) or length_dict[slice_index] is None
            if length_dict[slice_index] is None:
                size.append(slice_.stop - slice_.start)
            else:
                assert len(length_dict[slice_index]) == 1
                for (component_name, _) in length_dict[slice_index].items():
                    break
                current_size = OnlineSizeDict()
                current_size[component_name] = slice_.stop - slice_.start
                size.append(current_size)
        else:
            assert isinstance(length_dict[slice_index], OnlineSizeDict)
            assert length_dict[slice_index].keys() == slice_.start.keys()
            assert length_dict[slice_index].keys() == slice_.stop.keys()
            current_size = OnlineSizeDict()
            for component_name in length_dict[slice_index].keys():
                current_size[component_name] = slice_.stop[component_name] - slice_.start[component_name]
            size.append(current_size)
    return size
