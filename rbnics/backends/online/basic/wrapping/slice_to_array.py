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

import sys
from numpy import cumsum

def slice_to_array(obj, key, length_dict, index_dict):
    key = _check_key(obj, key)
    length_dict = _check_length_dict(key, length_dict)
    index_dict = _check_index_dict(key, index_dict)
    
    slices_start = list()
    slices_stop = list()
    for (slice_index, slice_) in enumerate(key):
        assert slice_.start is None
        assert slice_.step is None
        assert isinstance(slice_.stop, (int, dict))
        if isinstance(slice_.stop, int):
            slices_start.append(0)
            slices_stop.append(slice_.stop)
        else:
            current_slice_length = [0]*len(index_dict[slice_index])
            for (component_name, basis_component_index) in index_dict[slice_index].items():
                current_slice_length[basis_component_index] = length_dict[slice_index][component_name]
            current_slice_length_cumsum = cumsum(current_slice_length).tolist()
            del current_slice_length_cumsum[-1]
            current_slice_start = [0]
            current_slice_start.extend(current_slice_length_cumsum)
            current_slice_stop = [0]*len(index_dict[slice_index])
            for (component_name, basis_component_index) in index_dict[slice_index].items():
                current_slice_stop[basis_component_index] = current_slice_start[basis_component_index] + slice_.stop[component_name]
            assert len(current_slice_start) == len(current_slice_stop)
            slices_start.append(current_slice_start)
            slices_stop.append(current_slice_stop)
            
    slices = list()
    assert len(slices_start) == len(slices_stop)
    for (current_slice_start, current_slice_stop) in zip(slices_start, slices_stop):
        assert isinstance(current_slice_start, int) == isinstance(current_slice_stop, int)
        if isinstance(current_slice_start, int):
            slices.append(tuple(range(current_slice_start, current_slice_stop)))
        else:
            current_slice = list()
            for (current_slice_start_component, current_slice_stop_component) in zip(current_slice_start, current_slice_stop):
                current_slice.extend(list(range(current_slice_start_component, current_slice_stop_component)))
            slices.append(tuple(current_slice))
    
    assert len(slices) > 0
    if len(slices) is 1:
        return slices[0]
    else:
        return tuple(slices)
    
def _check_key(obj, key):
    if not isinstance(key, tuple):
        key = (key,)
    assert isinstance(key, tuple)
    assert all([isinstance(key_i, slice) for key_i in key])
    converted_key = list()
    for (slice_index, slice_) in enumerate(key):
        assert slice_.start is None or slice_.start == 0
        if slice_.start == 0:
            start = None
        else:
            start = slice_.start
        assert slice_.step is None
        step = slice_.step
        assert isinstance(slice_.stop, (int, dict)) or slice_.stop is None
        if (
            (isinstance(slice_.stop, int) and slice_.stop == sys.maxsize)
                or
            slice_.stop is None
        ):
            stop = obj.shape[slice_index]
        else:
            stop = slice_.stop
        converted_slice = slice(start, stop, step)
        converted_key.append(converted_slice)
    return converted_key
    
def _check_length_dict(key, length_dict):
    if length_dict is None:
        length_dict = (None, )*len(key)
    elif isinstance(length_dict, dict):
        length_dict = (length_dict, )
    assert isinstance(length_dict, tuple)
    assert all([isinstance(length_dict_i, dict) or length_dict_i is None for length_dict_i in length_dict])
    assert len(key) == len(length_dict)
    return length_dict
    
def _check_index_dict(key, index_dict):
    if index_dict is None:
        index_dict = (None, )*len(key)
    elif isinstance(index_dict, dict):
        index_dict = (index_dict, )
    assert isinstance(index_dict, tuple)
    assert all([isinstance(index_dict_i, dict) or index_dict_i is None for index_dict_i in index_dict])
    assert len(key) == len(index_dict)
    return index_dict
