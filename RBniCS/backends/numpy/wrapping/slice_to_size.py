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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

def slice_to_size(key, length_dict=None):
    if isinstance(key, slice):
        key = (key,)
    assert isinstance(key, tuple)
    assert isinstance(key[0], slice)
    
    if length_dict is None:
        length_dict = (None, )*len(key)
    elif isinstance(length_dict, dict):
        length_dict = (length_dict, )
    assert isinstance(length_dict, tuple)
    assert isinstance(length_dict[0], dict) or length_dict[0] is None
    
    assert len(key) == len(length_dict)
    
    size = list()
    for (slice_index, slice_) in enumerate(key):
        assert slice_.start is None
        assert slice_.step is None
        assert isinstance(slice_.stop, (int, dict))
        if isinstance(slice_.stop, int):
            assert isinstance(length_dict[slice_index], dict) or length_dict[slice_index] is None
            if length_dict[slice_index] is None:
                size.append(slice_.stop)
            else:
                assert len(length_dict[slice_index]) == 1
                current_size = dict()
                current_size[length_dict[slice_index].keys()[0]] = slice_.stop
                size.append(current_size)
        else:
            assert isinstance(length_dict[slice_index], dict)
            assert length_dict[slice_index].keys() == slice_.stop.keys()
            current_size = dict()
            for (component_name, component_size) in slice_.stop.iteritems():
                current_size[component_name] = component_size
            size.append(current_size)
    return size

    
