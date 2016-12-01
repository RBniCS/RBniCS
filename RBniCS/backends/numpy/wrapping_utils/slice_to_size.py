# Copyright (C) 2015-2016 by the RBniCS authors
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
    
    size = list()
    for slice_ in key:
        assert slice_.start is None
        assert slice_.step is None
        assert isinstance(slice_.stop, (int, dict))
        if isinstance(slice_.stop, int):
            assert isinstance(length_dict, dict) or length_dict is None
            if length_dict is None:
                size.append(slice_.stop)
            else:
                assert len(length_dict) == 1
                current_size = dict()
                current_size[length_dict.keys()[0]] = slice_.stop
                size.append(current_size)
        else:
            assert isinstance(length_dict, dict)
            assert length_dict.keys() == slice_.stop.keys()
            current_size = dict()
            for (component_name, component_size) in slice_.stop.iteritems():
                current_size[component_name] = component_size
            size.append(current_size)
    return size

    
