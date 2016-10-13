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

def slice_to_array(key, obj):
    if isinstance(key, slice):
        key = (key,)
        
    assert isinstance(key, tuple)
    assert isinstance(key[0], slice)
    
    slices_start = list()
    slices_stop = list()
    for slice_ in key:
        assert slice_.start is None 
        assert slice_.step is None
        assert isinstance(slice_.stop, (int, dict))
        if isinstance(slice_.stop, int):
            slices_start.append(0)
            slices_stop.append(slice_.stop)
        else:
            assert obj._basis_component_index_to_component_name is not None
            assert obj._component_name_to_basis_component_index is not None
            assert obj._component_name_to_basis_component_length is not None
            current_slice_start = [0]*len(obj._component_name_to_basis_component_index)
            current_slice_stop  = [0]*len(obj._component_name_to_basis_component_index)
            for (component_name, basis_component_index) in obj._component_name_to_basis_component_index.iteritems():
                current_slice_start[basis_component_index] = obj._component_name_to_basis_component_length[component_name]
                current_slice_stop[basis_component_index]  = current_slice_start[basis_component_index] + slice_.stop[component_name]
            slices_start.append(current_slice_start)
            slices_stop .append(current_slice_stop )
            
    slices = list()
    assert len(slices_start) == len(slices_stop)
    for (current_slice_start, current_slice_stop) in zip(slices_start, slices_stop):
        assert isinstance(current_slice_start, int) == isinstance(current_slice_stop, int)
        if isinstance(current_slice_start, int):
            slices.append(tuple(range(current_slice_start, current_slice_stop)))
        else:
            current_slice = list()
            for (current_slice_start_component, current_slice_stop_component) in zip(current_slice_start, current_slice_stop):
                current_slice.extend(range(current_slice_start_component, current_slice_stop_component))
            slices.append(tuple(current_slice))
    slices = tuple(slices)
    return slices
    
