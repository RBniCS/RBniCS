# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import sys
from numpy import cumsum
from rbnics.utils.io import ComponentNameToBasisComponentIndexDict, OnlineSizeDict


def slice_to_array(obj, key, length_dict, index_dict):
    key = _check_key(obj, key)
    length_dict = _check_length_dict(key, length_dict)
    index_dict = _check_index_dict(key, index_dict)

    slices_start = list()
    slices_stop = list()
    for (slice_index, slice_) in enumerate(key):
        assert isinstance(slice_.start, (int, OnlineSizeDict))
        assert slice_.step is None
        assert isinstance(slice_.stop, (int, OnlineSizeDict))
        assert isinstance(slice_.start, int) == isinstance(slice_.stop, int)
        if isinstance(slice_.stop, int):
            slices_start.append(slice_.start)
            slices_stop.append(slice_.stop)
        else:
            if len(index_dict[slice_index]) > 1:
                current_slice_length = [0] * len(index_dict[slice_index])
                for (component_name, basis_component_index) in index_dict[slice_index].items():
                    current_slice_length[basis_component_index] = length_dict[slice_index][component_name]
                current_slice_length_cumsum = [0] + cumsum(current_slice_length).tolist()[:-1]
            else:
                current_slice_length_cumsum = [0]
            current_slice_start = [0] * len(index_dict[slice_index])
            for (component_name, basis_component_index) in index_dict[slice_index].items():
                current_slice_start[basis_component_index] = current_slice_length_cumsum[
                    basis_component_index] + slice_.start[component_name]
            current_slice_stop = [0] * len(index_dict[slice_index])
            for (component_name, basis_component_index) in index_dict[slice_index].items():
                current_slice_stop[basis_component_index] = current_slice_length_cumsum[
                    basis_component_index] + slice_.stop[component_name]
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
            for (current_slice_start_component, current_slice_stop_component) in zip(
                    current_slice_start, current_slice_stop):
                current_slice.extend(list(range(current_slice_start_component, current_slice_stop_component)))
            slices.append(tuple(current_slice))

    assert len(slices) > 0
    if len(slices) == 1:
        return slices[0]
    else:
        return tuple(slices)


def _check_key(obj, key):
    if not isinstance(key, tuple):
        key = (key,)
    assert isinstance(key, tuple)
    assert all([isinstance(key_i, slice) for key_i in key])

    def shape_attribute(slice_index):
        return getattr(obj, _slice_shape_attribute[len(key)][slice_index])

    converted_key = list()
    for (slice_index, slice_) in enumerate(key):
        shape_index = shape_attribute(slice_index)
        assert isinstance(slice_.start, (int, OnlineSizeDict)) or slice_.start is None
        if ((isinstance(slice_.start, int) and slice_.start == 0)
                or slice_.start is None):
            assert isinstance(shape_index, (int, OnlineSizeDict))
            if isinstance(shape_index, int):
                start = 0
            elif isinstance(shape_index, OnlineSizeDict):
                start = OnlineSizeDict()
                for component_name in shape_index.keys():
                    start[component_name] = 0
            else:
                raise TypeError("Invalid shape")
        else:
            start = slice_.start
        assert slice_.step is None
        step = slice_.step
        assert isinstance(slice_.stop, (int, OnlineSizeDict)) or slice_.stop is None
        if ((isinstance(slice_.stop, int) and slice_.stop == sys.maxsize)
                or slice_.stop is None):
            stop = shape_index
        else:
            stop = slice_.stop
        converted_slice = slice(start, stop, step)
        converted_key.append(converted_slice)
    return converted_key


_slice_shape_attribute = {
    1: {
        0: "N",
    },
    2: {
        0: "M",
        1: "N"
    }
}


def _check_length_dict(key, length_dict):
    if length_dict is None:
        length_dict = (None, ) * len(key)
    elif isinstance(length_dict, OnlineSizeDict):
        length_dict = (length_dict, )
    assert isinstance(length_dict, tuple)
    assert all([isinstance(length_dict_i, OnlineSizeDict) or length_dict_i is None for length_dict_i in length_dict])
    assert len(key) == len(length_dict)
    return length_dict


def _check_index_dict(key, index_dict):
    if index_dict is None:
        index_dict = (None, ) * len(key)
    elif isinstance(index_dict, ComponentNameToBasisComponentIndexDict):
        index_dict = (index_dict, )
    assert isinstance(index_dict, tuple)
    assert all([isinstance(
        index_dict_i, ComponentNameToBasisComponentIndexDict) or index_dict_i is None for index_dict_i in index_dict])
    assert len(key) == len(index_dict)
    return index_dict
