# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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
