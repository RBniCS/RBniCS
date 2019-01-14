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

import os
from numbers import Number
from numpy import isclose
from rbnics.utils.decorators import dict_of, list_of, overload, tuple_of
from rbnics.utils.io import CSVIO, TextIO

def diff(reference_file, current_file):
    reference_ext = os.path.splitext(reference_file)[1]
    current_ext = os.path.splitext(current_file)[1]
    assert reference_ext == current_ext
    if reference_ext == ".txt":
        return _diff_txt(reference_file, current_file)
    elif reference_ext == ".csv":
        return _diff_csv(reference_file, current_file)
    else:
        raise ValueError("Invalid argument to diff")
    
def _diff_txt(reference_file, current_file):
    reference_content = TextIO.load_file("", reference_file)
    current_content = TextIO.load_file("", current_file)
    return _diff_content(reference_content, current_content, "")
    
def _diff_csv(reference_file, current_file):
    reference_lines = CSVIO.load_file("", reference_file)
    current_lines = CSVIO.load_file("", current_file)
    return _diff_content(reference_lines, current_lines, "")
    
@overload
def _diff_content(reference_items: (list_of(object), tuple_of(object)), current_items: (list_of(object), tuple_of(object)), tab: str):
    if len(reference_items) != len(current_items):
        return [
            tab + "@@ different lengths @@" + "\n" +
            tab + "- " + str(len(reference_items)) + "\n" +
            tab + "+ " + str(len(current_items)) + "\n"
        ]
    else:
        diff_items = list()
        for (item_number, (reference_item, current_item)) in enumerate(zip(reference_items, current_items)):
            diff_item = _diff_content(reference_item, current_item, tab + "\t")
            if len(diff_item) > 0:
                for d in diff_item:
                    diff_items.append(
                        tab + "@@ " + str(item_number) + " @@" + "\n" +
                        d
                    )
        return diff_items
        
@overload
def _diff_content(reference_items: dict_of(object, object), current_items: dict_of(object, object), tab: str):
    if len(reference_items) != len(current_items):
        return [
            tab + "@@ different lengths @@" + "\n" +
            tab + "- " + str(len(reference_items)) + "\n" +
            tab + "+ " + str(len(current_items)) + "\n"
        ]
    elif reference_items.keys() != current_items.keys():
        return [
            tab + "@@ different keys @@" + "\n" +
            tab + "- " + str(reference_items.keys()) + "\n" +
            tab + "+ " + str(current_items.keys()) + "\n"
        ]
    else:
        diff_items = list()
        for item_key in reference_items:
            diff_item = _diff_content(reference_items[item_key], current_items[item_key], tab + "\t")
            if len(diff_item) > 0:
                for d in diff_item:
                    diff_items.append(
                        tab + "@@ " + str(item_key) + " @@" + "\n" +
                        d
                    )
        return diff_items
        
@overload
def _diff_content(reference_item: str, current_item: str, tab: str):
    try:
        reference_item = float(reference_item)
        current_item = float(current_item)
    except ValueError:
        assert isinstance(reference_item, str)
        assert isinstance(current_item, str)
        if reference_item != current_item:
            return [
                tab + "- " + reference_item + "\n" +
                tab + "+ " + current_item + "\n"
            ]
        else:
            return []
    else:
        assert isinstance(reference_item, float)
        assert isinstance(current_item, float)
        return _diff_content(reference_item, current_item, tab)
        
@overload
def _diff_content(reference_item: Number, current_item: Number, tab: str):
    if not isclose(reference_item, current_item):
        return [
            tab + "- " + str(reference_item) + "\n" +
            tab + "+ " + str(current_item) + "\n"
        ]
    else:
        return []
