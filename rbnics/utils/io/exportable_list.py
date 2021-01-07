# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.io.numpy_io import NumpyIO
from rbnics.utils.io.pickle_io import PickleIO
from rbnics.utils.io.text_io import TextIO
from rbnics.utils.decorators import list_of, overload


class ExportableList(object):
    def __init__(self, import_export_backend, original_list=None):
        self._list = list()
        if import_export_backend == "numpy":
            self._FileIO = NumpyIO
        elif import_export_backend == "pickle":
            self._FileIO = PickleIO
        elif import_export_backend == "text":
            self._FileIO = TextIO
        else:
            raise ValueError("Invalid import/export backend")
        if original_list is not None:
            self._list.extend(original_list)

    def append(self, element):
        self._list.append(element)

    @overload(list_of(object))
    def extend(self, other_list):
        self._list.extend(other_list)

    @overload(lambda cls: cls)
    def extend(self, other_list):
        self._list.extend(other_list._list)

    def clear(self):
        self._list.clear()

    def save(self, directory, filename):
        self._FileIO.save_file(self._list, directory, filename)

    # Returns False if the list had been already imported so no further
    # action was needed.
    # Returns True if it was possible to import the list.
    # Raises an error if it was not possible to import the list.
    def load(self, directory, filename):
        if self._list:  # avoid loading multiple times
            return False
        if self._FileIO.exists_file(directory, filename):
            self._list = self._FileIO.load_file(directory, filename)
            return True
        else:
            raise OSError

    def __getitem__(self, key):
        return self._list[key]

    def __setitem__(self, key, item):
        self._list[key] = item

    def __iter__(self):
        return iter(self._list)

    def __len__(self):
        return len(self._list)

    def __str__(self):
        return str(self._list)
