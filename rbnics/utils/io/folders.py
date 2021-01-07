# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from rbnics.utils.decorators import overload
from rbnics.utils.mpi import parallel_io


class Folders(dict):  # dict from string to string

    # Auxiliary class
    class Folder(object):
        @overload(str)
        def __init__(self, name):
            self.name = name

        @overload(lambda cls: cls)
        def __init__(self, name):
            self.name = name.name

        # Returns True if it was necessary to create the folder
        # or if the folder was already created before, but it is
        # empty. Returs False otherwise.
        def create(self):
            def create_task():
                if os.path.exists(self.name) and len(os.listdir(self.name)) == 0:  # already created, but empty
                    return True
                if not os.path.exists(self.name):  # to be created
                    os.makedirs(self.name)
                    return True
                return False
            return parallel_io(create_task)

        def touch_file(self, filename):
            def touch_file_task():
                with open(os.path.join(self.name, filename), "a"):
                    os.utime(os.path.join(self.name, filename), None)
            parallel_io(touch_file_task)

        def __str__(self):
            return self.name

        def __repr__(self):
            return self.name

        def __add__(self, suffix):
            return Folders.Folder(str(self) + suffix)

        def __radd__(self, prefix):
            return Folders.Folder(prefix + str(self))

        def replace(self, old, new):
            return Folders.Folder(str(self).replace(old, new))

    def __init__(self, *args):
        dict.__init__(self, args)

    def __getitem__(self, key):
        # this will return a Folder object
        return dict.__getitem__(self, key)

    def __setitem__(self, key, val):
        # takes a string and initialize a Folder from it
        dict.__setitem__(self, key, Folders.Folder(val))

    # Returns True if it was necessary to create *at least* a folder
    # or if *at least* a folder was already created before, but it is
    # empty. Returs False otherwise.
    def create(self):
        global_return_value = False
        for key in self:
            return_value = self[key].create()
            global_return_value = global_return_value or return_value
        return global_return_value
