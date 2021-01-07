# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod, abstractonlinemethod


@AbstractBackend
class NonAffineExpansionStorage(object, metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractonlinemethod
    def save(self, directory, filename):
        pass

    @abstractonlinemethod
    def load(self, directory, filename):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractonlinemethod
    def __setitem__(self, key, item):
        pass

    @abstractmethod
    def __len__(self):
        pass
