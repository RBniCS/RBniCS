# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod


@AbstractBackend
class TensorsList(object, metaclass=ABCMeta):
    def __init__(self, space, empty_tensor):
        pass

    @abstractmethod
    def enrich(self, tensor):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def save(self, directory, filename):
        pass

    @abstractmethod
    def load(self, directory, filename):
        pass

    # self * other [used to compute S*eigv]
    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __len__(self):
        pass

    # key may be an integer
    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __iter__(self):
        pass
