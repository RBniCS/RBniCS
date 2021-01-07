# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod


@AbstractBackend
class BasisFunctionsMatrix(object, metaclass=ABCMeta):
    def __init__(self, space, component=None):
        pass

    @abstractmethod
    def init(self, components):
        pass

    @abstractmethod
    def enrich(self, functions, component=None, weights=None, copy=True):
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

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __len__(self):
        pass

    # key may be an integer or a slice
    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __setitem__(self, key, item):
        pass

    @abstractmethod
    def __iter__(self):
        pass
