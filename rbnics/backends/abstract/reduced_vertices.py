# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod


@AbstractBackend
class ReducedVertices(object, metaclass=ABCMeta):
    def __init__(self, space):
        pass

    @abstractmethod
    def append(self, vertex_and_component):
        pass

    @abstractmethod
    def save(self, directory, filename):
        pass

    @abstractmethod
    def load(self, directory, filename):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass
