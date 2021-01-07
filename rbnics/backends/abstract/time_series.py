# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod


@AbstractBackend
class TimeSeries(metaclass=ABCMeta):
    def __init__(self, *args):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def at(self, time):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __delitem__(self, key):
        pass

    @abstractmethod
    def append(self, item):
        pass

    @abstractmethod
    def extend(self, iterable):
        pass

    @abstractmethod
    def clear(self):
        pass
