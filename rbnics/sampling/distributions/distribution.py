# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from abc import ABCMeta, abstractmethod


class Distribution(object, metaclass=ABCMeta):
    @abstractmethod
    def sample(self, box, n):
        raise NotImplementedError("The method sample is distribution-specific and needs to be overridden.")

    # Override the following methods to use a Distribution as a dict key
    def __hash__(self):
        dict_for_hash = list()
        for (k, v) in self.__dict__.items():
            if isinstance(v, dict):
                dict_for_hash.append(tuple(v.values()))
            elif isinstance(v, list):
                dict_for_hash.append(tuple(v))
            else:
                dict_for_hash.append(v)
        return hash((type(self).__name__, tuple(dict_for_hash)))

    def __eq__(self, other):
        return (type(self).__name__, self.__dict__) == (type(other).__name__, other.__dict__)

    def __ne__(self, other):
        return not(self == other)
