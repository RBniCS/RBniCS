# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.sampling import ParameterSpaceSubset
from rbnics.utils.decorators import overload


class GreedySelectedParametersList(object):
    def __init__(self):
        self.parameter_space_subset = ParameterSpaceSubset()

    def save(self, directory, filename):
        self.parameter_space_subset.save(directory, filename)

    def load(self, directory, filename):
        return self.parameter_space_subset.load(directory, filename)

    def append(self, element):
        self.parameter_space_subset.append(element)

    def closest(self, M, mu):
        output = GreedySelectedParametersList()
        output.parameter_space_subset = self.parameter_space_subset.closest(M, mu)
        return output

    @overload
    def __getitem__(self, key: int):
        return self.parameter_space_subset[key]

    @overload
    def __getitem__(self, key: slice):
        output = GreedySelectedParametersList()
        output.parameter_space_subset = self.parameter_space_subset[key]
        return output

    def __iter__(self):
        return iter(self.parameter_space_subset)

    def __len__(self):
        return len(self.parameter_space_subset)

    def __str__(self):
        return str(self.parameter_space_subset)
