# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from math import ceil
from numpy import linspace
import itertools
from rbnics.sampling.distributions.distribution import Distribution


class EquispacedDistribution(Distribution):
    def sample(self, box, n):
        n_P_root = int(ceil(n**(1. / len(box))))
        grid = list()  # of linspaces
        for box_p in box:
            grid.append(linspace(box_p[0], box_p[1], num=n_P_root).tolist())
        set_itertools = itertools.product(*grid)
        set_ = list()  # of tuples
        for mu in set_itertools:
            set_.append(mu)
        return set_
