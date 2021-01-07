# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numpy import random
from rbnics.sampling.distributions.distribution import Distribution


class UniformDistribution(Distribution):
    def sample(self, box, n):
        set_ = list()  # of tuples
        for i in range(n):
            mu = list()  # of numbers
            for box_p in box:
                mu.append(random.uniform(box_p[0], box_p[1]))
            set_.append(tuple(mu))
        return set_
