# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numpy import pi, random
from rbnics.sampling.distributions.distribution import Distribution


class LinearlyDependentUniformDistribution(Distribution):
    def __init__(self):
        self.aux_box = [(0.5, 1.5), (0.5, 1.5), (0, pi / 6.)]

    def sample(self, _, n):
        set_ = list()  # of tuples
        for i in range(n):
            aux_mu = list()  # of numbers
            for aux_box_p in self.aux_box:
                aux_mu.append(random.uniform(aux_box_p[0], aux_box_p[1]))
            mu = (2 - aux_mu[0], aux_mu[1], aux_mu[0], aux_mu[0], 2 - aux_mu[1], aux_mu[2])
            set_.append(mu)
        return set_
