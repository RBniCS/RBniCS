# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.sampling.distributions.distribution import Distribution


class DiscreteDistribution(Distribution):
    def __init__(self, distribution, box_step_size):
        self.distribution = distribution
        self.box_step_size = box_step_size

    def sample(self, box, n):
        assert len(box) == len(self.box_step_size)
        set_ = self.distribution.sample(box, n)
        rounded_set = list()
        for mu in set_:
            rounded_mu = list()
            for (p, step_size) in enumerate(self.box_step_size):
                rounded_mu.append(round(mu[p] / step_size) * step_size)
            rounded_set.append(tuple(rounded_mu))
        return rounded_set
