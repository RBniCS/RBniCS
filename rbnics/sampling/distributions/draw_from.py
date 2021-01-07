# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.sampling.distributions.distribution import Distribution


class DrawFrom(Distribution):
    def __init__(self, generator, *args, **kwargs):
        self.generator = generator  # of a distribution in [0, 1]
        self.args = args
        self.kwargs = kwargs

    def sample(self, box, n):
        set_ = list()  # of tuples
        for i in range(n):
            mu = list()  # of numbers
            for box_p in box:
                mu.append(box_p[0] + self.generator(*self.args, **self.kwargs) * (box_p[1] - box_p[0]))
            set_.append(tuple(mu))
        return set_
