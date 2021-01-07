# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from scipy.stats import beta
from .weight import Weight


class BetaWeight(Weight):
    def __init__(self, a, b):
        assert isinstance(a, (list, tuple))
        assert isinstance(b, (list, tuple))
        assert len(a) == len(b)
        self.a = a
        self.b = b

    def density(self, box, samples):
        assert len(self.a) == len(box)
        assert len(self.b) == len(box)
        samples_density = list()
        for mu in samples:
            p_mu = 1.0
            assert len(mu) == len(box)
            for (mu_j, box_j, a_j, b_j) in zip(mu, box, self.a, self.b):
                p_mu *= beta.pdf((mu_j - box_j[0]) / (box_j[1] - box_j[0]), a=a_j, b=b_j)
            samples_density.append(p_mu)
        return samples_density
