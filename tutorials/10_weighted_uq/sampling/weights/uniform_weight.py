# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from scipy.stats import uniform
from .weight import Weight


class UniformWeight(Weight):
    def density(self, box, samples):
        samples_density = list()
        for mu in samples:
            p_mu = 1.0
            assert len(mu) == len(box)
            for (mu_j, box_j) in zip(mu, box):
                p_mu *= uniform.pdf(mu_j, box_j[0], box_j[1])
            samples_density.append(p_mu)
        return samples_density
