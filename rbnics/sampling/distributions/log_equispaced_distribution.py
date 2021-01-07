# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from math import exp, log
from rbnics.sampling.distributions.distribution import Distribution
from rbnics.sampling.distributions.equispaced_distribution import EquispacedDistribution


class LogEquispacedDistribution(Distribution):
    def __init__(self):
        self.equispaced_distribution = EquispacedDistribution()

    def sample(self, box, n):
        log_box = [(log(box_p[0]), log(box_p[1])) for box_p in box]
        log_set = self.equispaced_distribution.sample(log_box, n)
        return [tuple(exp(log_mu_p) for log_mu_p in log_mu) for log_mu in log_set]
