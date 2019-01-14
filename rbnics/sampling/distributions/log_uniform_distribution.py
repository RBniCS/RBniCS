# Copyright (C) 2015-2019 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

from math import exp, log
from rbnics.sampling.distributions.distribution import Distribution
from rbnics.sampling.distributions.uniform_distribution import UniformDistribution

class LogUniformDistribution(Distribution):
    def __init__(self):
        self.uniform_distribution = UniformDistribution()
        
    def sample(self, box, n):
        log_box = [(log(box_p[0]), log(box_p[1])) for box_p in box]
        log_set = self.uniform_distribution.sample(log_box, n)
        return [tuple(exp(log_mu_p) for log_mu_p in log_mu) for log_mu in log_set]
