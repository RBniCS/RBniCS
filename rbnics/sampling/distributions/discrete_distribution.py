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
                rounded_mu.append(round(mu[p]/step_size)*step_size)
            rounded_set.append(tuple(rounded_mu))
        return rounded_set
