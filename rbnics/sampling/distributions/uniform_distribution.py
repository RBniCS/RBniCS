# Copyright (C) 2015-2017 by the RBniCS authors
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

from numpy import random
from rbnics.sampling.distributions.distribution import Distribution

class UniformDistribution(Distribution):
    def sample(self, box, n):
        set_ = list() # of tuples
        for i in range(n):
            mu = list() # of numbers
            for box_p in box:
                mu.append(random.uniform(box_p[0], box_p[1]))
            set_.append(tuple(mu))
        return set_
