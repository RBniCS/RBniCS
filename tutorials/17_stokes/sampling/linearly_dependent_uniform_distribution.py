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

from numpy import pi, random
from RBniCS.sampling.distributions.distribution import Distribution
from RBniCS.utils.decorators import Extends, override

@Extends(Distribution)
class LinearlyDependentUniformDistribution(Distribution):
    def __init__(self):
        self.aux_box = [(0.5, 1.5), (0.5, 1.5), (0, pi/6.)]
        
    @override
    def sample(self, _, n):
        set_ = list() # of tuples
        for i in range(n):
            aux_mu = list() # of numbers
            for aux_box_p in self.aux_box:
                aux_mu.append(random.uniform(aux_box_p[0], aux_box_p[1]))
            mu = (2 - aux_mu[0], aux_mu[1], aux_mu[0], aux_mu[0], 2 - aux_mu[1], aux_mu[2])
            set_.append(mu)
        return set_
        
