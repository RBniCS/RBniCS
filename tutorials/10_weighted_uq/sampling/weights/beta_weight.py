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
                p_mu *= beta.pdf((mu_j - box_j[0])/(box_j[1] - box_j[0]), a=a_j, b=b_j)
            samples_density.append(p_mu)
        return samples_density
