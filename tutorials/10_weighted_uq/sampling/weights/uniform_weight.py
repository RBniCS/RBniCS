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
