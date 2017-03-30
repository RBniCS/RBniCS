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

from math import ceil
from numpy import linspace
import itertools
from rbnics.sampling.distributions.distribution import Distribution
from rbnics.utils.decorators import Extends, override

@Extends(Distribution)
class EquispacedDistribution(Distribution):
    @override
    def sample(self, box, n):
        n_P_root = int(ceil(n**(1./len(box))))
        grid = list() # of linspaces
        for box_p in box:
            grid.append( linspace(box_p[0], box_p[1], num=n_P_root).tolist() )
        set_itertools = itertools.product(*grid)
        set_ = list() # of tuples
        for mu in set_itertools:
            set_.append(mu)
        return set_
        
