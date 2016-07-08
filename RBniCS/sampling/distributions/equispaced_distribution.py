# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file distribution.py
#  @brief Type for distribution
#
#  @author Luca      Venturi  <luca.venturi@sissa.it>
#  @author Davide    Torlo    <davide.torlo@sissa.it>
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from math import ceil
from numpy import linspace
import itertools
from RBniCS.sampling.distributions import Distribution

class EquispacedDistribution(Distribution):
    def sample(self, box, n):
        n_P_root = int(ceil(n**(1./len(box))))
        grid = list() # of linspaces
        for p in range(len(box)):
            grid.append( linspace(box[p][0], box[p][1], num=n_P_root).tolist() )
        xi_itertools = itertools.product(*grid)
        xi = list() # of tuples
        for mu in xi_itertools:
            xi.append(mu)
        return xi
        
