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

from RBniCS.sampling.distributions.distribution import Distribution
from RBniCS.utils.decorators import Extends, override

@Extends(Distribution)
class DrawFrom(Distribution):
    def __init__(self, generator, *args, **kwargs):
        self.generator = generator # of a distribution in [0, 1]
        self.args = args
        self.kwargs = kwargs
        
    @override
    def sample(self, box, n):
        xi = list() # of tuples
        for i in range(n):
            mu = list() # of numbers
            for box_p in box:
                mu.append(box_p[0] + self.generator(*self.args, **self.kwargs)*(box_p[1] - box_p[0]))
            xi.append(tuple(mu))
        return xi
        
