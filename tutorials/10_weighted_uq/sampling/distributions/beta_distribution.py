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
from rbnics.sampling.distributions import CompositeDistribution, DrawFrom

class BetaDistribution(CompositeDistribution):
    def __init__(self, a, b):
        assert isinstance(a, (list, tuple))
        assert isinstance(b, (list, tuple))
        assert len(a) is len(b)
        CompositeDistribution.__init__(self, [DrawFrom(random.beta, a=a_p, b=b_p) for (a_p, b_p) in zip(a, b)])
