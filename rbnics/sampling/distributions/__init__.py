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
## @file __init__.py
#  @brief Init file for auxiliary sampling module
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.sampling.distributions.composite_distribution import CompositeDistribution
from RBniCS.sampling.distributions.discrete_distribution import DiscreteDistribution
from RBniCS.sampling.distributions.distribution import Distribution
from RBniCS.sampling.distributions.draw_from import DrawFrom
from RBniCS.sampling.distributions.equispaced_distribution import EquispacedDistribution
from RBniCS.sampling.distributions.log_uniform_distribution import LogUniformDistribution
from RBniCS.sampling.distributions.uniform_distribution import UniformDistribution

__all__ = [
    'CompositeDistribution',
    'DiscreteDistribution',
    'Distribution',
    'EquispacedDistribution',
    'LogUniformDistribution',
    'UniformDistribution'
]
