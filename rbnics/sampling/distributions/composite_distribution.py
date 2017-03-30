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
## @file distribution.py
#  @brief Type for distribution
#
#  @author Luca      Venturi  <luca.venturi@sissa.it>
#  @author Davide    Torlo    <davide.torlo@sissa.it>
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.sampling.distributions.distribution import Distribution
from RBniCS.sampling.distributions.equispaced_distribution import EquispacedDistribution
from RBniCS.utils.decorators import Extends, override

@Extends(Distribution)
class CompositeDistribution(Distribution):
    def __init__(self, distributions):
        self.distributions = distributions
        # Create a dict from scalar distribution to component
        self.distribution_to_components = dict()
        for (p, distribution) in enumerate(self.distributions):
            assert isinstance(distribution, Distribution)
            if distribution not in self.distribution_to_components:
                self.distribution_to_components[distribution] = list()
            self.distribution_to_components[distribution].append(p)
        
    @override
    def sample(self, box, n):
        # Divide box among the different distributions
        distribution_to_sub_box = dict()
        for (distribution, components) in self.distribution_to_components.iteritems():
            distribution_to_sub_box[distribution] = [box[p] for p in components]
        # Prepare a dict that will store the map from components to subset sub_set
        components_to_sub_set = dict()
        # Consider first equispaced distributions, because they may change the value of n
        for (distribution, sub_box) in distribution_to_sub_box.iteritems():
            if isinstance(distribution, EquispacedDistribution):
                sub_box = distribution_to_sub_box[distribution]
                sub_set = distribution.sample(sub_box, n)
                n = len(sub_set) # may be greater or equal than the one originally provided
                components = self.distribution_to_components[distribution]
                components_to_sub_set[tuple(components)] = sub_set
        assert len(components_to_sub_set) in (0, 1)
        # ... and the consider all the remaining distributions
        for (distribution, sub_box) in distribution_to_sub_box.iteritems():
            if not isinstance(distribution, EquispacedDistribution):
                components = self.distribution_to_components[distribution]
                components_to_sub_set[tuple(components)] = distribution.sample(sub_box, n)
        # Prepare a list that will store the set [mu_1, ... mu_n] ...
        set_as_list = [[None]*len(box) for _ in range(n)]
        for (components, sub_set) in components_to_sub_set.iteritems():
            assert len(sub_set) == n
            for (index, sub_mu) in enumerate(sub_set):
                assert len(components) == len(sub_mu)
                for (p, sub_mu_p) in zip(components, sub_mu):
                    set_as_list[index][p] = sub_mu_p
        # ... and convert each mu to a tuple
        set_ = [tuple(mu) for mu in set_as_list]
        return set_
        
