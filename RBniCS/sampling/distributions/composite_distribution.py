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
        # Create a reference equispaced distribution, because it is handled as a special case
        self.equispaced_distribution = EquispacedDistribution()
        
    @override
    def sample(self, box, n):
        # Divide box among the different distributions
        distribution_to_sub_box = dict()
        for (distribution, components) in self.distribution_to_components.iteritems():
            distribution_to_sub_box[distribution] = [box[p] for p in components]
        # Prepare a dict that will store the map from components to subset sub_xi
        components_to_sub_xi = dict()
        # Consider first equispaced distributions, because they may change the value of n
        equispaced_distribution = self.equispaced_distribution
        if equispaced_distribution in distribution_to_sub_box:
            sub_box = distribution_to_sub_box[equispaced_distribution]
            sub_xi = equispaced_distribution.sample(sub_box, n)
            n = len(sub_xi) # may be greater or equal than the one originally provided
            components = self.distribution_to_components[distribution]
            components_to_sub_xi[equispaced_distribution] = sub_xi
        # ... and the consider all the remaining distributions
        for (distribution, sub_box) in distribution_to_sub_box.iteritems():
            if distribution is not equispaced_distribution:
                components = self.distribution_to_components[distribution]
                components_to_sub_xi[components] = distribution.sample(sub_box, n)
        # Prepare a list that will store xi = [mu_1, ... mu_n] ...
        xi_as_list = [[None]*len(box) for _ in range(n)]
        for (components, sub_xi) in components_to_sub_xi.iteritems():
            assert len(sub_xi) == n
            for (index, sub_mu) in enumerate(sub_xi):
                assert len(components) == len(sub_mu)
                for (p, sub_mu_p) in zip(components, sub_mu):
                    xi_as_list[index][p] = sub_mu_p
        # ... and convert each mu to a tuple
        xi = [tuple(mu) for mu in xi_as_list]
        return xi
        
