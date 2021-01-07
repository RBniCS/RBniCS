# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.sampling.distributions.distribution import Distribution
from rbnics.sampling.distributions.equispaced_distribution import EquispacedDistribution


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

    def sample(self, box, n):
        # Divide box among the different distributions
        distribution_to_sub_box = dict()
        for (distribution, components) in self.distribution_to_components.items():
            distribution_to_sub_box[distribution] = [box[p] for p in components]
        # Prepare a dict that will store the map from components to subset sub_set
        components_to_sub_set = dict()
        # Consider first equispaced distributions, because they may change the value of n
        for (distribution, sub_box) in distribution_to_sub_box.items():
            if isinstance(distribution, EquispacedDistribution):
                sub_box = distribution_to_sub_box[distribution]
                sub_set = distribution.sample(sub_box, n)
                n = len(sub_set)  # may be greater or equal than the one originally provided
                components = self.distribution_to_components[distribution]
                components_to_sub_set[tuple(components)] = sub_set
        assert len(components_to_sub_set) in (0, 1)
        # ... and the consider all the remaining distributions
        for (distribution, sub_box) in distribution_to_sub_box.items():
            if not isinstance(distribution, EquispacedDistribution):
                components = self.distribution_to_components[distribution]
                components_to_sub_set[tuple(components)] = distribution.sample(sub_box, n)
        # Prepare a list that will store the set [mu_1, ... mu_n] ...
        set_as_list = [[None] * len(box) for _ in range(n)]
        for (components, sub_set) in components_to_sub_set.items():
            assert len(sub_set) == n
            for (index, sub_mu) in enumerate(sub_set):
                assert len(components) == len(sub_mu)
                for (p, sub_mu_p) in zip(components, sub_mu):
                    set_as_list[index][p] = sub_mu_p
        # ... and convert each mu to a tuple
        set_ = [tuple(mu) for mu in set_as_list]
        return set_
