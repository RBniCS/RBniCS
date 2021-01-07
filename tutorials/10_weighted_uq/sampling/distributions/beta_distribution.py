# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numpy import random
from rbnics.sampling.distributions import CompositeDistribution, DrawFrom


class BetaDistribution(CompositeDistribution):
    def __init__(self, a, b):
        assert isinstance(a, (list, tuple))
        assert isinstance(b, (list, tuple))
        assert len(a) == len(b)
        CompositeDistribution.__init__(self, [DrawFrom(random.beta, a=a_p, b=b_p) for (a_p, b_p) in zip(a, b)])
