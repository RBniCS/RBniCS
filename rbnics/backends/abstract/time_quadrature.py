# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod


@AbstractBackend
class TimeQuadrature(object, metaclass=ABCMeta):
    def __init__(self, time_interval, function_over_time):
        pass

    @abstractmethod
    def integrate(self):
        pass
