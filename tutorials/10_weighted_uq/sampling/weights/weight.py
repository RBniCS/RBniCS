# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from abc import ABCMeta, abstractmethod


class Weight(object, metaclass=ABCMeta):
    @abstractmethod
    def density(self, box, samples):
        raise NotImplementedError("The method density is weight-specific and needs to be overridden.")
