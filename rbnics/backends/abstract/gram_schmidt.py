# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod


@AbstractBackend
class GramSchmidt(object, metaclass=ABCMeta):
    def __init__(self, space, inner_product, component=None):
        pass

    # Apply one iteration of Gram Schmidt procedure to orthonormalize the new basis function
    # with respect to the provided basis functions matrix
    @abstractmethod
    def apply(self, new_basis_function, basis_functions, component=None):
        pass
