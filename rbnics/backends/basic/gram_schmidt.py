# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from math import sqrt
from rbnics.backends.abstract import GramSchmidt as AbstractGramSchmidt
from rbnics.utils.decorators import dict_of, overload


def GramSchmidt(backend, wrapping):

    class _GramSchmidt(AbstractGramSchmidt):
        def __init__(self, space, inner_product, component=None):
            if component is None:
                self.space = space
            else:
                self.space = wrapping.get_function_subspace(space, component)
            self.inner_product = inner_product

        def apply(self, new_basis_function, basis_functions, component=None):
            inner_product = self.inner_product

            transpose = backend.transpose

            new_basis_function = self._extend_or_restrict_if_needed(new_basis_function, component)

            for b in basis_functions:
                new_basis_function = wrapping.gram_schmidt_projection_step(new_basis_function, inner_product, b,
                                                                           transpose)
            norm_new_basis_function = sqrt(transpose(new_basis_function) * inner_product * new_basis_function)
            if norm_new_basis_function != 0.:
                new_basis_function /= norm_new_basis_function

            return new_basis_function

        @overload(backend.Function.Type(), (None, str))
        def _extend_or_restrict_if_needed(self, function, component):
            return wrapping.function_extend_or_restrict(function, component, self.space, component, weight=None,
                                                        copy=True)

        @overload(backend.Function.Type(), dict_of(str, str))
        def _extend_or_restrict_if_needed(self, function, component):
            assert len(component) == 1
            for (component_from, component_to) in component.items():
                break
            return wrapping.function_extend_or_restrict(function, component_from, self.space, component_to,
                                                        weight=None, copy=True)

    return _GramSchmidt
