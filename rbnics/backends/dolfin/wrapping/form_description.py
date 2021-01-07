# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.dolfin.wrapping.expression_description import expression_description
from rbnics.utils.decorators import ModuleWrapper


def basic_form_description(backend, wrapping):
    def _basic_form_description(form):
        coefficients_repr = dict()
        for integral in form.integrals():
            coefficients_repr_integral = wrapping.expression_description(integral.integrand())
            # Check consistency
            intersection = set(coefficients_repr_integral.keys()).intersection(set(coefficients_repr.keys()))
            for key in intersection:
                assert coefficients_repr_integral[key] == coefficients_repr[key]
            # Update
            coefficients_repr.update(coefficients_repr_integral)
        return coefficients_repr
    return _basic_form_description


backend = ModuleWrapper()
wrapping = ModuleWrapper(expression_description=expression_description)
form_description = basic_form_description(backend, wrapping)
