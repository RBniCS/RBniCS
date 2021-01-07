# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl.domain import extract_domains
from rbnics.backends.dolfin.wrapping.expression_replace import replace


def form_replace(form, replacements, replacement_type="nodes"):
    assert replacement_type in ("nodes", "measures")
    if replacement_type == "nodes":
        replaced_form = replace(form, replacements)
        for (integral, replaced_integral) in zip(form.integrals(), replaced_form.integrals()):
            replaced_integral_domains = extract_domains(replaced_integral.integrand())
            assert len(replaced_integral_domains) == 1
            integral_domains = extract_domains(integral.integrand())
            assert len(integral_domains) == 1
            assert replaced_integral_domains[0] is not integral_domains[0]
        return replaced_form
    elif replacement_type == "measures":
        replaced_form = 0
        for integral in form.integrals():
            measure = replacements[integral.integrand(), integral.integral_type(), integral.subdomain_id()]
            replaced_form += integral.integrand() * measure
        return replaced_form
    else:
        raise ValueError("Invalid replacement type")
