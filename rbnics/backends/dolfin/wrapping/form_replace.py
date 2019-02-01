# Copyright (C) 2015-2019 by the RBniCS authors
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
            replaced_form += integral.integrand()*measure
        return replaced_form
    else:
        raise ValueError("Invalid replacement type")
