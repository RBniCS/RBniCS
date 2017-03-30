# Copyright (C) 2015-2017 by the RBniCS authors
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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from rbnics.backends.fenics.wrapping.get_expression_description import get_expression_description

def get_form_description(form):
    coefficients_repr = {}
    for integral in form.integrals():
        coefficients_repr_integral = get_expression_description(integral.integrand())
        # Check consistency
        intersection = set(coefficients_repr_integral.keys()).intersection(set(coefficients_repr.keys()))
        for key in intersection:
            assert coefficients_repr_integral[key] == coefficients_repr[key]
        # Update
        coefficients_repr.update(coefficients_repr_integral)
    return coefficients_repr
