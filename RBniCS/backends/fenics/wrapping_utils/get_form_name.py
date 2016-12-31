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

from dolfin import __version__ as dolfin_version
from ufl.corealg.traversal import pre_traversal
import hashlib

def get_form_name(form):
    str_repr = ""
    coefficients_replacement = {}
    for integral in form.integrals():
        for n in pre_traversal(integral.integrand()):
            if hasattr(n, "cppcode"):
                coefficients_replacement[repr(n)] = str(n.cppcode)
                str_repr += repr(n.cppcode)
            else:
                str_repr += repr(n)
    for key, value in coefficients_replacement.iteritems():
        str_repr = str_repr.replace(key, value)
    hash_code = hashlib.sha1(
                    (str_repr + dolfin_version).encode("utf-8")
                ).hexdigest() # similar to dolfin/compilemodules/compilemodule.py
    return hash_code
