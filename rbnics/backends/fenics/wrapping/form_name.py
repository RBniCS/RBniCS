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

from dolfin import __version__ as dolfin_version
import hashlib
import rbnics.backends.fenics

def form_name(form, backend=None):
    if backend is None:
        backend = rbnics.backends.fenics
    
    str_repr = ""
    for integral in form.integrals():
        str_repr += backend.wrapping.expression_name(integral.integrand(), backend)
    hash_code = hashlib.sha1(
                    (str_repr + dolfin_version).encode("utf-8")
                ).hexdigest() # similar to dolfin/compilemodules/compilemodule.py
    return hash_code
    
