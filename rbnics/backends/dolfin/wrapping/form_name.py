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

import hashlib

def basic_form_name(backend, wrapping):
    def _basic_form_name(form):
        str_repr = ""
        for integral in form.integrals():
            str_repr += wrapping.expression_name(integral.integrand())
            str_repr += "measure(" + integral.integral_type() + ")[" + str(integral.subdomain_id()) + "]"
        hash_code = hashlib.sha1(str_repr.encode("utf-8")).hexdigest()
        return hash_code
    return _basic_form_name
    
from rbnics.utils.decorators import ModuleWrapper
from rbnics.backends.dolfin.wrapping.expression_name import expression_name
backend = ModuleWrapper()
wrapping = ModuleWrapper(expression_name=expression_name)
form_name = basic_form_name(backend, wrapping)
