# Copyright (C) 2015-2016 by the RBniCS authors
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

import dolfin

def DirichletBC(V, g, subdomains, subdomain_id):
    output = dolfin.DirichletBC(*args, **kwargs)
    output.subdomains = subdomains # this is currently not available in the public interface
    output.subdomain_id = subdomain_id # this is currently not available in the public interface
    output.value = g # this is available but it is cast to a base type, and it makes performing the sum not possible
    output.function_space = function_space # this is already available as a method, replace it with an attribute for consistency
    return output

