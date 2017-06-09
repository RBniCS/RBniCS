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

from dolfin import Constant, DirichletBC as dolfin_DirichletBC

def DirichletBC(V, g, subdomains, subdomain_id):
    output = dolfin_DirichletBC(V, g, subdomains, subdomain_id)
    output.subdomains = subdomains # this is currently not available in the public interface
    output.subdomain_id = subdomain_id # this is currently not available in the public interface
    output.value = g # this is available but it is cast to a base type, and it makes performing the sum not possible
    output.function_space = V # this is already available as a method, replace it with an attribute for consistency
    return output

# Add a multiplication operator by a scalar
def mul_by_scalar(self, other):
    if isinstance(other, (float, int)):
        V = self.function_space
        g = Constant(other)*self.value
        subdomains = self.subdomains
        subdomain_id = self.subdomain_id
        return DirichletBC(V, g, subdomains, subdomain_id)
    else:
        return NotImplemented
        
setattr(dolfin_DirichletBC, "__mul__", mul_by_scalar)
setattr(dolfin_DirichletBC, "__rmul__", mul_by_scalar)

class ProductOutputDirichletBC(list):
    # Define the __invert__ operator to be used in combination with __and__ operator of Matrix
    # to zero rows and columns associated to Dirichlet BCs
    def __invert__(self):
        return InvertProductOutputDirichletBC(self)
        
class InvertProductOutputDirichletBC(object):
    def __init__(self, bc_list):
        self.bc_list = bc_list
        
