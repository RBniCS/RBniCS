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
## @file sum.py
#  @brief sum function to assemble truth/reduced affine expansions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

###########################     OFFLINE AND ONLINE COMMON INTERFACES     ########################### 
## @defgroup OfflineOnlineInterfaces Common interfaces for offline and online
#  @{

from dolfin import DirichletBC
from RBniCS.linear_algebra.product import _DotProductOutput, _DirichletBCsProductOutput

# sum function to assemble truth/reduced affine expansions. To be used in combination with the product method.
__std_sum = sum
def sum(product_output):
    if isinstance(product_output, _DotProductOutput):
        return _EquationSide(product_output[0]) # sum has been already performed by the dot product
    elif isinstance(product_output, _DirichletBCsProductOutput): # we use this Dirichlet BCs with FEniCS
        boundary_id_to_function_space_map = dict() # first argument of the constructor
        boundary_id_to_function_map = dict() # second argument of the constructor
        boundary_id_to_boundary_mesh_map = dict() # third argument of the constructor
        for i in range(len(product_output)):
            for j in range(len(product_output[i])):
                # Each element of the list contains a tuple. Owing to FEniCS documentation (overloaded version with MeshFunction argument),
                # its fourth argument is the subdomain id, to be collected and used as map index.
                assert len(product_output[i][j]) == 4
                function_space = product_output[i][j][0]
                function = product_output[i][j][1]
                boundary_mesh = product_output[i][j][2]
                boundary_id = product_output[i][j][3]
                if not boundary_id in boundary_id_to_function_map:
                    assert not boundary_id in boundary_id_to_function_space_map
                    assert not boundary_id in boundary_id_to_boundary_mesh_map
                    boundary_id_to_function_space_map[boundary_id] = function_space
                    boundary_id_to_function_map[boundary_id] = function
                    boundary_id_to_boundary_mesh_map[boundary_id] = boundary_mesh
                else:
                    assert boundary_id_to_function_space_map[boundary_id] == function_space
                    assert boundary_id_to_boundary_mesh_map[boundary_id] == boundary_mesh
                    boundary_id_to_function_map[boundary_id] += function
        output = []
        for boundary_id in boundary_id_to_function_map.keys():
            assert boundary_id in boundary_id_to_function_space_map
            assert boundary_id in boundary_id_to_boundary_mesh_map
            output.append( \
                DirichletBC( \
                    boundary_id_to_function_space_map[boundary_id], \
                    boundary_id_to_function_map[boundary_id], \
                    boundary_id_to_boundary_mesh_map[boundary_id], \
                    boundary_id \
                ) \
            )
        return output
    else: # preserve the standard python sum function
        return __std_sum(product_output)

# The following two classes try to mimic a subset of the functionalities of Equation and Form
# classes of UFL in order to be able to write solve(A == F, ...) also in RBniCS, where A and F
# are assembled in an offline/online efficient way.
import types
from RBniCS.linear_algebra.truth_vector import TruthVector
from RBniCS.linear_algebra.truth_matrix import TruthMatrix
from RBniCS.linear_algebra.online_vector import OnlineVector_Type
from RBniCS.linear_algebra.online_matrix import OnlineMatrix_Type

def _EquationSide(content): # inspired by UFL Form class
    if \
        isinstance(content, TruthMatrix) or isinstance(content, TruthVector) \
            or \
        isinstance(content, OnlineMatrix_Type) or isinstance(content, OnlineVector_Type) \
    :
        if hasattr(content, "__eq__"):
            standard_eq = content.__eq__
        else:
            def standard_eq(self, other):
                raise AttributeError
        def __eq__(self, other):
            if hasattr(other, "equals"): # it means that also other was preprocessed by this function
                return _EquationPlaceholder(self, other)
            else:
                return standard_eq(other)
        content.__eq__ = types.MethodType(__eq__, content)
        content.equals = types.MethodType(standard_eq, content)
    return content
    
class _EquationPlaceholder(object): # inspired by UFL Equation class
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
    
    def __bool__(self):
        if type(self.lhs) != type(self.rhs):
            return False
        return self.lhs.equals(self.rhs)

#  @}
########################### end - OFFLINE AND ONLINE COMMON INTERFACES - end ########################### 
