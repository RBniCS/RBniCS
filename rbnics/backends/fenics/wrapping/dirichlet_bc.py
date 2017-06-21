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

import types
from numpy import zeros
from dolfin import Constant, DirichletBC as dolfin_DirichletBC, FunctionSpace, GenericFunction, MeshFunctionSizet, SubDomain

def DirichletBC(*args, **kwargs):
    # Call the constructor
    output = dolfin_DirichletBC(*args, **kwargs)
    # Deduce private variable values from arguments
    if len(args) == 1 and isinstance(args[0], dolfin_DirichletBC):
        assert len(kwargs) == 0
        _value = args[0]._value
        _function_space = args[0]._function_space
        _sorted_kwargs = args[0]._sorted_kwargs
        _identifier = args[0]._identifier
    else:
        _value = args[1]
        _function_space = args[0]
        _sorted_kwargs = list()
        for key in ["method", "check_midpoint"]:
            if key in kwargs:
                _sorted_kwargs.append(kwargs[key])
        _identifier = list()
        _identifier.extend(output.domain_args)
        _identifier.extend(_sorted_kwargs)
        _identifier = tuple(_identifier)
    # Override the value(), set_value() and homogenize() methods. These are already available in the public interface,
    # but it is cast the value to a base type (GenericFunction), which makes it not possible to perform the sum
    output._value = _value
    def value(self_):
        return self_._value
    output.value = types.MethodType(value, output)
    def set_value(self_, g):
        self_._value = g
        dolfin_DirichletBC.set_value(self_, g)
    output.set_value = types.MethodType(set_value, output)
    def homogenize(self_):
        self_._value = Constant(zeros(self_._value.ufl_shape))
        dolfin_DirichletBC.set_value(self_, self_._value)
    output.homogenize = types.MethodType(homogenize, output)
    # Override the function_space() method. This is already available in the public interface,
    # but it casts the function space to a C++ FunctionSpace and then wraps it into a python FunctionSpace,
    # losing all the customization that we have done in the function_space.py file
    output._function_space = _function_space
    def function_space(self_):
        return self_._function_space
    output.function_space = types.MethodType(function_space, output)
    # Define an identifier() method, that identifies whether BCs are defined on the same boundary
    output._identifier = _identifier
    def identifier(self_):
        return self_._identifier
    output.identifier = types.MethodType(identifier, output)
    # Store kwargs, in a sorted way (as in dolfin_DirichletBC)
    output._sorted_kwargs = _sorted_kwargs
    # Return
    return output

# Add a multiplication operator by a scalar
def mul_by_scalar(self, other):
    if isinstance(other, (float, int)):
        args = list()
        args.append(self.function_space())
        args.append(Constant(other)*self.value())
        args.extend(self.domain_args)
        args.extend(self._sorted_kwargs)
        return DirichletBC(*args)
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
        
