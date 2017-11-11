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

from ufl import Form
from dolfin import as_backend_type, assemble, has_pybind11
if has_pybind11():
    from dolfin.cpp.la import GenericMatrix
else:
    from dolfin import GenericMatrix
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.wrapping.dirichlet_bc import InvertProductOutputDirichletBC

def Matrix():
    raise NotImplementedError("This is dummy function (not required by the interface) just store the Type")
    
# Attach a Type() function
def Type():
    return GenericMatrix
Matrix.Type = Type

# Enable matrix*function product (i.e. matrix*function.vector())
original__mul__ = GenericMatrix.__mul__
def custom__mul__(self, other):
    if isinstance(other, Function.Type()):
        return original__mul__(self, other.vector())
    else:
        return original__mul__(self, other)
GenericMatrix.__mul__ = custom__mul__

# Preserve generator attribute in algebraic operators, as required by DEIM
def preserve_generator_attribute(operator):
    original_operator = getattr(GenericMatrix, operator)
    def custom_operator(self, other):
        if hasattr(self, "generator"):
            output = original_operator(self, other)
            output.generator = self.generator
            return output
        else:
            return original_operator(self, other)
    setattr(GenericMatrix, operator, custom_operator)
    
if has_pybind11():
    for operator in ("__add__", "__iadd__", "__sub__", "__isub__", "__mul__", "__imul__", "__truediv__", "__itruediv__"):
        preserve_generator_attribute(operator)
else:
    for operator in ("__add__", "__radd__", "__iadd__", "__sub__", "__rsub__", "__isub__", "__mul__", "__rmul__", "__imul__", "__truediv__", "__rtruediv__", "__itruediv__"):
        preserve_generator_attribute(operator)

# Allow sum and sub between matrix and form by assemblying the form. This is required because
# affine expansion storage is not assembled if it is parametrized, and it may happen that
# some terms are parametrized (and thus not assembled) while others are not parametrized
# (and thus assembled).
def arithmetic_with_form(operator):
    original_operator = getattr(GenericMatrix, operator)
    def custom_operator(self, other):
        if isinstance(other, Form):
            assert len(other.arguments()) is 2
            other = assemble(other, keep_diagonal=True)
        return original_operator(self, other)
    setattr(GenericMatrix, operator, custom_operator)

if has_pybind11():
    for operator in ("__add__", "__sub__"):
        arithmetic_with_form(operator)
else:
    for operator in ("__add__", "__radd__", "__sub__", "__rsub__"):
        arithmetic_with_form(operator)

# Define the __and__ operator to be used in combination with __invert__ operator
# of sum(product(theta, DirichletBCs)) to zero rows and columns associated to Dirichlet BCs
def custom__and__(self, other):
    if isinstance(other, InvertProductOutputDirichletBC):
        output = self.copy()
        mat = as_backend_type(output).mat()
        for bc in other.bc_list:
            constrained_dofs = [bc.function_space().dofmap().local_to_global_index(local_dof_index) for local_dof_index in bc.get_boundary_values().keys()]
            mat.zeroRowsColumns(constrained_dofs, 0.)
        return output
    else:
        return NotImplemented
setattr(GenericMatrix, "__and__", custom__and__)
