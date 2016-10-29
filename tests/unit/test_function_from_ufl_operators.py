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
## @file 
#  @brief 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from numpy import isclose
from dolfin import *
from RBniCS.backends.fenics import transpose
from RBniCS.backends.fenics.wrapping_utils import function_from_ufl_operators

def conversion_test(V, isclose):
    z1 = Function(V)
    z1.vector()[:] = 1.
    assert function_from_ufl_operators(z1) is z1
    
    _2_z1 = function_from_ufl_operators(2*z1)
    assert isclose(_2_z1.vector().array(), 2.).all()
    
    z1_2 = function_from_ufl_operators(z1*2)
    assert isclose(z1_2.vector().array(), 2.).all()
    
    z1_over_2 = function_from_ufl_operators(z1/2.)
    assert isclose(z1_over_2.vector().array(), 0.5).all()
    
    z2 = Function(V)
    z2.vector()[:] = 2.
    
    z1_plus_z2 = function_from_ufl_operators(z1 + z2)
    assert isclose(z1_plus_z2.vector().array(), 3.).all()
    
    z1_minus_z2 = function_from_ufl_operators(z1 - z2)
    assert isclose(z1_minus_z2.vector().array(), -1.).all()
    
    z1_minus_2_z2 = function_from_ufl_operators(z1 - 2*z2)
    assert isclose(z1_minus_2_z2.vector().array(), -3.).all()
    
    z1_minus_z2_2 = function_from_ufl_operators(z1 - z2*2)
    assert isclose(z1_minus_z2_2.vector().array(), -3.).all()
    
    z1_minus_3_z2_2 = function_from_ufl_operators(z1 - 3*z2*2)
    assert isclose(z1_minus_3_z2_2.vector().array(), -11.).all()
    
    z1_minus_z2_over_4 = function_from_ufl_operators(z1 - z2/4.)
    assert isclose(z1_minus_z2_over_4.vector().array(), 0.5).all()
    
    z3 = Function(V)
    z3.vector()[:] = 3.
    
    z1_minus_z2_plus_z3 = function_from_ufl_operators(z1 - z2 + z3)
    assert isclose(z1_minus_z2_plus_z3.vector().array(), 2.).all()
            
def normalization_test(V, A, isclose):
    z1 = Function(V)
    z1.vector()[:] = 2.
    
    z1_normalized = function_from_ufl_operators(z1/sqrt(transpose(z1)*A*z1))
    assert isclose(z1_normalized.vector().array(), 1).all()
    
def transpose_test(V, A, b, isclose):
    z1 = Function(V)
    z1.vector()[:] = 1.
    assert isclose(transpose(z1)*A*z1, 1.)
    assert isclose(transpose(b)*z1, 1.)
    assert isclose(transpose(z1)*b, 1.)

    assert isclose(transpose(z1)*A*(2*z1), 2.)
    assert isclose(transpose(2*z1)*A*z1, 2.)
    assert isclose(transpose(b)*(2*z1), 2.)
    assert isclose(transpose(2*z1)*b, 2.)

    assert isclose(transpose(z1)*A*(z1*2), 2.)
    assert isclose(transpose(z1*2)*A*z1, 2.)
    assert isclose(transpose(b)*(z1*2), 2.)
    assert isclose(transpose(z1*2)*b, 2.)

    assert isclose(transpose(z1)*A*(z1/2.), 0.5)
    assert isclose(transpose(z1/2.)*A*z1, 0.5)
    assert isclose(transpose(b)*(z1/2.), 0.5)
    assert isclose(transpose(z1/2.)*b, 0.5)

    z2 = Function(V)
    z2.vector()[:] = 2.
    
    assert isclose(transpose(z1)*A*(z1 + z2), 3.)
    assert isclose(transpose(z1 + z2)*A*z1, 3.)
    assert isclose(transpose(z1 + z2)*A*(z1 + z2), 9.)
    assert isclose(transpose(b)*(z1 + z2), 3.)
    assert isclose(transpose(z1 + z2)*b, 3.)

    assert isclose(transpose(z1)*A*(z1 - z2), -1.)
    assert isclose(transpose(z1 - z2)*A*z1, -1.)
    assert isclose(transpose(z1 - z2)*A*(z1 - z2), 1.)
    assert isclose(transpose(z1 - z2)*A*(z1 + z2), -3.)
    assert isclose(transpose(b)*(z1 - z2), -1.)
    assert isclose(transpose(z1 - z2)*b, -1.)

    assert isclose(transpose(z1)*A*(z1 - 2*z2), -3.)
    assert isclose(transpose(z1 - 2*z2)*A*z1, -3.)
    assert isclose(transpose(z1 - 2*z2)*A*(z1 - 2*z2), 9.)
    assert isclose(transpose(b)*(z1 - 2*z2), -3.)
    assert isclose(transpose(z1 - 2*z2)*b, -3.)

    assert isclose(transpose(z1)*A*(z1 - z2*2), -3.)
    assert isclose(transpose(z1 - z2*2)*A*z1, -3.)
    assert isclose(transpose(z1 - z2*2)*A*(z1 - z2*2), 9.)
    assert isclose(transpose(b)*(z1 - z2*2), -3.)
    assert isclose(transpose(z1 - z2*2)*b, -3.)

    assert isclose(transpose(z1)*A*(z1 - 3*z2*2), -11.)
    assert isclose(transpose(z1 - 3*z2*2)*A*z1, -11.)
    assert isclose(transpose(z1 - 3*z2*2)*A*(z1 - 3*z2*2), 121.)
    assert isclose(transpose(b)*(z1 - 3*z2*2), -11.)
    assert isclose(transpose(z1 - 3*z2*2)*b, -11.)

    assert isclose(transpose(z1)*A*(z1 - z2/4.), 0.5)
    assert isclose(transpose(z1 - z2/4.)*A*z1, 0.5)
    assert isclose(transpose(z1 - z2/4.)*A*(z1 - z2/4.), 0.25)
    assert isclose(transpose(b)*(z1 - z2/4.), 0.5)
    assert isclose(transpose(z1 - z2/4.)*b, 0.5)
    
    z3 = Function(V)
    z3.vector()[:] = 3.
    
    assert isclose(transpose(z1)*A*(z1 - z2 + z3), 2.)
    assert isclose(transpose(z1 - z2)*A*(z1 - z2 + z3), -2.)
    assert isclose(transpose(z1 - z2 + z3)*A*z1, 2.)
    assert isclose(transpose(z1 - z2 + z3)*A*(z1 - z2), -2.)
    assert isclose(transpose(z1 - z2 + z3)*A*(z1 - z2 + z3), 4.)
    assert isclose(transpose(b)*(z1 - z2 + z3), 2.)
    assert isclose(transpose(z1 - z2 + z3)*b, 2.)

mesh = UnitSquareMesh(10, 10)

# ~~~ Scalar case ~~~ #
V = FunctionSpace(mesh, "Lagrange", 2)
u = TrialFunction(V)
v = TestFunction(V)
A = assemble(u*v*dx)
b = assemble(v*dx)

def scalar_conversion_isclose(a, b):
    return isclose(a, b)
    
def scalar_normalization_isclose(a, b):
    return isclose(a, b)
    
def scalar_transpose_isclose(a, b):
    return isclose(a, b)
    
conversion_test(V, scalar_conversion_isclose)
normalization_test(V, A, scalar_normalization_isclose)
transpose_test(V, A, b, scalar_transpose_isclose)

# ~~~ Vector case ~~~ #
V = VectorFunctionSpace(mesh, "Lagrange", 2)
u = TrialFunction(V)
v = TestFunction(V)
A = assemble(u[0]*v[0]*dx + u[1]*v[1]*dx)
b = assemble(v[0]*dx + v[1]*dx)

def vector_conversion_isclose(a, b):
    return isclose(a, b)
    
def vector_normalization_isclose(a, b):
    return isclose(a, b/sqrt(2))
    
def vector_transpose_isclose(a, b):
    return isclose(a, 2*b)
    
conversion_test(V, vector_conversion_isclose)
normalization_test(V, A, vector_normalization_isclose)
transpose_test(V, A, b, vector_transpose_isclose)

# ~~~ Mixed case ~~~ #
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element)
u = TrialFunction(V)
v = TestFunction(V)
(u_0, u_1) = split(u)
(v_0, v_1) = split(v)
A = assemble(u_0[0]*v_0[0]*dx + u_0[1]*v_0[1]*dx + u_1*v_1*dx)
b = assemble(v_0[0]*dx + v_0[1]*dx + v_1*dx)

def mixed_conversion_isclose(a, b):
    return isclose(a, b)
    
def mixed_normalization_isclose(a, b):
    return isclose(a, b/sqrt(3))
    
def mixed_transpose_isclose(a, b):
    return isclose(a, 3*b)
    
conversion_test(V, mixed_conversion_isclose)
normalization_test(V, A, mixed_normalization_isclose)
transpose_test(V, A, b, mixed_transpose_isclose)
