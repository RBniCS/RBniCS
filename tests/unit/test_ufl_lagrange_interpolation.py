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

import pytest
from numpy import isclose
from dolfin import Constant, Expression, FiniteElement, Function, FunctionSpace, LagrangeInterpolator, MixedElement, project, UnitSquareMesh, VectorElement, VectorFunctionSpace
from rbnics.backends.dolfin.wrapping import ufl_lagrange_interpolation

# Mesh
@pytest.fixture(scope="module")
def mesh():
    return UnitSquareMesh(2, 2)
    
# Prepare a Function containing the coordinate x.
# We cannot use SpatialCoordinates since it is not implemented with dP
# Moreover, we use CG2 elements to show that the expression may involve FE functions of arbitrary degree
@pytest.fixture(scope="module")
def X(mesh):
    return VectorFunctionSpace(mesh, "Lagrange", 2)
    
@pytest.fixture(scope="module")
def x(X):
    return project(Expression(("x[0]", "x[1]"), degree=1), X)

# Prepare a Function on a mixed function space
@pytest.fixture(scope="module")
def XS(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    return FunctionSpace(mesh, element)
    
@pytest.fixture(scope="module")
def xs(XS):
    return project(Expression(("x[0]", "x[1]", "x[0] + x[1]"), degree=1), XS)

# ~~~ Scalar case ~~~ #
def ScalarSpace(mesh):
    return FunctionSpace(mesh, "Lagrange", 1)

def test_scalar_1(mesh):
    V = ScalarSpace(mesh)
    e1a = Expression("x[0] + pow(x[1], 2)", element=V.ufl_element())
    u1a = Function(V)
    LagrangeInterpolator.interpolate(u1a, e1a)
    e1b = e1a
    u1b = Function(V)
    ufl_lagrange_interpolation(u1b, e1b)
    assert isclose(u1a.vector().array(), u1b.vector().array()).all()

def test_scalar_2(mesh):
    V = ScalarSpace(mesh)
    e2a = Expression("x[0] + pow(x[1], 2) + 1", element=V.ufl_element())
    u2a = Function(V)
    LagrangeInterpolator.interpolate(u2a, e2a)
    e2b = Expression("x[0] + pow(x[1], 2)", element=V.ufl_element()) + 1
    u2b = Function(V)
    ufl_lagrange_interpolation(u2b, e2b)
    assert isclose(u2a.vector().array(), u2b.vector().array()).all()

def test_scalar_3(mesh, x):
    V = ScalarSpace(mesh)
    e3a = Expression("x[0] + pow(x[1], 2)", element=V.ufl_element())
    u3a = Function(V)
    LagrangeInterpolator.interpolate(u3a, e3a)
    e3b = x[0] + x[1]**2
    u3b = Function(V)
    ufl_lagrange_interpolation(u3b, e3b)
    assert isclose(u3a.vector().array(), u3b.vector().array()).all()

def test_scalar_4(mesh, x):
    V = ScalarSpace(mesh)
    e4a = Expression("x[0] + pow(x[1], 2) + 1", element=V.ufl_element())
    u4a = Function(V)
    LagrangeInterpolator.interpolate(u4a, e4a)
    e4b = x[0] + x[1]**2 + 1
    u4b = Function(V)
    ufl_lagrange_interpolation(u4b, e4b)
    assert isclose(u4a.vector().array(), u4b.vector().array()).all()

def test_scalar_5(mesh, xs):
    V = ScalarSpace(mesh)
    e5a = Expression("2*x[0] + x[1] + pow(x[1], 2)", element=V.ufl_element())
    u5a = Function(V)
    LagrangeInterpolator.interpolate(u5a, e5a)
    e5b = xs[0] + xs[1]**2 + xs[2]
    u5b = Function(V)
    ufl_lagrange_interpolation(u5b, e5b)
    assert isclose(u5a.vector().array(), u5b.vector().array()).all()

def test_scalar_6(mesh, xs):
    V = ScalarSpace(mesh)
    e6a = Expression("2*x[0] + x[1] + pow(x[1], 2) + 1", element=V.ufl_element())
    u6a = Function(V)
    LagrangeInterpolator.interpolate(u6a, e6a)
    e6b = xs[0] + xs[1]**2 + xs[2] + 1
    u6b = Function(V)
    ufl_lagrange_interpolation(u6b, e6b)
    assert isclose(u6a.vector().array(), u6b.vector().array()).all()

# ~~~ Vector case ~~~ #
def VectorSpace(mesh):
    return VectorFunctionSpace(mesh, "Lagrange", 1)

def test_vector_1(mesh):
    V = VectorSpace(mesh)
    e1a = Expression(("x[0] + pow(x[1], 2)", "pow(x[0], 3) + pow(x[1], 4)"), element=V.ufl_element())
    u1a = Function(V)
    LagrangeInterpolator.interpolate(u1a, e1a)
    e1b = e1a
    u1b = Function(V)
    ufl_lagrange_interpolation(u1b, e1b)
    assert isclose(u1a.vector().array(), u1b.vector().array()).all()

def test_vector_2(mesh):
    V = VectorSpace(mesh)
    e2a = Expression(("x[0] + pow(x[1], 2) + 1", "pow(x[0], 3) + pow(x[1], 4) + 2"), element=V.ufl_element())
    u2a = Function(V)
    LagrangeInterpolator.interpolate(u2a, e2a)
    e2b = Expression(("x[0] + pow(x[1], 2)", "pow(x[0], 3) + pow(x[1], 4)"), element=V.ufl_element()) + Constant((1, 2))
    u2b = Function(V)
    ufl_lagrange_interpolation(u2b, e2b)
    assert isclose(u2a.vector().array(), u2b.vector().array()).all()

def test_vector_3(mesh, X):
    V = VectorSpace(mesh)
    e3a = Expression(("x[0] + pow(x[1], 2)", "pow(x[0], 3) + pow(x[1], 4)"), element=V.ufl_element())
    u3a = Function(V)
    LagrangeInterpolator.interpolate(u3a, e3a)
    e3b = project(e3a, X)
    u3b = Function(V)
    ufl_lagrange_interpolation(u3b, e3b)
    assert isclose(u3a.vector().array(), u3b.vector().array()).all()

def test_vector_4(mesh, X):
    V = VectorSpace(mesh)
    e4a = Expression(("x[0] + pow(x[1], 2) + 1", "pow(x[0], 3) + pow(x[1], 4) + 2"), element=V.ufl_element())
    u4a = Function(V)
    LagrangeInterpolator.interpolate(u4a, e4a)
    e4b = project(Expression(("x[0] + pow(x[1], 2)", "pow(x[0], 3) + pow(x[1], 4)"), element=V.ufl_element()), X) + Constant((1, 2))
    u4b = Function(V)
    ufl_lagrange_interpolation(u4b, e4b)
    assert isclose(u4a.vector().array(), u4b.vector().array()).all()
