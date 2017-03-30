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

from numpy import isclose
from dolfin import *
import time
from rbnics.backends.fenics import transpose
from rbnics.backends.fenics.wrapping import ufl_lagrange_interpolation

mesh = UnitSquareMesh(2, 2)
interpolator = LagrangeInterpolator()

# Prepare a Function containing the coordinate x.
# We cannot use SpatialCoordinates since it is not implemented with dP
# Moreover, we use CG2 elements to show that the expression may involve FE functions of arbitrary degree
x_expression = Expression(("x[0]", "x[1]"), degree=1)
X = VectorFunctionSpace(mesh, "Lagrange", 2)
x = project(x_expression, X)

# Prepare a Function on a mixed function space
xs_expression = Expression(("x[0]", "x[1]", "x[0] + x[1]"), degree=1)
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
XS = FunctionSpace(mesh, element)
xs = project(xs_expression, XS)

# ~~~ Scalar case ~~~ #
V = FunctionSpace(mesh, "Lagrange", 1)

e1a = Expression("x[0] + pow(x[1], 2)", element=V.ufl_element())
u1a = Function(V)
interpolator.interpolate(u1a, e1a)
e1b = e1a
u1b = Function(V)
ufl_lagrange_interpolation(u1b, e1b)
assert isclose(u1a.vector().array(), u1b.vector().array()).all()

e2a = Expression("x[0] + pow(x[1], 2) + 1", element=V.ufl_element())
u2a = Function(V)
interpolator.interpolate(u2a, e2a)
e2b = e1b + 1
u2b = Function(V)
ufl_lagrange_interpolation(u2b, e2b)
assert isclose(u2a.vector().array(), u2b.vector().array()).all()

e3a = e1a
u3a = Function(V)
interpolator.interpolate(u3a, e3a)
e3b = x[0] + x[1]**2
u3b = Function(V)
ufl_lagrange_interpolation(u3b, e3b)
assert isclose(u3a.vector().array(), u3b.vector().array()).all()

e4a = e2a
u4a = Function(V)
interpolator.interpolate(u4a, e4a)
e4b = e3b + 1
u4b = Function(V)
ufl_lagrange_interpolation(u4b, e4b)
assert isclose(u4a.vector().array(), u4b.vector().array()).all()

e5a = Expression("2*x[0] + x[1] + pow(x[1], 2)", element=V.ufl_element())
u5a = Function(V)
interpolator.interpolate(u5a, e5a)
e5b = xs[0] + xs[1]**2 + xs[2]
u5b = Function(V)
ufl_lagrange_interpolation(u5b, e5b)
assert isclose(u5a.vector().array(), u5b.vector().array()).all()

e6a = Expression("2*x[0] + x[1] + pow(x[1], 2) + 1", element=V.ufl_element())
u6a = Function(V)
interpolator.interpolate(u6a, e6a)
e6b = e5b + 1
u6b = Function(V)
ufl_lagrange_interpolation(u6b, e6b)
assert isclose(u6a.vector().array(), u6b.vector().array()).all()

# ~~~ Vector case ~~~ #
V = VectorFunctionSpace(mesh, "Lagrange", 1)

e1a = Expression(("x[0] + pow(x[1], 2)", "pow(x[0], 3) + pow(x[1], 4)"), element=V.ufl_element())
u1a = Function(V)
interpolator.interpolate(u1a, e1a)
e1b = e1a
u1b = Function(V)
ufl_lagrange_interpolation(u1b, e1b)
assert isclose(u1a.vector().array(), u1b.vector().array()).all()

e2a = Expression(("x[0] + pow(x[1], 2) + 1", "pow(x[0], 3) + pow(x[1], 4) + 2"), element=V.ufl_element())
u2a = Function(V)
interpolator.interpolate(u2a, e2a)
e2b = e1b + Constant((1, 2))
u2b = Function(V)
ufl_lagrange_interpolation(u2b, e2b)
assert isclose(u2a.vector().array(), u2b.vector().array()).all()

e3a = e1a
u3a = Function(V)
interpolator.interpolate(u3a, e3a)
e3b = project(e3a, X)
u3b = Function(V)
ufl_lagrange_interpolation(u3b, e3b)
assert isclose(u3a.vector().array(), u3b.vector().array()).all()

e4a = e2a
u4a = Function(V)
interpolator.interpolate(u4a, e4a)
e4b = e3b + Constant((1, 2))
u4b = Function(V)
ufl_lagrange_interpolation(u4b, e4b)
assert isclose(u4a.vector().array(), u4b.vector().array()).all()

