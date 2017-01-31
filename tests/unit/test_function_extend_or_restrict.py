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
## @file 
#  @brief 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from numpy import isclose
from dolfin import *
from RBniCS.backends.fenics import Function
from RBniCS.backends.fenics.wrapping import function_extend_or_restrict, FunctionSpace

mesh = UnitSquareMesh(10, 10)

# ~~~ Scalar case ~~~ #
V = FunctionSpace(mesh, "Lagrange", 2)
W = V
u = Function(V)
u.vector()[:] = 1.

v = function_extend_or_restrict(u, None, W, None, weight=None, copy=False)
assert isclose(v.vector().array(), 1.).all()
assert u is v

v = function_extend_or_restrict(u, None, W, None, weight=None, copy=True)
assert isclose(v.vector().array(), 1.).all()
assert u is not v
assert v.vector().size() == W.dim()

v = function_extend_or_restrict(u, None, W, None, weight=2., copy=True)
assert isclose(v.vector().array(), 2.).all()
assert u is not v
assert v.vector().size() == W.dim()


# ~~~ Vector case ~~~ #
V = VectorFunctionSpace(mesh, "Lagrange", 2)
W = V
u = Function(V)
u.vector()[:] = 1.

v = function_extend_or_restrict(u, None, W, None, weight=None, copy=False)
assert isclose(v.vector().array(), 1.).all()
assert u is v

v = function_extend_or_restrict(u, None, W, None, weight=None, copy=True)
assert isclose(v.vector().array(), 1.).all()
assert u is not v
assert v.vector().size() == W.dim()

v = function_extend_or_restrict(u, None, W, None, weight=2., copy=True)
assert isclose(v.vector().array(), 2.).all()
assert u is not v
assert v.vector().size() == W.dim()

# ~~~ Mixed case: extension, automatic detection of components ~~~ #
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element_0)
W = FunctionSpace(mesh, element)
s = Function(V)
s.vector()[:] = 1.

try:
    extended_s = function_extend_or_restrict(s, None, W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extend functions without copying the vector"
    
extended_s = function_extend_or_restrict(s, None, W, None, weight=None, copy=True)
assert extended_s.vector().size() == W.dim()
(u, p) = extended_s.split(deepcopy=True)
assert isclose(u.vector().array(), 1.).all()
assert isclose(p.vector().array(), 0.).all()

extended_s = function_extend_or_restrict(s, None, W, None, weight=2., copy=True)
assert extended_s.vector().size() == W.dim()
(u, p) = extended_s.split(deepcopy=True)
assert isclose(u.vector().array(), 2.).all()
assert isclose(p.vector().array(), 0.).all()

# ~~~ Mixed case: extension, ambiguous extension due to failing automatic detection of components ~~~ #
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element_0)
W = FunctionSpace(mesh, element)
s = Function(V)
s.vector()[:] = 1.

try:
    extended_s = function_extend_or_restrict(s, None, W, None, weight=None, copy=False)
except RuntimeError as e:
    assert str(e) == "Ambiguity when querying _function_spaces_lt"
    
# ~~~ Mixed case: extension, avoid ambiguity thanks to user provided input components ~~~ #
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element_0)
s = Function(V)
s.vector()[:] = 1.

W = FunctionSpace(mesh, element)

try:
    extended_s = function_extend_or_restrict(s, None, W, 0, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"
    
extended_s = function_extend_or_restrict(s, None, W, 0, weight=None, copy=True)
assert extended_s.vector().size() == W.dim()
(u, p) = extended_s.split(deepcopy=True)
assert isclose(u.vector().array(), 1.).all()
assert isclose(p.vector().array(), 0.).all()

extended_s = function_extend_or_restrict(s, None, W, 0, weight=2., copy=True)
assert extended_s.vector().size() == W.dim()
(u, p) = extended_s.split(deepcopy=True)
assert isclose(u.vector().array(), 2.).all()
assert isclose(p.vector().array(), 0.).all()

W = FunctionSpace(mesh, element, components=[["u", "s"], "p"])

try:
    extended_s = function_extend_or_restrict(s, None, W, "s", weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"
    
extended_s = function_extend_or_restrict(s, None, W, "s", weight=None, copy=True)
assert extended_s.vector().size() == W.dim()
(u, p) = extended_s.split(deepcopy=True)
assert isclose(u.vector().array(), 1.).all()
assert isclose(p.vector().array(), 0.).all()

extended_s = function_extend_or_restrict(s, None, W, "s", weight=2., copy=True)
assert extended_s.vector().size() == W.dim()
(u, p) = extended_s.split(deepcopy=True)
assert isclose(u.vector().array(), 2.).all()
assert isclose(p.vector().array(), 0.).all()

# ~~~ Mixed case: extension from sub element, ambiguous extension due to failing automatic detection of components ~~~ #
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # Note that we need to use 2nd order FE otherwise
element_00 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)  # the automatic detection would extend s into the 
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # pressure component
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element_00)
W = FunctionSpace(mesh, element)
s = Function(V)
s.vector()[:] = 1.

try:
    extended_s = function_extend_or_restrict(s, None, W, None, weight=None, copy=False)
except RuntimeError as e:
    assert str(e) == "Ambiguity when querying _function_spaces_lt"
    
# ~~~ Mixed case: extension from sub element, avoid ambiguity thanks to user provided input components ~~~ #
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
element_00 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element_00)
s = Function(V)
s.vector()[:] = 1.

W = FunctionSpace(mesh, element)

try:
    extended_s = function_extend_or_restrict(s, None, W, (0, 0), weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"
    
extended_s = function_extend_or_restrict(s, None, W, (0, 0), weight=None, copy=True)
assert extended_s.vector().size() == W.dim()
(u, p) = extended_s.split(deepcopy=True)
(ux, uy) = u.split(deepcopy=True)
assert isclose(ux.vector().array(), 1.).all()
assert isclose(uy.vector().array(), 0.).all()
assert isclose(p.vector().array(), 0.).all()

extended_s = function_extend_or_restrict(s, None, W, (0, 0), weight=2., copy=True)
assert extended_s.vector().size() == W.dim()
(u, p) = extended_s.split(deepcopy=True)
(ux, uy) = u.split(deepcopy=True)
assert isclose(ux.vector().array(), 2.).all()
assert isclose(uy.vector().array(), 0.).all()
assert isclose(p.vector().array(), 0.).all()

W = FunctionSpace(mesh, element, components=[("ux", "uy"), "p"])

try:
    extended_s = function_extend_or_restrict(s, None, W, "ux", weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"
    
extended_s = function_extend_or_restrict(s, None, W, "ux", weight=None, copy=True)
assert extended_s.vector().size() == W.dim()
(u, p) = extended_s.split(deepcopy=True)
(ux, uy) = u.split(deepcopy=True)
assert isclose(ux.vector().array(), 1.).all()
assert isclose(uy.vector().array(), 0.).all()
assert isclose(p.vector().array(), 0.).all()

extended_s = function_extend_or_restrict(s, None, W, "ux", weight=2., copy=True)
assert extended_s.vector().size() == W.dim()
(u, p) = extended_s.split(deepcopy=True)
(ux, uy) = u.split(deepcopy=True)
assert isclose(ux.vector().array(), 2.).all()
assert isclose(uy.vector().array(), 0.).all()
assert isclose(p.vector().array(), 0.).all()
    
# ~~~ Mixed case: restriction, automatic detection of components ~~~ #
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element)
up = Function(V)
assign(up.sub(0), project(Constant((1., 1.)), V.sub(0).collapse()))
assign(up.sub(1), project(Constant(2.), V.sub(1).collapse()))

W = FunctionSpace(mesh, element_0)

try:
    u = function_extend_or_restrict(up, None, W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to restrict functions without copying the vector"

u = function_extend_or_restrict(up, None, W, None, weight=None, copy=True)
assert u.vector().size() == W.dim()
assert isclose(u.vector().array(), 1.).all()

u = function_extend_or_restrict(up, None, W, None, weight=2., copy=True)
assert u.vector().size() == W.dim()
assert isclose(u.vector().array(), 2.).all()

W = FunctionSpace(mesh, element_1)

try:
    p = function_extend_or_restrict(up, None, W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to restrict functions without copying the vector"

p = function_extend_or_restrict(up, None, W, None, weight=None, copy=True)
assert p.vector().size() == W.dim()
assert isclose(p.vector().array(), 2.).all()

p = function_extend_or_restrict(up, None, W, None, weight=2., copy=True)
assert p.vector().size() == W.dim()
assert isclose(p.vector().array(), 4.).all()

# ~~~ Mixed case: restriction, ambiguous extension due to failing automatic detection of components ~~~ #
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element)
up = Function(V)
assign(up.sub(0), project(Constant((1., 1.)), V.sub(0).collapse()))
assign(up.sub(1), project(Constant(2.), V.sub(1).collapse()))

W = FunctionSpace(mesh, element_0)

try:
    u = function_extend_or_restrict(up, None, W, None, weight=None, copy=False)
except RuntimeError as e:
    assert str(e) == "Ambiguity when querying _function_spaces_lt"

# ~~~ Mixed case: restriction, avoid ambiguity thanks to user provided input components ~~~ #
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element, components=["u", "p"])
up = Function(V)
assign(up.sub(0), project(Constant((1., 1.)), V.sub(0).collapse()))
assign(up.sub(1), project(Constant(2.), V.sub(1).collapse()))

W = FunctionSpace(mesh, element_0)

try:
    u = function_extend_or_restrict(up, 0, W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"

u = function_extend_or_restrict(up, 0, W, None, weight=None, copy=True)
assert u.vector().size() == W.dim()
assert isclose(u.vector().array(), 1.).all()

u = function_extend_or_restrict(up, 0, W, None, weight=2., copy=True)
assert u.vector().size() == W.dim()
assert isclose(u.vector().array(), 2.).all()

try:
    u = function_extend_or_restrict(up, "u", W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"

u = function_extend_or_restrict(up, "u", W, None, weight=None, copy=True)
assert u.vector().size() == W.dim()
assert isclose(u.vector().array(), 1.).all()

u = function_extend_or_restrict(up, "u", W, None, weight=2., copy=True)
assert u.vector().size() == W.dim()
assert isclose(u.vector().array(), 2.).all()

W = FunctionSpace(mesh, element_1)

try:
    p = function_extend_or_restrict(up, "p", W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"

p = function_extend_or_restrict(up, "p", W, None, weight=None, copy=True)
assert p.vector().size() == W.dim()
assert isclose(p.vector().array(), 2.).all()

p = function_extend_or_restrict(up, "p", W, None, weight=2., copy=True)
assert p.vector().size() == W.dim()
assert isclose(p.vector().array(), 4.).all()

# ~~~ Mixed case: restriction to sub element, ambiguous extension due to failing automatic detection of components ~~~ #
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # Note that we need to use 2nd order FE otherwise
element_00 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)  # the automatic detection would restrict the 
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # pressure component
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element)
up = Function(V)
assign(up.sub(0).sub(0), project(Constant(1.), V.sub(0).sub(0).collapse()))
assign(up.sub(0).sub(1), project(Constant(3.), V.sub(0).sub(1).collapse()))
assign(up.sub(1), project(Constant(2.), V.sub(1).collapse()))

W = FunctionSpace(mesh, element_00)

try:
    ux = function_extend_or_restrict(up, None, W, None, weight=None, copy=False)
except RuntimeError as e:
    assert str(e) == "Ambiguity when querying _function_spaces_lt"
    
# ~~~ Mixed case: restriction to sub element, avoid ambiguity thanks to user provided input components ~~~ #
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
element_00 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element, components=[("ux", "uy"), "p"])
up = Function(V)
assign(up.sub(0).sub(0), project(Constant(1.), V.sub(0).sub(0).collapse()))
assign(up.sub(0).sub(1), project(Constant(3.), V.sub(0).sub(1).collapse()))
assign(up.sub(1), project(Constant(2.), V.sub(1).collapse()))

W = FunctionSpace(mesh, element_00)

try:
    ux = function_extend_or_restrict(up, (0, 0), W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"

ux = function_extend_or_restrict(up, (0, 0), W, None, weight=None, copy=True)
assert ux.vector().size() == W.dim()
assert isclose(ux.vector().array(), 1.).all()

ux = function_extend_or_restrict(up, (0, 0), W, None, weight=2., copy=True)
assert ux.vector().size() == W.dim()
assert isclose(ux.vector().array(), 2.).all()

try:
    uy = function_extend_or_restrict(up, (0, 1), W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"

uy = function_extend_or_restrict(up, (0, 1), W, None, weight=None, copy=True)
assert uy.vector().size() == W.dim()
assert isclose(uy.vector().array(), 3.).all()

uy = function_extend_or_restrict(up, (0, 1), W, None, weight=2., copy=True)
assert uy.vector().size() == W.dim()
assert isclose(uy.vector().array(), 6.).all()

try:
    ux = function_extend_or_restrict(up, "ux", W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"

ux = function_extend_or_restrict(up, "ux", W, None, weight=None, copy=True)
assert ux.vector().size() == W.dim()
assert isclose(ux.vector().array(), 1.).all()

ux = function_extend_or_restrict(up, "ux", W, None, weight=2., copy=True)
assert ux.vector().size() == W.dim()
assert isclose(ux.vector().array(), 2.).all()

try:
    uy = function_extend_or_restrict(up, "uy", W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"

uy = function_extend_or_restrict(up, "uy", W, None, weight=None, copy=True)
assert uy.vector().size() == W.dim()
assert isclose(uy.vector().array(), 3.).all()

uy = function_extend_or_restrict(up, "uy", W, None, weight=2., copy=True)
assert uy.vector().size() == W.dim()
assert isclose(uy.vector().array(), 6.).all()

# ~~~ Mixed case to mixed case: copy only a component, in the same location ~~~ #
element_0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element, components=["ux", "uy"])
W = V
u = Function(V)
assign(u.sub(0), project(Constant(1.), V.sub(0).collapse()))
assign(u.sub(1), project(Constant(2.), V.sub(1).collapse()))

try:
    copied_u = function_extend_or_restrict(u, 0, W, 0, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"

copied_u = function_extend_or_restrict(u, 0, W, 0, weight=None, copy=True)
assert copied_u.vector().size() == W.dim()
(copied_ux, copied_uy) = copied_u.split(deepcopy=True)
assert isclose(copied_ux.vector().array(), 1.).all()
assert isclose(copied_uy.vector().array(), 0.).all()

copied_u = function_extend_or_restrict(u, 0, W, 0, weight=2., copy=True)
assert copied_u.vector().size() == W.dim()
(copied_ux, copied_uy) = copied_u.split(deepcopy=True)
assert isclose(copied_ux.vector().array(), 2.).all()
assert isclose(copied_uy.vector().array(), 0.).all()

try:
    copied_u = function_extend_or_restrict(u, "ux", W, "ux", weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"

copied_u = function_extend_or_restrict(u, "ux", W, "ux", weight=None, copy=True)
assert copied_u.vector().size() == W.dim()
(copied_ux, copied_uy) = copied_u.split(deepcopy=True)
assert isclose(copied_ux.vector().array(), 1.).all()
assert isclose(copied_uy.vector().array(), 0.).all()

copied_u = function_extend_or_restrict(u, "ux", W, "ux", weight=2., copy=True)
assert copied_u.vector().size() == W.dim()
(copied_ux, copied_uy) = copied_u.split(deepcopy=True)
assert isclose(copied_ux.vector().array(), 2.).all()
assert isclose(copied_uy.vector().array(), 0.).all()

# ~~~ Mixed case to mixed case: copy only a component, to a different location ~~~ #
element_0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element, components=["ux", "uy"])
W = V
u = Function(V)
assign(u.sub(0), project(Constant(1.), V.sub(0).collapse()))
assign(u.sub(1), project(Constant(2.), V.sub(1).collapse()))

try:
    copied_u = function_extend_or_restrict(u, 0, W, 1, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"

copied_u = function_extend_or_restrict(u, 0, W, 1, weight=None, copy=True)
assert copied_u.vector().size() == W.dim()
(copied_ux, copied_uy) = copied_u.split(deepcopy=True)
assert isclose(copied_ux.vector().array(), 0.).all()
assert isclose(copied_uy.vector().array(), 1.).all()

copied_u = function_extend_or_restrict(u, 0, W, 1, weight=2., copy=True)
assert copied_u.vector().size() == W.dim()
(copied_ux, copied_uy) = copied_u.split(deepcopy=True)
assert isclose(copied_ux.vector().array(), 0.).all()
assert isclose(copied_uy.vector().array(), 2.).all()

try:
    copied_u = function_extend_or_restrict(u, "ux", W, "uy", weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"

copied_u = function_extend_or_restrict(u, "ux", W, "uy", weight=None, copy=True)
assert copied_u.vector().size() == W.dim()
(copied_ux, copied_uy) = copied_u.split(deepcopy=True)
assert isclose(copied_ux.vector().array(), 0.).all()
assert isclose(copied_uy.vector().array(), 1.).all()

copied_u = function_extend_or_restrict(u, "ux", W, "uy", weight=2., copy=True)
assert copied_u.vector().size() == W.dim()
(copied_ux, copied_uy) = copied_u.split(deepcopy=True)
assert isclose(copied_ux.vector().array(), 0.).all()
assert isclose(copied_uy.vector().array(), 2.).all()

# ~~~ Mixed case to mixed case: copy only a sub component, in the same location ~~~ #
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element, components=[("uxx", "uxy"), ("uyx", "uyy")])
W = V
u = Function(V)
assign(u.sub(0), project(Constant((1., 2.)), V.sub(0).collapse()))
assign(u.sub(1), project(Constant((3., 4.)), V.sub(1).collapse()))

try:
    copied_u = function_extend_or_restrict(u, (0, 0), W, (0, 0), weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"

copied_u = function_extend_or_restrict(u, (0, 0), W, (0, 0), weight=None, copy=True)
assert copied_u.vector().size() == W.dim()
(copied_ux, copied_uy) = copied_u.split(deepcopy=True)
(copied_uxx, copied_uxy) = copied_ux.split(deepcopy=True)
(copied_uyx, copied_uyy) = copied_uy.split(deepcopy=True)
assert isclose(copied_uxx.vector().array(), 1.).all()
assert isclose(copied_uxy.vector().array(), 0.).all()
assert isclose(copied_uyx.vector().array(), 0.).all()
assert isclose(copied_uyy.vector().array(), 0.).all()

copied_u = function_extend_or_restrict(u, (0, 0), W, (0, 0), weight=2., copy=True)
assert copied_u.vector().size() == W.dim()
(copied_ux, copied_uy) = copied_u.split(deepcopy=True)
(copied_uxx, copied_uxy) = copied_ux.split(deepcopy=True)
(copied_uyx, copied_uyy) = copied_uy.split(deepcopy=True)
assert isclose(copied_uxx.vector().array(), 2.).all()
assert isclose(copied_uxy.vector().array(), 0.).all()
assert isclose(copied_uyx.vector().array(), 0.).all()
assert isclose(copied_uyy.vector().array(), 0.).all()

try:
    copied_u = function_extend_or_restrict(u, "uxx", W, "uxx", weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"

copied_u = function_extend_or_restrict(u, "uxx", W, "uxx", weight=None, copy=True)
assert copied_u.vector().size() == W.dim()
(copied_ux, copied_uy) = copied_u.split(deepcopy=True)
(copied_uxx, copied_uxy) = copied_ux.split(deepcopy=True)
(copied_uyx, copied_uyy) = copied_uy.split(deepcopy=True)
assert isclose(copied_uxx.vector().array(), 1.).all()
assert isclose(copied_uxy.vector().array(), 0.).all()
assert isclose(copied_uyx.vector().array(), 0.).all()
assert isclose(copied_uyy.vector().array(), 0.).all()

copied_u = function_extend_or_restrict(u, "uxx", W, "uxx", weight=2., copy=True)
assert copied_u.vector().size() == W.dim()
(copied_ux, copied_uy) = copied_u.split(deepcopy=True)
(copied_uxx, copied_uxy) = copied_ux.split(deepcopy=True)
(copied_uyx, copied_uyy) = copied_uy.split(deepcopy=True)
assert isclose(copied_uxx.vector().array(), 2.).all()
assert isclose(copied_uxy.vector().array(), 0.).all()
assert isclose(copied_uyx.vector().array(), 0.).all()
assert isclose(copied_uyy.vector().array(), 0.).all()

# ~~~ Mixed case to mixed case: copy only a sub component, to a different location ~~~ #
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element, components=[("uxx", "uxy"), ("uyx", "uyy")])
W = V
u = Function(V)
assign(u.sub(0), project(Constant((1., 2.)), V.sub(0).collapse()))
assign(u.sub(1), project(Constant((3., 4.)), V.sub(1).collapse()))

try:
    copied_u = function_extend_or_restrict(u, (0, 0), W, (1, 0), weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"

copied_u = function_extend_or_restrict(u, (0, 0), W, (1, 0), weight=None, copy=True)
assert copied_u.vector().size() == W.dim()
(copied_ux, copied_uy) = copied_u.split(deepcopy=True)
(copied_uxx, copied_uxy) = copied_ux.split(deepcopy=True)
(copied_uyx, copied_uyy) = copied_uy.split(deepcopy=True)
assert isclose(copied_uxx.vector().array(), 0.).all()
assert isclose(copied_uxy.vector().array(), 0.).all()
assert isclose(copied_uyx.vector().array(), 1.).all()
assert isclose(copied_uyy.vector().array(), 0.).all()

copied_u = function_extend_or_restrict(u, (0, 0), W, (1, 0), weight=2., copy=True)
assert copied_u.vector().size() == W.dim()
(copied_ux, copied_uy) = copied_u.split(deepcopy=True)
(copied_uxx, copied_uxy) = copied_ux.split(deepcopy=True)
(copied_uyx, copied_uyy) = copied_uy.split(deepcopy=True)
assert isclose(copied_uxx.vector().array(), 0.).all()
assert isclose(copied_uxy.vector().array(), 0.).all()
assert isclose(copied_uyx.vector().array(), 2.).all()
assert isclose(copied_uyy.vector().array(), 0.).all()

try:
    copied_u = function_extend_or_restrict(u, "uxx", W, "uyx", weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract function components without copying the vector"

copied_u = function_extend_or_restrict(u, "uxx", W, "uyx", weight=None, copy=True)
assert copied_u.vector().size() == W.dim()
(copied_ux, copied_uy) = copied_u.split(deepcopy=True)
(copied_uxx, copied_uxy) = copied_ux.split(deepcopy=True)
(copied_uyx, copied_uyy) = copied_uy.split(deepcopy=True)
assert isclose(copied_uxx.vector().array(), 0.).all()
assert isclose(copied_uxy.vector().array(), 0.).all()
assert isclose(copied_uyx.vector().array(), 1.).all()
assert isclose(copied_uyy.vector().array(), 0.).all()

copied_u = function_extend_or_restrict(u, "uxx", W, "uyx", weight=2., copy=True)
assert copied_u.vector().size() == W.dim()
(copied_ux, copied_uy) = copied_u.split(deepcopy=True)
(copied_uxx, copied_uxy) = copied_ux.split(deepcopy=True)
(copied_uyx, copied_uyy) = copied_uy.split(deepcopy=True)
assert isclose(copied_uxx.vector().array(), 0.).all()
assert isclose(copied_uxy.vector().array(), 0.).all()
assert isclose(copied_uyx.vector().array(), 2.).all()
assert isclose(copied_uyy.vector().array(), 0.).all()
