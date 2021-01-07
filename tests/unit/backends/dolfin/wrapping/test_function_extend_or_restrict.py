# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from numpy import isclose
from dolfin import (assign, Constant, FiniteElement, MixedElement, project, UnitSquareMesh, VectorElement,
                    VectorFunctionSpace)
from dolfin_utils.test import fixture as module_fixture
from rbnics.backends.dolfin import Function
from rbnics.backends.dolfin.wrapping import function_extend_or_restrict, FunctionSpace


# Mesh
@module_fixture
def mesh():
    return UnitSquareMesh(10, 10)


# ~~~ Scalar case ~~~ #
def ScalarSpaces(mesh):
    V = FunctionSpace(mesh, "Lagrange", 2)
    W = V
    return (V, W)


def ScalarFunction(V):
    u = Function(V)
    u.vector()[:] = 1.
    return u


def test_scalar_1(mesh):
    (V, W) = ScalarSpaces(mesh)
    u = ScalarFunction(V)
    v = function_extend_or_restrict(u, None, W, None, weight=None, copy=False)
    assert isclose(v.vector().get_local(), 1.).all()
    assert u is v


def test_scalar_2_copy(mesh):
    (V, W) = ScalarSpaces(mesh)
    u = ScalarFunction(V)
    v = function_extend_or_restrict(u, None, W, None, weight=None, copy=True)
    assert isclose(v.vector().get_local(), 1.).all()
    assert u is not v
    assert v.vector().size() == W.dim()


def test_scalar_3_weight_copy(mesh):
    (V, W) = ScalarSpaces(mesh)
    u = ScalarFunction(V)
    v = function_extend_or_restrict(u, None, W, None, weight=2., copy=True)
    assert isclose(v.vector().get_local(), 2.).all()
    assert u is not v
    assert v.vector().size() == W.dim()


# ~~~ Vector case ~~~ #
def VectorSpaces(mesh):
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    W = V
    return (V, W)


def VectorFunction(V):
    u = Function(V)
    u.vector()[:] = 1.
    return u


def test_vector_1(mesh):
    (V, W) = VectorSpaces(mesh)
    u = VectorFunction(V)
    v = function_extend_or_restrict(u, None, W, None, weight=None, copy=False)
    assert isclose(v.vector().get_local(), 1.).all()
    assert u is v


def test_vector_2_copy(mesh):
    (V, W) = VectorSpaces(mesh)
    u = VectorFunction(V)
    v = function_extend_or_restrict(u, None, W, None, weight=None, copy=True)
    assert isclose(v.vector().get_local(), 1.).all()
    assert u is not v
    assert v.vector().size() == W.dim()


def test_vector_3_weight_copy(mesh):
    (V, W) = VectorSpaces(mesh)
    u = VectorFunction(V)
    v = function_extend_or_restrict(u, None, W, None, weight=2., copy=True)
    assert isclose(v.vector().get_local(), 2.).all()
    assert u is not v
    assert v.vector().size() == W.dim()


# ~~~ Mixed case: extension, automatic detection of components ~~~ #
def MixedSpacesExtensionAutomatic(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    V = FunctionSpace(mesh, element_0)
    W = FunctionSpace(mesh, element)
    return (V, W)


def MixedFunctionExtensionAutomatic(V):
    s = Function(V)
    s.vector()[:] = 1.
    return s


def test_mixed_function_extension_automatic_1_fail(mesh):
    (V, W) = MixedSpacesExtensionAutomatic(mesh)
    s = MixedFunctionExtensionAutomatic(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(s, None, W, None, weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extend functions without copying the vector"


def test_mixed_function_extension_automatic_2_copy(mesh):
    (V, W) = MixedSpacesExtensionAutomatic(mesh)
    s = MixedFunctionExtensionAutomatic(V)
    extended_s = function_extend_or_restrict(s, None, W, None, weight=None, copy=True)
    assert extended_s.vector().size() == W.dim()
    (u, p) = extended_s.split(deepcopy=True)
    assert isclose(u.vector().get_local(), 1.).all()
    assert isclose(p.vector().get_local(), 0.).all()


def test_mixed_function_extension_automatic_3_weight_copy(mesh):
    (V, W) = MixedSpacesExtensionAutomatic(mesh)
    s = MixedFunctionExtensionAutomatic(V)
    extended_s = function_extend_or_restrict(s, None, W, None, weight=2., copy=True)
    assert extended_s.vector().size() == W.dim()
    (u, p) = extended_s.split(deepcopy=True)
    assert isclose(u.vector().get_local(), 2.).all()
    assert isclose(p.vector().get_local(), 0.).all()


# ~~~ Mixed case: extension, ambiguous extension due to failing automatic detection of components ~~~ #
def MixedSpacesExtensionAmbiguous(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    V = FunctionSpace(mesh, element_0)
    W = FunctionSpace(mesh, element)
    return (V, W)


def MixedFunctionExtensionAmbiguous(V):
    s = Function(V)
    s.vector()[:] = 1.
    return s


def test_mixed_function_extension_ambiguous_1_fail(mesh):
    (V, W) = MixedSpacesExtensionAmbiguous(mesh)
    s = MixedFunctionExtensionAmbiguous(V)
    with pytest.raises(RuntimeError) as excinfo:
        function_extend_or_restrict(s, None, W, None, weight=None, copy=False)
    assert str(excinfo.value) == "Ambiguity when querying _function_spaces_lt"


# ~~~ Mixed case: extension, non ambiguous vector element ~~~ #
def MixedSpacesExtensionNonAmbiguousVectorElement(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    V = FunctionSpace(mesh, element_0)
    W = FunctionSpace(mesh, element)
    return (V, W)


def MixedFunctionExtensionNonAmbiguousVectorElement(V):
    s = Function(V)
    s.vector()[:] = 1.
    return s


def test_mixed_function_extension_non_ambiguous_vector_element_1_fail(mesh):
    (V, W) = MixedSpacesExtensionNonAmbiguousVectorElement(mesh)
    s = MixedFunctionExtensionNonAmbiguousVectorElement(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(s, None, W, 0, weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_function_extension_non_ambiguous_vector_element_2_copy(mesh):
    (V, W) = MixedSpacesExtensionNonAmbiguousVectorElement(mesh)
    s = MixedFunctionExtensionNonAmbiguousVectorElement(V)
    extended_s = function_extend_or_restrict(s, None, W, 0, weight=None, copy=True)
    assert extended_s.vector().size() == W.dim()
    (u, p) = extended_s.split(deepcopy=True)
    assert isclose(u.vector().get_local(), 1.).all()
    assert isclose(p.vector().get_local(), 0.).all()


def test_mixed_function_extension_non_ambiguous_vector_element_3_weight_copy(mesh):
    (V, W) = MixedSpacesExtensionNonAmbiguousVectorElement(mesh)
    s = MixedFunctionExtensionNonAmbiguousVectorElement(V)
    extended_s = function_extend_or_restrict(s, None, W, 0, weight=2., copy=True)
    assert extended_s.vector().size() == W.dim()
    (u, p) = extended_s.split(deepcopy=True)
    assert isclose(u.vector().get_local(), 2.).all()
    assert isclose(p.vector().get_local(), 0.).all()


# ~~~ Mixed case: extension, avoid ambiguity thanks to user provided input components ~~~ #
def MixedSpacesExtensionSolveAmbiguityWithComponents(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    V = FunctionSpace(mesh, element_0)
    W = FunctionSpace(mesh, element, components=[["u", "s"], "p"])
    return (V, W)


def MixedFunctionExtensionSolveAmbiguityWithComponents(V):
    s = Function(V)
    s.vector()[:] = 1.
    return s


def test_mixed_function_extension_solve_ambiguity_with_components_1_fail(mesh):
    (V, W) = MixedSpacesExtensionSolveAmbiguityWithComponents(mesh)
    s = MixedFunctionExtensionSolveAmbiguityWithComponents(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(s, None, W, "s", weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_function_extension_solve_ambiguity_with_components_2_copy(mesh):
    (V, W) = MixedSpacesExtensionSolveAmbiguityWithComponents(mesh)
    s = MixedFunctionExtensionSolveAmbiguityWithComponents(V)
    extended_s = function_extend_or_restrict(s, None, W, "s", weight=None, copy=True)
    assert extended_s.vector().size() == W.dim()
    (u, p) = extended_s.split(deepcopy=True)
    assert isclose(u.vector().get_local(), 1.).all()
    assert isclose(p.vector().get_local(), 0.).all()


def test_mixed_function_extension_solve_ambiguity_with_components_3_weight_copy(mesh):
    (V, W) = MixedSpacesExtensionSolveAmbiguityWithComponents(mesh)
    s = MixedFunctionExtensionSolveAmbiguityWithComponents(V)
    extended_s = function_extend_or_restrict(s, None, W, "s", weight=2., copy=True)
    assert extended_s.vector().size() == W.dim()
    (u, p) = extended_s.split(deepcopy=True)
    assert isclose(u.vector().get_local(), 2.).all()
    assert isclose(p.vector().get_local(), 0.).all()


# ~~~ Mixed case: extension from sub element, ambiguous extension due to failing automatic detection of components ~~~ #
def MixedSpacesExtensionFromSubElementAmbiguous(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # Note that we need to use 2nd order FE otherwise
    element_00 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)  # the automatic detection would extend s into the
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # pressure component
    element = MixedElement(element_0, element_1)
    V = FunctionSpace(mesh, element_00)
    W = FunctionSpace(mesh, element)
    return (V, W)


def MixedFunctionExtensionFromSubElementAmbiguous(V):
    s = Function(V)
    s.vector()[:] = 1.
    return s


def test_mixed_function_extension_from_sub_element_ambiguous_1_fail(mesh):
    (V, W) = MixedSpacesExtensionFromSubElementAmbiguous(mesh)
    s = MixedFunctionExtensionFromSubElementAmbiguous(V)
    with pytest.raises(RuntimeError) as excinfo:
        function_extend_or_restrict(s, None, W, None, weight=None, copy=False)
    assert str(excinfo.value) == "Ambiguity when querying _function_spaces_lt"


# ~~~ Mixed case: extension from sub element, avoid ambiguity thanks to user provided input components ~~~ #
def MixedSpacesExtensionFromSubElementSolveAmbiguityWithComponents(mesh, components):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    element_00 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    V = FunctionSpace(mesh, element_00)
    assert components in (tuple, str)
    if components is tuple:
        W = FunctionSpace(mesh, element)
    else:
        W = FunctionSpace(mesh, element, components=[("ux", "uy"), "p"])
    return (V, W)


def MixedFunctionExtensionFromSubElementSolveAmbiguityWithComponents(V):
    s = Function(V)
    s.vector()[:] = 1.
    return s


def test_mixed_function_extension_from_sub_element_solve_ambiguity_with_components_tuple_1_fail(mesh):
    (V, W) = MixedSpacesExtensionFromSubElementSolveAmbiguityWithComponents(mesh, tuple)
    s = MixedFunctionExtensionFromSubElementSolveAmbiguityWithComponents(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(s, None, W, (0, 0), weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_function_extension_from_sub_element_solve_ambiguity_with_components_tuple_2_copy(mesh):
    (V, W) = MixedSpacesExtensionFromSubElementSolveAmbiguityWithComponents(mesh, tuple)
    s = MixedFunctionExtensionFromSubElementSolveAmbiguityWithComponents(V)
    extended_s = function_extend_or_restrict(s, None, W, (0, 0), weight=None, copy=True)
    assert extended_s.vector().size() == W.dim()
    (u, p) = extended_s.split(deepcopy=True)
    (ux, uy) = u.split(deepcopy=True)
    assert isclose(ux.vector().get_local(), 1.).all()
    assert isclose(uy.vector().get_local(), 0.).all()
    assert isclose(p.vector().get_local(), 0.).all()


def test_mixed_function_extension_from_sub_element_solve_ambiguity_with_components_tuple_3_weight_copy(mesh):
    (V, W) = MixedSpacesExtensionFromSubElementSolveAmbiguityWithComponents(mesh, tuple)
    s = MixedFunctionExtensionFromSubElementSolveAmbiguityWithComponents(V)
    extended_s = function_extend_or_restrict(s, None, W, (0, 0), weight=2., copy=True)
    assert extended_s.vector().size() == W.dim()
    (u, p) = extended_s.split(deepcopy=True)
    (ux, uy) = u.split(deepcopy=True)
    assert isclose(ux.vector().get_local(), 2.).all()
    assert isclose(uy.vector().get_local(), 0.).all()
    assert isclose(p.vector().get_local(), 0.).all()


def test_mixed_function_extension_from_sub_element_solve_ambiguity_with_components_str_1_fail(mesh):
    (V, W) = MixedSpacesExtensionFromSubElementSolveAmbiguityWithComponents(mesh, str)
    s = MixedFunctionExtensionFromSubElementSolveAmbiguityWithComponents(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(s, None, W, "ux", weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_function_extension_from_sub_element_solve_ambiguity_with_components_str_2_copy(mesh):
    (V, W) = MixedSpacesExtensionFromSubElementSolveAmbiguityWithComponents(mesh, str)
    s = MixedFunctionExtensionFromSubElementSolveAmbiguityWithComponents(V)
    extended_s = function_extend_or_restrict(s, None, W, "ux", weight=None, copy=True)
    assert extended_s.vector().size() == W.dim()
    (u, p) = extended_s.split(deepcopy=True)
    (ux, uy) = u.split(deepcopy=True)
    assert isclose(ux.vector().get_local(), 1.).all()
    assert isclose(uy.vector().get_local(), 0.).all()
    assert isclose(p.vector().get_local(), 0.).all()


def test_mixed_function_extension_from_sub_element_solve_ambiguity_with_components_str_3_weight_copy(mesh):
    (V, W) = MixedSpacesExtensionFromSubElementSolveAmbiguityWithComponents(mesh, str)
    s = MixedFunctionExtensionFromSubElementSolveAmbiguityWithComponents(V)
    extended_s = function_extend_or_restrict(s, None, W, "ux", weight=2., copy=True)
    assert extended_s.vector().size() == W.dim()
    (u, p) = extended_s.split(deepcopy=True)
    (ux, uy) = u.split(deepcopy=True)
    assert isclose(ux.vector().get_local(), 2.).all()
    assert isclose(uy.vector().get_local(), 0.).all()
    assert isclose(p.vector().get_local(), 0.).all()


# ~~~ Mixed case: restriction, automatic detection of components ~~~ #
def MixedSpacesRestrictionAutomatic(mesh, sub_element):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    V = FunctionSpace(mesh, element)
    assert sub_element in (0, 1)
    if sub_element == 0:
        W = FunctionSpace(mesh, element_0)
    elif sub_element == 1:
        W = FunctionSpace(mesh, element_1)
    return (V, W)


def MixedFunctionRestrictionAutomatic(V):
    up = Function(V)
    assign(up.sub(0), project(Constant((1., 1.)), V.sub(0).collapse()))
    assign(up.sub(1), project(Constant(2.), V.sub(1).collapse()))
    return up


def test_mixed_function_restriction_automatic_first_sub_element_1_fail(mesh):
    (V, W) = MixedSpacesRestrictionAutomatic(mesh, 0)
    up = MixedFunctionRestrictionAutomatic(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(up, None, W, None, weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to restrict functions without copying the vector"


def test_mixed_function_restriction_automatic_first_sub_element_2_copy(mesh):
    (V, W) = MixedSpacesRestrictionAutomatic(mesh, 0)
    up = MixedFunctionRestrictionAutomatic(V)
    u = function_extend_or_restrict(up, None, W, None, weight=None, copy=True)
    assert u.vector().size() == W.dim()
    assert isclose(u.vector().get_local(), 1.).all()


def test_mixed_function_restriction_automatic_first_sub_element_3_weight_copy(mesh):
    (V, W) = MixedSpacesRestrictionAutomatic(mesh, 0)
    up = MixedFunctionRestrictionAutomatic(V)
    u = function_extend_or_restrict(up, None, W, None, weight=2., copy=True)
    assert u.vector().size() == W.dim()
    assert isclose(u.vector().get_local(), 2.).all()


def test_mixed_function_restriction_automatic_second_sub_element_1_fail(mesh):
    (V, W) = MixedSpacesRestrictionAutomatic(mesh, 1)
    up = MixedFunctionRestrictionAutomatic(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(up, None, W, None, weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to restrict functions without copying the vector"


def test_mixed_function_restriction_automatic_second_sub_element_2_copy(mesh):
    (V, W) = MixedSpacesRestrictionAutomatic(mesh, 1)
    up = MixedFunctionRestrictionAutomatic(V)
    p = function_extend_or_restrict(up, None, W, None, weight=None, copy=True)
    assert p.vector().size() == W.dim()
    assert isclose(p.vector().get_local(), 2.).all()


def test_mixed_function_restriction_automatic_second_sub_element_3_weight_copy(mesh):
    (V, W) = MixedSpacesRestrictionAutomatic(mesh, 1)
    up = MixedFunctionRestrictionAutomatic(V)
    p = function_extend_or_restrict(up, None, W, None, weight=2., copy=True)
    assert p.vector().size() == W.dim()
    assert isclose(p.vector().get_local(), 4.).all()


# ~~~ Mixed case: restriction, ambiguous restriction due to failing automatic detection of components ~~~ #
def MixedSpacesRestrictionAmbiguous(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    V = FunctionSpace(mesh, element)
    W = FunctionSpace(mesh, element_0)
    return (V, W)


def MixedFunctionRestrictionAmbiguous(V):
    up = Function(V)
    assign(up.sub(0), project(Constant((1., 1.)), V.sub(0).collapse()))
    assign(up.sub(1), project(Constant(2.), V.sub(1).collapse()))
    return up


def test_mixed_function_restriction_ambiguous_1_fail(mesh):
    (V, W) = MixedSpacesRestrictionAmbiguous(mesh)
    up = MixedFunctionRestrictionAmbiguous(V)
    with pytest.raises(RuntimeError) as excinfo:
        function_extend_or_restrict(up, None, W, None, weight=None, copy=False)
    assert str(excinfo.value) == "Ambiguity when querying _function_spaces_lt"


# ~~~ Mixed case: restriction, avoid ambiguity thanks to user provided input components ~~~ #
def MixedSpacesRestrictionSolveAmbiguityWithComponents(mesh, sub_element):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    V = FunctionSpace(mesh, element, components=["u", "p"])
    assert sub_element in (0, 1)
    if sub_element == 0:
        W = FunctionSpace(mesh, element_0)
    elif sub_element == 1:
        W = FunctionSpace(mesh, element_1)
    return (V, W)


def MixedFunctionRestrictionSolveAmbiguityWithComponents(V):
    up = Function(V)
    assign(up.sub(0), project(Constant((1., 1.)), V.sub(0).collapse()))
    assign(up.sub(1), project(Constant(2.), V.sub(1).collapse()))
    return up


def test_mixed_function_restriction_solve_ambiguity_with_components_first_sub_element_1_int_fail(mesh):
    (V, W) = MixedSpacesRestrictionSolveAmbiguityWithComponents(mesh, 0)
    up = MixedFunctionRestrictionSolveAmbiguityWithComponents(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(up, 0, W, None, weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_function_restriction_solve_ambiguity_with_components_first_sub_element_2_int_copy(mesh):
    (V, W) = MixedSpacesRestrictionSolveAmbiguityWithComponents(mesh, 0)
    up = MixedFunctionRestrictionSolveAmbiguityWithComponents(V)
    u = function_extend_or_restrict(up, 0, W, None, weight=None, copy=True)
    assert u.vector().size() == W.dim()
    assert isclose(u.vector().get_local(), 1.).all()


def test_mixed_function_restriction_solve_ambiguity_with_components_first_sub_element_3_int_weight_copy(mesh):
    (V, W) = MixedSpacesRestrictionSolveAmbiguityWithComponents(mesh, 0)
    up = MixedFunctionRestrictionSolveAmbiguityWithComponents(V)
    u = function_extend_or_restrict(up, 0, W, None, weight=2., copy=True)
    assert u.vector().size() == W.dim()
    assert isclose(u.vector().get_local(), 2.).all()


def test_mixed_function_restriction_solve_ambiguity_with_components_first_sub_element_4_str_fail(mesh):
    (V, W) = MixedSpacesRestrictionSolveAmbiguityWithComponents(mesh, 0)
    up = MixedFunctionRestrictionSolveAmbiguityWithComponents(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(up, "u", W, None, weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_function_restriction_solve_ambiguity_with_components_first_sub_element_5_str_copy(mesh):
    (V, W) = MixedSpacesRestrictionSolveAmbiguityWithComponents(mesh, 0)
    up = MixedFunctionRestrictionSolveAmbiguityWithComponents(V)
    u = function_extend_or_restrict(up, "u", W, None, weight=None, copy=True)
    assert u.vector().size() == W.dim()
    assert isclose(u.vector().get_local(), 1.).all()


def test_mixed_function_restriction_solve_ambiguity_with_components_first_sub_element_6_str_weight_copy(mesh):
    (V, W) = MixedSpacesRestrictionSolveAmbiguityWithComponents(mesh, 0)
    up = MixedFunctionRestrictionSolveAmbiguityWithComponents(V)
    u = function_extend_or_restrict(up, "u", W, None, weight=2., copy=True)
    assert u.vector().size() == W.dim()
    assert isclose(u.vector().get_local(), 2.).all()


def test_mixed_function_restriction_solve_ambiguity_with_components_second_sub_element_1_str_fail(mesh):
    (V, W) = MixedSpacesRestrictionSolveAmbiguityWithComponents(mesh, 1)
    up = MixedFunctionRestrictionSolveAmbiguityWithComponents(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(up, "p", W, None, weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_function_restriction_solve_ambiguity_with_components_second_sub_element_2_str_copy(mesh):
    (V, W) = MixedSpacesRestrictionSolveAmbiguityWithComponents(mesh, 1)
    up = MixedFunctionRestrictionSolveAmbiguityWithComponents(V)
    p = function_extend_or_restrict(up, "p", W, None, weight=None, copy=True)
    assert p.vector().size() == W.dim()
    assert isclose(p.vector().get_local(), 2.).all()


def test_mixed_function_restriction_solve_ambiguity_with_components_second_sub_element_3_str_weight_copy(mesh):
    (V, W) = MixedSpacesRestrictionSolveAmbiguityWithComponents(mesh, 1)
    up = MixedFunctionRestrictionSolveAmbiguityWithComponents(V)
    p = function_extend_or_restrict(up, "p", W, None, weight=2., copy=True)
    assert p.vector().size() == W.dim()
    assert isclose(p.vector().get_local(), 4.).all()


# ~~~ Mixed case: restriction to sub element, ambiguous restriction due to failing automatic components detection ~~~ #
def MixedSpacesRestrictionToSubElementAmbiguous(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # Note that we need to use 2nd order FE otherwise
    element_00 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)  # the automatic detection would restrict the
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # pressure component
    element = MixedElement(element_0, element_1)
    V = FunctionSpace(mesh, element)
    W = FunctionSpace(mesh, element_00)
    return (V, W)


def MixedFunctionRestrictionToSubElementAmbiguous(V):
    up = Function(V)
    assign(up.sub(0).sub(0), project(Constant(1.), V.sub(0).sub(0).collapse()))
    assign(up.sub(0).sub(1), project(Constant(3.), V.sub(0).sub(1).collapse()))
    assign(up.sub(1), project(Constant(2.), V.sub(1).collapse()))
    return up


def test_mixed_function_restriction_to_sub_element_ambiguous_1_fail(mesh):
    (V, W) = MixedSpacesRestrictionToSubElementAmbiguous(mesh)
    up = MixedFunctionRestrictionToSubElementAmbiguous(V)
    with pytest.raises(RuntimeError) as excinfo:
        function_extend_or_restrict(up, None, W, None, weight=None, copy=False)
    assert str(excinfo.value) == "Ambiguity when querying _function_spaces_lt"


# ~~~ Mixed case: restriction to sub element, avoid ambiguity thanks to user provided input components ~~~ #
def MixedSpacesRestrictionToSubElementSolveAmbiguityWithComponents(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    element_00 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    V = FunctionSpace(mesh, element, components=[("ux", "uy"), "p"])
    W = FunctionSpace(mesh, element_00)
    return (V, W)


def MixedFunctionRestrictionToSubElementSolveAmbiguityWithComponents(V):
    up = Function(V)
    assign(up.sub(0).sub(0), project(Constant(1.), V.sub(0).sub(0).collapse()))
    assign(up.sub(0).sub(1), project(Constant(3.), V.sub(0).sub(1).collapse()))
    assign(up.sub(1), project(Constant(2.), V.sub(1).collapse()))
    return up


def test_mixed_function_restriction_to_sub_element_solve_ambiguity_with_components_1_tuple_x_fail(mesh):
    (V, W) = MixedSpacesRestrictionToSubElementSolveAmbiguityWithComponents(mesh)
    up = MixedFunctionRestrictionToSubElementSolveAmbiguityWithComponents(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(up, (0, 0), W, None, weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_function_restriction_to_sub_element_solve_ambiguity_with_components_2_tuple_x_copy(mesh):
    (V, W) = MixedSpacesRestrictionToSubElementSolveAmbiguityWithComponents(mesh)
    up = MixedFunctionRestrictionToSubElementSolveAmbiguityWithComponents(V)
    ux = function_extend_or_restrict(up, (0, 0), W, None, weight=None, copy=True)
    assert ux.vector().size() == W.dim()
    assert isclose(ux.vector().get_local(), 1.).all()


def test_mixed_function_restriction_to_sub_element_solve_ambiguity_with_components_3_tuple_x_weight_copy(mesh):
    (V, W) = MixedSpacesRestrictionToSubElementSolveAmbiguityWithComponents(mesh)
    up = MixedFunctionRestrictionToSubElementSolveAmbiguityWithComponents(V)
    ux = function_extend_or_restrict(up, (0, 0), W, None, weight=2., copy=True)
    assert ux.vector().size() == W.dim()
    assert isclose(ux.vector().get_local(), 2.).all()


def test_mixed_function_restriction_to_sub_element_solve_ambiguity_with_components_4_tuple_y_fail(mesh):
    (V, W) = MixedSpacesRestrictionToSubElementSolveAmbiguityWithComponents(mesh)
    up = MixedFunctionRestrictionToSubElementSolveAmbiguityWithComponents(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(up, (0, 1), W, None, weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_function_restriction_to_sub_element_solve_ambiguity_with_components_5_tuple_y_copy(mesh):
    (V, W) = MixedSpacesRestrictionToSubElementSolveAmbiguityWithComponents(mesh)
    up = MixedFunctionRestrictionToSubElementSolveAmbiguityWithComponents(V)
    uy = function_extend_or_restrict(up, (0, 1), W, None, weight=None, copy=True)
    assert uy.vector().size() == W.dim()
    assert isclose(uy.vector().get_local(), 3.).all()


def test_mixed_function_restriction_to_sub_element_solve_ambiguity_with_components_6_tuple_y_weight_copy(mesh):
    (V, W) = MixedSpacesRestrictionToSubElementSolveAmbiguityWithComponents(mesh)
    up = MixedFunctionRestrictionToSubElementSolveAmbiguityWithComponents(V)
    uy = function_extend_or_restrict(up, (0, 1), W, None, weight=2., copy=True)
    assert uy.vector().size() == W.dim()
    assert isclose(uy.vector().get_local(), 6.).all()


def test_mixed_function_restriction_to_sub_element_solve_ambiguity_with_components_1_str_x_fail(mesh):
    (V, W) = MixedSpacesRestrictionToSubElementSolveAmbiguityWithComponents(mesh)
    up = MixedFunctionRestrictionToSubElementSolveAmbiguityWithComponents(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(up, "ux", W, None, weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_function_restriction_to_sub_element_solve_ambiguity_with_components_2_str_x_copy(mesh):
    (V, W) = MixedSpacesRestrictionToSubElementSolveAmbiguityWithComponents(mesh)
    up = MixedFunctionRestrictionToSubElementSolveAmbiguityWithComponents(V)
    ux = function_extend_or_restrict(up, "ux", W, None, weight=None, copy=True)
    assert ux.vector().size() == W.dim()
    assert isclose(ux.vector().get_local(), 1.).all()


def test_mixed_function_restriction_to_sub_element_solve_ambiguity_with_components_3_str_x_weight_copy(mesh):
    (V, W) = MixedSpacesRestrictionToSubElementSolveAmbiguityWithComponents(mesh)
    up = MixedFunctionRestrictionToSubElementSolveAmbiguityWithComponents(V)
    ux = function_extend_or_restrict(up, "ux", W, None, weight=2., copy=True)
    assert ux.vector().size() == W.dim()
    assert isclose(ux.vector().get_local(), 2.).all()


def test_mixed_function_restriction_to_sub_element_solve_ambiguity_with_components_4_str_y_fail(mesh):
    (V, W) = MixedSpacesRestrictionToSubElementSolveAmbiguityWithComponents(mesh)
    up = MixedFunctionRestrictionToSubElementSolveAmbiguityWithComponents(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(up, "uy", W, None, weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_function_restriction_to_sub_element_solve_ambiguity_with_components_5_str_y_copy(mesh):
    (V, W) = MixedSpacesRestrictionToSubElementSolveAmbiguityWithComponents(mesh)
    up = MixedFunctionRestrictionToSubElementSolveAmbiguityWithComponents(V)
    uy = function_extend_or_restrict(up, "uy", W, None, weight=None, copy=True)
    assert uy.vector().size() == W.dim()
    assert isclose(uy.vector().get_local(), 3.).all()


def test_mixed_function_restriction_to_sub_element_solve_ambiguity_with_components_6_str_y_weight_copy(mesh):
    (V, W) = MixedSpacesRestrictionToSubElementSolveAmbiguityWithComponents(mesh)
    up = MixedFunctionRestrictionToSubElementSolveAmbiguityWithComponents(V)
    uy = function_extend_or_restrict(up, "uy", W, None, weight=2., copy=True)
    assert uy.vector().size() == W.dim()
    assert isclose(uy.vector().get_local(), 6.).all()


# ~~~ Mixed case to mixed case: copy only a component, in the same location ~~~ #
def MixedToMixedSpacesCopyComponentToSameLocation(mesh):
    element_0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    V = FunctionSpace(mesh, element, components=["ux", "uy"])
    W = V
    return (V, W)


def MixedToMixedFunctionCopyComponentToSameLocation(V):
    u = Function(V)
    assign(u.sub(0), project(Constant(1.), V.sub(0).collapse()))
    assign(u.sub(1), project(Constant(2.), V.sub(1).collapse()))
    return u


def test_mixed_to_mixed_function_copy_component_to_same_location_1_int_fail(mesh):
    (V, W) = MixedToMixedSpacesCopyComponentToSameLocation(mesh)
    u = MixedToMixedFunctionCopyComponentToSameLocation(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(u, 0, W, 0, weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_to_mixed_function_copy_component_to_same_location_2_int_copy(mesh):
    (V, W) = MixedToMixedSpacesCopyComponentToSameLocation(mesh)
    u = MixedToMixedFunctionCopyComponentToSameLocation(V)
    copied_u = function_extend_or_restrict(u, 0, W, 0, weight=None, copy=True)
    assert copied_u.vector().size() == W.dim()
    (copied_ux, copied_uy) = copied_u.split(deepcopy=True)
    assert isclose(copied_ux.vector().get_local(), 1.).all()
    assert isclose(copied_uy.vector().get_local(), 0.).all()


def test_mixed_to_mixed_function_copy_component_to_same_location_3_int_weight_copy(mesh):
    (V, W) = MixedToMixedSpacesCopyComponentToSameLocation(mesh)
    u = MixedToMixedFunctionCopyComponentToSameLocation(V)
    copied_u = function_extend_or_restrict(u, 0, W, 0, weight=2., copy=True)
    assert copied_u.vector().size() == W.dim()
    (copied_ux, copied_uy) = copied_u.split(deepcopy=True)
    assert isclose(copied_ux.vector().get_local(), 2.).all()
    assert isclose(copied_uy.vector().get_local(), 0.).all()


def test_mixed_to_mixed_function_copy_component_to_same_location_4_str_fail(mesh):
    (V, W) = MixedToMixedSpacesCopyComponentToSameLocation(mesh)
    u = MixedToMixedFunctionCopyComponentToSameLocation(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(u, "ux", W, "ux", weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_to_mixed_function_copy_component_to_same_location_5_str_copy(mesh):
    (V, W) = MixedToMixedSpacesCopyComponentToSameLocation(mesh)
    u = MixedToMixedFunctionCopyComponentToSameLocation(V)
    copied_u = function_extend_or_restrict(u, "ux", W, "ux", weight=None, copy=True)
    assert copied_u.vector().size() == W.dim()
    (copied_ux, copied_uy) = copied_u.split(deepcopy=True)
    assert isclose(copied_ux.vector().get_local(), 1.).all()
    assert isclose(copied_uy.vector().get_local(), 0.).all()


def test_mixed_to_mixed_function_copy_component_to_same_location_6_str_weight_copy(mesh):
    (V, W) = MixedToMixedSpacesCopyComponentToSameLocation(mesh)
    u = MixedToMixedFunctionCopyComponentToSameLocation(V)
    copied_u = function_extend_or_restrict(u, "ux", W, "ux", weight=2., copy=True)
    assert copied_u.vector().size() == W.dim()
    (copied_ux, copied_uy) = copied_u.split(deepcopy=True)
    assert isclose(copied_ux.vector().get_local(), 2.).all()
    assert isclose(copied_uy.vector().get_local(), 0.).all()


# ~~~ Mixed case to mixed case: copy only a component, to a different location ~~~ #
def MixedToMixedSpacesCopyComponentToDifferentLocation(mesh):
    element_0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    V = FunctionSpace(mesh, element, components=["ux", "uy"])
    W = V
    return (V, W)


def MixedToMixedFunctionCopyComponentToDifferentLocation(V):
    u = Function(V)
    assign(u.sub(0), project(Constant(1.), V.sub(0).collapse()))
    assign(u.sub(1), project(Constant(2.), V.sub(1).collapse()))
    return u


def test_mixed_to_mixed_function_copy_component_to_different_location_1_int_fail(mesh):
    (V, W) = MixedToMixedSpacesCopyComponentToDifferentLocation(mesh)
    u = MixedToMixedFunctionCopyComponentToDifferentLocation(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(u, 0, W, 1, weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_to_mixed_function_copy_component_to_different_location_2_int_copy(mesh):
    (V, W) = MixedToMixedSpacesCopyComponentToDifferentLocation(mesh)
    u = MixedToMixedFunctionCopyComponentToDifferentLocation(V)
    copied_u = function_extend_or_restrict(u, 0, W, 1, weight=None, copy=True)
    assert copied_u.vector().size() == W.dim()
    (copied_ux, copied_uy) = copied_u.split(deepcopy=True)
    assert isclose(copied_ux.vector().get_local(), 0.).all()
    assert isclose(copied_uy.vector().get_local(), 1.).all()


def test_mixed_to_mixed_function_copy_component_to_different_location_3_int_weight_copy(mesh):
    (V, W) = MixedToMixedSpacesCopyComponentToDifferentLocation(mesh)
    u = MixedToMixedFunctionCopyComponentToDifferentLocation(V)
    copied_u = function_extend_or_restrict(u, 0, W, 1, weight=2., copy=True)
    assert copied_u.vector().size() == W.dim()
    (copied_ux, copied_uy) = copied_u.split(deepcopy=True)
    assert isclose(copied_ux.vector().get_local(), 0.).all()
    assert isclose(copied_uy.vector().get_local(), 2.).all()


def test_mixed_to_mixed_function_copy_component_to_different_location_4_str_fail(mesh):
    (V, W) = MixedToMixedSpacesCopyComponentToDifferentLocation(mesh)
    u = MixedToMixedFunctionCopyComponentToDifferentLocation(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(u, "ux", W, "uy", weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_to_mixed_function_copy_component_to_different_location_5_str_copy(mesh):
    (V, W) = MixedToMixedSpacesCopyComponentToDifferentLocation(mesh)
    u = MixedToMixedFunctionCopyComponentToDifferentLocation(V)
    copied_u = function_extend_or_restrict(u, "ux", W, "uy", weight=None, copy=True)
    assert copied_u.vector().size() == W.dim()
    (copied_ux, copied_uy) = copied_u.split(deepcopy=True)
    assert isclose(copied_ux.vector().get_local(), 0.).all()
    assert isclose(copied_uy.vector().get_local(), 1.).all()


def test_mixed_to_mixed_function_copy_component_to_different_location_6_str_weight_copy(mesh):
    (V, W) = MixedToMixedSpacesCopyComponentToDifferentLocation(mesh)
    u = MixedToMixedFunctionCopyComponentToDifferentLocation(V)
    copied_u = function_extend_or_restrict(u, "ux", W, "uy", weight=2., copy=True)
    assert copied_u.vector().size() == W.dim()
    (copied_ux, copied_uy) = copied_u.split(deepcopy=True)
    assert isclose(copied_ux.vector().get_local(), 0.).all()
    assert isclose(copied_uy.vector().get_local(), 2.).all()


# ~~~ Mixed case to mixed case: copy only a sub component, in the same location ~~~ #
def MixedToMixedSpacesCopySubComponentToSameLocation(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    element_1 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    V = FunctionSpace(mesh, element, components=[("uxx", "uxy"), ("uyx", "uyy")])
    W = V
    return (V, W)


def MixedToMixedFunctionCopySubComponentToSameLocation(V):
    u = Function(V)
    assign(u.sub(0), project(Constant((1., 2.)), V.sub(0).collapse()))
    assign(u.sub(1), project(Constant((3., 4.)), V.sub(1).collapse()))
    return u


def test_mixed_to_mixed_function_copy_sub_component_to_same_location_1_tuple_fail(mesh):
    (V, W) = MixedToMixedSpacesCopySubComponentToSameLocation(mesh)
    u = MixedToMixedFunctionCopySubComponentToSameLocation(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(u, (0, 0), W, (0, 0), weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_to_mixed_function_copy_sub_component_to_same_location_2_tuple_copy(mesh):
    (V, W) = MixedToMixedSpacesCopySubComponentToSameLocation(mesh)
    u = MixedToMixedFunctionCopySubComponentToSameLocation(V)
    copied_u = function_extend_or_restrict(u, (0, 0), W, (0, 0), weight=None, copy=True)
    assert copied_u.vector().size() == W.dim()
    (copied_ux, copied_uy) = copied_u.split(deepcopy=True)
    (copied_uxx, copied_uxy) = copied_ux.split(deepcopy=True)
    (copied_uyx, copied_uyy) = copied_uy.split(deepcopy=True)
    assert isclose(copied_uxx.vector().get_local(), 1.).all()
    assert isclose(copied_uxy.vector().get_local(), 0.).all()
    assert isclose(copied_uyx.vector().get_local(), 0.).all()
    assert isclose(copied_uyy.vector().get_local(), 0.).all()


def test_mixed_to_mixed_function_copy_sub_component_to_same_location_3_tuple_weight_copy(mesh):
    (V, W) = MixedToMixedSpacesCopySubComponentToSameLocation(mesh)
    u = MixedToMixedFunctionCopySubComponentToSameLocation(V)
    copied_u = function_extend_or_restrict(u, (0, 0), W, (0, 0), weight=2., copy=True)
    assert copied_u.vector().size() == W.dim()
    (copied_ux, copied_uy) = copied_u.split(deepcopy=True)
    (copied_uxx, copied_uxy) = copied_ux.split(deepcopy=True)
    (copied_uyx, copied_uyy) = copied_uy.split(deepcopy=True)
    assert isclose(copied_uxx.vector().get_local(), 2.).all()
    assert isclose(copied_uxy.vector().get_local(), 0.).all()
    assert isclose(copied_uyx.vector().get_local(), 0.).all()
    assert isclose(copied_uyy.vector().get_local(), 0.).all()


def test_mixed_to_mixed_function_copy_sub_component_to_same_location_4_str_fail(mesh):
    (V, W) = MixedToMixedSpacesCopySubComponentToSameLocation(mesh)
    u = MixedToMixedFunctionCopySubComponentToSameLocation(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(u, "uxx", W, "uxx", weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_to_mixed_function_copy_sub_component_to_same_location_5_str_copy(mesh):
    (V, W) = MixedToMixedSpacesCopySubComponentToSameLocation(mesh)
    u = MixedToMixedFunctionCopySubComponentToSameLocation(V)
    copied_u = function_extend_or_restrict(u, "uxx", W, "uxx", weight=None, copy=True)
    assert copied_u.vector().size() == W.dim()
    (copied_ux, copied_uy) = copied_u.split(deepcopy=True)
    (copied_uxx, copied_uxy) = copied_ux.split(deepcopy=True)
    (copied_uyx, copied_uyy) = copied_uy.split(deepcopy=True)
    assert isclose(copied_uxx.vector().get_local(), 1.).all()
    assert isclose(copied_uxy.vector().get_local(), 0.).all()
    assert isclose(copied_uyx.vector().get_local(), 0.).all()
    assert isclose(copied_uyy.vector().get_local(), 0.).all()


def test_mixed_to_mixed_function_copy_sub_component_to_same_location_6_str_weight_copy(mesh):
    (V, W) = MixedToMixedSpacesCopySubComponentToSameLocation(mesh)
    u = MixedToMixedFunctionCopySubComponentToSameLocation(V)
    copied_u = function_extend_or_restrict(u, "uxx", W, "uxx", weight=2., copy=True)
    assert copied_u.vector().size() == W.dim()
    (copied_ux, copied_uy) = copied_u.split(deepcopy=True)
    (copied_uxx, copied_uxy) = copied_ux.split(deepcopy=True)
    (copied_uyx, copied_uyy) = copied_uy.split(deepcopy=True)
    assert isclose(copied_uxx.vector().get_local(), 2.).all()
    assert isclose(copied_uxy.vector().get_local(), 0.).all()
    assert isclose(copied_uyx.vector().get_local(), 0.).all()
    assert isclose(copied_uyy.vector().get_local(), 0.).all()


# ~~~ Mixed case to mixed case: copy only a sub component, to a different location ~~~ #
def MixedToMixedSpacesCopySubComponentToDifferentLocation(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    element_1 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    V = FunctionSpace(mesh, element, components=[("uxx", "uxy"), ("uyx", "uyy")])
    W = V
    return (V, W)


def MixedToMixedFunctionCopySubComponentToDifferentLocation(V):
    u = Function(V)
    assign(u.sub(0), project(Constant((1., 2.)), V.sub(0).collapse()))
    assign(u.sub(1), project(Constant((3., 4.)), V.sub(1).collapse()))
    return u


def test_mixed_to_mixed_function_copy_sub_component_to_different_location_1_tuple_fail(mesh):
    (V, W) = MixedToMixedSpacesCopySubComponentToDifferentLocation(mesh)
    u = MixedToMixedFunctionCopySubComponentToDifferentLocation(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(u, (0, 0), W, (1, 0), weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_to_mixed_function_copy_sub_component_to_different_location_2_tuple_copy(mesh):
    (V, W) = MixedToMixedSpacesCopySubComponentToDifferentLocation(mesh)
    u = MixedToMixedFunctionCopySubComponentToDifferentLocation(V)
    copied_u = function_extend_or_restrict(u, (0, 0), W, (1, 0), weight=None, copy=True)
    assert copied_u.vector().size() == W.dim()
    (copied_ux, copied_uy) = copied_u.split(deepcopy=True)
    (copied_uxx, copied_uxy) = copied_ux.split(deepcopy=True)
    (copied_uyx, copied_uyy) = copied_uy.split(deepcopy=True)
    assert isclose(copied_uxx.vector().get_local(), 0.).all()
    assert isclose(copied_uxy.vector().get_local(), 0.).all()
    assert isclose(copied_uyx.vector().get_local(), 1.).all()
    assert isclose(copied_uyy.vector().get_local(), 0.).all()


def test_mixed_to_mixed_function_copy_sub_component_to_different_location_3_tuple_weight_copy(mesh):
    (V, W) = MixedToMixedSpacesCopySubComponentToDifferentLocation(mesh)
    u = MixedToMixedFunctionCopySubComponentToDifferentLocation(V)
    copied_u = function_extend_or_restrict(u, (0, 0), W, (1, 0), weight=2., copy=True)
    assert copied_u.vector().size() == W.dim()
    (copied_ux, copied_uy) = copied_u.split(deepcopy=True)
    (copied_uxx, copied_uxy) = copied_ux.split(deepcopy=True)
    (copied_uyx, copied_uyy) = copied_uy.split(deepcopy=True)
    assert isclose(copied_uxx.vector().get_local(), 0.).all()
    assert isclose(copied_uxy.vector().get_local(), 0.).all()
    assert isclose(copied_uyx.vector().get_local(), 2.).all()
    assert isclose(copied_uyy.vector().get_local(), 0.).all()


def test_mixed_to_mixed_function_copy_sub_component_to_different_location_4_str_fail(mesh):
    (V, W) = MixedToMixedSpacesCopySubComponentToDifferentLocation(mesh)
    u = MixedToMixedFunctionCopySubComponentToDifferentLocation(V)
    with pytest.raises(AssertionError) as excinfo:
        function_extend_or_restrict(u, "uxx", W, "uyx", weight=None, copy=False)
    assert str(excinfo.value) == "It is not possible to extract function components without copying the vector"


def test_mixed_to_mixed_function_copy_sub_component_to_different_location_5_str_copy(mesh):
    (V, W) = MixedToMixedSpacesCopySubComponentToDifferentLocation(mesh)
    u = MixedToMixedFunctionCopySubComponentToDifferentLocation(V)
    copied_u = function_extend_or_restrict(u, "uxx", W, "uyx", weight=None, copy=True)
    assert copied_u.vector().size() == W.dim()
    (copied_ux, copied_uy) = copied_u.split(deepcopy=True)
    (copied_uxx, copied_uxy) = copied_ux.split(deepcopy=True)
    (copied_uyx, copied_uyy) = copied_uy.split(deepcopy=True)
    assert isclose(copied_uxx.vector().get_local(), 0.).all()
    assert isclose(copied_uxy.vector().get_local(), 0.).all()
    assert isclose(copied_uyx.vector().get_local(), 1.).all()
    assert isclose(copied_uyy.vector().get_local(), 0.).all()


def test_mixed_to_mixed_function_copy_sub_component_to_different_location_6_str_weight_copy(mesh):
    (V, W) = MixedToMixedSpacesCopySubComponentToDifferentLocation(mesh)
    u = MixedToMixedFunctionCopySubComponentToDifferentLocation(V)
    copied_u = function_extend_or_restrict(u, "uxx", W, "uyx", weight=2., copy=True)
    assert copied_u.vector().size() == W.dim()
    (copied_ux, copied_uy) = copied_u.split(deepcopy=True)
    (copied_uxx, copied_uxy) = copied_ux.split(deepcopy=True)
    (copied_uyx, copied_uyy) = copied_uy.split(deepcopy=True)
    assert isclose(copied_uxx.vector().get_local(), 0.).all()
    assert isclose(copied_uxy.vector().get_local(), 0.).all()
    assert isclose(copied_uyx.vector().get_local(), 2.).all()
    assert isclose(copied_uyy.vector().get_local(), 0.).all()
