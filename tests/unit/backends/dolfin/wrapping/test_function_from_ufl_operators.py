# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from math import sqrt
from numpy import isclose
from dolfin import (assemble, dx, FiniteElement, Function, FunctionSpace, MixedElement, split, TestFunction,
                    TrialFunction, UnitSquareMesh, VectorElement, VectorFunctionSpace)
from dolfin_utils.test import fixture as module_fixture
from rbnics.backends.dolfin import transpose
from rbnics.backends.dolfin.wrapping import function_from_ufl_operators


# Mesh
@module_fixture
def mesh():
    return UnitSquareMesh(10, 10)


# Scalar fixtures
def ScalarSpace(mesh):
    return FunctionSpace(mesh, "Lagrange", 2)


def scalar_linear_form(V):
    v = TestFunction(V)
    return v * dx


def scalar_bilinear_form(V):
    u = TrialFunction(V)
    v = TestFunction(V)
    return u * v * dx


def scalar_conversion_isclose(a, b):
    return isclose(a, b)


def scalar_normalization_isclose(a, b):
    return isclose(a, b)


def scalar_transpose_isclose(a, b):
    return isclose(a, b)


# Vector fixtures
def VectorSpace(mesh):
    return VectorFunctionSpace(mesh, "Lagrange", 2)


def vector_linear_form(V):
    v = TestFunction(V)
    return v[0] * dx + v[1] * dx


def vector_bilinear_form(V):
    u = TrialFunction(V)
    v = TestFunction(V)
    return u[0] * v[0] * dx + u[1] * v[1] * dx


def vector_conversion_isclose(a, b):
    return isclose(a, b)


def vector_normalization_isclose(a, b):
    return isclose(a, b / sqrt(2))


def vector_transpose_isclose(a, b):
    return isclose(a, 2 * b)


# Mixed fixtures
def MixedSpace(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    return FunctionSpace(mesh, element)


def mixed_linear_form(V):
    v = TestFunction(V)
    (v_0, v_1) = split(v)
    return v_0[0] * dx + v_0[1] * dx + v_1 * dx


def mixed_bilinear_form(V):
    u = TrialFunction(V)
    v = TestFunction(V)
    (u_0, u_1) = split(u)
    (v_0, v_1) = split(v)
    return u_0[0] * v_0[0] * dx + u_0[1] * v_0[1] * dx + u_1 * v_1 * dx


def mixed_conversion_isclose(a, b):
    return isclose(a, b)


def mixed_normalization_isclose(a, b):
    return isclose(a, b / sqrt(3))


def mixed_transpose_isclose(a, b):
    return isclose(a, 3 * b)


# Tests
@pytest.mark.parametrize("FunctionSpace, isclose", [
    (ScalarSpace, scalar_conversion_isclose),
    (VectorSpace, vector_conversion_isclose),
    (MixedSpace, mixed_conversion_isclose)
])
def test_conversion(mesh, FunctionSpace, isclose):
    V = FunctionSpace(mesh)

    z1 = Function(V)
    z1.vector()[:] = 1.
    assert function_from_ufl_operators(z1) is z1

    _2_z1 = function_from_ufl_operators(2 * z1)
    assert isclose(_2_z1.vector().get_local(), 2.).all()

    z1_2 = function_from_ufl_operators(z1 * 2)
    assert isclose(z1_2.vector().get_local(), 2.).all()

    z1_over_2 = function_from_ufl_operators(z1 / 2.)
    assert isclose(z1_over_2.vector().get_local(), 0.5).all()

    z2 = Function(V)
    z2.vector()[:] = 2.

    z1_plus_z2 = function_from_ufl_operators(z1 + z2)
    assert isclose(z1_plus_z2.vector().get_local(), 3.).all()

    z1_minus_z2 = function_from_ufl_operators(z1 - z2)
    assert isclose(z1_minus_z2.vector().get_local(), -1.).all()

    z1_minus_2_z2 = function_from_ufl_operators(z1 - 2 * z2)
    assert isclose(z1_minus_2_z2.vector().get_local(), -3.).all()

    z1_minus_z2_2 = function_from_ufl_operators(z1 - z2 * 2)
    assert isclose(z1_minus_z2_2.vector().get_local(), -3.).all()

    z1_minus_3_z2_2 = function_from_ufl_operators(z1 - 3 * z2 * 2)
    assert isclose(z1_minus_3_z2_2.vector().get_local(), -11.).all()

    z1_minus_z2_over_4 = function_from_ufl_operators(z1 - z2 / 4.)
    assert isclose(z1_minus_z2_over_4.vector().get_local(), 0.5).all()

    z1_minus_z2_over_2 = function_from_ufl_operators((z1 - z2) / 2.)
    assert isclose(z1_minus_z2_over_2.vector().get_local(), -0.5).all()

    z3 = Function(V)
    z3.vector()[:] = 3.

    z1_minus_z2_plus_z3 = function_from_ufl_operators(z1 - z2 + z3)
    assert isclose(z1_minus_z2_plus_z3.vector().get_local(), 2.).all()


@pytest.mark.parametrize("FunctionSpace, bilinear_form, isclose", [
    (ScalarSpace, scalar_bilinear_form, scalar_normalization_isclose),
    (VectorSpace, vector_bilinear_form, vector_normalization_isclose),
    (MixedSpace, mixed_bilinear_form, mixed_normalization_isclose)
])
def test_normalization(mesh, FunctionSpace, bilinear_form, isclose):
    V = FunctionSpace(mesh)

    A = assemble(bilinear_form(V))

    z1 = Function(V)
    z1.vector()[:] = 2.

    z1_normalized = function_from_ufl_operators(z1 / sqrt(transpose(z1) * A * z1))
    assert isclose(z1_normalized.vector().get_local(), 1).all()


@pytest.mark.parametrize("FunctionSpace, bilinear_form, linear_form, isclose", [
    (ScalarSpace, scalar_bilinear_form, scalar_linear_form, scalar_transpose_isclose),
    (VectorSpace, vector_bilinear_form, vector_linear_form, vector_transpose_isclose),
    (MixedSpace, mixed_bilinear_form, mixed_linear_form, mixed_transpose_isclose)
])
def test_transpose(mesh, FunctionSpace, bilinear_form, linear_form, isclose):
    V = FunctionSpace(mesh)

    A = assemble(bilinear_form(V))
    b = assemble(linear_form(V))

    z1 = Function(V)
    z1.vector()[:] = 1.
    assert isclose(transpose(z1) * A * z1, 1.)
    assert isclose(transpose(b) * z1, 1.)
    assert isclose(transpose(z1) * b, 1.)

    assert isclose(transpose(z1) * A * (2 * z1), 2.)
    assert isclose(transpose(2 * z1) * A * z1, 2.)
    assert isclose(transpose(b) * (2 * z1), 2.)
    assert isclose(transpose(2 * z1) * b, 2.)

    assert isclose(transpose(z1) * A * (z1 * 2), 2.)
    assert isclose(transpose(z1 * 2) * A * z1, 2.)
    assert isclose(transpose(b) * (z1 * 2), 2.)
    assert isclose(transpose(z1 * 2) * b, 2.)

    assert isclose(transpose(z1) * A * (z1 / 2.), 0.5)
    assert isclose(transpose(z1 / 2.) * A * z1, 0.5)
    assert isclose(transpose(b) * (z1 / 2.), 0.5)
    assert isclose(transpose(z1 / 2.) * b, 0.5)

    z2 = Function(V)
    z2.vector()[:] = 2.

    assert isclose(transpose(z1) * A * (z1 + z2), 3.)
    assert isclose(transpose(z1 + z2) * A * z1, 3.)
    assert isclose(transpose(z1 + z2) * A * (z1 + z2), 9.)
    assert isclose(transpose(b) * (z1 + z2), 3.)
    assert isclose(transpose(z1 + z2) * b, 3.)

    assert isclose(transpose(z1) * A * (z1 - z2), -1.)
    assert isclose(transpose(z1 - z2) * A * z1, -1.)
    assert isclose(transpose(z1 - z2) * A * (z1 - z2), 1.)
    assert isclose(transpose(z1 - z2) * A * (z1 + z2), -3.)
    assert isclose(transpose(b) * (z1 - z2), -1.)
    assert isclose(transpose(z1 - z2) * b, -1.)

    assert isclose(transpose(z1) * A * (z1 - 2 * z2), -3.)
    assert isclose(transpose(z1 - 2 * z2) * A * z1, -3.)
    assert isclose(transpose(z1 - 2 * z2) * A * (z1 - 2 * z2), 9.)
    assert isclose(transpose(b) * (z1 - 2 * z2), -3.)
    assert isclose(transpose(z1 - 2 * z2) * b, -3.)

    assert isclose(transpose(z1) * A * (z1 - z2 * 2), -3.)
    assert isclose(transpose(z1 - z2 * 2) * A * z1, -3.)
    assert isclose(transpose(z1 - z2 * 2) * A * (z1 - z2 * 2), 9.)
    assert isclose(transpose(b) * (z1 - z2 * 2), -3.)
    assert isclose(transpose(z1 - z2 * 2) * b, -3.)

    assert isclose(transpose(z1) * A * (z1 - 3 * z2 * 2), -11.)
    assert isclose(transpose(z1 - 3 * z2 * 2) * A * z1, -11.)
    assert isclose(transpose(z1 - 3 * z2 * 2) * A * (z1 - 3 * z2 * 2), 121.)
    assert isclose(transpose(b) * (z1 - 3 * z2 * 2), -11.)
    assert isclose(transpose(z1 - 3 * z2 * 2) * b, -11.)

    assert isclose(transpose(z1) * A * (z1 - z2 / 4.), 0.5)
    assert isclose(transpose(z1 - z2 / 4.) * A * z1, 0.5)
    assert isclose(transpose(z1 - z2 / 4.) * A * (z1 - z2 / 4.), 0.25)
    assert isclose(transpose(b) * (z1 - z2 / 4.), 0.5)
    assert isclose(transpose(z1 - z2 / 4.) * b, 0.5)

    assert isclose(transpose(z1) * A * ((z1 - z2) / 2.), -0.5)
    assert isclose(transpose((z1 - z2) / 2.) * A * z1, -0.5)
    assert isclose(transpose((z1 - z2) / 2.) * A * ((z1 - z2) / 2.), 0.25)
    assert isclose(transpose(b) * ((z1 - z2) / 2.), -0.5)
    assert isclose(transpose((z1 - z2) / 2.) * b, -0.5)

    z3 = Function(V)
    z3.vector()[:] = 3.

    assert isclose(transpose(z1) * A * (z1 - z2 + z3), 2.)
    assert isclose(transpose(z1 - z2) * A * (z1 - z2 + z3), -2.)
    assert isclose(transpose(z1 - z2 + z3) * A * z1, 2.)
    assert isclose(transpose(z1 - z2 + z3) * A * (z1 - z2), -2.)
    assert isclose(transpose(z1 - z2 + z3) * A * (z1 - z2 + z3), 4.)
    assert isclose(transpose(b) * (z1 - z2 + z3), 2.)
    assert isclose(transpose(z1 - z2 + z3) * b, 2.)
