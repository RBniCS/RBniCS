# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from numpy import isclose
from dolfin import (assemble, dx, FiniteElement, FunctionSpace, inner, MixedElement, split, TestFunction,
                    TrialFunction, UnitSquareMesh, VectorElement)
from dolfin_utils.test import fixture as module_fixture
from rbnics.backends.dolfin import evaluate as _evaluate, ParametrizedTensorFactory
from rbnics.backends.dolfin.export import tensor_save
from rbnics.backends.dolfin.import_ import tensor_load
from rbnics.eim.utils.decorators import add_to_map_from_parametrized_expression_to_problem


# Meshes
@module_fixture
def mesh():
    return UnitSquareMesh(10, 10)


# Forms: elliptic case
def generate_elliptic_linear_form_space(mesh):
    return (FunctionSpace(mesh, "Lagrange", 2), )


def generate_elliptic_linear_form(V):
    v = TestFunction(V)
    return v * dx


def generate_elliptic_bilinear_form_space(mesh):
    return generate_elliptic_linear_form_space(mesh) + generate_elliptic_linear_form_space(mesh)


def generate_elliptic_bilinear_form(V1, V2):
    assert V1.ufl_element() == V2.ufl_element()
    u = TrialFunction(V1)
    v = TestFunction(V2)
    return u * v * dx


# Forms: mixed case
def generate_mixed_linear_form_space(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    return (FunctionSpace(mesh, element), )


def generate_mixed_linear_form(V):
    v = TestFunction(V)
    (v_0, v_1) = split(v)
    return v_0[0] * dx + v_0[1] * dx + v_1 * dx


def generate_mixed_bilinear_form_space(mesh):
    return generate_mixed_linear_form_space(mesh) + generate_mixed_linear_form_space(mesh)


def generate_mixed_bilinear_form(V1, V2):
    assert V1.ufl_element() == V2.ufl_element()
    u = TrialFunction(V1)
    v = TestFunction(V2)
    (u_0, u_1) = split(u)
    (v_0, v_1) = split(v)
    return inner(u_0, v_0) * dx + u_1 * v_1 * dx + u_0[0] * v_1 * dx + u_1 * v_0[1] * dx


# Forms: collapsed case
def generate_collapsed_linear_form_space(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    U = FunctionSpace(mesh, element)
    V = U.sub(0).collapse()
    return (V, )


def generate_collapsed_linear_form(V):
    v = TestFunction(V)
    return v[0] * dx + v[1] * dx


def generate_collapsed_bilinear_form_space(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    U = FunctionSpace(mesh, element)
    V = U.sub(0).collapse()
    return (V, U)


def generate_collapsed_bilinear_form(V, U):
    u = TrialFunction(U)
    (u_0, u_1) = split(u)
    v = TestFunction(V)
    return inner(u_0, v) * dx + u_1 * v[0] * dx


# Forms decorator
generate_form_spaces_and_forms = pytest.mark.parametrize("generate_form_space, generate_form", [
    (generate_elliptic_linear_form_space, generate_elliptic_linear_form),
    (generate_elliptic_bilinear_form_space, generate_elliptic_bilinear_form),
    (generate_mixed_linear_form_space, generate_mixed_linear_form),
    (generate_mixed_bilinear_form_space, generate_mixed_bilinear_form),
    (generate_collapsed_linear_form_space, generate_collapsed_linear_form),
    (generate_collapsed_bilinear_form_space, generate_collapsed_bilinear_form)
])


# Mock problem to avoid triggering an assert
class Problem(object):
    mu = None


def evaluate(tensor):
    add_to_map_from_parametrized_expression_to_problem(tensor, Problem())
    return _evaluate(tensor)


# Prepare tensor storage for load
class Generator(object):
    def __init__(self, form):
        self._form = form


def zero_for_load(form):
    tensor = assemble(form, keep_diagonal=True)
    tensor.zero()
    tensor.generator = Generator(form)
    return tensor


# Tests
@generate_form_spaces_and_forms
def test_tensor_save(mesh, generate_form_space, generate_form, save_tempdir):
    space = generate_form_space(mesh)
    form = generate_form(*space)
    tensor = ParametrizedTensorFactory(form)
    evaluated_tensor = evaluate(tensor)
    tensor_save(evaluated_tensor, save_tempdir, "evaluated_tensor")


@generate_form_spaces_and_forms
def test_tensor_load(mesh, generate_form_space, generate_form, load_tempdir):
    space = generate_form_space(mesh)
    form = generate_form(*space)
    tensor = ParametrizedTensorFactory(form)
    expected_evaluated_tensor = evaluate(tensor)
    loaded_evaluated_tensor = zero_for_load(form)
    tensor_load(loaded_evaluated_tensor, load_tempdir, "evaluated_tensor")
    assert len(space) in (1, 2)
    if len(space) == 1:
        assert isclose(loaded_evaluated_tensor.get_local(), expected_evaluated_tensor.get_local()).all()
    elif len(space) == 2:
        assert isclose(loaded_evaluated_tensor.array(), expected_evaluated_tensor.array()).all()


@generate_form_spaces_and_forms
def test_tensor_io(mesh, generate_form_space, generate_form, tempdir):
    test_tensor_save(mesh, generate_form_space, generate_form, tempdir)
    test_tensor_load(mesh, generate_form_space, generate_form, tempdir)
