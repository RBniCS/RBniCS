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
from dolfin import assemble, dx, FiniteElement, FunctionSpace, inner, MixedElement, split, TestFunction, TrialFunction, UnitSquareMesh, VectorElement
from rbnics.backends.dolfin import evaluate as _evaluate, ParametrizedTensorFactory
from rbnics.backends.dolfin.export import tensor_save
from rbnics.backends.dolfin.import_ import tensor_load
from rbnics.eim.utils.decorators import add_to_map_from_parametrized_expression_to_EIM_approximation

# Meshes
@pytest.fixture(scope="module")
def mesh():
    return UnitSquareMesh(10, 10)

# Forms: elliptic case
def generate_elliptic_linear_form_space(mesh):
    return (FunctionSpace(mesh, "Lagrange", 2), )
    
def generate_elliptic_linear_form(V):
    v = TestFunction(V)
    return v*dx
    
def generate_elliptic_bilinear_form_space(mesh):
    return generate_elliptic_linear_form_space(mesh) + generate_elliptic_linear_form_space(mesh)
    
def generate_elliptic_bilinear_form(V1, V2):
    assert V1.ufl_element() == V2.ufl_element()
    u = TrialFunction(V1)
    v = TestFunction(V2)
    return u*v*dx
    
# Forms: mixed case
def generate_mixed_linear_form_space(mesh):
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    return (FunctionSpace(mesh, element), )
    
def generate_mixed_linear_form(V):
    v = TestFunction(V)
    (v_0, v_1) = split(v)
    return v_0[0]*dx + v_0[1]*dx + v_1*dx
    
def generate_mixed_bilinear_form_space(mesh):
    return generate_mixed_linear_form_space(mesh) + generate_mixed_linear_form_space(mesh)
    
def generate_mixed_bilinear_form(V1, V2):
    assert V1.ufl_element() == V2.ufl_element()
    u = TrialFunction(V1)
    v = TestFunction(V2)
    (u_0, u_1) = split(u)
    (v_0, v_1) = split(v)
    return inner(u_0, v_0)*dx + u_1*v_1*dx + u_0[0]*v_1*dx + u_1*v_0[1]*dx

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
    return v[0]*dx + v[1]*dx
    
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
    return inner(u_0, v)*dx + u_1*v[0]*dx
    
# Forms decorator
generate_form_spaces_and_forms = pytest.mark.parametrize("generate_form_space, generate_form", [
    (generate_elliptic_linear_form_space, generate_elliptic_linear_form),
    (generate_elliptic_bilinear_form_space, generate_elliptic_bilinear_form),
    (generate_mixed_linear_form_space, generate_mixed_linear_form),
    (generate_mixed_bilinear_form_space, generate_mixed_bilinear_form),
    (generate_collapsed_linear_form_space, generate_collapsed_linear_form),
    (generate_collapsed_bilinear_form_space, generate_collapsed_bilinear_form)
])
    
# Mock an EIM approximation to avoid triggering an assert
def evaluate(tensor):
    add_to_map_from_parametrized_expression_to_EIM_approximation(tensor, None)
    return _evaluate(tensor)
    
# Prepare tensor storage for load
class Generator(object):
    def __init__(self, form):
        self._form = form
        
def zero_for_load(form):
    tensor = assemble(form)
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
    loaded_successfully = tensor_load(loaded_evaluated_tensor, load_tempdir, "evaluated_tensor")
    assert loaded_successfully
    assert isclose(loaded_evaluated_tensor.array(), expected_evaluated_tensor.array()).all()
    
@generate_form_spaces_and_forms
def test_tensor_io(mesh, generate_form_space, generate_form, tempdir):
    test_tensor_save(mesh, generate_form_space, generate_form, tempdir)
    test_tensor_load(mesh, generate_form_space, generate_form, tempdir)
