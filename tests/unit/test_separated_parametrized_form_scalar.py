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
from dolfin import CellSize, Constant, ds, dx, Expression, Function, FunctionSpace, grad, inner, log, MPI, mpi_comm_world, PROGRESS, set_log_level, split, TensorFunctionSpace, TestFunction, TrialFunction, UnitSquareMesh, VectorFunctionSpace
set_log_level(PROGRESS)
from rbnics.backends.dolfin import SeparatedParametrizedForm
from rbnics.utils.decorators.store_map_from_solution_to_problem import _solution_to_problem_map

# Common variables
mesh = UnitSquareMesh(10, 10)

V = FunctionSpace(mesh, "Lagrange", 2)

expr1 = Expression("x[0]", mu_0=0., element=V.ufl_element()) # f_4
expr2 = Expression("x[1]", mu_0=0., element=V.ufl_element()) # f_5
expr3 = Expression("x[0]", mu_0=0., element=V.ufl_element()) # f_6
expr4 = Expression("x[1]", mu_0=0., element=V.ufl_element()) # f_7
expr5 = Expression(("x[0]", "x[1]"), mu_0=0., degree=1, cell=mesh.ufl_cell()) # f_8
expr6 = Expression((("1*x[0]", "2*x[1]"), ("3*x[0]", "4*x[1]")), mu_0=0., degree=1, cell=mesh.ufl_cell()) # f_9
expr7 = Expression("x[0]", element=V.ufl_element()) # f_10
expr8 = Expression("x[1]", element=V.ufl_element()) # f_11
expr9 = Expression("x[0]", element=V.ufl_element()) # f_12
expr10 = Constant(5) # f_13
expr11 = Constant(((1, 2), (3, 4))) # f_14

vector_V = VectorFunctionSpace(mesh, "Lagrange", 3)
tensor_V = TensorFunctionSpace(mesh, "Lagrange", 1)

expr12 = Function(V) # f_19
expr13 = Function(vector_V) # f_22
expr13_split = split(expr13)
expr14 = Function(tensor_V) # f_25
expr15 = Function(V) # f_28
expr16 = Function(vector_V) # f_31
expr16_split = split(expr16)
expr17 = Function(tensor_V) # f_34

class Problem(object):
    def __init__(self, name):
        self._name = name
    
    def name(self):
        return self._name
        
_solution_to_problem_map[expr12] = Problem("problem12")
_solution_to_problem_map[expr13] = Problem("problem13")
_solution_to_problem_map[expr14] = Problem("problem14")

u = TrialFunction(V)
v = TestFunction(V)

# Fixtures
skip_in_parallel = pytest.mark.skipif(MPI.size(mpi_comm_world()) > 1, reason="Numbering of functions changes in parallel.")

# Tests
@skip_in_parallel
@pytest.mark.dependency(name="1")
def test_separated_parametrized_forms_scalar_1():
    a1 = expr3*expr2*(1 + expr1*expr2)*inner(grad(u), grad(v))*dx + expr2*u.dx(0)*v*dx + expr3*u*v*dx
    a1_sep = SeparatedParametrizedForm(a1)
    log(PROGRESS, "*** ###              FORM 1             ### ***")
    log(PROGRESS, "This is a basic advection-diffusion-reaction parametrized form, with all parametrized coefficients")
    a1_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a1_sep.coefficients)) + "\n"
        )
    assert 3 == len(a1_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a1_sep.coefficients[0])) + "\n" +
        "\t\t" + str(len(a1_sep.coefficients[1])) + "\n" +
        "\t\t" + str(len(a1_sep.coefficients[2])) + "\n"
        )
    assert 1 == len(a1_sep.coefficients[0])
    assert 1 == len(a1_sep.coefficients[1])
    assert 1 == len(a1_sep.coefficients[2])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a1_sep.coefficients[0][0]) + "\n" +
        "\t\t" + str(a1_sep.coefficients[1][0]) + "\n" +
        "\t\t" + str(a1_sep.coefficients[2][0]) + "\n"
        )
    assert "(1 + f_4 * f_5) * f_5 * f_6" == str(a1_sep.coefficients[0][0])
    assert "f_5" == str(a1_sep.coefficients[1][0])
    assert "f_6" == str(a1_sep.coefficients[2][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a1_sep._placeholders[0][0]) + "\n" +
        "\t\t" + str(a1_sep._placeholders[1][0]) + "\n" +
        "\t\t" + str(a1_sep._placeholders[2][0]) + "\n"
        )
    assert "f_37" == str(a1_sep._placeholders[0][0])
    assert "f_38" == str(a1_sep._placeholders[1][0])
    assert "f_39" == str(a1_sep._placeholders[2][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a1_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a1_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a1_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
        )
    assert "f_37 * (sum_{i_8} (grad(v_0))[i_8] * (grad(v_1))[i_8] )" == str(a1_sep._form_with_placeholders[0].integrals()[0].integrand())
    assert "v_0 * (grad(v_1))[0] * f_38" == str(a1_sep._form_with_placeholders[1].integrals()[0].integrand())
    assert "v_0 * v_1 * f_39" == str(a1_sep._form_with_placeholders[2].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a1_sep._form_unchanged)) + "\n"
        )
    assert 0 == len(a1_sep._form_unchanged)

@skip_in_parallel
@pytest.mark.dependency(name="2", depends=["1"])
def test_separated_parametrized_forms_scalar_2():
    a2 = expr3*(1 + expr1*expr2)*inner(grad(u), grad(v))*expr2*dx + expr2*u.dx(0)*v*dx + expr3*u*v*dx
    a2_sep = SeparatedParametrizedForm(a2)
    log(PROGRESS, "*** ###              FORM 2             ### ***")
    log(PROGRESS, "We change the order of the product in the diffusion coefficient: note that coefficient extraction was forced to extract two coefficients because they were separated in the UFL tree")
    a2_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a2_sep.coefficients)) + "\n"
        )
    assert 3 == len(a2_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a2_sep.coefficients[0])) + "\n" +
        "\t\t" + str(len(a2_sep.coefficients[1])) + "\n" +
        "\t\t" + str(len(a2_sep.coefficients[2])) + "\n"
        )
    assert 2 == len(a2_sep.coefficients[0])
    assert 1 == len(a2_sep.coefficients[1])
    assert 1 == len(a2_sep.coefficients[2])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t(" + str(a2_sep.coefficients[0][0]) + ", " + str(a2_sep.coefficients[0][1]) + ")\n" +
        "\t\t" + str(a2_sep.coefficients[1][0]) + "\n" +
        "\t\t" + str(a2_sep.coefficients[2][0]) + "\n"
        )
    assert "(f_6 * (1 + f_4 * f_5), f_5)" == "(" + str(a2_sep.coefficients[0][0]) + ", " + str(a2_sep.coefficients[0][1]) + ")"
    assert "f_5" == str(a2_sep.coefficients[1][0])
    assert "f_6" == str(a2_sep.coefficients[2][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t(" + str(a2_sep._placeholders[0][0]) + ", " + str(a2_sep._placeholders[0][1]) + ")\n" +
        "\t\t" + str(a2_sep._placeholders[1][0]) + "\n" +
        "\t\t" + str(a2_sep._placeholders[2][0]) + "\n"
        )
    assert "(f_40, f_41)" == "(" + str(a2_sep._placeholders[0][0]) + ", " + str(a2_sep._placeholders[0][1]) + ")"
    assert "f_42" == str(a2_sep._placeholders[1][0])
    assert "f_43" == str(a2_sep._placeholders[2][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a2_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a2_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a2_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
        )
    assert "f_41 * f_40 * (sum_{i_9} (grad(v_0))[i_9] * (grad(v_1))[i_9] )" == str(a2_sep._form_with_placeholders[0].integrals()[0].integrand())
    assert "v_0 * (grad(v_1))[0] * f_42" == str(a2_sep._form_with_placeholders[1].integrals()[0].integrand())
    assert "v_0 * v_1 * f_43" == str(a2_sep._form_with_placeholders[2].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a2_sep._form_unchanged)) + "\n"
        )
    assert 0 == len(a2_sep._form_unchanged)

@skip_in_parallel
@pytest.mark.dependency(name="3", depends=["2"])
def test_separated_parametrized_forms_scalar_3():
    a3 = inner(expr3*expr2*(1 + expr1*expr2)*grad(u), grad(v))*dx + expr2*u.dx(0)*v*dx + expr3*u*v*dx
    a3_sep = SeparatedParametrizedForm(a3)
    log(PROGRESS, "*** ###              FORM 3             ### ***")
    log(PROGRESS, "We move the diffusion coefficient inside the inner product. Everything works as in form 1")
    a3_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a3_sep.coefficients)) + "\n"
        )
    assert 3 == len(a3_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a3_sep.coefficients[0])) + "\n" +
        "\t\t" + str(len(a3_sep.coefficients[1])) + "\n" +
        "\t\t" + str(len(a3_sep.coefficients[2])) + "\n"
        )
    assert 1 == len(a3_sep.coefficients[0])
    assert 1 == len(a3_sep.coefficients[1])
    assert 1 == len(a3_sep.coefficients[2])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a3_sep.coefficients[0][0]) + "\n" +
        "\t\t" + str(a3_sep.coefficients[1][0]) + "\n" +
        "\t\t" + str(a3_sep.coefficients[2][0]) + "\n"
        )
    assert "(1 + f_4 * f_5) * f_5 * f_6" == str(a3_sep.coefficients[0][0])
    assert "f_5" == str(a3_sep.coefficients[1][0])
    assert "f_6" == str(a3_sep.coefficients[2][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a3_sep._placeholders[0][0]) + "\n" +
        "\t\t" + str(a3_sep._placeholders[1][0]) + "\n" +
        "\t\t" + str(a3_sep._placeholders[2][0]) + "\n"
        )
    assert "f_44" == str(a3_sep._placeholders[0][0])
    assert "f_45" == str(a3_sep._placeholders[1][0])
    assert "f_46" == str(a3_sep._placeholders[2][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a3_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a3_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a3_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
        )
    assert "sum_{i_{11}} ({ A | A_{i_{10}} = (grad(v_1))[i_{10}] * f_44 })[i_{11}] * (grad(v_0))[i_{11}] " == str(a3_sep._form_with_placeholders[0].integrals()[0].integrand())
    assert "v_0 * (grad(v_1))[0] * f_45" == str(a3_sep._form_with_placeholders[1].integrals()[0].integrand())
    assert "v_0 * v_1 * f_46" == str(a3_sep._form_with_placeholders[2].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a3_sep._form_unchanged)) + "\n"
        )
    assert 0 == len(a3_sep._form_unchanged)

@skip_in_parallel
@pytest.mark.dependency(name="4", depends=["3"])
def test_separated_parametrized_forms_scalar_4():
    a4 = inner(expr6*grad(u), grad(v))*dx + inner(expr5, grad(u))*v*dx + expr3*u*v*dx
    a4_sep = SeparatedParametrizedForm(a4)
    log(PROGRESS, "*** ###              FORM 4             ### ***")
    log(PROGRESS, "We use a diffusivity tensor now. The extraction is able to correctly detect the matrix.")
    a4_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a4_sep.coefficients)) + "\n"
        )
    assert 3 == len(a4_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a4_sep.coefficients[0])) + "\n" +
        "\t\t" + str(len(a4_sep.coefficients[1])) + "\n" +
        "\t\t" + str(len(a4_sep.coefficients[2])) + "\n"
        )
    assert 1 == len(a4_sep.coefficients[0])
    assert 1 == len(a4_sep.coefficients[1])
    assert 1 == len(a4_sep.coefficients[2])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a4_sep.coefficients[0][0]) + "\n" +
        "\t\t" + str(a4_sep.coefficients[1][0]) + "\n" +
        "\t\t" + str(a4_sep.coefficients[2][0]) + "\n"
        )
    assert "f_9" == str(a4_sep.coefficients[0][0])
    assert "f_8" == str(a4_sep.coefficients[1][0])
    assert "f_6" == str(a4_sep.coefficients[2][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a4_sep._placeholders[0][0]) + "\n" +
        "\t\t" + str(a4_sep._placeholders[1][0]) + "\n" +
        "\t\t" + str(a4_sep._placeholders[2][0]) + "\n"
        )
    assert "f_47" == str(a4_sep._placeholders[0][0])
    assert "f_48" == str(a4_sep._placeholders[1][0])
    assert "f_49" == str(a4_sep._placeholders[2][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a4_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a4_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a4_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
        )
    assert "sum_{i_{14}} ({ A | A_{i_{12}} = sum_{i_{13}} f_47[i_{12}, i_{13}] * (grad(v_1))[i_{13}]  })[i_{14}] * (grad(v_0))[i_{14}] " == str(a4_sep._form_with_placeholders[0].integrals()[0].integrand())
    assert "v_0 * (sum_{i_{15}} f_48[i_{15}] * (grad(v_1))[i_{15}] )" == str(a4_sep._form_with_placeholders[1].integrals()[0].integrand())
    assert "v_0 * v_1 * f_49" == str(a4_sep._form_with_placeholders[2].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a4_sep._form_unchanged)) + "\n"
        )
    assert 0 == len(a4_sep._form_unchanged)

@skip_in_parallel
@pytest.mark.dependency(name="5", depends=["4"])
def test_separated_parametrized_forms_scalar_5():
    a5 = expr3*expr2*(1 + expr1*expr2)*inner(grad(u), grad(v))*ds + expr2*u.dx(0)*v*ds + expr3*u*v*ds
    a5_sep = SeparatedParametrizedForm(a5)
    log(PROGRESS, "*** ###              FORM 5             ### ***")
    log(PROGRESS, "We change the integration domain to be the boundary. The result is the same as form 1")
    a5_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a5_sep.coefficients)) + "\n"
        )
    assert 3 == len(a5_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a5_sep.coefficients[0])) + "\n" +
        "\t\t" + str(len(a5_sep.coefficients[1])) + "\n" +
        "\t\t" + str(len(a5_sep.coefficients[2])) + "\n"
        )
    assert 1 == len(a5_sep.coefficients[0])
    assert 1 == len(a5_sep.coefficients[1])
    assert 1 == len(a5_sep.coefficients[2])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a5_sep.coefficients[0][0]) + "\n" +
        "\t\t" + str(a5_sep.coefficients[1][0]) + "\n" +
        "\t\t" + str(a5_sep.coefficients[2][0]) + "\n"
        )
    assert "(1 + f_4 * f_5) * f_5 * f_6" == str(a5_sep.coefficients[0][0])
    assert "f_5" == str(a5_sep.coefficients[1][0])
    assert "f_6" == str(a5_sep.coefficients[2][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a5_sep._placeholders[0][0]) + "\n" +
        "\t\t" + str(a5_sep._placeholders[1][0]) + "\n" +
        "\t\t" + str(a5_sep._placeholders[2][0]) + "\n"
        )
    assert "f_50" == str(a5_sep._placeholders[0][0])
    assert "f_51" == str(a5_sep._placeholders[1][0])
    assert "f_52" == str(a5_sep._placeholders[2][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a5_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a5_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a5_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
        )
    assert "f_50 * (sum_{i_{16}} (grad(v_0))[i_{16}] * (grad(v_1))[i_{16}] )" == str(a5_sep._form_with_placeholders[0].integrals()[0].integrand())
    assert "v_0 * (grad(v_1))[0] * f_51" == str(a5_sep._form_with_placeholders[1].integrals()[0].integrand())
    assert "v_0 * v_1 * f_52" == str(a5_sep._form_with_placeholders[2].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a5_sep._form_unchanged)) + "\n"
        )
    assert 0 == len(a5_sep._form_unchanged)

@skip_in_parallel
@pytest.mark.dependency(name="6", depends=["5"])
def test_separated_parametrized_forms_scalar_6():
    h = CellSize(mesh)
    a6 = expr3*h*u*v*dx
    a6_sep = SeparatedParametrizedForm(a6)
    log(PROGRESS, "*** ###              FORM 6             ### ***")
    log(PROGRESS, "We add a term depending on the mesh size. The extracted coefficient retain the mesh size factor")
    a6_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a6_sep.coefficients)) + "\n"
        )
    assert 1 == len(a6_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a6_sep.coefficients[0])) + "\n"
        )
    assert 1 == len(a6_sep.coefficients[0])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a6_sep.coefficients[0][0]) + "\n"
        )
    assert "f_6 * 2.0 * circumradius" == str(a6_sep.coefficients[0][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a6_sep._placeholders[0][0]) + "\n"
        )
    assert "f_53" == str(a6_sep._placeholders[0][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a6_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n"
        )
    assert "v_0 * v_1 * f_53" == str(a6_sep._form_with_placeholders[0].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a6_sep._form_unchanged)) + "\n"
        )
    assert 0 == len(a6_sep._form_unchanged)

@skip_in_parallel
@pytest.mark.dependency(name="7", depends=["6"])
def test_separated_parametrized_forms_scalar_7():
    a7 = expr9*expr8*(1 + expr7*expr8)*inner(grad(u), grad(v))*dx + expr8*u.dx(0)*v*dx + expr9*u*v*dx
    a7_sep = SeparatedParametrizedForm(a7)
    log(PROGRESS, "*** ###              FORM 7             ### ***")
    log(PROGRESS, "We change the coefficients to be non-parametrized. No (parametrized) coefficients are extracted this time.")
    a7_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a7_sep.coefficients)) + "\n"
        )
    assert 0 == len(a7_sep.coefficients)
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a7_sep._form_unchanged)) + "\n"
        )
    assert 3 == len(a7_sep._form_unchanged)
    log(PROGRESS, "\tUnchanged forms:\n" +
        "\t\t" + str(a7_sep._form_unchanged[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a7_sep._form_unchanged[1].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a7_sep._form_unchanged[2].integrals()[0].integrand()) + "\n"
        )
    assert "(1 + f_10 * f_11) * f_11 * f_12 * (sum_{i_{17}} (grad(v_0))[i_{17}] * (grad(v_1))[i_{17}] )" == str(a7_sep._form_unchanged[0].integrals()[0].integrand())
    assert "v_0 * (grad(v_1))[0] * f_11" == str(a7_sep._form_unchanged[1].integrals()[0].integrand())
    assert "v_0 * v_1 * f_12" == str(a7_sep._form_unchanged[2].integrals()[0].integrand())

@skip_in_parallel
@pytest.mark.dependency(name="8", depends=["7"])
def test_separated_parametrized_forms_scalar_8():
    a8 = expr2*(1 + expr1*expr2)*expr9*inner(grad(u), grad(v))*dx + expr8*u.dx(0)*v*dx + expr9*u*v*dx
    a8_sep = SeparatedParametrizedForm(a8)
    log(PROGRESS, "*** ###              FORM 8             ### ***")
    log(PROGRESS, "A part of the diffusion coefficient is parametrized (advection-reaction are not parametrized). Only the parametrized part is extracted.")
    a8_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a8_sep.coefficients)) + "\n"
        )
    assert 1 == len(a8_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a8_sep.coefficients[0])) + "\n"
        )
    assert 1 == len(a8_sep.coefficients[0])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a8_sep.coefficients[0][0]) + "\n"
        )
    assert "f_5 * (1 + f_4 * f_5)" == str(a8_sep.coefficients[0][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a8_sep._placeholders[0][0]) + "\n"
        )
    assert "f_54" == str(a8_sep._placeholders[0][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a8_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n"
        )
    assert "f_12 * f_54 * (sum_{i_{18}} (grad(v_0))[i_{18}] * (grad(v_1))[i_{18}] )" == str(a8_sep._form_with_placeholders[0].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a8_sep._form_unchanged)) + "\n"
        )
    assert 2 == len(a8_sep._form_unchanged)
    log(PROGRESS, "\tUnchanged forms:\n" +
        "\t\t" + str(a8_sep._form_unchanged[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a8_sep._form_unchanged[1].integrals()[0].integrand()) + "\n"
        )
    assert "v_0 * (grad(v_1))[0] * f_11" == str(a8_sep._form_unchanged[0].integrals()[0].integrand())
    assert "v_0 * v_1 * f_12" == str(a8_sep._form_unchanged[1].integrals()[0].integrand())

@skip_in_parallel
@pytest.mark.dependency(name="9", depends=["8"])
def test_separated_parametrized_forms_scalar_9():
    a9 = expr9*expr2*(1 + expr1*expr2)*inner(grad(u), grad(v))*dx + expr8*u.dx(0)*v*dx + expr9*u*v*dx
    a9_sep = SeparatedParametrizedForm(a9)
    log(PROGRESS, "*** ###              FORM 9             ### ***")
    log(PROGRESS, "A part of the diffusion coefficient is parametrized (advection-reaction are not parametrized), where the terms are written in a different order when compared to form 8. Due to the UFL tree this would entail to extract two (\"sub\") coefficients, but this is not done for the sake of efficiency")
    a9_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a9_sep.coefficients)) + "\n"
        )
    assert 1 == len(a9_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a9_sep.coefficients[0])) + "\n"
        )
    assert 1 == len(a9_sep.coefficients[0])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a9_sep.coefficients[0][0]) + "\n"
        )
    assert "(1 + f_4 * f_5) * f_5 * f_12" == str(a9_sep.coefficients[0][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a9_sep._placeholders[0][0]) + "\n"
        )
    assert "f_55" == str(a9_sep._placeholders[0][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a9_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n"
        )
    assert "f_55 * (sum_{i_{19}} (grad(v_0))[i_{19}] * (grad(v_1))[i_{19}] )" == str(a9_sep._form_with_placeholders[0].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a9_sep._form_unchanged)) + "\n"
        )
    assert 2 == len(a9_sep._form_unchanged)
    log(PROGRESS, "\tUnchanged forms:\n" +
        "\t\t" + str(a9_sep._form_unchanged[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a9_sep._form_unchanged[1].integrals()[0].integrand()) + "\n"
        )
    assert "v_0 * (grad(v_1))[0] * f_11" == str(a9_sep._form_unchanged[0].integrals()[0].integrand())
    assert "v_0 * v_1 * f_12" == str(a9_sep._form_unchanged[1].integrals()[0].integrand())

@skip_in_parallel
@pytest.mark.dependency(name="10", depends=["9"])
def test_separated_parametrized_forms_scalar_10():
    h = CellSize(mesh)
    a10 = expr9*h*u*v*dx
    a10_sep = SeparatedParametrizedForm(a10)
    log(PROGRESS, "*** ###              FORM 10             ### ***")
    log(PROGRESS, "Similarly to form 6, we add a term depending on the mesh size multiplied by a non-parametrized coefficient. Neither are retained")
    a10_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a10_sep.coefficients)) + "\n"
        )
    assert 0 == len(a10_sep.coefficients)
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a10_sep._form_unchanged)) + "\n"
        )
    assert 1 == len(a10_sep._form_unchanged)
    log(PROGRESS, "\tUnchanged forms:\n" +
        "\t\t" + str(a10_sep._form_unchanged[0].integrals()[0].integrand()) + "\n"
        )
    assert "v_0 * v_1 * f_12 * 2.0 * circumradius" == str(a10_sep._form_unchanged[0].integrals()[0].integrand())

@skip_in_parallel
@pytest.mark.dependency(name="11", depends=["10"])
def test_separated_parametrized_forms_scalar_11():
    h = CellSize(mesh)
    a11 = expr9*expr3*h*u*v*dx
    a11_sep = SeparatedParametrizedForm(a11)
    log(PROGRESS, "*** ###              FORM 11             ### ***")
    log(PROGRESS, "Similarly to form 6, we add a term depending on the mesh size multiplied by the product of parametrized and a non-parametrized coefficients. In this case, in contrast to form 6, the extraction retains the non-parametrized coefficient but not the mesh size")
    a11_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a11_sep.coefficients)) + "\n"
        )
    assert 1 == len(a11_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a11_sep.coefficients[0])) + "\n"
        )
    assert 1 == len(a11_sep.coefficients[0])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a11_sep.coefficients[0][0]) + "\n"
        )
    assert "f_6" == str(a11_sep.coefficients[0][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a11_sep._placeholders[0][0]) + "\n"
        )
    assert "f_56" == str(a11_sep._placeholders[0][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a11_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n"
        )
    assert "v_0 * v_1 * 2.0 * circumradius * f_12 * f_56" == str(a11_sep._form_with_placeholders[0].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a11_sep._form_unchanged)) + "\n"
        )
    assert 0 == len(a11_sep._form_unchanged)

@skip_in_parallel
@pytest.mark.dependency(name="12", depends=["11"])
def test_separated_parametrized_forms_scalar_12():
    h = CellSize(mesh)
    a12 = expr9*(expr3*h)*u*v*dx
    a12_sep = SeparatedParametrizedForm(a12)
    log(PROGRESS, "*** ###              FORM 12             ### ***")
    log(PROGRESS, "We change form 11 adding parenthesis around the multiplication between parametrized coefficient and h. In this case the extraction retains the non-parametrized coefficient and the mesh size")
    a12_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a12_sep.coefficients)) + "\n"
        )
    assert 1 == len(a12_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a12_sep.coefficients[0])) + "\n"
        )
    assert 1 == len(a12_sep.coefficients[0])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a12_sep.coefficients[0][0]) + "\n"
        )
    assert "f_6 * 2.0 * circumradius" == str(a12_sep.coefficients[0][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a12_sep._placeholders[0][0]) + "\n"
        )
    assert "f_57" == str(a12_sep._placeholders[0][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a12_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n"
        )
    assert "v_0 * v_1 * f_12 * f_57" == str(a12_sep._form_with_placeholders[0].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a12_sep._form_unchanged)) + "\n"
        )
    assert 0 == len(a12_sep._form_unchanged)

@skip_in_parallel
@pytest.mark.dependency(name="13", depends=["12"])
def test_separated_parametrized_forms_scalar_13():
    a13 = inner(expr11*expr6*expr10*grad(u), grad(v))*dx + inner(expr10*expr5, grad(u))*v*dx + expr10*expr3*u*v*dx
    a13_sep = SeparatedParametrizedForm(a13)
    log(PROGRESS, "*** ###              FORM 13             ### ***")
    log(PROGRESS, "Constants are factored out.")
    a13_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a13_sep.coefficients)) + "\n"
        )
    assert 3 == len(a13_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a13_sep.coefficients[0])) + "\n" +
        "\t\t" + str(len(a13_sep.coefficients[1])) + "\n" +
        "\t\t" + str(len(a13_sep.coefficients[2])) + "\n"
        )
    assert 1 == len(a13_sep.coefficients[0])
    assert 1 == len(a13_sep.coefficients[1])
    assert 1 == len(a13_sep.coefficients[2])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a13_sep.coefficients[0][0]) + "\n" +
        "\t\t" + str(a13_sep.coefficients[1][0]) + "\n" +
        "\t\t" + str(a13_sep.coefficients[2][0]) + "\n"
        )
    assert "f_9" == str(a13_sep.coefficients[0][0])
    assert "f_8" == str(a13_sep.coefficients[1][0])
    assert "f_6" == str(a13_sep.coefficients[2][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a13_sep._placeholders[0][0]) + "\n" +
        "\t\t" + str(a13_sep._placeholders[1][0]) + "\n" +
        "\t\t" + str(a13_sep._placeholders[2][0]) + "\n"
        )
    assert "f_58" == str(a13_sep._placeholders[0][0])
    assert "f_59" == str(a13_sep._placeholders[1][0])
    assert "f_60" == str(a13_sep._placeholders[2][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a13_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a13_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a13_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
        )
    assert "sum_{i_{28}} ({ A | A_{i_{25}} = sum_{i_{26}} ({ A | A_{i_{23}, i_{24}} = ({ A | A_{i_{20}, i_{21}} = sum_{i_{22}} f_14[i_{20}, i_{22}] * f_58[i_{22}, i_{21}]  })[i_{23}, i_{24}] * f_13 })[i_{25}, i_{26}] * (grad(v_1))[i_{26}]  })[i_{28}] * (grad(v_0))[i_{28}] " == str(a13_sep._form_with_placeholders[0].integrals()[0].integrand())
    assert "v_0 * (sum_{i_{29}} ({ A | A_{i_{27}} = f_59[i_{27}] * f_13 })[i_{29}] * (grad(v_1))[i_{29}] )" == str(a13_sep._form_with_placeholders[1].integrals()[0].integrand())
    assert "v_0 * v_1 * f_13 * f_60" == str(a13_sep._form_with_placeholders[2].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a13_sep._form_unchanged)) + "\n"
        )
    assert 0 == len(a13_sep._form_unchanged)

@skip_in_parallel
@pytest.mark.dependency(name="14", depends=["13"])
def test_separated_parametrized_forms_scalar_14():
    a14 = expr3*expr2*(1 + expr1*expr2)*expr12*inner(grad(u), grad(v))*dx + expr2*expr12*u.dx(0)*v*dx + expr3*expr12*u*v*dx
    a14_sep = SeparatedParametrizedForm(a14)
    log(PROGRESS, "*** ###              FORM 14             ### ***")
    log(PROGRESS, "This form is similar to form 1, but each term is multiplied by a scalar Function, which is the solution of a parametrized problem")
    a14_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a14_sep.coefficients)) + "\n"
        )
    assert 3 == len(a14_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a14_sep.coefficients[0])) + "\n" +
        "\t\t" + str(len(a14_sep.coefficients[1])) + "\n" +
        "\t\t" + str(len(a14_sep.coefficients[2])) + "\n"
        )
    assert 1 == len(a14_sep.coefficients[0])
    assert 1 == len(a14_sep.coefficients[1])
    assert 1 == len(a14_sep.coefficients[2])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a14_sep.coefficients[0][0]) + "\n" +
        "\t\t" + str(a14_sep.coefficients[1][0]) + "\n" +
        "\t\t" + str(a14_sep.coefficients[2][0]) + "\n"
        )
    assert "f_19 * (1 + f_4 * f_5) * f_5 * f_6" == str(a14_sep.coefficients[0][0])
    assert "f_5 * f_19" == str(a14_sep.coefficients[1][0])
    assert "f_6 * f_19" == str(a14_sep.coefficients[2][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a14_sep._placeholders[0][0]) + "\n" +
        "\t\t" + str(a14_sep._placeholders[1][0]) + "\n" +
        "\t\t" + str(a14_sep._placeholders[2][0]) + "\n"
        )
    assert "f_65" == str(a14_sep._placeholders[0][0])
    assert "f_66" == str(a14_sep._placeholders[1][0])
    assert "f_67" == str(a14_sep._placeholders[2][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a14_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a14_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a14_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
        )
    assert "f_65 * (sum_{i_{30}} (grad(v_0))[i_{30}] * (grad(v_1))[i_{30}] )" == str(a14_sep._form_with_placeholders[0].integrals()[0].integrand())
    assert "v_0 * (grad(v_1))[0] * f_66" == str(a14_sep._form_with_placeholders[1].integrals()[0].integrand())
    assert "v_0 * v_1 * f_67" == str(a14_sep._form_with_placeholders[2].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a14_sep._form_unchanged)) + "\n"
        )
    assert 0 == len(a14_sep._form_unchanged)

@skip_in_parallel
@pytest.mark.dependency(name="15", depends=["14"])
def test_separated_parametrized_forms_scalar_15():
    a15 = inner(expr14*expr6*grad(u), grad(v))*dx + inner((expr13 + expr5), grad(u))*v*dx + expr12*expr3*u*v*dx
    a15_sep = SeparatedParametrizedForm(a15)
    log(PROGRESS, "*** ###              FORM 15             ### ***")
    log(PROGRESS, "This form is similar to form 4, but each term is multiplied/added by/to a solution of a parametrized problem, either scalar, vector or tensor shaped")
    a15_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a15_sep.coefficients)) + "\n"
        )
    assert 3 == len(a15_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a15_sep.coefficients[0])) + "\n" +
        "\t\t" + str(len(a15_sep.coefficients[1])) + "\n" +
        "\t\t" + str(len(a15_sep.coefficients[2])) + "\n"
        )
    assert 1 == len(a15_sep.coefficients[0])
    assert 1 == len(a15_sep.coefficients[1])
    assert 1 == len(a15_sep.coefficients[2])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a15_sep.coefficients[0][0]) + "\n" +
        "\t\t" + str(a15_sep.coefficients[1][0]) + "\n" +
        "\t\t" + str(a15_sep.coefficients[2][0]) + "\n"
        )
    assert "{ A | A_{i_{31}, i_{32}} = sum_{i_{33}} f_9[i_{33}, i_{32}] * f_25[i_{31}, i_{33}]  }" == str(a15_sep.coefficients[0][0])
    assert "f_8 + f_22" == str(a15_sep.coefficients[1][0])
    assert "f_6 * f_19" == str(a15_sep.coefficients[2][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a15_sep._placeholders[0][0]) + "\n" +
        "\t\t" + str(a15_sep._placeholders[1][0]) + "\n" +
        "\t\t" + str(a15_sep._placeholders[2][0]) + "\n"
        )
    assert "f_68" == str(a15_sep._placeholders[0][0])
    assert "f_69" == str(a15_sep._placeholders[1][0])
    assert "f_70" == str(a15_sep._placeholders[2][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a15_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a15_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a15_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
        )
    assert "sum_{i_{36}} ({ A | A_{i_{34}} = sum_{i_{35}} f_68[i_{34}, i_{35}] * (grad(v_1))[i_{35}]  })[i_{36}] * (grad(v_0))[i_{36}] " == str(a15_sep._form_with_placeholders[0].integrals()[0].integrand())
    assert "v_0 * (sum_{i_{37}} f_69[i_{37}] * (grad(v_1))[i_{37}] )" == str(a15_sep._form_with_placeholders[1].integrals()[0].integrand())
    assert "v_0 * v_1 * f_70" == str(a15_sep._form_with_placeholders[2].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a15_sep._form_unchanged)) + "\n"
        )
    assert 0 == len(a15_sep._form_unchanged)

@skip_in_parallel
@pytest.mark.dependency(name="16", depends=["15"])
def test_separated_parametrized_forms_scalar_16():
    a16 = inner(expr9*expr14*expr8*(1 + expr7*expr8)*grad(u), grad(v))*dx + inner(expr13, grad(u))*v*dx + expr12*u*v*dx
    a16_sep = SeparatedParametrizedForm(a16)
    log(PROGRESS, "*** ###              FORM 16             ### ***")
    log(PROGRESS, "We change the coefficients of form 14 to be non-parametrized (as in form 7). Due to the presence of solutions of parametrized problems in contrast to form 7 all coefficients are now parametrized.")
    a16_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a16_sep.coefficients)) + "\n"
        )
    assert 3 == len(a16_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a16_sep.coefficients[0])) + "\n" +
        "\t\t" + str(len(a16_sep.coefficients[1])) + "\n" +
        "\t\t" + str(len(a16_sep.coefficients[2])) + "\n"
        )
    assert 1 == len(a16_sep.coefficients[0])
    assert 1 == len(a16_sep.coefficients[1])
    assert 1 == len(a16_sep.coefficients[2])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a16_sep.coefficients[0][0]) + "\n" +
        "\t\t" + str(a16_sep.coefficients[1][0]) + "\n" +
        "\t\t" + str(a16_sep.coefficients[2][0]) + "\n"
        )
    assert "f_25" == str(a16_sep.coefficients[0][0])
    assert "f_22" == str(a16_sep.coefficients[1][0])
    assert "f_19" == str(a16_sep.coefficients[2][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a16_sep._placeholders[0][0]) + "\n" +
        "\t\t" + str(a16_sep._placeholders[1][0]) + "\n" +
        "\t\t" + str(a16_sep._placeholders[2][0]) + "\n"
        )
    assert "f_71" == str(a16_sep._placeholders[0][0])
    assert "f_72" == str(a16_sep._placeholders[1][0])
    assert "f_73" == str(a16_sep._placeholders[2][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a16_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a16_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a16_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
        )
    assert "sum_{i_{46}} ({ A | A_{i_{44}} = sum_{i_{45}} ({ A | A_{i_{42}, i_{43}} = ({ A | A_{i_{40}, i_{41}} = ({ A | A_{i_{38}, i_{39}} = f_71[i_{38}, i_{39}] * f_12 })[i_{40}, i_{41}] * f_11 })[i_{42}, i_{43}] * (1 + f_10 * f_11) })[i_{44}, i_{45}] * (grad(v_1))[i_{45}]  })[i_{46}] * (grad(v_0))[i_{46}] " == str(a16_sep._form_with_placeholders[0].integrals()[0].integrand())
    assert "v_0 * (sum_{i_{47}} f_72[i_{47}] * (grad(v_1))[i_{47}] )" == str(a16_sep._form_with_placeholders[1].integrals()[0].integrand())
    assert "v_0 * v_1 * f_73" == str(a16_sep._form_with_placeholders[2].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a16_sep._form_unchanged)) + "\n"
        )
    assert 0 == len(a16_sep._form_unchanged)

@skip_in_parallel
@pytest.mark.dependency(name="17", depends=["16"])
def test_separated_parametrized_forms_scalar_17():
    a17 = expr13_split[0]*u*v*dx + expr13_split[1]*u.dx(0)*v*dx
    a17_sep = SeparatedParametrizedForm(a17)
    log(PROGRESS, "*** ###              FORM 17             ### ***")
    log(PROGRESS, "This form is similar to form 15, but each term is multiplied to a component of a solution of a parametrized problem")
    a17_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a17_sep.coefficients)) + "\n"
        )
    assert 2 == len(a17_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a17_sep.coefficients[0])) + "\n" +
        "\t\t" + str(len(a17_sep.coefficients[1])) + "\n"
        )
    assert 1 == len(a17_sep.coefficients[0])
    assert 1 == len(a17_sep.coefficients[1])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a17_sep.coefficients[0][0]) + "\n" +
        "\t\t" + str(a17_sep.coefficients[1][0]) + "\n"
        )
    assert "f_22[0]" == str(a17_sep.coefficients[0][0])
    assert "f_22[1]" == str(a17_sep.coefficients[1][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a17_sep._placeholders[0][0]) + "\n" +
        "\t\t" + str(a17_sep._placeholders[1][0]) + "\n"
        )
    assert "f_74" == str(a17_sep._placeholders[0][0])
    assert "f_75" == str(a17_sep._placeholders[1][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a17_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a17_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n"
        )
    assert "v_0 * v_1 * f_74" == str(a17_sep._form_with_placeholders[0].integrals()[0].integrand())
    assert "v_0 * (grad(v_1))[0] * f_75" == str(a17_sep._form_with_placeholders[1].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a17_sep._form_unchanged)) + "\n"
        )
    assert 0 == len(a17_sep._form_unchanged)
    
@skip_in_parallel
@pytest.mark.dependency(name="18", depends=["17"])
def test_separated_parametrized_forms_scalar_18():
    a18 = expr3*expr2*(1 + expr1*expr2)*expr15*inner(grad(u), grad(v))*dx + expr2*expr15*u.dx(0)*v*dx + expr3*expr15*u*v*dx
    a18_sep = SeparatedParametrizedForm(a18)
    log(PROGRESS, "*** ###              FORM 18             ### ***")
    log(PROGRESS, "This form is similar to form 14, but each term is multiplied by a scalar Function, which is not the solution of a parametrized problem")
    a18_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a18_sep.coefficients)) + "\n"
        )
    assert 3 == len(a18_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a18_sep.coefficients[0])) + "\n" +
        "\t\t" + str(len(a18_sep.coefficients[1])) + "\n" +
        "\t\t" + str(len(a18_sep.coefficients[2])) + "\n"
        )
    assert 1 == len(a18_sep.coefficients[0])
    assert 1 == len(a18_sep.coefficients[1])
    assert 1 == len(a18_sep.coefficients[2])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a18_sep.coefficients[0][0]) + "\n" +
        "\t\t" + str(a18_sep.coefficients[1][0]) + "\n" +
        "\t\t" + str(a18_sep.coefficients[2][0]) + "\n"
        )
    assert "(1 + f_4 * f_5) * f_5 * f_6" == str(a18_sep.coefficients[0][0])
    assert "f_5" == str(a18_sep.coefficients[1][0])
    assert "f_6" == str(a18_sep.coefficients[2][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a18_sep._placeholders[0][0]) + "\n" +
        "\t\t" + str(a18_sep._placeholders[1][0]) + "\n" +
        "\t\t" + str(a18_sep._placeholders[2][0]) + "\n"
        )
    assert "f_76" == str(a18_sep._placeholders[0][0])
    assert "f_77" == str(a18_sep._placeholders[1][0])
    assert "f_78" == str(a18_sep._placeholders[2][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a18_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a18_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a18_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
        )
    assert "f_28 * f_76 * (sum_{i_{48}} (grad(v_0))[i_{48}] * (grad(v_1))[i_{48}] )" == str(a18_sep._form_with_placeholders[0].integrals()[0].integrand())
    assert "v_0 * (grad(v_1))[0] * f_28 * f_77" == str(a18_sep._form_with_placeholders[1].integrals()[0].integrand())
    assert "v_0 * v_1 * f_28 * f_78" == str(a18_sep._form_with_placeholders[2].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a18_sep._form_unchanged)) + "\n"
        )
    assert 0 == len(a18_sep._form_unchanged)
    
@skip_in_parallel
@pytest.mark.dependency(name="19", depends=["18"])
def test_separated_parametrized_forms_scalar_19():
    a19 = inner(expr17*expr6*grad(u), grad(v))*dx + inner((expr16 + expr5), grad(u))*v*dx + expr15*expr3*u*v*dx
    a19_sep = SeparatedParametrizedForm(a19)
    log(PROGRESS, "*** ###              FORM 19             ### ***")
    log(PROGRESS, "This form is similar to form 15, but each term is multiplied/added by/to a Function which is not the solution of a parametrized problem")
    a19_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a19_sep.coefficients)) + "\n"
        )
    assert 3 == len(a19_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a19_sep.coefficients[0])) + "\n" +
        "\t\t" + str(len(a19_sep.coefficients[1])) + "\n" +
        "\t\t" + str(len(a19_sep.coefficients[2])) + "\n"
        )
    assert 1 == len(a19_sep.coefficients[0])
    assert 1 == len(a19_sep.coefficients[1])
    assert 1 == len(a19_sep.coefficients[2])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a19_sep.coefficients[0][0]) + "\n" +
        "\t\t" + str(a19_sep.coefficients[1][0]) + "\n" +
        "\t\t" + str(a19_sep.coefficients[2][0]) + "\n"
        )
    assert "f_9" == str(a19_sep.coefficients[0][0])
    assert "f_8" == str(a19_sep.coefficients[1][0])
    assert "f_6" == str(a19_sep.coefficients[2][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a19_sep._placeholders[0][0]) + "\n" +
        "\t\t" + str(a19_sep._placeholders[1][0]) + "\n" +
        "\t\t" + str(a19_sep._placeholders[2][0]) + "\n"
        )
    assert "f_79" == str(a19_sep._placeholders[0][0])
    assert "f_80" == str(a19_sep._placeholders[1][0])
    assert "f_81" == str(a19_sep._placeholders[2][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a19_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a19_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a19_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
        )
    assert "sum_{i_{54}} ({ A | A_{i_{52}} = sum_{i_{53}} ({ A | A_{i_{49}, i_{50}} = sum_{i_{51}} f_34[i_{49}, i_{51}] * f_79[i_{51}, i_{50}]  })[i_{52}, i_{53}] * (grad(v_1))[i_{53}]  })[i_{54}] * (grad(v_0))[i_{54}] " == str(a19_sep._form_with_placeholders[0].integrals()[0].integrand())
    assert "v_0 * (sum_{i_{55}} (f_31 + f_80)[i_{55}] * (grad(v_1))[i_{55}] )" == str(a19_sep._form_with_placeholders[1].integrals()[0].integrand())
    assert "v_0 * v_1 * f_28 * f_81" == str(a19_sep._form_with_placeholders[2].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a19_sep._form_unchanged)) + "\n"
        )
    assert 0 == len(a19_sep._form_unchanged)
    
@skip_in_parallel
@pytest.mark.dependency(name="20", depends=["19"])
def test_separated_parametrized_forms_scalar_20():
    a20 = inner(expr9*expr17*expr8*(1 + expr7*expr8)*grad(u), grad(v))*dx + inner(expr16, grad(u))*v*dx + expr15*u*v*dx
    a20_sep = SeparatedParametrizedForm(a20)
    log(PROGRESS, "*** ###              FORM 20             ### ***")
    log(PROGRESS, "This form is similar to form 16, but each term is multiplied by a scalar Function, which is not the solution of a parametrized problem. As in form 7, no parametrized coefficients are detected")
    a20_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a20_sep.coefficients)) + "\n"
        )
    assert 0 == len(a20_sep.coefficients)
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a20_sep._form_unchanged)) + "\n"
        )
    assert 3 == len(a20_sep._form_unchanged)
    log(PROGRESS, "\tUnchanged forms:\n" +
        "\t\t" + str(a20_sep._form_unchanged[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a20_sep._form_unchanged[1].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a20_sep._form_unchanged[2].integrals()[0].integrand()) + "\n"
        )
    assert "sum_{i_{64}} ({ A | A_{i_{62}} = sum_{i_{63}} ({ A | A_{i_{60}, i_{61}} = ({ A | A_{i_{58}, i_{59}} = ({ A | A_{i_{56}, i_{57}} = f_34[i_{56}, i_{57}] * f_12 })[i_{58}, i_{59}] * f_11 })[i_{60}, i_{61}] * (1 + f_10 * f_11) })[i_{62}, i_{63}] * (grad(v_1))[i_{63}]  })[i_{64}] * (grad(v_0))[i_{64}] " == str(a20_sep._form_unchanged[0].integrals()[0].integrand())
    assert "v_0 * (sum_{i_{65}} f_31[i_{65}] * (grad(v_1))[i_{65}] )" == str(a20_sep._form_unchanged[1].integrals()[0].integrand())
    assert "v_0 * v_1 * f_28" == str(a20_sep._form_unchanged[2].integrals()[0].integrand())
    
@skip_in_parallel
@pytest.mark.dependency(name="21", depends=["20"])
def test_separated_parametrized_forms_scalar_21():
    a21 = expr16_split[0]*u*v*dx + expr16_split[1]*u.dx(0)*v*dx
    a21_sep = SeparatedParametrizedForm(a21)
    log(PROGRESS, "*** ###              FORM 21             ### ***")
    log(PROGRESS, "This form is similar to form 17, but each term is multiplied to a component of a Function which is not the solution of a parametrized problem. This results in no parametrized coefficients")
    a21_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a21_sep.coefficients)) + "\n"
        )
    assert 0 == len(a21_sep.coefficients)
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a21_sep._form_unchanged)) + "\n"
        )
    assert 2 == len(a21_sep._form_unchanged)
    log(PROGRESS, "\tUnchanged forms:\n" +
        "\t\t" + str(a21_sep._form_unchanged[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a21_sep._form_unchanged[1].integrals()[0].integrand()) + "\n"
        )
    assert "v_0 * f_31[0] * v_1" == str(a21_sep._form_unchanged[0].integrals()[0].integrand())
    assert "v_0 * (grad(v_1))[0] * f_31[1]" == str(a21_sep._form_unchanged[1].integrals()[0].integrand())
    
@skip_in_parallel
@pytest.mark.dependency(name="22", depends=["21"])
def test_separated_parametrized_forms_scalar_22():
    a22 = inner(grad(expr13_split[0]), grad(u))*v*dx + expr13_split[1].dx(0)*u.dx(0)*v*dx
    a22_sep = SeparatedParametrizedForm(a22)
    log(PROGRESS, "*** ###              FORM 22             ### ***")
    log(PROGRESS, "This form is similar to form 17, but each term is multiplied to the gradient/partial derivative of a component of a solution of a parametrized problem")
    a22_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a22_sep.coefficients)) + "\n"
        )
    assert 2 == len(a22_sep.coefficients)
    log(PROGRESS, "\tSublen coefficients:\n" +
        "\t\t" + str(len(a22_sep.coefficients[0])) + "\n" +
        "\t\t" + str(len(a22_sep.coefficients[1])) + "\n"
        )
    assert 1 == len(a22_sep.coefficients[0])
    assert 1 == len(a22_sep.coefficients[1])
    log(PROGRESS, "\tCoefficients:\n" +
        "\t\t" + str(a22_sep.coefficients[0][0]) + "\n" +
        "\t\t" + str(a22_sep.coefficients[1][0]) + "\n"
        )
    assert "grad(f_22)" == str(a22_sep.coefficients[0][0])
    assert "(grad(f_22))[1, 0]" == str(a22_sep.coefficients[1][0])
    log(PROGRESS, "\tPlaceholders:\n" +
        "\t\t" + str(a22_sep._placeholders[0][0]) + "\n" +
        "\t\t" + str(a22_sep._placeholders[1][0]) + "\n"
        )
    assert "f_82" == str(a22_sep._placeholders[0][0])
    assert "f_83" == str(a22_sep._placeholders[1][0])
    log(PROGRESS, "\tForms with placeholders:\n" +
        "\t\t" + str(a22_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a22_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n"
        )
    assert "v_0 * (sum_{i_{66}} f_82[0, i_{66}] * (grad(v_1))[i_{66}] )" == str(a22_sep._form_with_placeholders[0].integrals()[0].integrand())
    assert "v_0 * (grad(v_1))[0] * f_83" == str(a22_sep._form_with_placeholders[1].integrals()[0].integrand())
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a22_sep._form_unchanged)) + "\n"
        )
    assert 0 == len(a22_sep._form_unchanged)
    
@skip_in_parallel
@pytest.mark.dependency(name="23", depends=["22"])
def test_separated_parametrized_forms_scalar_23():
    a23 = inner(grad(expr16_split[0]), grad(u))*v*dx + expr16_split[1].dx(0)*u.dx(0)*v*dx
    a23_sep = SeparatedParametrizedForm(a23)
    log(PROGRESS, "*** ###              FORM 23             ### ***")
    log(PROGRESS, "This form is similar to form 22, but each term is multiplied to the gradient/partial derivative of a component of a Function which is not the solution of a parametrized problem. This results in no parametrized coefficients")
    a23_sep.separate()
    log(PROGRESS, "\tLen coefficients:\n" +
        "\t\t" + str(len(a23_sep.coefficients)) + "\n"
        )
    assert 0 == len(a23_sep.coefficients)
    log(PROGRESS, "\tLen unchanged forms:\n" +
        "\t\t" + str(len(a23_sep._form_unchanged)) + "\n"
        )
    assert 2 == len(a23_sep._form_unchanged)
    log(PROGRESS, "\tUnchanged forms:\n" +
        "\t\t" + str(a23_sep._form_unchanged[0].integrals()[0].integrand()) + "\n" +
        "\t\t" + str(a23_sep._form_unchanged[1].integrals()[0].integrand()) + "\n"
        )
    assert "v_0 * (sum_{i_{69}} (grad(f_31))[0, i_{69}] * (grad(v_1))[i_{69}] )" == str(a23_sep._form_unchanged[0].integrals()[0].integrand())
    assert "v_0 * (grad(v_1))[0] * (grad(f_31))[1, 0]" == str(a23_sep._form_unchanged[1].integrals()[0].integrand())
