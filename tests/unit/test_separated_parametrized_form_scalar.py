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

from dolfin import *
set_log_level(PROGRESS)
from rbnics.backends.dolfin import SeparatedParametrizedForm

assert MPI.size(mpi_comm_world()) == 1, "Numbering of functions changes in parallel"

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

u = TrialFunction(V)
v = TestFunction(V)

log(PROGRESS, "*** ### @@@       SCALAR FORMS      @@@ ### ***")

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
assert "f_28" == str(a1_sep._placeholders[0][0])
assert "f_29" == str(a1_sep._placeholders[1][0])
assert "f_30" == str(a1_sep._placeholders[2][0])
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a1_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a1_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a1_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
)
assert "f_28 * (sum_{i_8} (grad(v_0))[i_8] * (grad(v_1))[i_8] )" == str(a1_sep._form_with_placeholders[0].integrals()[0].integrand())
assert "v_0 * (grad(v_1))[0] * f_29" == str(a1_sep._form_with_placeholders[1].integrals()[0].integrand())
assert "v_0 * v_1 * f_30" == str(a1_sep._form_with_placeholders[2].integrals()[0].integrand())
log(PROGRESS, "\tLen unchanged forms:\n" +
    "\t\t" + str(len(a1_sep._form_unchanged)) + "\n"
)
assert 0 == len(a1_sep._form_unchanged)

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
assert "(f_31, f_32)" == "(" + str(a2_sep._placeholders[0][0]) + ", " + str(a2_sep._placeholders[0][1]) + ")"
assert "f_33" == str(a2_sep._placeholders[1][0])
assert "f_34" == str(a2_sep._placeholders[2][0])
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a2_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a2_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a2_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
)
assert "f_32 * f_31 * (sum_{i_9} (grad(v_0))[i_9] * (grad(v_1))[i_9] )" == str(a2_sep._form_with_placeholders[0].integrals()[0].integrand())
assert "v_0 * (grad(v_1))[0] * f_33" == str(a2_sep._form_with_placeholders[1].integrals()[0].integrand())
assert "v_0 * v_1 * f_34" == str(a2_sep._form_with_placeholders[2].integrals()[0].integrand())
log(PROGRESS, "\tLen unchanged forms:\n" +
    "\t\t" + str(len(a2_sep._form_unchanged)) + "\n"
)
assert 0 == len(a2_sep._form_unchanged)

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
assert "f_35" == str(a3_sep._placeholders[0][0])
assert "f_36" == str(a3_sep._placeholders[1][0])
assert "f_37" == str(a3_sep._placeholders[2][0])
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a3_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a3_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a3_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
)
assert "sum_{i_{11}} ({ A | A_{i_{10}} = (grad(v_1))[i_{10}] * f_35 })[i_{11}] * (grad(v_0))[i_{11}] " == str(a3_sep._form_with_placeholders[0].integrals()[0].integrand())
assert "v_0 * (grad(v_1))[0] * f_36" == str(a3_sep._form_with_placeholders[1].integrals()[0].integrand())
assert "v_0 * v_1 * f_37" == str(a3_sep._form_with_placeholders[2].integrals()[0].integrand())
log(PROGRESS, "\tLen unchanged forms:\n" +
    "\t\t" + str(len(a3_sep._form_unchanged)) + "\n"
)
assert 0 == len(a3_sep._form_unchanged)

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
assert "f_38" == str(a4_sep._placeholders[0][0])
assert "f_39" == str(a4_sep._placeholders[1][0])
assert "f_40" == str(a4_sep._placeholders[2][0])
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a4_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a4_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a4_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
)
assert "sum_{i_{14}} ({ A | A_{i_{12}} = sum_{i_{13}} f_38[i_{12}, i_{13}] * (grad(v_1))[i_{13}]  })[i_{14}] * (grad(v_0))[i_{14}] " == str(a4_sep._form_with_placeholders[0].integrals()[0].integrand())
assert "v_0 * (sum_{i_{15}} f_39[i_{15}] * (grad(v_1))[i_{15}] )" == str(a4_sep._form_with_placeholders[1].integrals()[0].integrand())
assert "v_0 * v_1 * f_40" == str(a4_sep._form_with_placeholders[2].integrals()[0].integrand())
log(PROGRESS, "\tLen unchanged forms:\n" +
    "\t\t" + str(len(a4_sep._form_unchanged)) + "\n"
)
assert 0 == len(a4_sep._form_unchanged)

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
assert "f_41" == str(a5_sep._placeholders[0][0])
assert "f_42" == str(a5_sep._placeholders[1][0])
assert "f_43" == str(a5_sep._placeholders[2][0])
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a5_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a5_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a5_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
)
assert "f_41 * (sum_{i_{16}} (grad(v_0))[i_{16}] * (grad(v_1))[i_{16}] )" == str(a5_sep._form_with_placeholders[0].integrals()[0].integrand())
assert "v_0 * (grad(v_1))[0] * f_42" == str(a5_sep._form_with_placeholders[1].integrals()[0].integrand())
assert "v_0 * v_1 * f_43" == str(a5_sep._form_with_placeholders[2].integrals()[0].integrand())
log(PROGRESS, "\tLen unchanged forms:\n" +
    "\t\t" + str(len(a5_sep._form_unchanged)) + "\n"
)
assert 0 == len(a5_sep._form_unchanged)

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
assert "f_44" == str(a6_sep._placeholders[0][0])
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a6_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n"
)
assert "v_0 * v_1 * f_44" == str(a6_sep._form_with_placeholders[0].integrals()[0].integrand())
log(PROGRESS, "\tLen unchanged forms:\n" +
    "\t\t" + str(len(a6_sep._form_unchanged)) + "\n"
)
assert 0 == len(a6_sep._form_unchanged)

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
assert "f_45" == str(a8_sep._placeholders[0][0])
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a8_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n"
)
assert "f_12 * f_45 * (sum_{i_{18}} (grad(v_0))[i_{18}] * (grad(v_1))[i_{18}] )" == str(a8_sep._form_with_placeholders[0].integrals()[0].integrand())
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
assert "f_46" == str(a9_sep._placeholders[0][0])
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a9_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n"
)
assert "f_46 * (sum_{i_{19}} (grad(v_0))[i_{19}] * (grad(v_1))[i_{19}] )" == str(a9_sep._form_with_placeholders[0].integrals()[0].integrand())
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
assert "f_47" == str(a11_sep._placeholders[0][0])
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a11_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n"
)
assert "v_0 * v_1 * 2.0 * circumradius * f_12 * f_47" == str(a11_sep._form_with_placeholders[0].integrals()[0].integrand())
log(PROGRESS, "\tLen unchanged forms:\n" +
    "\t\t" + str(len(a11_sep._form_unchanged)) + "\n"
)
assert 0 == len(a11_sep._form_unchanged)

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
assert "f_48" == str(a12_sep._placeholders[0][0])
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a12_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n"
)
assert "v_0 * v_1 * f_12 * f_48" == str(a12_sep._form_with_placeholders[0].integrals()[0].integrand())
log(PROGRESS, "\tLen unchanged forms:\n" +
    "\t\t" + str(len(a12_sep._form_unchanged)) + "\n"
)
assert 0 == len(a12_sep._form_unchanged)

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
assert "f_49" == str(a13_sep._placeholders[0][0])
assert "f_50" == str(a13_sep._placeholders[1][0])
assert "f_51" == str(a13_sep._placeholders[2][0])
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a13_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a13_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a13_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
)
assert "sum_{i_{28}} ({ A | A_{i_{25}} = sum_{i_{26}} ({ A | A_{i_{23}, i_{24}} = ({ A | A_{i_{20}, i_{21}} = sum_{i_{22}} f_14[i_{20}, i_{22}] * f_49[i_{22}, i_{21}]  })[i_{23}, i_{24}] * f_13 })[i_{25}, i_{26}] * (grad(v_1))[i_{26}]  })[i_{28}] * (grad(v_0))[i_{28}] " == str(a13_sep._form_with_placeholders[0].integrals()[0].integrand())
assert "v_0 * (sum_{i_{29}} ({ A | A_{i_{27}} = f_50[i_{27}] * f_13 })[i_{29}] * (grad(v_1))[i_{29}] )" == str(a13_sep._form_with_placeholders[1].integrals()[0].integrand())
assert "v_0 * v_1 * f_13 * f_51" == str(a13_sep._form_with_placeholders[2].integrals()[0].integrand())
log(PROGRESS, "\tLen unchanged forms:\n" +
    "\t\t" + str(len(a13_sep._form_unchanged)) + "\n"
)
assert 0 == len(a13_sep._form_unchanged)

a14 = expr3*expr2*(1 + expr1*expr2)*expr12*inner(grad(u), grad(v))*dx + expr2*expr12*u.dx(0)*v*dx + expr3*expr12*u*v*dx
a14_sep = SeparatedParametrizedForm(a14)
log(PROGRESS, "*** ###              FORM 14             ### ***")
log(PROGRESS, "This form is similar to form 1, but each term is multiplied by a scalar Function, which could result from a nonlinear problem")
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
assert "f_52" == str(a14_sep._placeholders[0][0])
assert "f_53" == str(a14_sep._placeholders[1][0])
assert "f_54" == str(a14_sep._placeholders[2][0])
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a14_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a14_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a14_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
)
assert "f_52 * (sum_{i_{30}} (grad(v_0))[i_{30}] * (grad(v_1))[i_{30}] )" == str(a14_sep._form_with_placeholders[0].integrals()[0].integrand())
assert "v_0 * (grad(v_1))[0] * f_53" == str(a14_sep._form_with_placeholders[1].integrals()[0].integrand())
assert "v_0 * v_1 * f_54" == str(a14_sep._form_with_placeholders[2].integrals()[0].integrand())
log(PROGRESS, "\tLen unchanged forms:\n" +
    "\t\t" + str(len(a14_sep._form_unchanged)) + "\n"
)
assert 0 == len(a14_sep._form_unchanged)

a15 = inner(expr14*expr6*grad(u), grad(v))*dx + inner((expr13 + expr5), grad(u))*v*dx + expr12*expr3*u*v*dx
a15_sep = SeparatedParametrizedForm(a15)
log(PROGRESS, "*** ###              FORM 15             ### ***")
log(PROGRESS, "This form is similar to form 4, but each term is multiplied/added by/to a Function, either scalar, vector or tensor shaped")
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
assert "f_55" == str(a15_sep._placeholders[0][0])
assert "f_56" == str(a15_sep._placeholders[1][0])
assert "f_57" == str(a15_sep._placeholders[2][0])
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a15_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a15_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a15_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
)
assert "sum_{i_{36}} ({ A | A_{i_{34}} = sum_{i_{35}} f_55[i_{34}, i_{35}] * (grad(v_1))[i_{35}]  })[i_{36}] * (grad(v_0))[i_{36}] " == str(a15_sep._form_with_placeholders[0].integrals()[0].integrand())
assert "v_0 * (sum_{i_{37}} f_56[i_{37}] * (grad(v_1))[i_{37}] )" == str(a15_sep._form_with_placeholders[1].integrals()[0].integrand())
assert "v_0 * v_1 * f_57" == str(a15_sep._form_with_placeholders[2].integrals()[0].integrand())
log(PROGRESS, "\tLen unchanged forms:\n" +
    "\t\t" + str(len(a15_sep._form_unchanged)) + "\n"
)
assert 0 == len(a15_sep._form_unchanged)

a16 = inner(expr9*expr14*expr8*(1 + expr7*expr8)*grad(u), grad(v))*dx + inner(expr13, grad(u))*v*dx + expr12*u*v*dx
a16_sep = SeparatedParametrizedForm(a16)
log(PROGRESS, "*** ###              FORM 16             ### ***")
log(PROGRESS, "We change the coefficients of form 14 to be non-parametrized (as in form 7). Due to the presence of functions (which are assumed to be parametrized, since they usually come as the solution itself in nonlinear problems) in contrast to form 7 all coefficients are now parametrized.")
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
assert "f_58" == str(a16_sep._placeholders[0][0])
assert "f_59" == str(a16_sep._placeholders[1][0])
assert "f_60" == str(a16_sep._placeholders[2][0])
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a16_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a16_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a16_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
)
assert "sum_{i_{46}} ({ A | A_{i_{44}} = sum_{i_{45}} ({ A | A_{i_{42}, i_{43}} = ({ A | A_{i_{40}, i_{41}} = ({ A | A_{i_{38}, i_{39}} = f_58[i_{38}, i_{39}] * f_12 })[i_{40}, i_{41}] * f_11 })[i_{42}, i_{43}] * (1 + f_10 * f_11) })[i_{44}, i_{45}] * (grad(v_1))[i_{45}]  })[i_{46}] * (grad(v_0))[i_{46}] " == str(a16_sep._form_with_placeholders[0].integrals()[0].integrand())
assert "v_0 * (sum_{i_{47}} f_59[i_{47}] * (grad(v_1))[i_{47}] )" == str(a16_sep._form_with_placeholders[1].integrals()[0].integrand())
assert "v_0 * v_1 * f_60" == str(a16_sep._form_with_placeholders[2].integrals()[0].integrand())
log(PROGRESS, "\tLen unchanged forms:\n" +
    "\t\t" + str(len(a16_sep._form_unchanged)) + "\n"
)
assert 0 == len(a16_sep._form_unchanged)

a17 = expr13_split[0]*u*v*dx + expr13_split[1]*u.dx(0)*v*dx
a17_sep = SeparatedParametrizedForm(a17)
log(PROGRESS, "*** ###              FORM 17             ### ***")
log(PROGRESS, "This form is similar to form 15, but each term is multiplied to a component of a Function")
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
assert "f_61" == str(a17_sep._placeholders[0][0])
assert "f_62" == str(a17_sep._placeholders[1][0])
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a17_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a17_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n"
)
assert "v_0 * v_1 * f_61" == str(a17_sep._form_with_placeholders[0].integrals()[0].integrand())
assert "v_0 * (grad(v_1))[0] * f_62" == str(a17_sep._form_with_placeholders[1].integrals()[0].integrand())
log(PROGRESS, "\tLen unchanged forms:\n" +
    "\t\t" + str(len(a17_sep._form_unchanged)) + "\n"
)
assert 0 == len(a17_sep._form_unchanged)
