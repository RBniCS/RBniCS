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

mesh = UnitSquareMesh(10, 10)

V = VectorFunctionSpace(mesh, "Lagrange", 2)

expr1 = Expression("x[0]", mu_0=0., degree=1, cell=mesh.ufl_cell()) # f_4
expr2 = Expression(("x[0]", "x[1]"), mu_0=0., degree=1, cell=mesh.ufl_cell()) # f_5
expr3 = Expression((("1*x[0]", "2*x[1]"), ("3*x[0]", "4*x[1]")), mu_0=0., degree=1, cell=mesh.ufl_cell()) # f_6
expr4 = Expression((("4*x[0]", "3*x[1]"), ("2*x[0]", "1*x[1]")), mu_0=0., degree=1, cell=mesh.ufl_cell()) # f_7
expr5 = Expression("x[0]", degree=1, cell=mesh.ufl_cell()) # f_8
expr6 = Expression(("x[0]", "x[1]"), degree=1, cell=mesh.ufl_cell()) # f_9
expr7 = Expression((("1*x[0]", "2*x[1]"), ("3*x[0]", "4*x[1]")), degree=1, cell=mesh.ufl_cell()) # f_10
expr8 = Expression((("4*x[0]", "3*x[1]"), ("2*x[0]", "1*x[1]")), degree=1, cell=mesh.ufl_cell()) # f_11
expr9 = Constant(((1, 2), (3, 4))) # f_12

u = TrialFunction(V)
v = TestFunction(V)

log(PROGRESS, "*** ### @@@       VECTOR FORMS      @@@ ### ***")

a1 = inner(expr3*grad(u), grad(v))*dx + inner(grad(u)*expr2, v)*dx + expr1*inner(u, v)*dx
a1_sep = SeparatedParametrizedForm(a1)
log(PROGRESS, "*** ###              FORM 1             ### ***")
log(PROGRESS, "This is a basic vector advection-diffusion-reaction parametrized form, with all parametrized coefficients")
a1_sep.separate()
log(PROGRESS, "\tExpected len coefficients vs extracted len coefficients:\n" +
    "\t\t3 vs " + str(len(a1_sep.coefficients)) + "\n"
)
log(PROGRESS, "\tExpected sublen coefficients vs extracted sublen coefficients:\n" +
    "\t\t1 vs " + str(len(a1_sep.coefficients[0])) + "\n" +
    "\t\t1 vs " + str(len(a1_sep.coefficients[1])) + "\n" +
    "\t\t1 vs " + str(len(a1_sep.coefficients[2])) + "\n"
)
log(PROGRESS, "\tExpected coefficients vs extracted coefficients:\n" +
    "\t\tf_6 vs " + str(a1_sep.coefficients[0][0]) + "\n" +
    "\t\tf_5 vs " + str(a1_sep.coefficients[1][0]) + "\n" +
    "\t\tf_4 vs " + str(a1_sep.coefficients[2][0]) + "\n"
)
log(PROGRESS, "\tPlaceholders:\n" +
    "\t\t" + str(a1_sep._placeholders[0][0]) + "\n" +
    "\t\t" + str(a1_sep._placeholders[1][0]) + "\n" +
    "\t\t" + str(a1_sep._placeholders[2][0]) + "\n"
)
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a1_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a1_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a1_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
)
log(PROGRESS, "\tExpected len unchanged forms vs extracted len unchanged forms:\n" +
    "\t\t0 vs " + str(len(a1_sep._form_unchanged)) + "\n"
)

a2 = inner(expr3*expr4*grad(u), grad(v))*dx + inner(grad(u)*expr2, v)*dx + expr1*inner(u, v)*dx
a2_sep = SeparatedParametrizedForm(a2)
log(PROGRESS, "*** ###              FORM 2             ### ***")
log(PROGRESS, "In this case the diffusivity tensor is given by the product of two expressions")
a2_sep.separate()
log(PROGRESS, "\tExpected len coefficients vs extracted len coefficients:\n" +
    "\t\t3 vs " + str(len(a2_sep.coefficients)) + "\n"
)
log(PROGRESS, "\tExpected sublen coefficients vs extracted sublen coefficients:\n" +
    "\t\t1 vs " + str(len(a2_sep.coefficients[0])) + "\n" +
    "\t\t1 vs " + str(len(a2_sep.coefficients[1])) + "\n" +
    "\t\t1 vs " + str(len(a2_sep.coefficients[2])) + "\n"
)
log(PROGRESS, "\tExpected coefficients:\n" +
    "\t\tf_6 * f_7 vs " + str(a2_sep.coefficients[0][0]) + "\n" +
    "\t\tf_5 vs " + str(a2_sep.coefficients[1][0]) + "\n" +
    "\t\tf_4 vs " + str(a2_sep.coefficients[2][0]) + "\n"
)
log(PROGRESS, "\tPlaceholders:\n" +
    "\t\t" + str(a2_sep._placeholders[0][0]) + "\n" +
    "\t\t" + str(a2_sep._placeholders[1][0]) + "\n" +
    "\t\t" + str(a2_sep._placeholders[2][0]) + "\n"
)
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a2_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a2_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a2_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
)
log(PROGRESS, "\tExpected len unchanged forms vs extracted len unchanged forms:\n" +
    "\t\t0 vs " + str(len(a2_sep._form_unchanged)) + "\n"
)

a3 = inner(det(expr3)*(expr4 + expr3*expr3)*expr1, grad(v))*dx + inner(grad(u)*expr2, v)*dx + expr1*inner(u, v)*dx
a3_sep = SeparatedParametrizedForm(a3)
log(PROGRESS, "*** ###              FORM 3             ### ***")
log(PROGRESS, "We try now with a more complex expression of for each coefficient")
a3_sep.separate()
log(PROGRESS, "\tExpected len coefficients vs extracted len coefficients:\n" +
    "\t\t3 vs " + str(len(a3_sep.coefficients)) + "\n"
)
log(PROGRESS, "\tExpected sublen coefficients vs extracted sublen coefficients:\n" +
    "\t\t1 vs " + str(len(a3_sep.coefficients[0])) + "\n" +
    "\t\t1 vs " + str(len(a3_sep.coefficients[1])) + "\n" +
    "\t\t1 vs " + str(len(a3_sep.coefficients[2])) + "\n"
)
log(PROGRESS, "\tExpected coefficients:\n" +
    "\t\tdet(f_6) * ( f_7 + f_6 * f_6 ) * f_4 vs " + str(a3_sep.coefficients[0][0]) + "\n" +
    "\t\tf_5 vs " + str(a3_sep.coefficients[1][0]) + "\n" +
    "\t\tf_4 vs " + str(a3_sep.coefficients[2][0]) + "\n"
)
log(PROGRESS, "\tPlaceholders:\n" +
    "\t\t" + str(a3_sep._placeholders[0][0]) + "\n" +
    "\t\t" + str(a3_sep._placeholders[1][0]) + "\n" +
    "\t\t" + str(a3_sep._placeholders[2][0]) + "\n"
)
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a3_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a3_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a3_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
)
log(PROGRESS, "\tExpected len unchanged forms vs extracted len unchanged forms:\n" +
    "\t\t0 vs " + str(len(a3_sep._form_unchanged)) + "\n"
)

h = CellSize(mesh)
a4 = inner(expr3*h*grad(u), grad(v))*dx + inner(grad(u)*expr2*h, v)*dx + expr1*h*inner(u, v)*dx
a4_sep = SeparatedParametrizedForm(a4)
log(PROGRESS, "*** ###              FORM 4             ### ***")
log(PROGRESS, "We add a term depending on the mesh size. The extracted coefficients may retain the mesh size factor depending on the UFL tree")
a4_sep.separate()
log(PROGRESS, "\tExpected len coefficients vs extracted len coefficients:\n" +
    "\t\t3 vs " + str(len(a4_sep.coefficients)) + "\n"
)
log(PROGRESS, "\tExpected sublen coefficients vs extracted sublen coefficients:\n" +
    "\t\t1 vs " + str(len(a4_sep.coefficients[0])) + "\n" +
    "\t\t1 vs " + str(len(a4_sep.coefficients[1])) + "\n" +
    "\t\t1 vs " + str(len(a4_sep.coefficients[2])) + "\n"
)
log(PROGRESS, "\tExpected coefficients vs extracted coefficients:\n" +
    "\t\tf_6 * h vs " + str(a4_sep.coefficients[0][0]) + "\n" +
    "\t\tf_5 vs " + str(a4_sep.coefficients[1][0]) + "\n" +
    "\t\tf_4 * h vs " + str(a4_sep.coefficients[2][0]) + "\n"
)
log(PROGRESS, "\tPlaceholders:\n" +
    "\t\t" + str(a4_sep._placeholders[0][0]) + "\n" +
    "\t\t" + str(a4_sep._placeholders[1][0]) + "\n" +
    "\t\t" + str(a4_sep._placeholders[2][0]) + "\n"
)
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a4_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a4_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a4_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
)
log(PROGRESS, "\tExpected len unchanged forms vs extracted len unchanged forms:\n" +
    "\t\t0 vs " + str(len(a4_sep._form_unchanged)) + "\n"
)

a5 = inner((expr3*h)*grad(u), grad(v))*dx + inner(grad(u)*(expr2*h), v)*dx + (expr1*h)*inner(u, v)*dx
a5_sep = SeparatedParametrizedForm(a5)
log(PROGRESS, "*** ###              FORM 5             ### ***")
log(PROGRESS, "Starting from form 4, use parenthesis to make sure that the extracted coefficients retain the mesh size factor")
a5_sep.separate()
log(PROGRESS, "\tExpected len coefficients vs extracted len coefficients:\n" +
    "\t\t3 vs " + str(len(a5_sep.coefficients)) + "\n"
)
log(PROGRESS, "\tExpected sublen coefficients vs extracted sublen coefficients:\n" +
    "\t\t1 vs " + str(len(a5_sep.coefficients[0])) + "\n" +
    "\t\t1 vs " + str(len(a5_sep.coefficients[1])) + "\n" +
    "\t\t1 vs " + str(len(a5_sep.coefficients[2])) + "\n"
)
log(PROGRESS, "\tExpected coefficients vs extracted coefficients:\n" +
    "\t\tf_6 * h vs " + str(a5_sep.coefficients[0][0]) + "\n" +
    "\t\tf_5 * h vs " + str(a5_sep.coefficients[1][0]) + "\n" +
    "\t\tf_4 * h vs " + str(a5_sep.coefficients[2][0]) + "\n"
)
log(PROGRESS, "\tPlaceholders:\n" +
    "\t\t" + str(a5_sep._placeholders[0][0]) + "\n" +
    "\t\t" + str(a5_sep._placeholders[1][0]) + "\n" +
    "\t\t" + str(a5_sep._placeholders[2][0]) + "\n"
)
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a5_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a5_sep._form_with_placeholders[1].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a5_sep._form_with_placeholders[2].integrals()[0].integrand()) + "\n"
)
log(PROGRESS, "\tExpected len unchanged forms vs extracted len unchanged forms:\n" +
    "\t\t0 vs " + str(len(a5_sep._form_unchanged)) + "\n"
)

a6 = inner(expr7*grad(u), grad(v))*dx + inner(grad(u)*expr6, v)*dx + expr5*inner(u, v)*dx
a6_sep = SeparatedParametrizedForm(a6)
log(PROGRESS, "*** ###              FORM 6             ### ***")
log(PROGRESS, "We change the coefficients to be non-parametrized. No (parametrized) coefficients are extracted this time")
a6_sep.separate()
log(PROGRESS, "\tExpected len coefficients vs extracted len coefficients:\n" +
    "\t\t0 vs " + str(len(a6_sep.coefficients)) + "\n"
)
log(PROGRESS, "\tExpected len unchanged forms vs extracted len unchanged forms:\n" +
    "\t\t3 vs " + str(len(a6_sep._form_unchanged)) + "\n"
)
log(PROGRESS, "\tUnchanged forms:\n" +
    "\t\t" + str(a6_sep._form_unchanged[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a6_sep._form_unchanged[1].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a6_sep._form_unchanged[2].integrals()[0].integrand()) + "\n"
)

a7 = inner(expr7*(expr3*expr4)*grad(u), grad(v))*dx + inner(grad(u)*expr6, v)*dx + expr5*inner(u, v)*dx
a7_sep = SeparatedParametrizedForm(a7)
log(PROGRESS, "*** ###              FORM 7             ### ***")
log(PROGRESS, "A part of the diffusion coefficient is parametrized (advection-reaction are not parametrized). Only the parametrized part is extracted.")
a7_sep.separate()
log(PROGRESS, "\tExpected len coefficients vs extracted len coefficients:\n" +
    "\t\t1 vs " + str(len(a7_sep.coefficients)) + "\n"
)
log(PROGRESS, "\tExpected sublen coefficients vs extracted sublen coefficients:\n" +
    "\t\t1 vs " + str(len(a7_sep.coefficients[0])) + "\n"
)
log(PROGRESS, "\tExpected coefficients vs extracted coefficients:\n" +
    "\t\tf_6 * f_7 vs " + str(a7_sep.coefficients[0][0]) + "\n"
)
log(PROGRESS, "\tPlaceholders:\n" +
    "\t\t" + str(a7_sep._placeholders[0][0]) + "\n"
)
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a7_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n"
)
log(PROGRESS, "\tExpected len unchanged forms vs extracted len unchanged forms:\n" +
    "\t\t2 vs " + str(len(a7_sep._form_unchanged)) + "\n"
)
log(PROGRESS, "\tUnchanged forms:\n" +
    "\t\t" + str(a7_sep._form_unchanged[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a7_sep._form_unchanged[1].integrals()[0].integrand()) + "\n"
)

a8 = inner(expr3*expr7*expr4*grad(u), grad(v))*dx + inner(grad(u)*expr6, v)*dx + expr5*inner(u, v)*dx
a8_sep = SeparatedParametrizedForm(a8)
log(PROGRESS, "*** ###              FORM 8             ### ***")
log(PROGRESS, "This case is similar to form 7, but the order of the matrix multiplication is different. In order not to extract separately f_6 and f_7, the whole product (even with the non-parametrized part) is extracted.")
a8_sep.separate()
log(PROGRESS, "\tExpected len coefficients vs extracted len coefficients:\n" +
    "\t\t1 vs " + str(len(a8_sep.coefficients)) + "\n"
)
log(PROGRESS, "\tExpected sublen coefficients vs extracted sublen coefficients:\n" +
    "\t\t1 vs " + str(len(a8_sep.coefficients[0])) + "\n"
)
log(PROGRESS, "\tExpected coefficients vs extracted coefficients:\n" +
    "\t\tf_6 * f_10 * f_7 vs " + str(a8_sep.coefficients[0][0]) + "\n"
)
log(PROGRESS, "\tPlaceholders:\n" +
    "\t\t" + str(a8_sep._placeholders[0][0]) + "\n"
)
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a8_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n"
)
log(PROGRESS, "\tExpected len unchanged forms vs extracted len unchanged forms:\n" +
    "\t\t2 vs " + str(len(a8_sep._form_unchanged)) + "\n"
)
log(PROGRESS, "\tUnchanged forms:\n" +
    "\t\t" + str(a8_sep._form_unchanged[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a8_sep._form_unchanged[1].integrals()[0].integrand()) + "\n"
)

a9 = inner(expr9*(expr3*expr4)*grad(u), grad(v))*dx + inner(grad(u)*expr6, v)*dx + expr5*inner(u, v)*dx
a9_sep = SeparatedParametrizedForm(a9)
log(PROGRESS, "*** ###              FORM 9             ### ***")
log(PROGRESS, "This is similar to form 7, showing the trivial constants can be factored out")
a9_sep.separate()
log(PROGRESS, "\tExpected len coefficients vs extracted len coefficients:\n" +
    "\t\t1 vs " + str(len(a9_sep.coefficients)) + "\n"
)
log(PROGRESS, "\tExpected sublen coefficients vs extracted sublen coefficients:\n" +
    "\t\t1 vs " + str(len(a9_sep.coefficients[0])) + "\n"
)
log(PROGRESS, "\tExpected coefficients vs extracted coefficients:\n" +
    "\t\tf_6 * f_7 vs " + str(a9_sep.coefficients[0][0]) + "\n"
)
log(PROGRESS, "\tPlaceholders:\n" +
    "\t\t" + str(a9_sep._placeholders[0][0]) + "\n"
)
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a9_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n"
)
log(PROGRESS, "\tExpected len unchanged forms vs extracted len unchanged forms:\n" +
    "\t\t2 vs " + str(len(a9_sep._form_unchanged)) + "\n"
)
log(PROGRESS, "\tUnchanged forms:\n" +
    "\t\t" + str(a9_sep._form_unchanged[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a9_sep._form_unchanged[1].integrals()[0].integrand()) + "\n"
)

a10 = inner(expr3*expr9*expr4*grad(u), grad(v))*dx + inner(grad(u)*expr6, v)*dx + expr5*inner(u, v)*dx
a10_sep = SeparatedParametrizedForm(a10)
log(PROGRESS, "*** ###              FORM 10             ### ***")
log(PROGRESS, "This is similar to form 8, showing a case where constants cannot be factored out")
a10_sep.separate()
log(PROGRESS, "\tExpected len coefficients vs extracted len coefficients:\n" +
    "\t\t1 vs " + str(len(a10_sep.coefficients)) + "\n"
)
log(PROGRESS, "\tExpected sublen coefficients vs extracted sublen coefficients:\n" +
    "\t\t1 vs " + str(len(a10_sep.coefficients[0])) + "\n"
)
log(PROGRESS, "\tExpected coefficients vs extracted coefficients:\n" +
    "\t\tf_6 * f_12 * f_7 vs " + str(a10_sep.coefficients[0][0]) + "\n"
)
log(PROGRESS, "\tPlaceholders:\n" +
    "\t\t" + str(a10_sep._placeholders[0][0]) + "\n"
)
log(PROGRESS, "\tForms with placeholders:\n" +
    "\t\t" + str(a10_sep._form_with_placeholders[0].integrals()[0].integrand()) + "\n"
)
log(PROGRESS, "\tExpected len unchanged forms vs extracted len unchanged forms:\n" +
    "\t\t2 vs " + str(len(a10_sep._form_unchanged)) + "\n"
)
log(PROGRESS, "\tUnchanged forms:\n" +
    "\t\t" + str(a10_sep._form_unchanged[0].integrals()[0].integrand()) + "\n" +
    "\t\t" + str(a10_sep._form_unchanged[1].integrals()[0].integrand()) + "\n"
)
