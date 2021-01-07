# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from logging import DEBUG, getLogger
from mpi4py import MPI
from dolfin import (CellDiameter, Constant, ds, dx, Expression, Function, FunctionSpace, grad, inner, split,
                    TensorFunctionSpace, TestFunction, TrialFunction, UnitSquareMesh, VectorFunctionSpace)
from rbnics.backends.dolfin import SeparatedParametrizedForm
from rbnics.backends.dolfin.separated_parametrized_form import logger as separated_parametrized_form_logger
from rbnics.utils.decorators.store_map_from_solution_to_problem import _solution_to_problem_map
from rbnics.utils.test import enable_logging

# Logger
test_logger = getLogger("tests/unit/test_separated_parametrized_forms_scalar.py")
enable_separated_parametrized_form_logging = enable_logging({
    separated_parametrized_form_logger: DEBUG, test_logger: DEBUG})

# Common variables
mesh = UnitSquareMesh(10, 10)

V = FunctionSpace(mesh, "Lagrange", 2)

expr1 = Expression("x[0]", mu_0=0., element=V.ufl_element())  # f_5
expr2 = Expression("x[1]", mu_0=0., element=V.ufl_element())  # f_6
expr3 = Expression("x[0]", mu_0=0., element=V.ufl_element())  # f_7
expr4 = Expression("x[1]", mu_0=0., element=V.ufl_element())  # f_8
expr5 = Expression(("x[0]", "x[1]"), mu_0=0., degree=1, cell=mesh.ufl_cell())  # f_9
expr6 = Expression((("1*x[0]", "2*x[1]"), ("3*x[0]", "4*x[1]")), mu_0=0., degree=1, cell=mesh.ufl_cell())  # f_10
expr7 = Expression("x[0]", element=V.ufl_element())  # f_11
expr8 = Expression("x[1]", element=V.ufl_element())  # f_12
expr9 = Expression("x[0]", element=V.ufl_element())  # f_13
expr10 = Constant(5)  # f_14
expr11 = Constant(((1, 2), (3, 4)))  # f_15

vector_V = VectorFunctionSpace(mesh, "Lagrange", 3)
tensor_V = TensorFunctionSpace(mesh, "Lagrange", 1)

expr12 = Function(V)  # f_20
expr13 = Function(vector_V)  # f_23
expr13_split = split(expr13)
expr14 = Function(tensor_V)  # f_26
expr15 = Function(V)  # f_29
expr16 = Function(vector_V)  # f_32
expr16_split = split(expr16)
expr17 = Function(tensor_V)  # f_35


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
skip_in_parallel = pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Numbering of functions changes in parallel.")


# Tests
@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="1")
def test_separated_parametrized_forms_scalar_1():
    a1 = expr3 * expr2 * inner(grad(u), grad(v)) * dx + expr2 * u.dx(0) * v * dx + expr3 * u * v * dx
    a1_sep = SeparatedParametrizedForm(a1)
    test_logger.log(DEBUG, "*** ###              FORM 1             ### ***")
    test_logger.log(DEBUG, "This is a basic advection-diffusion-reaction parametrized form, with all"
                    + " parametrized coefficients")
    a1_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep.coefficients)))

    assert 3 == len(a1_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep.coefficients[2])))

    assert 2 == len(a1_sep.coefficients[0])
    assert 1 == len(a1_sep.coefficients[1])
    assert 1 == len(a1_sep.coefficients[2])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t(" + str(a1_sep.coefficients[0][0]) + ", " + str(a1_sep.coefficients[0][1]) + ")")
    test_logger.log(DEBUG, "\t\t" + str(a1_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep.coefficients[2][0]))

    assert "(f_7, f_6)" == "(" + str(a1_sep.coefficients[0][0]) + ", " + str(a1_sep.coefficients[0][1]) + ")"
    assert "f_6" == str(a1_sep.coefficients[1][0])
    assert "f_7" == str(a1_sep.coefficients[2][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t(" + str(a1_sep._placeholders[0][0]) + ", " + str(a1_sep._placeholders[0][1]) + ")")
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._placeholders[2][0]))

    assert "(f_38, f_39)" == "(" + str(a1_sep._placeholders[0][0]) + ", " + str(a1_sep._placeholders[0][1]) + ")"
    assert "f_40" == str(a1_sep._placeholders[1][0])
    assert "f_41" == str(a1_sep._placeholders[2][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._form_with_placeholders[2].integrals()[0].integrand()))

    assert ("f_39 * f_38 * (sum_{i_8} (grad(v_0))[i_8] * (grad(v_1))[i_8] )"
            == str(a1_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert "f_40 * (grad(v_1))[0] * v_0" == str(a1_sep._form_with_placeholders[1].integrals()[0].integrand())
    assert "f_41 * v_0 * v_1" == str(a1_sep._form_with_placeholders[2].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep._form_unchanged)))

    assert 0 == len(a1_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="2", depends=["1"])
def test_separated_parametrized_forms_scalar_2():
    a2 = inner(expr3 * grad(u), expr2 * grad(v)) * dx + expr2 * u.dx(0) * v * dx + expr3 * u * v * dx
    a2_sep = SeparatedParametrizedForm(a2)
    test_logger.log(DEBUG, "*** ###              FORM 2             ### ***")
    test_logger.log(DEBUG, "We move the diffusion coefficient inside the inner product, splitting it into"
                    + " two coefficients: note that coefficient extraction was forced to extract two coefficients"
                    + " because they were separated in the UFL tree")
    a2_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep.coefficients)))

    assert 3 == len(a2_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep.coefficients[2])))

    assert 2 == len(a2_sep.coefficients[0])
    assert 1 == len(a2_sep.coefficients[1])
    assert 1 == len(a2_sep.coefficients[2])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t(" + str(a2_sep.coefficients[0][0]) + ", " + str(a2_sep.coefficients[0][1]) + ")")
    test_logger.log(DEBUG, "\t\t" + str(a2_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep.coefficients[2][0]))

    assert "(f_7, f_6)" == "(" + str(a2_sep.coefficients[0][0]) + ", " + str(a2_sep.coefficients[0][1]) + ")"
    assert "f_6" == str(a2_sep.coefficients[1][0])
    assert "f_7" == str(a2_sep.coefficients[2][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t(" + str(a2_sep._placeholders[0][0]) + ", " + str(a2_sep._placeholders[0][1]) + ")")
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._placeholders[2][0]))

    assert "(f_42, f_43)" == "(" + str(a2_sep._placeholders[0][0]) + ", " + str(a2_sep._placeholders[0][1]) + ")"
    assert "f_44" == str(a2_sep._placeholders[1][0])
    assert "f_45" == str(a2_sep._placeholders[2][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._form_with_placeholders[2].integrals()[0].integrand()))

    assert ("sum_{i_{11}} ({ A | A_{i_9} = (grad(v_1))[i_9] * f_42 })[i_{11}] * ({ A | A_{i_{10}} = (grad(v_0))"
            + "[i_{10}] * f_43 })[i_{11}] "
            == str(a2_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert "v_0 * (grad(v_1))[0] * f_44" == str(a2_sep._form_with_placeholders[1].integrals()[0].integrand())
    assert "v_0 * v_1 * f_45" == str(a2_sep._form_with_placeholders[2].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep._form_unchanged)))

    assert 0 == len(a2_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="3", depends=["2"])
def test_separated_parametrized_forms_scalar_3():
    a3 = (expr3 * expr2 * (1 + expr1 * expr2) * inner(grad(u), grad(v)) * dx
          + expr2 * (1 + expr2 * expr3) * u.dx(0) * v * dx
          + expr3 * (1 + expr1 * expr2) * u * v * dx)
    a3_sep = SeparatedParametrizedForm(a3)
    test_logger.log(DEBUG, "*** ###              FORM 3             ### ***")
    test_logger.log(DEBUG, "This form tests the expansion of a sum of coefficients")
    a3_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients)))

    assert 6 == len(a3_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients[2])))
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients[3])))
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients[4])))
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients[5])))

    assert 3 == len(a3_sep.coefficients[0])
    assert 2 == len(a3_sep.coefficients[1])
    assert 1 == len(a3_sep.coefficients[2])
    assert 2 == len(a3_sep.coefficients[3])
    assert 1 == len(a3_sep.coefficients[4])
    assert 3 == len(a3_sep.coefficients[5])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t(" + str(a3_sep.coefficients[0][0]) + ", " + str(a3_sep.coefficients[0][1]) + ", "
                    + str(a3_sep.coefficients[0][2]) + ")")
    test_logger.log(DEBUG, "\t\t(" + str(a3_sep.coefficients[1][0]) + ", " + str(a3_sep.coefficients[1][1]) + ")")
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[2][0]))
    test_logger.log(DEBUG, "\t\t(" + str(a3_sep.coefficients[3][0]) + ", " + str(a3_sep.coefficients[3][1]) + ")")
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[4][0]))
    test_logger.log(DEBUG, "\t\t(" + str(a3_sep.coefficients[5][0]) + ", " + str(a3_sep.coefficients[5][1]) + ", "
                    + str(a3_sep.coefficients[5][2]) + ")")

    assert "f_5" == str(a3_sep.coefficients[0][0])
    assert "f_7" == str(a3_sep.coefficients[0][1])
    assert "f_6 * f_6" == str(a3_sep.coefficients[0][2])
    assert "f_7" == str(a3_sep.coefficients[1][0])
    assert "f_6" == str(a3_sep.coefficients[1][1])
    assert "f_6" == str(a3_sep.coefficients[2][0])
    assert "f_7" == str(a3_sep.coefficients[3][0])
    assert "f_6 * f_6" == str(a3_sep.coefficients[3][1])
    assert "f_7" == str(a3_sep.coefficients[4][0])
    assert "f_5" == str(a3_sep.coefficients[5][0])
    assert "f_7" == str(a3_sep.coefficients[5][1])
    assert "f_6" == str(a3_sep.coefficients[5][2])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t(" + str(a3_sep._placeholders[0][0]) + ", " + str(a3_sep._placeholders[0][1]) + ", "
                    + str(a3_sep._placeholders[0][2]) + ")")
    test_logger.log(DEBUG, "\t\t(" + str(a3_sep._placeholders[1][0]) + ", " + str(a3_sep._placeholders[1][1]) + ")")
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[2][0]))
    test_logger.log(DEBUG, "\t\t(" + str(a3_sep._placeholders[3][0]) + ", " + str(a3_sep._placeholders[3][1]) + ")")
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[4][0]))
    test_logger.log(DEBUG, "\t\t(" + str(a3_sep._placeholders[5][0]) + ", " + str(a3_sep._placeholders[5][1]) + ", "
                    + str(a3_sep._placeholders[5][2]) + ")")

    assert "f_46" == str(a3_sep._placeholders[0][0])
    assert "f_47" == str(a3_sep._placeholders[0][1])
    assert "f_48" == str(a3_sep._placeholders[0][2])
    assert "f_49" == str(a3_sep._placeholders[1][0])
    assert "f_50" == str(a3_sep._placeholders[1][1])
    assert "f_51" == str(a3_sep._placeholders[2][0])
    assert "f_52" == str(a3_sep._placeholders[3][0])
    assert "f_53" == str(a3_sep._placeholders[3][1])
    assert "f_54" == str(a3_sep._placeholders[4][0])
    assert "f_55" == str(a3_sep._placeholders[5][0])
    assert "f_56" == str(a3_sep._placeholders[5][1])
    assert "f_57" == str(a3_sep._placeholders[5][2])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[3].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[4].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[5].integrals()[0].integrand()))

    assert ("f_48 * f_47 * f_46 * (sum_{i_{12}} (grad(v_0))[i_{12}] * (grad(v_1))[i_{12}] )"
            == str(a3_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("f_50 * f_49 * (sum_{i_{12}} (grad(v_0))[i_{12}] * (grad(v_1))[i_{12}] )"
            == str(a3_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert "f_51 * (grad(v_1))[0] * v_0" == str(a3_sep._form_with_placeholders[2].integrals()[0].integrand())
    assert "f_53 * f_52 * (grad(v_1))[0] * v_0" == str(a3_sep._form_with_placeholders[3].integrals()[0].integrand())
    assert "f_54 * v_0 * v_1" == str(a3_sep._form_with_placeholders[4].integrals()[0].integrand())
    assert "f_57 * f_56 * v_0 * v_1 * f_55" == str(a3_sep._form_with_placeholders[5].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep._form_unchanged)))

    assert 0 == len(a3_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="4", depends=["3"])
def test_separated_parametrized_forms_scalar_4():
    a4 = (inner(expr6 * (1 + expr1 * expr2) * grad(u), grad(v)) * dx + inner(expr5, grad(u)) * v * dx
          + expr3 * u * v * dx)
    a4_sep = SeparatedParametrizedForm(a4)
    test_logger.log(DEBUG, "*** ###              FORM 4             ### ***")
    test_logger.log(DEBUG, "We use a diffusivity tensor now. The extraction is able to correctly detect the matrix.")
    a4_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep.coefficients)))

    assert 4 == len(a4_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep.coefficients[2])))
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep.coefficients[3])))

    assert 1 == len(a4_sep.coefficients[0])
    assert 1 == len(a4_sep.coefficients[1])
    assert 1 == len(a4_sep.coefficients[2])
    assert 1 == len(a4_sep.coefficients[3])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a4_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep.coefficients[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep.coefficients[3][0]))

    assert "{ A | A_{i_{13}, i_{14}} = f_10[i_{13}, i_{14}] }" == str(a4_sep.coefficients[0][0])
    assert "{ A | A_{i_{13}, i_{14}} = f_5 * f_10[i_{13}, i_{14}] * f_6 }" == str(a4_sep.coefficients[1][0])
    assert "f_9" == str(a4_sep.coefficients[2][0])
    assert "f_7" == str(a4_sep.coefficients[3][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._placeholders[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._placeholders[3][0]))

    assert "f_58" == str(a4_sep._placeholders[0][0])
    assert "f_59" == str(a4_sep._placeholders[1][0])
    assert "f_60" == str(a4_sep._placeholders[2][0])
    assert "f_61" == str(a4_sep._placeholders[3][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._form_with_placeholders[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._form_with_placeholders[3].integrals()[0].integrand()))

    assert ("sum_{i_{17}} ({ A | A_{i_{15}} = sum_{i_{16}} f_58[i_{15}, i_{16}] * (grad(v_1))[i_{16}]  })[i_{17}]"
            + " * (grad(v_0))[i_{17}] "
            == str(a4_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("sum_{i_{17}} ({ A | A_{i_{15}} = sum_{i_{16}} f_59[i_{15}, i_{16}] * (grad(v_1))[i_{16}]  })[i_{17}]"
            + " * (grad(v_0))[i_{17}] "
            == str(a4_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert ("v_0 * (sum_{i_{18}} f_60[i_{18}] * (grad(v_1))[i_{18}] )"
            == str(a4_sep._form_with_placeholders[2].integrals()[0].integrand()))
    assert "f_61 * v_0 * v_1" == str(a4_sep._form_with_placeholders[3].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep._form_unchanged)))

    assert 0 == len(a4_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="5", depends=["4"])
def test_separated_parametrized_forms_scalar_5():
    a5 = expr3 * expr2 * inner(grad(u), grad(v)) * ds + expr2 * u.dx(0) * v * ds + expr3 * u * v * ds
    a5_sep = SeparatedParametrizedForm(a5)
    test_logger.log(DEBUG, "*** ###              FORM 5             ### ***")
    test_logger.log(DEBUG, "We change the integration domain to be the boundary. The result is the same as form 1")
    a5_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep.coefficients)))

    assert 3 == len(a5_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep.coefficients[2])))

    assert 2 == len(a5_sep.coefficients[0])
    assert 1 == len(a5_sep.coefficients[1])
    assert 1 == len(a5_sep.coefficients[2])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t(" + str(a5_sep.coefficients[0][0]) + ", " + str(a5_sep.coefficients[0][1]) + ")")
    test_logger.log(DEBUG, "\t\t" + str(a5_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep.coefficients[2][0]))

    assert "(f_7, f_6)" == "(" + str(a5_sep.coefficients[0][0]) + ", " + str(a5_sep.coefficients[0][1]) + ")"
    assert "f_6" == str(a5_sep.coefficients[1][0])
    assert "f_7" == str(a5_sep.coefficients[2][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t(" + str(a5_sep._placeholders[0][0]) + ", " + str(a5_sep._placeholders[0][1]) + ")")
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._placeholders[2][0]))

    assert "(f_62, f_63)" == "(" + str(a5_sep._placeholders[0][0]) + ", " + str(a5_sep._placeholders[0][1]) + ")"
    assert "f_64" == str(a5_sep._placeholders[1][0])
    assert "f_65" == str(a5_sep._placeholders[2][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._form_with_placeholders[2].integrals()[0].integrand()))

    assert ("f_63 * f_62 * (sum_{i_{19}} (grad(v_0))[i_{19}] * (grad(v_1))[i_{19}] )"
            == str(a5_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert "f_64 * (grad(v_1))[0] * v_0" == str(a5_sep._form_with_placeholders[1].integrals()[0].integrand())
    assert "f_65 * v_0 * v_1" == str(a5_sep._form_with_placeholders[2].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep._form_unchanged)))

    assert 0 == len(a5_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="6", depends=["5"])
def test_separated_parametrized_forms_scalar_6():
    h = CellDiameter(mesh)
    a6 = expr3 * h * u * v * dx
    a6_sep = SeparatedParametrizedForm(a6, strict=True)
    test_logger.log(DEBUG, "*** ###              FORM 6             ### ***")
    test_logger.log(DEBUG, "We add a term depending on the mesh size. The extracted coefficient does not retain"
                    + " the mesh size factor")
    a6_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a6_sep.coefficients)))

    assert 1 == len(a6_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a6_sep.coefficients[0])))

    assert 1 == len(a6_sep.coefficients[0])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a6_sep.coefficients[0][0]))

    assert "f_7" == str(a6_sep.coefficients[0][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a6_sep._placeholders[0][0]))

    assert "f_66" == str(a6_sep._placeholders[0][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a6_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert "v_0 * v_1 * diameter * f_66" == str(a6_sep._form_with_placeholders[0].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a6_sep._form_unchanged)))

    assert 0 == len(a6_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="7", depends=["6"])
def test_separated_parametrized_forms_scalar_7():
    a7 = expr9 * expr8 * inner(grad(u), grad(v)) * dx + expr8 * u.dx(0) * v * dx + expr9 * u * v * dx
    a7_sep = SeparatedParametrizedForm(a7)
    test_logger.log(DEBUG, "*** ###              FORM 7             ### ***")
    test_logger.log(DEBUG, "We change the coefficients to be non-parametrized. No (parametrized) coefficients"
                    + " are extracted this time.")
    a7_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a7_sep.coefficients)))

    assert 0 == len(a7_sep.coefficients)
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a7_sep._form_unchanged)))

    assert 3 == len(a7_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a7_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a7_sep._form_unchanged[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a7_sep._form_unchanged[2].integrals()[0].integrand()))

    assert ("f_12 * f_13 * (sum_{i_{20}} (grad(v_0))[i_{20}] * (grad(v_1))[i_{20}] )"
            == str(a7_sep._form_unchanged[0].integrals()[0].integrand()))
    assert "f_12 * (grad(v_1))[0] * v_0" == str(a7_sep._form_unchanged[1].integrals()[0].integrand())
    assert "f_13 * v_0 * v_1" == str(a7_sep._form_unchanged[2].integrals()[0].integrand())


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="8", depends=["7"])
def test_separated_parametrized_forms_scalar_8():
    a8 = expr2 * expr9 * inner(grad(u), grad(v)) * dx + expr8 * u.dx(0) * v * dx + expr9 * u * v * dx
    a8_sep = SeparatedParametrizedForm(a8)
    test_logger.log(DEBUG, "*** ###              FORM 8             ### ***")
    test_logger.log(DEBUG, "A part of the diffusion coefficient is parametrized (advection-reaction are"
                    + " not parametrized). Only the parametrized part is extracted.")
    a8_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a8_sep.coefficients)))

    assert 1 == len(a8_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a8_sep.coefficients[0])))

    assert 1 == len(a8_sep.coefficients[0])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a8_sep.coefficients[0][0]))

    assert "f_6" == str(a8_sep.coefficients[0][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a8_sep._placeholders[0][0]))

    assert "f_67" == str(a8_sep._placeholders[0][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a8_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert ("f_67 * f_13 * (sum_{i_{21}} (grad(v_0))[i_{21}] * (grad(v_1))[i_{21}] )"
            == str(a8_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a8_sep._form_unchanged)))

    assert 2 == len(a8_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a8_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a8_sep._form_unchanged[1].integrals()[0].integrand()))

    assert "v_0 * (grad(v_1))[0] * f_12" == str(a8_sep._form_unchanged[0].integrals()[0].integrand())
    assert "f_13 * v_0 * v_1" == str(a8_sep._form_unchanged[1].integrals()[0].integrand())


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="9", depends=["8"])
def test_separated_parametrized_forms_scalar_9():
    a9 = expr2 * expr9 / expr1 * inner(grad(u), grad(v)) * dx + expr8 * u.dx(0) * v * dx + expr9 * u * v * dx
    a9_sep = SeparatedParametrizedForm(a9)
    test_logger.log(DEBUG, "*** ###              FORM 9             ### ***")
    test_logger.log(DEBUG, "A part of the diffusion coefficient is parametrized (advection-reaction are"
                    + " not parametrized), where the terms are written in a different way when compared to form 8."
                    + " Due to the UFL tree this would entail to extract two (\"sub\") coefficients, but this is"
                    + " not done for the sake of efficiency")
    a9_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a9_sep.coefficients)))

    assert 1 == len(a9_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a9_sep.coefficients[0])))

    assert 1 == len(a9_sep.coefficients[0])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a9_sep.coefficients[0][0]))

    assert "f_6 * f_13 * 1.0 / f_5" == str(a9_sep.coefficients[0][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a9_sep._placeholders[0][0]))

    assert "f_68" == str(a9_sep._placeholders[0][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a9_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert ("f_68 * (sum_{i_{22}} (grad(v_0))[i_{22}] * (grad(v_1))[i_{22}] )"
            == str(a9_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a9_sep._form_unchanged)))

    assert 2 == len(a9_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a9_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a9_sep._form_unchanged[1].integrals()[0].integrand()))

    assert "v_0 * (grad(v_1))[0] * f_12" == str(a9_sep._form_unchanged[0].integrals()[0].integrand())
    assert "f_13 * v_0 * v_1" == str(a9_sep._form_unchanged[1].integrals()[0].integrand())


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="10", depends=["9"])
def test_separated_parametrized_forms_scalar_10():
    h = CellDiameter(mesh)
    a10 = expr9 * h * u * v * dx
    a10_sep = SeparatedParametrizedForm(a10)
    test_logger.log(DEBUG, "*** ###              FORM 10             ### ***")
    test_logger.log(DEBUG, "Similarly to form 6, we add a term depending on the mesh size multiplied by a"
                    + " non-parametrized coefficient. Neither are retained")
    a10_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a10_sep.coefficients)))

    assert 0 == len(a10_sep.coefficients)
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a10_sep._form_unchanged)))

    assert 1 == len(a10_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a10_sep._form_unchanged[0].integrals()[0].integrand()))

    assert "v_0 * v_1 * diameter * f_13" == str(a10_sep._form_unchanged[0].integrals()[0].integrand())


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="11", depends=["10"])
def test_separated_parametrized_forms_scalar_11():
    h = CellDiameter(mesh)
    a11 = expr9 * expr3 * h * u * v * dx
    a11_sep = SeparatedParametrizedForm(a11)
    test_logger.log(DEBUG, "*** ###              FORM 11             ### ***")
    test_logger.log(DEBUG, "Similarly to form 6, we add a term depending on the mesh size multiplied by"
                    + " the product of parametrized and a non-parametrized coefficients. Neither the"
                    + " non-parametrized coefficient nor the mesh size are retained")
    a11_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a11_sep.coefficients)))

    assert 1 == len(a11_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a11_sep.coefficients[0])))

    assert 1 == len(a11_sep.coefficients[0])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a11_sep.coefficients[0][0]))

    assert "f_7" == str(a11_sep.coefficients[0][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._placeholders[0][0]))

    assert "f_69" == str(a11_sep._placeholders[0][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert "v_0 * v_1 * diameter * f_13 * f_69" == str(a11_sep._form_with_placeholders[0].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a11_sep._form_unchanged)))

    assert 0 == len(a11_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="12", depends=["11"])
def test_separated_parametrized_forms_scalar_12():
    h = CellDiameter(mesh)
    a12 = expr3 * h / expr9 * u * v * dx
    a12_sep = SeparatedParametrizedForm(a12)
    test_logger.log(DEBUG, "*** ###              FORM 12             ### ***")
    test_logger.log(DEBUG, "We change form 11 with a slightly different coefficient. In this case the extraction"
                    + " retains the mesh size")
    a12_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a12_sep.coefficients)))

    assert 1 == len(a12_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a12_sep.coefficients[0])))

    assert 1 == len(a12_sep.coefficients[0])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a12_sep.coefficients[0][0]))

    assert "diameter * f_7" == str(a12_sep.coefficients[0][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a12_sep._placeholders[0][0]))

    assert "f_70" == str(a12_sep._placeholders[0][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a12_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert "v_0 * v_1 * f_70 * 1.0 / f_13" == str(a12_sep._form_with_placeholders[0].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a12_sep._form_unchanged)))

    assert 0 == len(a12_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="13", depends=["12"])
def test_separated_parametrized_forms_scalar_13():
    a13 = (inner(expr11 * expr6 * expr10 * grad(u), grad(v)) * dx + inner(expr10 * expr5, grad(u)) * v * dx
           + expr10 * expr3 * u * v * dx)
    a13_sep = SeparatedParametrizedForm(a13)
    test_logger.log(DEBUG, "*** ###              FORM 13             ### ***")
    test_logger.log(DEBUG, "Constants are factored out.")
    a13_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a13_sep.coefficients)))

    assert 3 == len(a13_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a13_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a13_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a13_sep.coefficients[2])))

    assert 1 == len(a13_sep.coefficients[0])
    assert 1 == len(a13_sep.coefficients[1])
    assert 1 == len(a13_sep.coefficients[2])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a13_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a13_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a13_sep.coefficients[2][0]))

    assert "f_10" == str(a13_sep.coefficients[0][0])
    assert "f_9" == str(a13_sep.coefficients[1][0])
    assert "f_7" == str(a13_sep.coefficients[2][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a13_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a13_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a13_sep._placeholders[2][0]))

    assert "f_71" == str(a13_sep._placeholders[0][0])
    assert "f_72" == str(a13_sep._placeholders[1][0])
    assert "f_73" == str(a13_sep._placeholders[2][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a13_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a13_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a13_sep._form_with_placeholders[2].integrals()[0].integrand()))

    assert ("sum_{i_{31}} ({ A | A_{i_{28}} = sum_{i_{29}} ({ A | A_{i_{26}, i_{27}} = ({ A | A_{i_{23}, i_{24}}"
            + " = sum_{i_{25}} f_15[i_{23}, i_{25}] * f_71[i_{25}, i_{24}]  })[i_{26}, i_{27}] * f_14 })"
            + "[i_{28}, i_{29}] * (grad(v_1))[i_{29}]  })[i_{31}] * (grad(v_0))[i_{31}] "
            == str(a13_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("v_0 * (sum_{i_{32}} ({ A | A_{i_{30}} = f_72[i_{30}] * f_14 })[i_{32}] * (grad(v_1))[i_{32}] )"
            == str(a13_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert "f_14 * f_73 * v_0 * v_1" == str(a13_sep._form_with_placeholders[2].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a13_sep._form_unchanged)))

    assert 0 == len(a13_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="14", depends=["13"])
def test_separated_parametrized_forms_scalar_14():
    a14 = (expr3 * expr12 * inner(grad(u), grad(v)) * dx + expr12 / expr2 * u.dx(0) * v * dx
           + expr3 * expr12 * u * v * dx)
    a14_sep = SeparatedParametrizedForm(a14)
    test_logger.log(DEBUG, "*** ###              FORM 14             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 1, but each term is multiplied by a scalar Function,"
                    + " which is the solution of a parametrized problem")
    a14_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a14_sep.coefficients)))

    assert 3 == len(a14_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a14_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a14_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a14_sep.coefficients[2])))

    assert 2 == len(a14_sep.coefficients[0])
    assert 1 == len(a14_sep.coefficients[1])
    assert 2 == len(a14_sep.coefficients[2])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t(" + str(a14_sep.coefficients[0][0]) + ", " + str(a14_sep.coefficients[0][1]) + ")")
    test_logger.log(DEBUG, "\t\t" + str(a14_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t(" + str(a14_sep.coefficients[2][0]) + ", " + str(a14_sep.coefficients[2][1]) + ")")

    assert "(f_20, f_7)" == "(" + str(a14_sep.coefficients[0][0]) + ", " + str(a14_sep.coefficients[0][1]) + ")"
    assert "f_20 * 1.0 / f_6" == str(a14_sep.coefficients[1][0])
    assert "(f_20, f_7)" == "(" + str(a14_sep.coefficients[2][0]) + ", " + str(a14_sep.coefficients[2][1]) + ")"
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t(" + str(a14_sep._placeholders[0][0]) + ", " + str(a14_sep._placeholders[0][1]) + ")")
    test_logger.log(DEBUG, "\t\t" + str(a14_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t(" + str(a14_sep._placeholders[2][0]) + ", " + str(a14_sep._placeholders[2][1]) + ")")

    assert "(f_78, f_79)" == "(" + str(a14_sep._placeholders[0][0]) + ", " + str(a14_sep._placeholders[0][1]) + ")"
    assert "f_80" == str(a14_sep._placeholders[1][0])
    assert "(f_81, f_82)" == "(" + str(a14_sep._placeholders[2][0]) + ", " + str(a14_sep._placeholders[2][1]) + ")"
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a14_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a14_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a14_sep._form_with_placeholders[2].integrals()[0].integrand()))

    assert ("f_79 * f_78 * (sum_{i_{33}} (grad(v_0))[i_{33}] * (grad(v_1))[i_{33}] )"
            == str(a14_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert "v_0 * (grad(v_1))[0] * f_80" == str(a14_sep._form_with_placeholders[1].integrals()[0].integrand())
    assert "f_82 * f_81 * v_0 * v_1" == str(a14_sep._form_with_placeholders[2].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a14_sep._form_unchanged)))

    assert 0 == len(a14_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="15", depends=["14"])
def test_separated_parametrized_forms_scalar_15():
    a15 = inner(expr14 * grad(u), grad(v)) * dx + inner(expr13, grad(u)) * v * dx + expr12 * u * v * dx
    a15_sep = SeparatedParametrizedForm(a15)
    test_logger.log(DEBUG, "*** ###              FORM 15             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 4, but each term is replace by a solution of"
                    + " a parametrized problem, either scalar, vector or tensor shaped")
    a15_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a15_sep.coefficients)))

    assert 3 == len(a15_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a15_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a15_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a15_sep.coefficients[2])))

    assert 1 == len(a15_sep.coefficients[0])
    assert 1 == len(a15_sep.coefficients[1])
    assert 1 == len(a15_sep.coefficients[2])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a15_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a15_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a15_sep.coefficients[2][0]))

    assert "f_26" == str(a15_sep.coefficients[0][0])
    assert "f_23" == str(a15_sep.coefficients[1][0])
    assert "f_20" == str(a15_sep.coefficients[2][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a15_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a15_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a15_sep._placeholders[2][0]))

    assert "f_83" == str(a15_sep._placeholders[0][0])
    assert "f_84" == str(a15_sep._placeholders[1][0])
    assert "f_85" == str(a15_sep._placeholders[2][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a15_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a15_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a15_sep._form_with_placeholders[2].integrals()[0].integrand()))

    assert ("sum_{i_{36}} ({ A | A_{i_{34}} = sum_{i_{35}} f_83[i_{34}, i_{35}] * (grad(v_1))[i_{35}]  })[i_{36}]"
            + " * (grad(v_0))[i_{36}] "
            == str(a15_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("v_0 * (sum_{i_{37}} f_84[i_{37}] * (grad(v_1))[i_{37}] )"
            == str(a15_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert "v_1 * v_0 * f_85" == str(a15_sep._form_with_placeholders[2].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a15_sep._form_unchanged)))

    assert 0 == len(a15_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="16", depends=["15"])
def test_separated_parametrized_forms_scalar_16():
    a16 = inner(expr9 * expr14 * expr8 * grad(u), grad(v)) * dx + inner(expr13, grad(u)) * v * dx + expr12 * u * v * dx
    a16_sep = SeparatedParametrizedForm(a16)
    test_logger.log(DEBUG, "*** ###              FORM 16             ### ***")
    test_logger.log(DEBUG, "We change the coefficients of form 14 to be non-parametrized (as in form 7)."
                    + " Due to the presence of solutions of parametrized problems in contrast to form 7"
                    + " all coefficients are now parametrized.")
    a16_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a16_sep.coefficients)))

    assert 3 == len(a16_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a16_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a16_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a16_sep.coefficients[2])))

    assert 1 == len(a16_sep.coefficients[0])
    assert 1 == len(a16_sep.coefficients[1])
    assert 1 == len(a16_sep.coefficients[2])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a16_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a16_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a16_sep.coefficients[2][0]))

    assert "f_26" == str(a16_sep.coefficients[0][0])
    assert "f_23" == str(a16_sep.coefficients[1][0])
    assert "f_20" == str(a16_sep.coefficients[2][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a16_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a16_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a16_sep._placeholders[2][0]))

    assert "f_86" == str(a16_sep._placeholders[0][0])
    assert "f_87" == str(a16_sep._placeholders[1][0])
    assert "f_88" == str(a16_sep._placeholders[2][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a16_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a16_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a16_sep._form_with_placeholders[2].integrals()[0].integrand()))

    assert ("sum_{i_{44}} ({ A | A_{i_{42}} = sum_{i_{43}} ({ A | A_{i_{40}, i_{41}} = ({ A | A_{i_{38}, i_{39}}"
            + " = f_86[i_{38}, i_{39}] * f_13 })[i_{40}, i_{41}] * f_12 })[i_{42}, i_{43}] * (grad(v_1))[i_{43}]  })"
            + "[i_{44}] * (grad(v_0))[i_{44}] "
            == str(a16_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("v_0 * (sum_{i_{45}} f_87[i_{45}] * (grad(v_1))[i_{45}] )"
            == str(a16_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert "v_1 * v_0 * f_88" == str(a16_sep._form_with_placeholders[2].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a16_sep._form_unchanged)))

    assert 0 == len(a16_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="17", depends=["16"])
def test_separated_parametrized_forms_scalar_17():
    a17 = expr13_split[0] * u * v * dx + expr13_split[1] * u.dx(0) * v * dx
    a17_sep = SeparatedParametrizedForm(a17)
    test_logger.log(DEBUG, "*** ###              FORM 17             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 15, but each term is multiplied to a component of"
                    + " a solution of a parametrized problem")
    a17_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a17_sep.coefficients)))

    assert 2 == len(a17_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a17_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a17_sep.coefficients[1])))

    assert 1 == len(a17_sep.coefficients[0])
    assert 1 == len(a17_sep.coefficients[1])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a17_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep.coefficients[1][0]))

    assert "f_23[0]" == str(a17_sep.coefficients[0][0])
    assert "f_23[1]" == str(a17_sep.coefficients[1][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._placeholders[1][0]))

    assert "f_89" == str(a17_sep._placeholders[0][0])
    assert "f_90" == str(a17_sep._placeholders[1][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._form_with_placeholders[1].integrals()[0].integrand()))

    assert "v_0 * v_1 * f_89" == str(a17_sep._form_with_placeholders[0].integrals()[0].integrand())
    assert "(grad(v_1))[0] * v_0 * f_90" == str(a17_sep._form_with_placeholders[1].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a17_sep._form_unchanged)))

    assert 0 == len(a17_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="18", depends=["17"])
def test_separated_parametrized_forms_scalar_18():
    a18 = (expr3 * expr15 * inner(grad(u), grad(v)) * dx + expr2 * expr15 * u.dx(0) * v * dx
           + expr3 * expr15 * u * v * dx)
    a18_sep = SeparatedParametrizedForm(a18)
    test_logger.log(DEBUG, "*** ###              FORM 18             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 14, but each term is multiplied by a scalar Function,"
                    + " which is not the solution of a parametrized problem")
    a18_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a18_sep.coefficients)))

    assert 3 == len(a18_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a18_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a18_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a18_sep.coefficients[2])))

    assert 1 == len(a18_sep.coefficients[0])
    assert 1 == len(a18_sep.coefficients[1])
    assert 1 == len(a18_sep.coefficients[2])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a18_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a18_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a18_sep.coefficients[2][0]))

    assert "f_7" == str(a18_sep.coefficients[0][0])
    assert "f_6" == str(a18_sep.coefficients[1][0])
    assert "f_7" == str(a18_sep.coefficients[2][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a18_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a18_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a18_sep._placeholders[2][0]))

    assert "f_91" == str(a18_sep._placeholders[0][0])
    assert "f_92" == str(a18_sep._placeholders[1][0])
    assert "f_93" == str(a18_sep._placeholders[2][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a18_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a18_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a18_sep._form_with_placeholders[2].integrals()[0].integrand()))

    assert ("f_91 * f_29 * (sum_{i_{46}} (grad(v_0))[i_{46}] * (grad(v_1))[i_{46}] )"
            == str(a18_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert "f_29 * v_0 * (grad(v_1))[0] * f_92" == str(a18_sep._form_with_placeholders[1].integrals()[0].integrand())
    assert "f_93 * f_29 * v_0 * v_1" == str(a18_sep._form_with_placeholders[2].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a18_sep._form_unchanged)))

    assert 0 == len(a18_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="19", depends=["18"])
def test_separated_parametrized_forms_scalar_19():
    a19 = inner(expr9 * expr17 * expr8 * grad(u), grad(v)) * dx + inner(expr16, grad(u)) * v * dx + expr15 * u * v * dx
    a19_sep = SeparatedParametrizedForm(a19)
    test_logger.log(DEBUG, "*** ###              FORM 19             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 16, but each term is multiplied by a scalar Function,"
                    + " which is not the solution of a parametrized problem. As in form 7, no parametrized"
                    + " coefficients are detected")
    a19_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a19_sep.coefficients)))

    assert 0 == len(a19_sep.coefficients)
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a19_sep._form_unchanged)))

    assert 3 == len(a19_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a19_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a19_sep._form_unchanged[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a19_sep._form_unchanged[2].integrals()[0].integrand()))

    assert ("sum_{i_{53}} ({ A | A_{i_{51}} = sum_{i_{52}} ({ A | A_{i_{49}, i_{50}} = ({ A | A_{i_{47}, i_{48}}"
            + " = f_35[i_{47}, i_{48}] * f_13 })[i_{49}, i_{50}] * f_12 })[i_{51}, i_{52}] * (grad(v_1))[i_{52}]  })"
            + "[i_{53}] * (grad(v_0))[i_{53}] "
            == str(a19_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("v_0 * (sum_{i_{54}} f_32[i_{54}] * (grad(v_1))[i_{54}] )"
            == str(a19_sep._form_unchanged[1].integrals()[0].integrand()))
    assert "v_1 * v_0 * f_29" == str(a19_sep._form_unchanged[2].integrals()[0].integrand())


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="20", depends=["19"])
def test_separated_parametrized_forms_scalar_20():
    a20 = expr16_split[0] * u * v * dx + expr16_split[1] * u.dx(0) * v * dx
    a20_sep = SeparatedParametrizedForm(a20)
    test_logger.log(DEBUG, "*** ###              FORM 20             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 17, but each term is multiplied to a component of"
                    + " a Function which is not the solution of a parametrized problem. This results in"
                    + " no parametrized coefficients")
    a20_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a20_sep.coefficients)))

    assert 0 == len(a20_sep.coefficients)
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a20_sep._form_unchanged)))

    assert 2 == len(a20_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a20_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a20_sep._form_unchanged[1].integrals()[0].integrand()))

    assert "v_0 * f_32[0] * v_1" == str(a20_sep._form_unchanged[0].integrals()[0].integrand())
    assert "(grad(v_1))[0] * f_32[1] * v_0" == str(a20_sep._form_unchanged[1].integrals()[0].integrand())


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="21", depends=["20"])
def test_separated_parametrized_forms_scalar_21():
    a21 = inner(grad(expr13_split[0]), grad(u)) * v * dx + expr13_split[1].dx(0) * u.dx(0) * v * dx
    a21_sep = SeparatedParametrizedForm(a21)
    test_logger.log(DEBUG, "*** ###              FORM 21             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 17, but each term is multiplied to the gradient/"
                    + "partial derivative of a component of a solution of a parametrized problem")
    a21_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a21_sep.coefficients)))

    assert 2 == len(a21_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a21_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a21_sep.coefficients[1])))

    assert 1 == len(a21_sep.coefficients[0])
    assert 1 == len(a21_sep.coefficients[1])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a21_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a21_sep.coefficients[1][0]))

    assert "grad(f_23)" == str(a21_sep.coefficients[0][0])
    assert "(grad(f_23))[1, 0]" == str(a21_sep.coefficients[1][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a21_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a21_sep._placeholders[1][0]))

    assert "f_94" == str(a21_sep._placeholders[0][0])
    assert "f_95" == str(a21_sep._placeholders[1][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a21_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a21_sep._form_with_placeholders[1].integrals()[0].integrand()))

    assert ("v_0 * (sum_{i_{55}} f_94[0, i_{55}] * (grad(v_1))[i_{55}] )"
            == str(a21_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert "v_0 * (grad(v_1))[0] * f_95" == str(a21_sep._form_with_placeholders[1].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a21_sep._form_unchanged)))

    assert 0 == len(a21_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="22", depends=["21"])
def test_separated_parametrized_forms_scalar_22():
    a22 = inner(grad(expr16_split[0]), grad(u)) * v * dx + expr16_split[1].dx(0) * u.dx(0) * v * dx
    a22_sep = SeparatedParametrizedForm(a22)
    test_logger.log(DEBUG, "*** ###              FORM 22             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 21, but each term is multiplied to the gradient/"
                    + "partial derivative of a component of a Function which is not the solution of"
                    + " a parametrized problem. This results in no parametrized coefficients")
    a22_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a22_sep.coefficients)))

    assert 0 == len(a22_sep.coefficients)
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a22_sep._form_unchanged)))

    assert 2 == len(a22_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a22_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a22_sep._form_unchanged[1].integrals()[0].integrand()))

    assert ("v_0 * (sum_{i_{58}} (grad(f_32))[0, i_{58}] * (grad(v_1))[i_{58}] )"
            == str(a22_sep._form_unchanged[0].integrals()[0].integrand()))
    assert "v_0 * (grad(v_1))[0] * (grad(f_32))[1, 0]" == str(a22_sep._form_unchanged[1].integrals()[0].integrand())


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="23", depends=["22"])
def test_separated_parametrized_forms_scalar_23():
    a23 = expr3 / expr2 * u * v * dx
    a23_sep = SeparatedParametrizedForm(a23)
    test_logger.log(DEBUG, "*** ###              FORM 23             ### ***")
    test_logger.log(DEBUG, "This form tests a division between two expressions, which can be collected as"
                    + " one coefficient")
    a23_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a23_sep.coefficients)))

    assert 1 == len(a23_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a23_sep.coefficients[0])))

    assert 1 == len(a23_sep.coefficients[0])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a23_sep.coefficients[0][0]))

    assert "f_7 * 1.0 / f_6" == str(a23_sep.coefficients[0][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a23_sep._placeholders[0][0]))

    assert "f_96" == str(a23_sep._placeholders[0][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a23_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert "v_0 * v_1 * f_96" == str(a23_sep._form_with_placeholders[0].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a23_sep._form_unchanged)))

    assert 0 == len(a23_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="24", depends=["23"])
def test_separated_parametrized_forms_scalar_24():
    a24 = expr3 * u * v / expr2 * dx
    a24_sep = SeparatedParametrizedForm(a24)
    test_logger.log(DEBUG, "*** ###              FORM 24             ### ***")
    test_logger.log(DEBUG, "This form tests a division between two expressions, which (in contrast to form 24)"
                    + " cannot be collected as one coefficient")
    a24_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a24_sep.coefficients)))

    assert 1 == len(a24_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a24_sep.coefficients[0])))

    assert 2 == len(a24_sep.coefficients[0])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t(" + str(a24_sep.coefficients[0][0]) + ", " + str(a24_sep.coefficients[0][1]) + ")")

    assert "(1.0 / f_6, f_7)" == "(" + str(a24_sep.coefficients[0][0]) + ", " + str(a24_sep.coefficients[0][1]) + ")"
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t(" + str(a24_sep._placeholders[0][0]) + ", " + str(a24_sep._placeholders[0][1]) + ")")

    assert "(f_97, f_98)" == "(" + str(a24_sep._placeholders[0][0]) + ", " + str(a24_sep._placeholders[0][1]) + ")"
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a24_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert "f_97 * v_0 * v_1 * f_98" == str(a24_sep._form_with_placeholders[0].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a24_sep._form_unchanged)))

    assert 0 == len(a24_sep._form_unchanged)
