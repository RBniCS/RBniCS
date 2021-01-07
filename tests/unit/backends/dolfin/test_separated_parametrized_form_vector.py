# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from logging import DEBUG, getLogger
from mpi4py import MPI
from dolfin import (CellDiameter, Constant, det, dx, Expression, Function, FunctionSpace, grad, inner,
                    TensorFunctionSpace, TestFunction, TrialFunction, UnitSquareMesh, VectorFunctionSpace)
from rbnics.backends.dolfin import SeparatedParametrizedForm
from rbnics.backends.dolfin.separated_parametrized_form import logger as separated_parametrized_form_logger
from rbnics.utils.decorators.store_map_from_solution_to_problem import _solution_to_problem_map
from rbnics.utils.test import enable_logging

# Logger
test_logger = getLogger("tests/unit/test_separated_parametrized_forms_vector.py")
enable_separated_parametrized_form_logging = enable_logging({
    separated_parametrized_form_logger: DEBUG, test_logger: DEBUG})

# Common variables
mesh = UnitSquareMesh(10, 10)

V = VectorFunctionSpace(mesh, "Lagrange", 2)

expr1 = Expression("x[0]", mu_0=0., degree=1, cell=mesh.ufl_cell())  # f_5
expr2 = Expression(("x[0]", "x[1]"), mu_0=0., degree=1, cell=mesh.ufl_cell())  # f_6
expr3 = Expression((("1*x[0]", "2*x[1]"), ("3*x[0]", "4*x[1]")), mu_0=0., degree=1, cell=mesh.ufl_cell())  # f_7
expr4 = Expression((("4*x[0]", "3*x[1]"), ("2*x[0]", "1*x[1]")), mu_0=0., degree=1, cell=mesh.ufl_cell())  # f_8
expr5 = Expression("x[0]", degree=1, cell=mesh.ufl_cell())  # f_9
expr6 = Expression(("x[0]", "x[1]"), degree=1, cell=mesh.ufl_cell())  # f_10
expr7 = Expression((("1*x[0]", "2*x[1]"), ("3*x[0]", "4*x[1]")), degree=1, cell=mesh.ufl_cell())  # f_11
expr8 = Expression((("4*x[0]", "3*x[1]"), ("2*x[0]", "1*x[1]")), degree=1, cell=mesh.ufl_cell())  # f_12
expr9 = Constant(((1, 2), (3, 4)))  # f_13

scalar_V = FunctionSpace(mesh, "Lagrange", 3)
tensor_V = TensorFunctionSpace(mesh, "Lagrange", 1)
expr10 = Function(scalar_V)  # f_18
expr11 = Function(V)  # f_21
expr12 = Function(tensor_V)  # f_24
expr13 = Function(scalar_V)  # f_27
expr14 = Function(V)  # f_30
expr15 = Function(tensor_V)  # f_33


class Problem(object):
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


_solution_to_problem_map[expr10] = Problem("problem10")
_solution_to_problem_map[expr11] = Problem("problem11")
_solution_to_problem_map[expr12] = Problem("problem12")

u = TrialFunction(V)
v = TestFunction(V)


# Fixtures
skip_in_parallel = pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Numbering of functions changes in parallel.")


# Tests
@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="1")
def test_separated_parametrized_forms_vector_1():
    a1 = inner(expr3 * grad(u), grad(v)) * dx + inner(grad(u) * expr2, v) * dx + expr1 * inner(u, v) * dx
    a1_sep = SeparatedParametrizedForm(a1)
    test_logger.log(DEBUG, "*** ###              FORM 1             ### ***")
    test_logger.log(DEBUG, "This is a basic vector advection-diffusion-reaction parametrized form,"
                    + " with all parametrized coefficients")
    a1_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep.coefficients)))

    assert 3 == len(a1_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep.coefficients[2])))

    assert 1 == len(a1_sep.coefficients[0])
    assert 1 == len(a1_sep.coefficients[1])
    assert 1 == len(a1_sep.coefficients[2])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a1_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep.coefficients[2][0]))

    assert "f_7" == str(a1_sep.coefficients[0][0])
    assert "f_6" == str(a1_sep.coefficients[1][0])
    assert "f_5" == str(a1_sep.coefficients[2][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._placeholders[2][0]))

    assert "f_36" == str(a1_sep._placeholders[0][0])
    assert "f_37" == str(a1_sep._placeholders[1][0])
    assert "f_38" == str(a1_sep._placeholders[2][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._form_with_placeholders[2].integrals()[0].integrand()))

    assert ("sum_{i_{14}} sum_{i_{13}} ({ A | A_{i_8, i_9} = sum_{i_{10}} f_36[i_8, i_{10}] * (grad(v_1))"
            + "[i_{10}, i_9]  })[i_{13}, i_{14}] * (grad(v_0))[i_{13}, i_{14}]  "
            == str(a1_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("sum_{i_{15}} ({ A | A_{i_{11}} = sum_{i_{12}} f_37[i_{12}] * (grad(v_1))[i_{11}, i_{12}]  })"
            + "[i_{15}] * v_0[i_{15}] "
            == str(a1_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert ("f_38 * (sum_{i_{16}} v_0[i_{16}] * v_1[i_{16}] )"
            == str(a1_sep._form_with_placeholders[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep._form_unchanged)))

    assert 0 == len(a1_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="2", depends=["1"])
def test_separated_parametrized_forms_vector_2():
    a2 = inner(expr3 * expr4 * grad(u), grad(v)) * dx + inner(grad(u) * expr2, v) * dx + expr1 * inner(u, v) * dx
    a2_sep = SeparatedParametrizedForm(a2)
    test_logger.log(DEBUG, "*** ###              FORM 2             ### ***")
    test_logger.log(DEBUG, "In this case the diffusivity tensor is given by the product of two expressions")
    a2_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep.coefficients)))

    assert 3 == len(a2_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep.coefficients[2])))

    assert 1 == len(a2_sep.coefficients[0])
    assert 1 == len(a2_sep.coefficients[1])
    assert 1 == len(a2_sep.coefficients[2])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a2_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep.coefficients[2][0]))

    assert ("{ A | A_{i_{17}, i_{18}} = sum_{i_{19}} f_7[i_{17}, i_{19}] * f_8[i_{19}, i_{18}]  }"
            == str(a2_sep.coefficients[0][0]))
    assert "f_6" == str(a2_sep.coefficients[1][0])
    assert "f_5" == str(a2_sep.coefficients[2][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._placeholders[2][0]))

    assert "f_39" == str(a2_sep._placeholders[0][0])
    assert "f_40" == str(a2_sep._placeholders[1][0])
    assert "f_41" == str(a2_sep._placeholders[2][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._form_with_placeholders[2].integrals()[0].integrand()))

    assert ("sum_{i_{26}} sum_{i_{25}} ({ A | A_{i_{20}, i_{21}} = sum_{i_{22}} f_39[i_{20}, i_{22}] * (grad(v_1))"
            + "[i_{22}, i_{21}]  })[i_{25}, i_{26}] * (grad(v_0))[i_{25}, i_{26}]  "
            == str(a2_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("sum_{i_{27}} ({ A | A_{i_{23}} = sum_{i_{24}} f_40[i_{24}] * (grad(v_1))[i_{23}, i_{24}]  })"
            + "[i_{27}] * v_0[i_{27}] "
            == str(a2_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert ("f_41 * (sum_{i_{28}} v_0[i_{28}] * v_1[i_{28}] )"
            == str(a2_sep._form_with_placeholders[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep._form_unchanged)))

    assert 0 == len(a2_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="3", depends=["2"])
def test_separated_parametrized_forms_vector_3():
    a3 = (inner(det(expr3) * (expr4 + expr3 * expr3) * expr1, grad(v)) * dx + inner(grad(u) * expr2, v) * dx
          + expr1 * inner(u, v) * dx)
    a3_sep = SeparatedParametrizedForm(a3)
    test_logger.log(DEBUG, "*** ###              FORM 3             ### ***")
    test_logger.log(DEBUG, "We try now with a more complex expression of for each coefficient")
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

    assert 1 == len(a3_sep.coefficients[0])
    assert 1 == len(a3_sep.coefficients[1])
    assert 1 == len(a3_sep.coefficients[2])
    assert 1 == len(a3_sep.coefficients[3])
    assert 1 == len(a3_sep.coefficients[4])
    assert 1 == len(a3_sep.coefficients[5])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[3][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[4][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[5][0]))

    assert ("{ A | A_{i_{34}, i_{35}} = ({ A | A_{i_{32}, i_{33}} = ({ A | A_{i_{29}, i_{30}} = sum_{i_{31}}"
            + " f_7[i_{31}, i_{30}] * f_7[i_{29}, i_{31}]  })[i_{32}, i_{33}] * f_7[0, 0] * f_7[1, 1] })"
            + "[i_{34}, i_{35}] * f_5 }"
            == str(a3_sep.coefficients[0][0]))
    assert ("{ A | A_{i_{34}, i_{35}} = ({ A | A_{i_{32}, i_{33}} = -1 * f_7[0, 1] * f_7[1, 0]"
            + " * f_8[i_{32}, i_{33}] })[i_{34}, i_{35}] * f_5 }"
            == str(a3_sep.coefficients[1][0]))
    assert ("{ A | A_{i_{34}, i_{35}} = ({ A | A_{i_{32}, i_{33}} = -1 * f_7[0, 1] * f_7[1, 0] * ({ A | A_{i_{29},"
            + " i_{30}} = sum_{i_{31}} f_7[i_{31}, i_{30}] * f_7[i_{29}, i_{31}]  })[i_{32}, i_{33}] })"
            + "[i_{34}, i_{35}] * f_5 }"
            == str(a3_sep.coefficients[2][0]))
    assert ("{ A | A_{i_{34}, i_{35}} = ({ A | A_{i_{32}, i_{33}} = f_8[i_{32}, i_{33}] * f_7[0, 0] * f_7[1, 1] })"
            + "[i_{34}, i_{35}] * f_5 }"
            == str(a3_sep.coefficients[3][0]))
    assert "f_6" == str(a3_sep.coefficients[4][0])
    assert "f_5" == str(a3_sep.coefficients[5][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[3][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[4][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[5][0]))

    assert "f_42" == str(a3_sep._placeholders[0][0])
    assert "f_43" == str(a3_sep._placeholders[1][0])
    assert "f_44" == str(a3_sep._placeholders[2][0])
    assert "f_45" == str(a3_sep._placeholders[3][0])
    assert "f_46" == str(a3_sep._placeholders[4][0])
    assert "f_47" == str(a3_sep._placeholders[5][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[3].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[4].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[5].integrals()[0].integrand()))

    assert ("sum_{i_{39}} sum_{i_{38}} f_42[i_{38}, i_{39}] * (grad(v_0))[i_{38}, i_{39}]  "
            == str(a3_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("sum_{i_{39}} sum_{i_{38}} f_43[i_{38}, i_{39}] * (grad(v_0))[i_{38}, i_{39}]  "
            == str(a3_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert ("sum_{i_{39}} sum_{i_{38}} f_44[i_{38}, i_{39}] * (grad(v_0))[i_{38}, i_{39}]  "
            == str(a3_sep._form_with_placeholders[2].integrals()[0].integrand()))
    assert ("sum_{i_{39}} sum_{i_{38}} f_45[i_{38}, i_{39}] * (grad(v_0))[i_{38}, i_{39}]  "
            == str(a3_sep._form_with_placeholders[3].integrals()[0].integrand()))
    assert ("sum_{i_{40}} ({ A | A_{i_{36}} = sum_{i_{37}} f_46[i_{37}] * (grad(v_1))[i_{36}, i_{37}]  })[i_{40}]"
            + " * v_0[i_{40}] "
            == str(a3_sep._form_with_placeholders[4].integrals()[0].integrand()))
    assert ("f_47 * (sum_{i_{41}} v_0[i_{41}] * v_1[i_{41}] )"
            == str(a3_sep._form_with_placeholders[5].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep._form_unchanged)))

    assert 0 == len(a3_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="4", depends=["3"])
def test_separated_parametrized_forms_vector_4():
    h = CellDiameter(mesh)
    a4 = inner(expr3 * h * grad(u), grad(v)) * dx + inner(grad(u) * expr2 * h, v) * dx + expr1 * h * inner(u, v) * dx
    a4_sep = SeparatedParametrizedForm(a4)
    test_logger.log(DEBUG, "*** ###              FORM 4             ### ***")
    test_logger.log(DEBUG, "We add a term depending on the mesh size. The extracted coefficients may retain"
                    + " the mesh size factor depending on the UFL tree")
    a4_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep.coefficients)))

    assert 3 == len(a4_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep.coefficients[2])))

    assert 1 == len(a4_sep.coefficients[0])
    assert 1 == len(a4_sep.coefficients[1])
    assert 1 == len(a4_sep.coefficients[2])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a4_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep.coefficients[2][0]))

    assert "{ A | A_{i_{42}, i_{43}} = diameter * f_7[i_{42}, i_{43}] }" == str(a4_sep.coefficients[0][0])
    assert "f_6" == str(a4_sep.coefficients[1][0])
    assert "f_5" == str(a4_sep.coefficients[2][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._placeholders[2][0]))

    assert "f_48" == str(a4_sep._placeholders[0][0])
    assert "f_49" == str(a4_sep._placeholders[1][0])
    assert "f_50" == str(a4_sep._placeholders[2][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._form_with_placeholders[2].integrals()[0].integrand()))

    assert ("sum_{i_{51}} sum_{i_{50}} ({ A | A_{i_{44}, i_{45}} = sum_{i_{46}} f_48[i_{44}, i_{46}] * (grad(v_1))"
            + "[i_{46}, i_{45}]  })[i_{50}, i_{51}] * (grad(v_0))[i_{50}, i_{51}]  "
            == str(a4_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("sum_{i_{52}} ({ A | A_{i_{49}} = diameter * ({ A | A_{i_{47}} = sum_{i_{48}} f_49[i_{48}] * (grad(v_1))"
            + "[i_{47}, i_{48}]  })[i_{49}] })[i_{52}] * v_0[i_{52}] "
            == str(a4_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert ("diameter * f_50 * (sum_{i_{53}} v_0[i_{53}] * v_1[i_{53}] )"
            == str(a4_sep._form_with_placeholders[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep._form_unchanged)))

    assert 0 == len(a4_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="5", depends=["4"])
def test_separated_parametrized_forms_vector_5():
    h = CellDiameter(mesh)
    a5 = (inner((expr3 * h) * grad(u), grad(v)) * dx + inner(grad(u) * (expr2 * h), v) * dx
          + (expr1 * h) * inner(u, v) * dx)
    a5_sep = SeparatedParametrizedForm(a5)
    test_logger.log(DEBUG, "*** ###              FORM 5             ### ***")
    test_logger.log(DEBUG, "Starting from form 4, use parenthesis to make sure that the extracted coefficients"
                    + " retain the mesh size factor")
    a5_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep.coefficients)))

    assert 3 == len(a5_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep.coefficients[2])))

    assert 1 == len(a5_sep.coefficients[0])
    assert 1 == len(a5_sep.coefficients[1])
    assert 1 == len(a5_sep.coefficients[2])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a5_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep.coefficients[2][0]))

    assert "{ A | A_{i_{54}, i_{55}} = diameter * f_7[i_{54}, i_{55}] }" == str(a5_sep.coefficients[0][0])
    assert "{ A | A_{i_{59}} = diameter * f_6[i_{59}] }" == str(a5_sep.coefficients[1][0])
    assert "f_5" == str(a5_sep.coefficients[2][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._placeholders[2][0]))

    assert "f_51" == str(a5_sep._placeholders[0][0])
    assert "f_52" == str(a5_sep._placeholders[1][0])
    assert "f_53" == str(a5_sep._placeholders[2][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._form_with_placeholders[2].integrals()[0].integrand()))

    assert ("sum_{i_{63}} sum_{i_{62}} ({ A | A_{i_{56}, i_{57}} = sum_{i_{58}} f_51[i_{56}, i_{58}] * (grad(v_1))"
            + "[i_{58}, i_{57}]  })[i_{62}, i_{63}] * (grad(v_0))[i_{62}, i_{63}]  "
            == str(a5_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("sum_{i_{64}} ({ A | A_{i_{60}} = sum_{i_{61}} f_52[i_{61}] * (grad(v_1))[i_{60}, i_{61}]  })[i_{64}]"
            + " * v_0[i_{64}] "
            == str(a5_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert ("diameter * f_53 * (sum_{i_{65}} v_0[i_{65}] * v_1[i_{65}] )"
            == str(a5_sep._form_with_placeholders[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep._form_unchanged)))

    assert 0 == len(a5_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="6", depends=["5"])
def test_separated_parametrized_forms_vector_6():
    a6 = inner(expr7 * grad(u), grad(v)) * dx + inner(grad(u) * expr6, v) * dx + expr5 * inner(u, v) * dx
    a6_sep = SeparatedParametrizedForm(a6)
    test_logger.log(DEBUG, "*** ###              FORM 6             ### ***")
    test_logger.log(DEBUG, "We change the coefficients to be non-parametrized. No (parametrized) coefficients"
                    + " are extracted this time")
    a6_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a6_sep.coefficients)))

    assert 0 == len(a6_sep.coefficients)
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a6_sep._form_unchanged)))

    assert 3 == len(a6_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a6_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a6_sep._form_unchanged[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a6_sep._form_unchanged[2].integrals()[0].integrand()))

    assert ("sum_{i_{72}} sum_{i_{71}} ({ A | A_{i_{66}, i_{67}} = sum_{i_{68}} f_11[i_{66}, i_{68}] * (grad(v_1))"
            + "[i_{68}, i_{67}]  })[i_{71}, i_{72}] * (grad(v_0))[i_{71}, i_{72}]  "
            == str(a6_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("sum_{i_{73}} ({ A | A_{i_{69}} = sum_{i_{70}} f_10[i_{70}] * (grad(v_1))[i_{69}, i_{70}]  })[i_{73}]"
            + " * v_0[i_{73}] "
            == str(a6_sep._form_unchanged[1].integrals()[0].integrand()))
    assert ("f_9 * (sum_{i_{74}} v_0[i_{74}] * v_1[i_{74}] )"
            == str(a6_sep._form_unchanged[2].integrals()[0].integrand()))


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="7", depends=["6"])
def test_separated_parametrized_forms_vector_7():
    a7 = (inner(expr7 * (expr3 * expr4) * grad(u), grad(v)) * dx + inner(grad(u) * expr6, v) * dx
          + expr5 * inner(u, v) * dx)
    a7_sep = SeparatedParametrizedForm(a7)
    test_logger.log(DEBUG, "*** ###              FORM 7             ### ***")
    test_logger.log(DEBUG, "A part of the diffusion coefficient is parametrized (advection-reaction are"
                    + " not parametrized). Only the parametrized part is extracted.")
    a7_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a7_sep.coefficients)))

    assert 1 == len(a7_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a7_sep.coefficients[0])))

    assert 1 == len(a7_sep.coefficients[0])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a7_sep.coefficients[0][0]))

    assert ("{ A | A_{i_{75}, i_{76}} = sum_{i_{77}} f_7[i_{75}, i_{77}] * f_8[i_{77}, i_{76}]  }"
            == str(a7_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a7_sep._placeholders[0][0]))

    assert "f_54" == str(a7_sep._placeholders[0][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a7_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert ("sum_{i_{87}} sum_{i_{86}} ({ A | A_{i_{81}, i_{82}} = sum_{i_{83}} ({ A | A_{i_{78}, i_{79}}"
            + " = sum_{i_{80}} f_11[i_{78}, i_{80}] * f_54[i_{80}, i_{79}]  })[i_{81}, i_{83}] * (grad(v_1))"
            + "[i_{83}, i_{82}]  })[i_{86}, i_{87}] * (grad(v_0))[i_{86}, i_{87}]  "
            == str(a7_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a7_sep._form_unchanged)))

    assert 2 == len(a7_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a7_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a7_sep._form_unchanged[1].integrals()[0].integrand()))

    assert ("sum_{i_{88}} ({ A | A_{i_{84}} = sum_{i_{85}} f_10[i_{85}] * (grad(v_1))[i_{84}, i_{85}]  })[i_{88}]"
            + " * v_0[i_{88}] "
            == str(a7_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("f_9 * (sum_{i_{89}} v_0[i_{89}] * v_1[i_{89}] )"
            == str(a7_sep._form_unchanged[1].integrals()[0].integrand()))


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="8", depends=["7"])
def test_separated_parametrized_forms_vector_8():
    a8 = (inner(expr3 * expr7 * expr4 * grad(u), grad(v)) * dx + inner(grad(u) * expr6, v) * dx
          + expr5 * inner(u, v) * dx)
    a8_sep = SeparatedParametrizedForm(a8)
    test_logger.log(DEBUG, "*** ###              FORM 8             ### ***")
    test_logger.log(DEBUG, "This case is similar to form 7, but the order of the matrix multiplication is different."
                    + " In order not to extract separately f_7 and f_8, the whole product (even with the"
                    + " non-parametrized part) is extracted.")
    a8_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a8_sep.coefficients)))

    assert 1 == len(a8_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a8_sep.coefficients[0])))

    assert 1 == len(a8_sep.coefficients[0])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a8_sep.coefficients[0][0]))

    assert ("{ A | A_{i_{93}, i_{94}} = sum_{i_{95}} ({ A | A_{i_{90}, i_{91}} = sum_{i_{92}} f_7[i_{90}, i_{92}]"
            + " * f_11[i_{92}, i_{91}]  })[i_{93}, i_{95}] * f_8[i_{95}, i_{94}]  }"
            == str(a8_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a8_sep._placeholders[0][0]))

    assert "f_55" == str(a8_sep._placeholders[0][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a8_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert ("sum_{i_{102}} sum_{i_{101}} ({ A | A_{i_{96}, i_{97}} = sum_{i_{98}} f_55[i_{96}, i_{98}] * (grad(v_1))"
            + "[i_{98}, i_{97}]  })[i_{101}, i_{102}] * (grad(v_0))[i_{101}, i_{102}]  "
            == str(a8_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a8_sep._form_unchanged)))

    assert 2 == len(a8_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a8_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a8_sep._form_unchanged[1].integrals()[0].integrand()))

    assert ("sum_{i_{103}} ({ A | A_{i_{99}} = sum_{i_{100}} f_10[i_{100}] * (grad(v_1))[i_{99}, i_{100}]  })"
            + "[i_{103}] * v_0[i_{103}] "
            == str(a8_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("f_9 * (sum_{i_{104}} v_0[i_{104}] * v_1[i_{104}] )"
            == str(a8_sep._form_unchanged[1].integrals()[0].integrand()))


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="9", depends=["8"])
def test_separated_parametrized_forms_vector_9():
    a9 = (inner(expr9 * (expr3 * expr4) * grad(u), grad(v)) * dx + inner(grad(u) * expr6, v) * dx
          + expr5 * inner(u, v) * dx)
    a9_sep = SeparatedParametrizedForm(a9)
    test_logger.log(DEBUG, "*** ###              FORM 9             ### ***")
    test_logger.log(DEBUG, "This is similar to form 7, showing the trivial constants can be factored out")
    a9_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a9_sep.coefficients)))

    assert 1 == len(a9_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a9_sep.coefficients[0])))

    assert 1 == len(a9_sep.coefficients[0])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a9_sep.coefficients[0][0]))

    assert ("{ A | A_{i_{105}, i_{106}} = sum_{i_{107}} f_7[i_{105}, i_{107}] * f_8[i_{107}, i_{106}]  }"
            == str(a9_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a9_sep._placeholders[0][0]))

    assert "f_56" == str(a9_sep._placeholders[0][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a9_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert ("sum_{i_{117}} sum_{i_{116}} ({ A | A_{i_{111}, i_{112}} = sum_{i_{113}} ({ A | A_{i_{108}, i_{109}}"
            + " = sum_{i_{110}} f_13[i_{108}, i_{110}] * f_56[i_{110}, i_{109}]  })[i_{111}, i_{113}] * (grad(v_1))"
            + "[i_{113}, i_{112}]  })[i_{116}, i_{117}] * (grad(v_0))[i_{116}, i_{117}]  "
            == str(a9_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a9_sep._form_unchanged)))

    assert 2 == len(a9_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a9_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a9_sep._form_unchanged[1].integrals()[0].integrand()))

    assert ("sum_{i_{118}} ({ A | A_{i_{114}} = sum_{i_{115}} f_10[i_{115}] * (grad(v_1))[i_{114}, i_{115}]  })"
            + "[i_{118}] * v_0[i_{118}] "
            == str(a9_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("f_9 * (sum_{i_{119}} v_0[i_{119}] * v_1[i_{119}] )"
            == str(a9_sep._form_unchanged[1].integrals()[0].integrand()))


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="10", depends=["9"])
def test_separated_parametrized_forms_vector_10():
    a10 = (inner(expr3 * expr9 * expr4 * grad(u), grad(v)) * dx + inner(grad(u) * expr6, v) * dx
           + expr5 * inner(u, v) * dx)
    a10_sep = SeparatedParametrizedForm(a10)
    test_logger.log(DEBUG, "*** ###              FORM 10             ### ***")
    test_logger.log(DEBUG, "This is similar to form 8, showing a case where constants cannot be factored out")
    a10_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a10_sep.coefficients)))

    assert 1 == len(a10_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a10_sep.coefficients[0])))

    assert 1 == len(a10_sep.coefficients[0])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a10_sep.coefficients[0][0]))

    assert ("{ A | A_{i_{123}, i_{124}} = sum_{i_{125}} ({ A | A_{i_{120}, i_{121}} = sum_{i_{122}} f_7[i_{120},"
            + " i_{122}] * f_13[i_{122}, i_{121}]  })[i_{123}, i_{125}] * f_8[i_{125}, i_{124}]  }"
            == str(a10_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a10_sep._placeholders[0][0]))

    assert "f_57" == str(a10_sep._placeholders[0][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a10_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert ("sum_{i_{132}} sum_{i_{131}} ({ A | A_{i_{126}, i_{127}} = sum_{i_{128}} f_57[i_{126}, i_{128}]"
            + " * (grad(v_1))[i_{128}, i_{127}]  })[i_{131}, i_{132}] * (grad(v_0))[i_{131}, i_{132}]  "
            == str(a10_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a10_sep._form_unchanged)))

    assert 2 == len(a10_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a10_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a10_sep._form_unchanged[1].integrals()[0].integrand()))

    assert ("sum_{i_{133}} ({ A | A_{i_{129}} = sum_{i_{130}} f_10[i_{130}] * (grad(v_1))[i_{129}, i_{130}]  })"
            + "[i_{133}] * v_0[i_{133}] "
            == str(a10_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("f_9 * (sum_{i_{134}} v_0[i_{134}] * v_1[i_{134}] )"
            == str(a10_sep._form_unchanged[1].integrals()[0].integrand()))


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="11", depends=["10"])
def test_separated_parametrized_forms_vector_11():
    a11 = inner(expr12 * grad(u), grad(v)) * dx + inner(grad(u) * expr11, v) * dx + expr10 * inner(u, v) * dx
    a11_sep = SeparatedParametrizedForm(a11)
    test_logger.log(DEBUG, "*** ###              FORM 11             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 1, but each term is multiplied by a Function,"
                    + " which is the solution of a parametrized problem")
    a11_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a11_sep.coefficients)))

    assert 3 == len(a11_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a11_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a11_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a11_sep.coefficients[2])))

    assert 1 == len(a11_sep.coefficients[0])
    assert 1 == len(a11_sep.coefficients[1])
    assert 1 == len(a11_sep.coefficients[2])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a11_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep.coefficients[2][0]))

    assert "f_24" == str(a11_sep.coefficients[0][0])
    assert "f_21" == str(a11_sep.coefficients[1][0])
    assert "f_18" == str(a11_sep.coefficients[2][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._placeholders[2][0]))

    assert "f_62" == str(a11_sep._placeholders[0][0])
    assert "f_63" == str(a11_sep._placeholders[1][0])
    assert "f_64" == str(a11_sep._placeholders[2][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._form_with_placeholders[2].integrals()[0].integrand()))

    assert ("sum_{i_{141}} sum_{i_{140}} ({ A | A_{i_{135}, i_{136}} = sum_{i_{137}} f_62[i_{135}, i_{137}]"
            + " * (grad(v_1))[i_{137}, i_{136}]  })[i_{140}, i_{141}] * (grad(v_0))[i_{140}, i_{141}]  "
            == str(a11_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("sum_{i_{142}} ({ A | A_{i_{138}} = sum_{i_{139}} f_63[i_{139}] * (grad(v_1))[i_{138}, i_{139}]  })"
            + "[i_{142}] * v_0[i_{142}] "
            == str(a11_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert ("f_64 * (sum_{i_{143}} v_0[i_{143}] * v_1[i_{143}] )"
            == str(a11_sep._form_with_placeholders[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a11_sep._form_unchanged)))

    assert 0 == len(a11_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="12", depends=["11"])
def test_separated_parametrized_forms_vector_12():
    a12 = expr11[0] * inner(u, v) * dx
    a12_sep = SeparatedParametrizedForm(a12)
    test_logger.log(DEBUG, "*** ###              FORM 12             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 11, but each term is multiplied by a component of"
                    + " a solution of a parametrized problem, resulting in an Indexed coefficient")
    a12_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a12_sep.coefficients)))

    assert 1 == len(a12_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a12_sep.coefficients[0])))

    assert 1 == len(a12_sep.coefficients[0])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a12_sep.coefficients[0][0]))

    assert "f_21[0]" == str(a12_sep.coefficients[0][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a12_sep._placeholders[0][0]))

    assert "f_65" == str(a12_sep._placeholders[0][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a12_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert ("f_65 * (sum_{i_{144}} v_0[i_{144}] * v_1[i_{144}] )"
            == str(a12_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a12_sep._form_unchanged)))

    assert 0 == len(a12_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="13", depends=["12"])
def test_separated_parametrized_forms_vector_13():
    a13 = inner(expr15 * grad(u), grad(v)) * dx + inner(grad(u) * expr14, v) * dx + expr13 * inner(u, v) * dx
    a13_sep = SeparatedParametrizedForm(a13)
    test_logger.log(DEBUG, "*** ###              FORM 13             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 11, but each term is multiplied by a Function,"
                    + " which is not the solution of a parametrized problem. This results in no parametrized"
                    + " coefficients")
    a13_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a13_sep.coefficients)))

    assert 0 == len(a13_sep.coefficients)
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a13_sep._form_unchanged)))

    assert 3 == len(a13_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a13_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a13_sep._form_unchanged[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a13_sep._form_unchanged[2].integrals()[0].integrand()))

    assert ("sum_{i_{151}} sum_{i_{150}} ({ A | A_{i_{145}, i_{146}} = sum_{i_{147}} f_33[i_{145}, i_{147}]"
            + " * (grad(v_1))[i_{147}, i_{146}]  })[i_{150}, i_{151}] * (grad(v_0))[i_{150}, i_{151}]  "
            == str(a13_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("sum_{i_{152}} ({ A | A_{i_{148}} = sum_{i_{149}} f_30[i_{149}] * (grad(v_1))[i_{148}, i_{149}]  })"
            + "[i_{152}] * v_0[i_{152}] "
            == str(a13_sep._form_unchanged[1].integrals()[0].integrand()))
    assert ("f_27 * (sum_{i_{153}} v_0[i_{153}] * v_1[i_{153}] )"
            == str(a13_sep._form_unchanged[2].integrals()[0].integrand()))


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="14", depends=["13"])
def test_separated_parametrized_forms_vector_14():
    a14 = expr14[0] * inner(u, v) * dx
    a14_sep = SeparatedParametrizedForm(a14)
    test_logger.log(DEBUG, "*** ###              FORM 14             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 12, but each term is multiplied by a component of"
                    + " a Function which is not solution of a parametrized problem. This results in no"
                    + " parametrized coefficients")
    a14_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a14_sep.coefficients)))

    assert 0 == len(a14_sep.coefficients)
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a14_sep._form_unchanged)))

    assert 1 == len(a14_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a14_sep._form_unchanged[0].integrals()[0].integrand()))

    assert ("f_30[0] * (sum_{i_{154}} v_0[i_{154}] * v_1[i_{154}] )"
            == str(a14_sep._form_unchanged[0].integrals()[0].integrand()))


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="15", depends=["14"])
def test_separated_parametrized_forms_vector_15():
    a15 = (inner(grad(expr11) * grad(u), grad(v)) * dx + inner(grad(u) * grad(expr10), v) * dx
           + expr10.dx(0) * inner(u, v) * dx)
    a15_sep = SeparatedParametrizedForm(a15)
    test_logger.log(DEBUG, "*** ###              FORM 15             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 11, but each term is multiplied by the gradient/"
                    + "partial derivative of a Function, which is the solution of a parametrized problem")
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

    assert "grad(f_21)" == str(a15_sep.coefficients[0][0])
    assert "grad(f_18)" == str(a15_sep.coefficients[1][0])
    assert "(grad(f_18))[0]" == str(a15_sep.coefficients[2][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a15_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a15_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a15_sep._placeholders[2][0]))

    assert "f_66" == str(a15_sep._placeholders[0][0])
    assert "f_67" == str(a15_sep._placeholders[1][0])
    assert "f_68" == str(a15_sep._placeholders[2][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a15_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a15_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a15_sep._form_with_placeholders[2].integrals()[0].integrand()))

    assert ("sum_{i_{161}} sum_{i_{160}} ({ A | A_{i_{155}, i_{156}} = sum_{i_{157}} f_66[i_{155}, i_{157}]"
            + " * (grad(v_1))[i_{157}, i_{156}]  })[i_{160}, i_{161}] * (grad(v_0))[i_{160}, i_{161}]  "
            == str(a15_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("sum_{i_{162}} ({ A | A_{i_{158}} = sum_{i_{159}} f_67[i_{159}] * (grad(v_1))[i_{158}, i_{159}]  })"
            + "[i_{162}] * v_0[i_{162}] "
            == str(a15_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert ("f_68 * (sum_{i_{163}} v_0[i_{163}] * v_1[i_{163}] )"
            == str(a15_sep._form_with_placeholders[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a15_sep._form_unchanged)))

    assert 0 == len(a15_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="16", depends=["15"])
def test_separated_parametrized_forms_vector_16():
    a16 = inner(grad(expr11[0]), u) * v[0] * dx + expr11[0].dx(0) * inner(u, v) * dx
    a16_sep = SeparatedParametrizedForm(a16)
    test_logger.log(DEBUG, "*** ###              FORM 16             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 12, but each term is multiplied by the gradient/"
                    + "partial derivative of a component of a solution of a parametrized problem")
    a16_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a16_sep.coefficients)))

    assert 2 == len(a16_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a16_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a16_sep.coefficients[1])))

    assert 1 == len(a16_sep.coefficients[0])
    assert 1 == len(a16_sep.coefficients[1])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a16_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a16_sep.coefficients[1][0]))

    assert "grad(f_21)" == str(a16_sep.coefficients[0][0])
    assert "(grad(f_21))[0, 0]" == str(a16_sep.coefficients[1][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a16_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a16_sep._placeholders[1][0]))

    assert "f_69" == str(a16_sep._placeholders[0][0])
    assert "f_70" == str(a16_sep._placeholders[1][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a16_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a16_sep._form_with_placeholders[1].integrals()[0].integrand()))

    assert ("v_0[0] * (sum_{i_{164}} f_69[0, i_{164}] * v_1[i_{164}] )"
            == str(a16_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("f_70 * (sum_{i_{165}} v_0[i_{165}] * v_1[i_{165}] )"
            == str(a16_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a16_sep._form_unchanged)))

    assert 0 == len(a16_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="17", depends=["16"])
def test_separated_parametrized_forms_vector_17():
    a17 = (inner(grad(expr14) * grad(u), grad(v)) * dx + inner(grad(u) * grad(expr13), v) * dx
           + expr13.dx(0) * inner(u, v) * dx)
    a17_sep = SeparatedParametrizedForm(a17)
    test_logger.log(DEBUG, "*** ###              FORM 17             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 13, but each term is multiplied by a the gradient/"
                    + "partial derivative of Function, which is not the solution of a parametrized problem."
                    + " This results in no parametrized coefficients")
    a17_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a17_sep.coefficients)))

    assert 0 == len(a17_sep.coefficients)
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a17_sep._form_unchanged)))

    assert 3 == len(a17_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._form_unchanged[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._form_unchanged[2].integrals()[0].integrand()))

    assert ("sum_{i_{174}} sum_{i_{173}} ({ A | A_{i_{168}, i_{169}} = sum_{i_{170}} (grad(v_1))[i_{170}, i_{169}]"
            + " * (grad(f_30))[i_{168}, i_{170}]  })[i_{173}, i_{174}] * (grad(v_0))[i_{173}, i_{174}]  "
            == str(a17_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("sum_{i_{175}} ({ A | A_{i_{171}} = sum_{i_{172}} (grad(v_1))[i_{171}, i_{172}] * (grad(f_27))"
            + "[i_{172}]  })[i_{175}] * v_0[i_{175}] "
            == str(a17_sep._form_unchanged[1].integrals()[0].integrand()))
    assert ("(grad(f_27))[0] * (sum_{i_{176}} v_0[i_{176}] * v_1[i_{176}] )"
            == str(a17_sep._form_unchanged[2].integrals()[0].integrand()))


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="18", depends=["17"])
def test_separated_parametrized_forms_vector_18():
    a18 = (inner(grad(expr14[0]), u) * v[0] * dx + expr14[0].dx(0) * inner(u, v) * dx)
    a18_sep = SeparatedParametrizedForm(a18)
    test_logger.log(DEBUG, "*** ###              FORM 18             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 14, but each term is multiplied by the gradient/"
                    + "partial derivative of a component of a Function which is not solution of a"
                    + " parametrized problem. This results in no parametrized coefficients")
    a18_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a18_sep.coefficients)))

    assert 0 == len(a18_sep.coefficients)
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a18_sep._form_unchanged)))

    assert 2 == len(a18_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a18_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a18_sep._form_unchanged[1].integrals()[0].integrand()))

    assert ("v_0[0] * (sum_{i_{177}} (grad(f_30))[0, i_{177}] * v_1[i_{177}] )"
            == str(a18_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("(grad(f_30))[0, 0] * (sum_{i_{178}} v_0[i_{178}] * v_1[i_{178}] )"
            == str(a18_sep._form_unchanged[1].integrals()[0].integrand()))
