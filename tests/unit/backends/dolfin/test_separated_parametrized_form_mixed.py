# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from logging import DEBUG, getLogger
from mpi4py import MPI
from dolfin import (CellDiameter, Constant, det, div, dx, Expression, FiniteElement, Function, FunctionSpace, grad,
                    inner, MixedElement, split, TensorFunctionSpace, TestFunction, tr, TrialFunction, UnitSquareMesh,
                    VectorElement, VectorFunctionSpace)
from rbnics.backends.dolfin import SeparatedParametrizedForm
from rbnics.backends.dolfin.separated_parametrized_form import logger as separated_parametrized_form_logger
from rbnics.utils.decorators.store_map_from_solution_to_problem import _solution_to_problem_map
from rbnics.utils.test import enable_logging

# Logger
test_logger = getLogger("tests/unit/test_separated_parametrized_forms_mixed.py")
enable_separated_parametrized_form_logging = enable_logging({
    separated_parametrized_form_logger: DEBUG, test_logger: DEBUG})

# Common variables
mesh = UnitSquareMesh(10, 10)

element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element)

expr1 = Expression("x[0]", mu_0=0., degree=1, cell=mesh.ufl_cell())  # f_5
expr2 = Expression(("x[0]", "x[1]"), mu_0=0., degree=1, cell=mesh.ufl_cell())  # f_6
expr3 = Expression((("1*x[0]", "2*x[1]"), ("3*x[0]", "4*x[1]")), mu_0=0., degree=1, cell=mesh.ufl_cell())  # f_7
expr4 = Expression((("4*x[0]", "3*x[1]"), ("2*x[0]", "1*x[1]")), mu_0=0., degree=1, cell=mesh.ufl_cell())  # f_8
expr5 = Expression("x[0]", degree=1, cell=mesh.ufl_cell())  # f_9
expr6 = Expression(("x[0]", "x[1]"), degree=1, cell=mesh.ufl_cell())  # f_10
expr7 = Expression((("1*x[0]", "2*x[1]"), ("3*x[0]", "4*x[1]")), degree=1, cell=mesh.ufl_cell())  # f_11
expr8 = Expression((("4*x[0]", "3*x[1]"), ("2*x[0]", "1*x[1]")), degree=1, cell=mesh.ufl_cell())  # f_12
expr9 = Constant(((1, 2), (3, 4)))  # f_13

scalar_V = FunctionSpace(mesh, "Lagrange", 2)
vector_V = VectorFunctionSpace(mesh, "Lagrange", 3)
tensor_V = TensorFunctionSpace(mesh, "Lagrange", 1)
expr10 = Function(scalar_V)  # f_20
expr11 = Function(vector_V)  # f_23
expr12 = Function(tensor_V)  # f_26
expr13 = Function(V)  # f_29
expr13_split = split(expr13)
expr13_split_0_split = split(expr13_split[0])
expr14 = Function(scalar_V)  # f_32
expr15 = Function(vector_V)  # f_35
expr16 = Function(tensor_V)  # f_38
expr17 = Function(V)  # f_41
expr17_split = split(expr17)
expr17_split_0_split = split(expr17_split[0])


class Problem(object):
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


_solution_to_problem_map[expr10] = Problem("problem10")
_solution_to_problem_map[expr11] = Problem("problem11")
_solution_to_problem_map[expr12] = Problem("problem12")
_solution_to_problem_map[expr13] = Problem("problem13")

u, p = split(TrialFunction(V))
v, q = split(TestFunction(V))

scalar_trial = TrialFunction(scalar_V)
scalar_test = TestFunction(scalar_V)
vector_trial = TrialFunction(vector_V)
vector_test = TestFunction(vector_V)


# Fixtures
skip_in_parallel = pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Numbering of functions changes in parallel.")


# Tests
@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="1")
def test_separated_parametrized_forms_mixed_1():
    a1 = (inner(expr3 * grad(u), grad(v)) * dx + inner(grad(u) * expr2, v) * dx + expr1 * inner(u, v) * dx
          - p * tr(expr4 * grad(v)) * dx - expr1 * q * div(u) * dx - expr2[0] * p * q * dx)
    a1_sep = SeparatedParametrizedForm(a1)
    test_logger.log(DEBUG, "*** ###              FORM 1             ### ***")
    test_logger.log(DEBUG, "This is a basic mixed parametrized form, with all parametrized coefficients")
    a1_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep.coefficients)))

    assert 6 == len(a1_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep.coefficients[2])))
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep.coefficients[3])))
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep.coefficients[4])))
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep.coefficients[5])))

    assert 1 == len(a1_sep.coefficients[0])
    assert 1 == len(a1_sep.coefficients[1])
    assert 1 == len(a1_sep.coefficients[2])
    assert 1 == len(a1_sep.coefficients[3])
    assert 1 == len(a1_sep.coefficients[4])
    assert 1 == len(a1_sep.coefficients[5])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a1_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep.coefficients[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep.coefficients[3][0]))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep.coefficients[4][0]))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep.coefficients[5][0]))

    assert "f_7" == str(a1_sep.coefficients[0][0])
    assert "f_6" == str(a1_sep.coefficients[1][0])
    assert "f_5" == str(a1_sep.coefficients[2][0])
    assert "f_8" == str(a1_sep.coefficients[3][0])
    assert "-1 * f_5" == str(a1_sep.coefficients[4][0])
    assert "f_6[0]" == str(a1_sep.coefficients[5][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._placeholders[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._placeholders[3][0]))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._placeholders[4][0]))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._placeholders[5][0]))

    assert "f_44" == str(a1_sep._placeholders[0][0])
    assert "f_45" == str(a1_sep._placeholders[1][0])
    assert "f_46" == str(a1_sep._placeholders[2][0])
    assert "f_47" == str(a1_sep._placeholders[3][0])
    assert "f_48" == str(a1_sep._placeholders[4][0])
    assert "f_49" == str(a1_sep._placeholders[5][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._form_with_placeholders[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._form_with_placeholders[3].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._form_with_placeholders[4].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a1_sep._form_with_placeholders[5].integrals()[0].integrand()))

    assert ("sum_{i_{17}} sum_{i_{16}} ([{ A | A_{i_{23}} = (grad(v_0))[0, i_{23}] },"
            + " { A | A_{i_{22}} = (grad(v_0))[1, i_{22}] }])[i_{16}, i_{17}] * ({ A | A_{i_8, i_9}"
            + " = sum_{i_{10}} ([{ A | A_{i_{25}} = (grad(v_1))[0, i_{25}] }, { A | A_{i_{24}}"
            + " = (grad(v_1))[1, i_{24}] }])[i_{10}, i_9] * f_44[i_8, i_{10}]  })[i_{16}, i_{17}]  "
            == str(a1_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("sum_{i_{18}} ([v_0[0], v_0[1]])[i_{18}] * ({ A | A_{i_{11}} = sum_{i_{12}} ([{ A | A_{i_{27}}"
            + " = (grad(v_1))[0, i_{27}] }, { A | A_{i_{26}} = (grad(v_1))[1, i_{26}] }])[i_{11}, i_{12}]"
            + " * f_45[i_{12}]  })[i_{18}] "
            == str(a1_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert ("f_46 * (sum_{i_{19}} ([v_0[0], v_0[1]])[i_{19}] * ([v_1[0], v_1[1]])[i_{19}] )"
            == str(a1_sep._form_with_placeholders[2].integrals()[0].integrand()))
    assert ("-1 * v_1[2] * (sum_{i_{20}} ({ A | A_{i_{13}, i_{14}} = sum_{i_{15}} ([{ A | A_{i_{29}}"
            + " = (grad(v_0))[0, i_{29}] }, { A | A_{i_{28}} = (grad(v_0))[1, i_{28}] }])[i_{15}, i_{14}]"
            + " * f_47[i_{13}, i_{15}]  })[i_{20}, i_{20}] )"
            == str(a1_sep._form_with_placeholders[3].integrals()[0].integrand()))
    assert ("v_0[2] * f_48 * (sum_{i_{21}} ([{ A | A_{i_{31}} = (grad(v_1))[0, i_{31}] }, { A | A_{i_{30}}"
            + " = (grad(v_1))[1, i_{30}] }])[i_{21}, i_{21}] )"
            == str(a1_sep._form_with_placeholders[4].integrals()[0].integrand()))
    assert "f_49 * v_0[2] * -1 * v_1[2]" == str(a1_sep._form_with_placeholders[5].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a1_sep._form_unchanged)))

    assert 0 == len(a1_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="2", depends=["1"])
def test_separated_parametrized_forms_mixed_2():
    a2 = (inner(expr3 * expr4 * grad(u), grad(v)) * dx + inner(grad(u) * expr2, v) * dx + expr1 * inner(u, v) * dx
          - p * tr(expr4 * grad(v)) * dx - expr1 * q * div(u) * dx - expr2[0] * p * q * dx)
    a2_sep = SeparatedParametrizedForm(a2)
    test_logger.log(DEBUG, "*** ###              FORM 2             ### ***")
    test_logger.log(DEBUG, "In this case the diffusivity tensor is given by the product of two expressions")
    a2_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep.coefficients)))

    assert 6 == len(a2_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep.coefficients[2])))
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep.coefficients[3])))
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep.coefficients[4])))
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep.coefficients[5])))

    assert 1 == len(a2_sep.coefficients[0])
    assert 1 == len(a2_sep.coefficients[1])
    assert 1 == len(a2_sep.coefficients[2])
    assert 1 == len(a2_sep.coefficients[3])
    assert 1 == len(a2_sep.coefficients[4])
    assert 1 == len(a2_sep.coefficients[5])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a2_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep.coefficients[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep.coefficients[3][0]))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep.coefficients[4][0]))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep.coefficients[5][0]))

    assert ("{ A | A_{i_{33}, i_{34}} = sum_{i_{35}} f_7[i_{33}, i_{35}] * f_8[i_{35}, i_{34}]  }"
            == str(a2_sep.coefficients[0][0]))
    assert "f_6" == str(a2_sep.coefficients[1][0])
    assert "f_5" == str(a2_sep.coefficients[2][0])
    assert "f_8" == str(a2_sep.coefficients[3][0])
    assert "-1 * f_5" == str(a2_sep.coefficients[4][0])
    assert "f_6[0]" == str(a2_sep.coefficients[5][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._placeholders[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._placeholders[3][0]))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._placeholders[4][0]))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._placeholders[5][0]))

    assert "f_50" == str(a2_sep._placeholders[0][0])
    assert "f_51" == str(a2_sep._placeholders[1][0])
    assert "f_52" == str(a2_sep._placeholders[2][0])
    assert "f_53" == str(a2_sep._placeholders[3][0])
    assert "f_54" == str(a2_sep._placeholders[4][0])
    assert "f_55" == str(a2_sep._placeholders[5][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._form_with_placeholders[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._form_with_placeholders[3].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._form_with_placeholders[4].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a2_sep._form_with_placeholders[5].integrals()[0].integrand()))

    assert ("sum_{i_{45}} sum_{i_{44}} ([{ A | A_{i_{51}} = (grad(v_0))[0, i_{51}] }, { A | A_{i_{50}}"
            + " = (grad(v_0))[1, i_{50}] }])[i_{44}, i_{45}] * ({ A | A_{i_{36}, i_{37}} = sum_{i_{38}}"
            + " ([{ A | A_{i_{53}} = (grad(v_1))[0, i_{53}] }, { A | A_{i_{52}} = (grad(v_1))[1, i_{52}] }])"
            + "[i_{38}, i_{37}] * f_50[i_{36}, i_{38}]  })[i_{44}, i_{45}]  "
            == str(a2_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("sum_{i_{46}} ([v_0[0], v_0[1]])[i_{46}] * ({ A | A_{i_{39}} = sum_{i_{40}} ([{ A | A_{i_{55}}"
            + " = (grad(v_1))[0, i_{55}] }, { A | A_{i_{54}} = (grad(v_1))[1, i_{54}] }])[i_{39}, i_{40}]"
            + " * f_51[i_{40}]  })[i_{46}] "
            == str(a2_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert ("f_52 * (sum_{i_{47}} ([v_0[0], v_0[1]])[i_{47}] * ([v_1[0], v_1[1]])[i_{47}] )"
            == str(a2_sep._form_with_placeholders[2].integrals()[0].integrand()))
    assert ("-1 * v_1[2] * (sum_{i_{48}} ({ A | A_{i_{41}, i_{42}} = sum_{i_{43}} ([{ A | A_{i_{57}}"
            + " = (grad(v_0))[0, i_{57}] }, { A | A_{i_{56}} = (grad(v_0))[1, i_{56}] }])[i_{43}, i_{42}]"
            + " * f_53[i_{41}, i_{43}]  })[i_{48}, i_{48}] )"
            == str(a2_sep._form_with_placeholders[3].integrals()[0].integrand()))
    assert ("v_0[2] * f_54 * (sum_{i_{49}} ([{ A | A_{i_{59}} = (grad(v_1))[0, i_{59}] }, { A | A_{i_{58}}"
            + " = (grad(v_1))[1, i_{58}] }])[i_{49}, i_{49}] )"
            == str(a2_sep._form_with_placeholders[4].integrals()[0].integrand()))
    assert "f_55 * v_0[2] * -1 * v_1[2]" == str(a2_sep._form_with_placeholders[5].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a2_sep._form_unchanged)))

    assert 0 == len(a2_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="3", depends=["2"])
def test_separated_parametrized_forms_mixed_3():
    a3 = (inner(det(expr3) * (expr4 + expr3 * expr3) * expr1 * grad(u), grad(v)) * dx + inner(grad(u) * expr2, v) * dx
          + expr1 * inner(u, v) * dx - p * tr(expr4 * grad(v)) * dx - expr1 * q * div(u) * dx - expr2[0] * p * q * dx)
    a3_sep = SeparatedParametrizedForm(a3)
    test_logger.log(DEBUG, "*** ###              FORM 3             ### ***")
    test_logger.log(DEBUG, "We try now with a more complex expression of for each coefficient")
    a3_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients)))

    assert 9 == len(a3_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients[2])))
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients[3])))
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients[4])))
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients[5])))
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients[6])))
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients[7])))
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep.coefficients[8])))

    assert 1 == len(a3_sep.coefficients[0])
    assert 1 == len(a3_sep.coefficients[1])
    assert 1 == len(a3_sep.coefficients[2])
    assert 1 == len(a3_sep.coefficients[3])
    assert 1 == len(a3_sep.coefficients[4])
    assert 1 == len(a3_sep.coefficients[5])
    assert 1 == len(a3_sep.coefficients[6])
    assert 1 == len(a3_sep.coefficients[7])
    assert 1 == len(a3_sep.coefficients[8])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[3][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[4][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[5][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[6][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[7][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep.coefficients[8][0]))

    assert ("{ A | A_{i_{66}, i_{67}} = ({ A | A_{i_{64}, i_{65}} = f_7[0, 0] * f_7[1, 1]"
            + " * ({ A | A_{i_{61}, i_{62}} = sum_{i_{63}} f_7[i_{63}, i_{62}] * f_7[i_{61}, i_{63}]  })"
            + "[i_{64}, i_{65}] })[i_{66}, i_{67}] * f_5 }"
            == str(a3_sep.coefficients[0][0]))
    assert ("{ A | A_{i_{66}, i_{67}} = ({ A | A_{i_{64}, i_{65}} = f_7[0, 0] * f_7[1, 1] * f_8[i_{64}, i_{65}] })"
            + "[i_{66}, i_{67}] * f_5 }"
            == str(a3_sep.coefficients[1][0]))
    assert ("{ A | A_{i_{66}, i_{67}} = ({ A | A_{i_{64}, i_{65}} = -1 * f_7[0, 1] * f_7[1, 0]"
            + " * f_8[i_{64}, i_{65}] })[i_{66}, i_{67}] * f_5 }"
            == str(a3_sep.coefficients[2][0]))
    assert ("{ A | A_{i_{66}, i_{67}} = ({ A | A_{i_{64}, i_{65}} = -1 * f_7[0, 1] * f_7[1, 0]"
            + " * ({ A | A_{i_{61}, i_{62}} = sum_{i_{63}} f_7[i_{63}, i_{62}] * f_7[i_{61}, i_{63}]  })"
            + "[i_{64}, i_{65}] })[i_{66}, i_{67}] * f_5 }"
            == str(a3_sep.coefficients[3][0]))
    assert "f_6" == str(a3_sep.coefficients[4][0])
    assert "f_5" == str(a3_sep.coefficients[5][0])
    assert "f_8" == str(a3_sep.coefficients[6][0])
    assert "f_5" == str(a3_sep.coefficients[7][0])
    assert "f_6[0]" == str(a3_sep.coefficients[8][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[3][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[4][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[5][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[6][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[7][0]))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._placeholders[8][0]))

    assert "f_56" == str(a3_sep._placeholders[0][0])
    assert "f_57" == str(a3_sep._placeholders[1][0])
    assert "f_58" == str(a3_sep._placeholders[2][0])
    assert "f_59" == str(a3_sep._placeholders[3][0])
    assert "f_60" == str(a3_sep._placeholders[4][0])
    assert "f_61" == str(a3_sep._placeholders[5][0])
    assert "f_62" == str(a3_sep._placeholders[6][0])
    assert "f_63" == str(a3_sep._placeholders[7][0])
    assert "f_64" == str(a3_sep._placeholders[8][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[3].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[4].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[5].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[6].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[7].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a3_sep._form_with_placeholders[8].integrals()[0].integrand()))

    assert ("sum_{i_{77}} sum_{i_{76}} ([{ A | A_{i_{83}} = (grad(v_0))[0, i_{83}] }, { A | A_{i_{82}}"
            + " = (grad(v_0))[1, i_{82}] }])[i_{76}, i_{77}] * ({ A | A_{i_{68}, i_{69}} = sum_{i_{70}}"
            + " ([{ A | A_{i_{85}} = (grad(v_1))[0, i_{85}] }, { A | A_{i_{84}} = (grad(v_1))[1, i_{84}] }])"
            + "[i_{70}, i_{69}] * f_56[i_{68}, i_{70}]  })[i_{76}, i_{77}]  "
            == str(a3_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("sum_{i_{77}} sum_{i_{76}} ([{ A | A_{i_{83}} = (grad(v_0))[0, i_{83}] }, { A | A_{i_{82}}"
            + " = (grad(v_0))[1, i_{82}] }])[i_{76}, i_{77}] * ({ A | A_{i_{68}, i_{69}} = sum_{i_{70}}"
            + " ([{ A | A_{i_{85}} = (grad(v_1))[0, i_{85}] }, { A | A_{i_{84}} = (grad(v_1))[1, i_{84}] }])"
            + "[i_{70}, i_{69}] * f_57[i_{68}, i_{70}]  })[i_{76}, i_{77}]  "
            == str(a3_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert ("sum_{i_{77}} sum_{i_{76}} ([{ A | A_{i_{83}} = (grad(v_0))[0, i_{83}] }, { A | A_{i_{82}}"
            + " = (grad(v_0))[1, i_{82}] }])[i_{76}, i_{77}] * ({ A | A_{i_{68}, i_{69}} = sum_{i_{70}}"
            + " ([{ A | A_{i_{85}} = (grad(v_1))[0, i_{85}] }, { A | A_{i_{84}} = (grad(v_1))[1, i_{84}] }])"
            + "[i_{70}, i_{69}] * f_58[i_{68}, i_{70}]  })[i_{76}, i_{77}]  "
            == str(a3_sep._form_with_placeholders[2].integrals()[0].integrand()))
    assert ("sum_{i_{77}} sum_{i_{76}} ([{ A | A_{i_{83}} = (grad(v_0))[0, i_{83}] }, { A | A_{i_{82}}"
            + " = (grad(v_0))[1, i_{82}] }])[i_{76}, i_{77}] * ({ A | A_{i_{68}, i_{69}} = sum_{i_{70}}"
            + " ([{ A | A_{i_{85}} = (grad(v_1))[0, i_{85}] }, { A | A_{i_{84}} = (grad(v_1))[1, i_{84}] }])"
            + "[i_{70}, i_{69}] * f_59[i_{68}, i_{70}]  })[i_{76}, i_{77}]  "
            == str(a3_sep._form_with_placeholders[3].integrals()[0].integrand()))
    assert ("sum_{i_{78}} ([v_0[0], v_0[1]])[i_{78}] * ({ A | A_{i_{71}} = sum_{i_{72}} ([{ A | A_{i_{87}}"
            + " = (grad(v_1))[0, i_{87}] }, { A | A_{i_{86}} = (grad(v_1))[1, i_{86}] }])[i_{71}, i_{72}]"
            + " * f_60[i_{72}]  })[i_{78}] "
            == str(a3_sep._form_with_placeholders[4].integrals()[0].integrand()))
    assert ("f_61 * (sum_{i_{79}} ([v_0[0], v_0[1]])[i_{79}] * ([v_1[0], v_1[1]])[i_{79}] )"
            == str(a3_sep._form_with_placeholders[5].integrals()[0].integrand()))
    assert ("-1 * v_1[2] * (sum_{i_{80}} ({ A | A_{i_{73}, i_{74}} = sum_{i_{75}} ([{ A | A_{i_{89}}"
            + " = (grad(v_0))[0, i_{89}] }, { A | A_{i_{88}} = (grad(v_0))[1, i_{88}] }])[i_{75}, i_{74}]"
            + " * f_62[i_{73}, i_{75}]  })[i_{80}, i_{80}] )"
            == str(a3_sep._form_with_placeholders[6].integrals()[0].integrand()))
    assert ("-1 * v_0[2] * f_63 * (sum_{i_{81}} ([{ A | A_{i_{91}} = (grad(v_1))[0, i_{91}] },"
            + " { A | A_{i_{90}} = (grad(v_1))[1, i_{90}] }])[i_{81}, i_{81}] )"
            == str(a3_sep._form_with_placeholders[7].integrals()[0].integrand()))
    assert "-1 * f_64 * v_0[2] * v_1[2]" == str(a3_sep._form_with_placeholders[8].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a3_sep._form_unchanged)))

    assert 0 == len(a3_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="4", depends=["3"])
def test_separated_parametrized_forms_mixed_4():
    h = CellDiameter(mesh)
    a4 = (inner(expr3 * h * grad(u), grad(v)) * dx + inner(grad(u) * expr2 * h, v) * dx + expr1 * h * inner(u, v) * dx
          - p * tr(expr4 * h * grad(v)) * dx - expr1 * h * q * div(u) * dx - expr2[0] * h * p * q * dx)
    a4_sep = SeparatedParametrizedForm(a4)
    test_logger.log(DEBUG, "*** ###              FORM 4             ### ***")
    test_logger.log(DEBUG, "We add a term depending on the mesh size. The extracted coefficients may retain"
                    + " the mesh size factor depending on the UFL tree")
    a4_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep.coefficients)))

    assert 6 == len(a4_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep.coefficients[2])))
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep.coefficients[3])))
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep.coefficients[4])))
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep.coefficients[5])))

    assert 1 == len(a4_sep.coefficients[0])
    assert 1 == len(a4_sep.coefficients[1])
    assert 1 == len(a4_sep.coefficients[2])
    assert 1 == len(a4_sep.coefficients[3])
    assert 1 == len(a4_sep.coefficients[4])
    assert 1 == len(a4_sep.coefficients[5])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a4_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep.coefficients[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep.coefficients[3][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep.coefficients[4][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep.coefficients[5][0]))

    assert "{ A | A_{i_{93}, i_{94}} = diameter * f_7[i_{93}, i_{94}] }" == str(a4_sep.coefficients[0][0])
    assert "f_6" == str(a4_sep.coefficients[1][0])
    assert "f_5" == str(a4_sep.coefficients[2][0])
    assert "{ A | A_{i_{101}, i_{102}} = diameter * f_8[i_{101}, i_{102}] }" == str(a4_sep.coefficients[3][0])
    assert "-1 * diameter * f_5" == str(a4_sep.coefficients[4][0])
    assert "f_6[0]" == str(a4_sep.coefficients[5][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._placeholders[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._placeholders[3][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._placeholders[4][0]))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._placeholders[5][0]))

    assert "f_65" == str(a4_sep._placeholders[0][0])
    assert "f_66" == str(a4_sep._placeholders[1][0])
    assert "f_67" == str(a4_sep._placeholders[2][0])
    assert "f_68" == str(a4_sep._placeholders[3][0])
    assert "f_69" == str(a4_sep._placeholders[4][0])
    assert "f_70" == str(a4_sep._placeholders[5][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._form_with_placeholders[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._form_with_placeholders[3].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._form_with_placeholders[4].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a4_sep._form_with_placeholders[5].integrals()[0].integrand()))

    assert ("sum_{i_{107}} sum_{i_{106}} ([{ A | A_{i_{113}} = (grad(v_0))[0, i_{113}] }, { A | A_{i_{112}}"
            + " = (grad(v_0))[1, i_{112}] }])[i_{106}, i_{107}] * ({ A | A_{i_{95}, i_{96}} = sum_{i_{97}}"
            + " ([{ A | A_{i_{115}} = (grad(v_1))[0, i_{115}] }, { A | A_{i_{114}} = (grad(v_1))[1, i_{114}] }])"
            + "[i_{97}, i_{96}] * f_65[i_{95}, i_{97}]  })[i_{106}, i_{107}]  "
            == str(a4_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("sum_{i_{108}} ([v_0[0], v_0[1]])[i_{108}] * ({ A | A_{i_{100}} = diameter * ({ A | A_{i_{98}}"
            + " = sum_{i_{99}} ([{ A | A_{i_{117}} = (grad(v_1))[0, i_{117}] }, { A | A_{i_{116}} = (grad(v_1))"
            + "[1, i_{116}] }])[i_{98}, i_{99}] * f_66[i_{99}]  })[i_{100}] })[i_{108}] "
            == str(a4_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert ("f_67 * diameter * (sum_{i_{109}} ([v_0[0], v_0[1]])[i_{109}] * ([v_1[0], v_1[1]])[i_{109}] )"
            == str(a4_sep._form_with_placeholders[2].integrals()[0].integrand()))
    assert ("-1 * v_1[2] * (sum_{i_{110}} ({ A | A_{i_{103}, i_{104}} = sum_{i_{105}} ([{ A | A_{i_{119}}"
            + " = (grad(v_0))[0, i_{119}] }, { A | A_{i_{118}} = (grad(v_0))[1, i_{118}] }])[i_{105}, i_{104}]"
            + " * f_68[i_{103}, i_{105}]  })[i_{110}, i_{110}] )"
            == str(a4_sep._form_with_placeholders[3].integrals()[0].integrand()))
    assert ("v_0[2] * f_69 * (sum_{i_{111}} ([{ A | A_{i_{121}} = (grad(v_1))[0, i_{121}] }, { A | A_{i_{120}}"
            + " = (grad(v_1))[1, i_{120}] }])[i_{111}, i_{111}] )"
            == str(a4_sep._form_with_placeholders[4].integrals()[0].integrand()))
    assert ("f_70 * v_0[2] * -1 * diameter * v_1[2]"
            == str(a4_sep._form_with_placeholders[5].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a4_sep._form_unchanged)))

    assert 0 == len(a4_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="5", depends=["4"])
def test_separated_parametrized_forms_mixed_5():
    h = CellDiameter(mesh)
    a5 = (inner((expr3 * h) * grad(u), grad(v)) * dx + inner(grad(u) * (expr2 * h), v) * dx
          + (expr1 * h) * inner(u, v) * dx - p * tr((expr4 * h) * grad(v)) * dx - (expr1 * h) * q * div(u) * dx
          - (expr2[0] * h) * p * q * dx)
    a5_sep = SeparatedParametrizedForm(a5)
    test_logger.log(DEBUG, "*** ###              FORM 5             ### ***")
    test_logger.log(DEBUG, "Starting from form 4, use parenthesis to make sure that the extracted coefficients"
                    + " retain the mesh size factor")
    a5_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep.coefficients)))

    assert 6 == len(a5_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep.coefficients[2])))
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep.coefficients[3])))
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep.coefficients[4])))
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep.coefficients[5])))

    assert 1 == len(a5_sep.coefficients[0])
    assert 1 == len(a5_sep.coefficients[1])
    assert 1 == len(a5_sep.coefficients[2])
    assert 1 == len(a5_sep.coefficients[3])
    assert 1 == len(a5_sep.coefficients[4])
    assert 1 == len(a5_sep.coefficients[5])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a5_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep.coefficients[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep.coefficients[3][0]))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep.coefficients[4][0]))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep.coefficients[5][0]))

    assert "{ A | A_{i_{123}, i_{124}} = diameter * f_7[i_{123}, i_{124}] }" == str(a5_sep.coefficients[0][0])
    assert "{ A | A_{i_{128}} = diameter * f_6[i_{128}] }" == str(a5_sep.coefficients[1][0])
    assert "f_5" == str(a5_sep.coefficients[2][0])
    assert "{ A | A_{i_{131}, i_{132}} = diameter * f_8[i_{131}, i_{132}] }" == str(a5_sep.coefficients[3][0])
    assert "-1 * diameter * f_5" == str(a5_sep.coefficients[4][0])
    assert "f_6[0]" == str(a5_sep.coefficients[5][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._placeholders[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._placeholders[3][0]))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._placeholders[4][0]))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._placeholders[5][0]))

    assert "f_71" == str(a5_sep._placeholders[0][0])
    assert "f_72" == str(a5_sep._placeholders[1][0])
    assert "f_73" == str(a5_sep._placeholders[2][0])
    assert "f_74" == str(a5_sep._placeholders[3][0])
    assert "f_75" == str(a5_sep._placeholders[4][0])
    assert "f_76" == str(a5_sep._placeholders[5][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._form_with_placeholders[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._form_with_placeholders[3].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._form_with_placeholders[4].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a5_sep._form_with_placeholders[5].integrals()[0].integrand()))

    assert ("sum_{i_{137}} sum_{i_{136}} ([{ A | A_{i_{143}} = (grad(v_0))[0, i_{143}] }, { A | A_{i_{142}}"
            + " = (grad(v_0))[1, i_{142}] }])[i_{136}, i_{137}] * ({ A | A_{i_{125}, i_{126}} = sum_{i_{127}}"
            + " ([{ A | A_{i_{145}} = (grad(v_1))[0, i_{145}] }, { A | A_{i_{144}} = (grad(v_1))[1, i_{144}] }])"
            + "[i_{127}, i_{126}] * f_71[i_{125}, i_{127}]  })[i_{136}, i_{137}]  "
            == str(a5_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("sum_{i_{138}} ([v_0[0], v_0[1]])[i_{138}] * ({ A | A_{i_{129}} = sum_{i_{130}} ([{ A | A_{i_{147}}"
            + " = (grad(v_1))[0, i_{147}] }, { A | A_{i_{146}} = (grad(v_1))[1, i_{146}] }])[i_{129}, i_{130}]"
            + " * f_72[i_{130}]  })[i_{138}] "
            == str(a5_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert ("f_73 * diameter * (sum_{i_{139}} ([v_0[0], v_0[1]])[i_{139}] * ([v_1[0], v_1[1]])[i_{139}] )"
            == str(a5_sep._form_with_placeholders[2].integrals()[0].integrand()))
    assert ("-1 * v_1[2] * (sum_{i_{140}} ({ A | A_{i_{133}, i_{134}} = sum_{i_{135}} ([{ A | A_{i_{149}}"
            + " = (grad(v_0))[0, i_{149}] }, { A | A_{i_{148}} = (grad(v_0))[1, i_{148}] }])[i_{135}, i_{134}]"
            + " * f_74[i_{133}, i_{135}]  })[i_{140}, i_{140}] )"
            == str(a5_sep._form_with_placeholders[3].integrals()[0].integrand()))
    assert ("v_0[2] * f_75 * (sum_{i_{141}} ([{ A | A_{i_{151}} = (grad(v_1))[0, i_{151}] }, { A | A_{i_{150}}"
            + " = (grad(v_1))[1, i_{150}] }])[i_{141}, i_{141}] )"
            == str(a5_sep._form_with_placeholders[4].integrals()[0].integrand()))
    assert ("f_76 * v_0[2] * -1 * diameter * v_1[2]"
            == str(a5_sep._form_with_placeholders[5].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a5_sep._form_unchanged)))

    assert 0 == len(a5_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="6", depends=["5"])
def test_separated_parametrized_forms_mixed_6():
    a6 = (inner(expr7 * grad(u), grad(v)) * dx + inner(grad(u) * expr6, v) * dx + expr5 * inner(u, v) * dx
          - p * tr(expr7 * grad(v)) * dx - expr5 * q * div(u) * dx - expr6[0] * p * q * dx)
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

    assert 6 == len(a6_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a6_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a6_sep._form_unchanged[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a6_sep._form_unchanged[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a6_sep._form_unchanged[3].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a6_sep._form_unchanged[4].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a6_sep._form_unchanged[5].integrals()[0].integrand()))

    assert ("sum_{i_{162}} sum_{i_{161}} ([{ A | A_{i_{168}} = (grad(v_0))[0, i_{168}] }, { A | A_{i_{167}}"
            + " = (grad(v_0))[1, i_{167}] }])[i_{161}, i_{162}] * ({ A | A_{i_{153}, i_{154}} = sum_{i_{155}}"
            + " ([{ A | A_{i_{170}} = (grad(v_1))[0, i_{170}] }, { A | A_{i_{169}} = (grad(v_1))[1, i_{169}] }])"
            + "[i_{155}, i_{154}] * f_11[i_{153}, i_{155}]  })[i_{161}, i_{162}]  "
            == str(a6_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("sum_{i_{163}} ([v_0[0], v_0[1]])[i_{163}] * ({ A | A_{i_{156}} = sum_{i_{157}} ([{ A | A_{i_{172}}"
            + " = (grad(v_1))[0, i_{172}] }, { A | A_{i_{171}} = (grad(v_1))[1, i_{171}] }])[i_{156}, i_{157}]"
            + " * f_10[i_{157}]  })[i_{163}] "
            == str(a6_sep._form_unchanged[1].integrals()[0].integrand()))
    assert ("f_9 * (sum_{i_{164}} ([v_0[0], v_0[1]])[i_{164}] * ([v_1[0], v_1[1]])[i_{164}] )"
            == str(a6_sep._form_unchanged[2].integrals()[0].integrand()))
    assert ("-1 * v_1[2] * (sum_{i_{165}} ({ A | A_{i_{158}, i_{159}} = sum_{i_{160}} ([{ A | A_{i_{174}}"
            + " = (grad(v_0))[0, i_{174}] }, { A | A_{i_{173}} = (grad(v_0))[1, i_{173}] }])[i_{160}, i_{159}]"
            + " * f_11[i_{158}, i_{160}]  })[i_{165}, i_{165}] )"
            == str(a6_sep._form_unchanged[3].integrals()[0].integrand()))
    assert ("v_0[2] * -1 * f_9 * (sum_{i_{166}} ([{ A | A_{i_{176}} = (grad(v_1))[0, i_{176}] }, { A | A_{i_{175}}"
            + " = (grad(v_1))[1, i_{175}] }])[i_{166}, i_{166}] )"
            == str(a6_sep._form_unchanged[4].integrals()[0].integrand()))
    assert "f_10[0] * v_0[2] * -1 * v_1[2]" == str(a6_sep._form_unchanged[5].integrals()[0].integrand())


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="7", depends=["6"])
def test_separated_parametrized_forms_mixed_7():
    a7 = (inner(expr7 * (expr3 * expr4) * grad(u), grad(v)) * dx + inner(grad(u) * expr6, v) * dx
          + expr5 * inner(u, v) * dx - p * tr(expr7 * grad(v)) * dx - expr5 * q * div(u) * dx - expr6[0] * p * q * dx)
    a7_sep = SeparatedParametrizedForm(a7)
    test_logger.log(DEBUG, "*** ###              FORM 7             ### ***")
    test_logger.log(DEBUG, "A part of the diffusion coefficient is parametrized (the others are not). "
                    + "Only the parametrized part is extracted.")
    a7_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a7_sep.coefficients)))

    assert 1 == len(a7_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a7_sep.coefficients[0])))

    assert 1 == len(a7_sep.coefficients[0])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a7_sep.coefficients[0][0]))

    assert ("{ A | A_{i_{178}, i_{179}} = sum_{i_{180}} f_7[i_{178}, i_{180}] * f_8[i_{180}, i_{179}]  }"
            == str(a7_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a7_sep._placeholders[0][0]))

    assert "f_77" == str(a7_sep._placeholders[0][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a7_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert ("sum_{i_{193}} sum_{i_{192}} ([{ A | A_{i_{199}} = (grad(v_0))[0, i_{199}] }, { A | A_{i_{198}}"
            + " = (grad(v_0))[1, i_{198}] }])[i_{192}, i_{193}] * ({ A | A_{i_{184}, i_{185}} = sum_{i_{186}}"
            + " ([{ A | A_{i_{201}} = (grad(v_1))[0, i_{201}] }, { A | A_{i_{200}} = (grad(v_1))[1, i_{200}] }])"
            + "[i_{186}, i_{185}] * ({ A | A_{i_{181}, i_{182}} = sum_{i_{183}} f_11[i_{181}, i_{183}]"
            + " * f_77[i_{183}, i_{182}]  })[i_{184}, i_{186}]  })[i_{192}, i_{193}]  "
            == str(a7_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a7_sep._form_unchanged)))

    assert 5 == len(a7_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a7_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a7_sep._form_unchanged[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a7_sep._form_unchanged[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a7_sep._form_unchanged[3].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a7_sep._form_unchanged[4].integrals()[0].integrand()))

    assert ("sum_{i_{194}} ([v_0[0], v_0[1]])[i_{194}] * ({ A | A_{i_{187}} = sum_{i_{188}} ([{ A | A_{i_{203}}"
            + " = (grad(v_1))[0, i_{203}] }, { A | A_{i_{202}} = (grad(v_1))[1, i_{202}] }])[i_{187}, i_{188}]"
            + " * f_10[i_{188}]  })[i_{194}] "
            == str(a7_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("f_9 * (sum_{i_{195}} ([v_0[0], v_0[1]])[i_{195}] * ([v_1[0], v_1[1]])[i_{195}] )"
            == str(a7_sep._form_unchanged[1].integrals()[0].integrand()))
    assert ("-1 * v_1[2] * (sum_{i_{196}} ({ A | A_{i_{189}, i_{190}} = sum_{i_{191}} ([{ A | A_{i_{205}}"
            + " = (grad(v_0))[0, i_{205}] }, { A | A_{i_{204}} = (grad(v_0))[1, i_{204}] }])[i_{191}, i_{190}]"
            + " * f_11[i_{189}, i_{191}]  })[i_{196}, i_{196}] )"
            == str(a7_sep._form_unchanged[2].integrals()[0].integrand()))
    assert ("v_0[2] * -1 * f_9 * (sum_{i_{197}} ([{ A | A_{i_{207}} = (grad(v_1))[0, i_{207}] }, { A | A_{i_{206}}"
            + " = (grad(v_1))[1, i_{206}] }])[i_{197}, i_{197}] )"
            == str(a7_sep._form_unchanged[3].integrals()[0].integrand()))
    assert "f_10[0] * v_0[2] * -1 * v_1[2]" == str(a7_sep._form_unchanged[4].integrals()[0].integrand())


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="8", depends=["7"])
def test_separated_parametrized_forms_mixed_8():
    a8 = (inner(expr3 * expr7 * expr4 * grad(u), grad(v)) * dx + inner(grad(u) * expr6, v) * dx
          + expr5 * inner(u, v) * dx - p * tr(expr7 * grad(v)) * dx - expr5 * q * div(u) * dx - expr6[0] * p * q * dx)
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

    assert ("{ A | A_{i_{212}, i_{213}} = sum_{i_{214}} ({ A | A_{i_{209}, i_{210}} = sum_{i_{211}}"
            + " f_7[i_{209}, i_{211}] * f_11[i_{211}, i_{210}]  })[i_{212}, i_{214}] * f_8[i_{214}, i_{213}]  }"
            == str(a8_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a8_sep._placeholders[0][0]))

    assert "f_78" == str(a8_sep._placeholders[0][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a8_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert ("sum_{i_{224}} sum_{i_{223}} ([{ A | A_{i_{230}} = (grad(v_0))[0, i_{230}] }, { A | A_{i_{229}}"
            + " = (grad(v_0))[1, i_{229}] }])[i_{223}, i_{224}] * ({ A | A_{i_{215}, i_{216}} = sum_{i_{217}}"
            + " ([{ A | A_{i_{232}} = (grad(v_1))[0, i_{232}] }, { A | A_{i_{231}} = (grad(v_1))[1, i_{231}] }])"
            + "[i_{217}, i_{216}] * f_78[i_{215}, i_{217}]  })[i_{223}, i_{224}]  "
            == str(a8_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a8_sep._form_unchanged)))

    assert 5 == len(a8_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a8_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a8_sep._form_unchanged[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a8_sep._form_unchanged[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a8_sep._form_unchanged[3].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a8_sep._form_unchanged[4].integrals()[0].integrand()))

    assert ("sum_{i_{225}} ([v_0[0], v_0[1]])[i_{225}] * ({ A | A_{i_{218}} = sum_{i_{219}} ([{ A | A_{i_{234}}"
            + " = (grad(v_1))[0, i_{234}] }, { A | A_{i_{233}} = (grad(v_1))[1, i_{233}] }])[i_{218}, i_{219}]"
            + " * f_10[i_{219}]  })[i_{225}] "
            == str(a8_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("f_9 * (sum_{i_{226}} ([v_0[0], v_0[1]])[i_{226}] * ([v_1[0], v_1[1]])[i_{226}] )"
            == str(a8_sep._form_unchanged[1].integrals()[0].integrand()))
    assert ("-1 * v_1[2] * (sum_{i_{227}} ({ A | A_{i_{220}, i_{221}} = sum_{i_{222}} ([{ A | A_{i_{236}}"
            + " = (grad(v_0))[0, i_{236}] }, { A | A_{i_{235}} = (grad(v_0))[1, i_{235}] }])[i_{222}, i_{221}]"
            + " * f_11[i_{220}, i_{222}]  })[i_{227}, i_{227}] )"
            == str(a8_sep._form_unchanged[2].integrals()[0].integrand()))
    assert ("v_0[2] * -1 * f_9 * (sum_{i_{228}} ([{ A | A_{i_{238}} = (grad(v_1))[0, i_{238}] }, { A | A_{i_{237}}"
            + " = (grad(v_1))[1, i_{237}] }])[i_{228}, i_{228}] )"
            == str(a8_sep._form_unchanged[3].integrals()[0].integrand()))
    assert "f_10[0] * v_0[2] * -1 * v_1[2]" == str(a8_sep._form_unchanged[4].integrals()[0].integrand())


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="9", depends=["8"])
def test_separated_parametrized_forms_mixed_9():
    a9 = (inner(expr9 * (expr3 * expr4) * grad(u), grad(v)) * dx + inner(grad(u) * expr6, v) * dx
          + expr5 * inner(u, v) * dx - p * tr(expr7 * grad(v)) * dx - expr5 * q * div(u) * dx - expr6[0] * p * q * dx)
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

    assert ("{ A | A_{i_{240}, i_{241}} = sum_{i_{242}} f_7[i_{240}, i_{242}] * f_8[i_{242}, i_{241}]  }"
            == str(a9_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a9_sep._placeholders[0][0]))

    assert "f_79" == str(a9_sep._placeholders[0][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a9_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert ("sum_{i_{255}} sum_{i_{254}} ([{ A | A_{i_{261}} = (grad(v_0))[0, i_{261}] }, { A | A_{i_{260}}"
            + " = (grad(v_0))[1, i_{260}] }])[i_{254}, i_{255}] * ({ A | A_{i_{246}, i_{247}} = sum_{i_{248}}"
            + " ([{ A | A_{i_{263}} = (grad(v_1))[0, i_{263}] }, { A | A_{i_{262}} = (grad(v_1))[1, i_{262}] }])"
            + "[i_{248}, i_{247}] * ({ A | A_{i_{243}, i_{244}} = sum_{i_{245}} f_13[i_{243}, i_{245}]"
            + " * f_79[i_{245}, i_{244}]  })[i_{246}, i_{248}]  })[i_{254}, i_{255}]  "
            == str(a9_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a9_sep._form_unchanged)))

    assert 5 == len(a9_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a9_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a9_sep._form_unchanged[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a9_sep._form_unchanged[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a9_sep._form_unchanged[3].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a9_sep._form_unchanged[4].integrals()[0].integrand()))

    assert ("sum_{i_{256}} ([v_0[0], v_0[1]])[i_{256}] * ({ A | A_{i_{249}} = sum_{i_{250}} ([{ A | A_{i_{265}}"
            + " = (grad(v_1))[0, i_{265}] }, { A | A_{i_{264}} = (grad(v_1))[1, i_{264}] }])[i_{249}, i_{250}]"
            + " * f_10[i_{250}]  })[i_{256}] "
            == str(a9_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("f_9 * (sum_{i_{257}} ([v_0[0], v_0[1]])[i_{257}] * ([v_1[0], v_1[1]])[i_{257}] )"
            == str(a9_sep._form_unchanged[1].integrals()[0].integrand()))
    assert ("-1 * v_1[2] * (sum_{i_{258}} ({ A | A_{i_{251}, i_{252}} = sum_{i_{253}} ([{ A | A_{i_{267}}"
            + " = (grad(v_0))[0, i_{267}] }, { A | A_{i_{266}} = (grad(v_0))[1, i_{266}] }])[i_{253}, i_{252}]"
            + " * f_11[i_{251}, i_{253}]  })[i_{258}, i_{258}] )"
            == str(a9_sep._form_unchanged[2].integrals()[0].integrand()))
    assert ("v_0[2] * -1 * f_9 * (sum_{i_{259}} ([{ A | A_{i_{269}} = (grad(v_1))[0, i_{269}] }, { A | A_{i_{268}}"
            + " = (grad(v_1))[1, i_{268}] }])[i_{259}, i_{259}] )"
            == str(a9_sep._form_unchanged[3].integrals()[0].integrand()))
    assert "f_10[0] * v_0[2] * -1 * v_1[2]" == str(a9_sep._form_unchanged[4].integrals()[0].integrand())


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="10", depends=["9"])
def test_separated_parametrized_forms_mixed_10():
    a10 = (inner(expr3 * expr9 * expr4 * grad(u), grad(v)) * dx + inner(grad(u) * expr6, v) * dx
           + expr5 * inner(u, v) * dx - p * tr(expr7 * grad(v)) * dx - expr5 * q * div(u) * dx - expr6[0] * p * q * dx)
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

    assert ("{ A | A_{i_{274}, i_{275}} = sum_{i_{276}} ({ A | A_{i_{271}, i_{272}} = sum_{i_{273}}"
            + " f_7[i_{271}, i_{273}] * f_13[i_{273}, i_{272}]  })[i_{274}, i_{276}] * f_8[i_{276}, i_{275}]  }"
            == str(a10_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a10_sep._placeholders[0][0]))

    assert "f_80" == str(a10_sep._placeholders[0][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a10_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert ("sum_{i_{286}} sum_{i_{285}} ([{ A | A_{i_{292}} = (grad(v_0))[0, i_{292}] }, { A | A_{i_{291}}"
            + " = (grad(v_0))[1, i_{291}] }])[i_{285}, i_{286}] * ({ A | A_{i_{277}, i_{278}} = sum_{i_{279}}"
            + " ([{ A | A_{i_{294}} = (grad(v_1))[0, i_{294}] }, { A | A_{i_{293}} = (grad(v_1))[1, i_{293}] }])"
            + "[i_{279}, i_{278}] * f_80[i_{277}, i_{279}]  })[i_{285}, i_{286}]  "
            == str(a10_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a10_sep._form_unchanged)))

    assert 5 == len(a10_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a10_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a10_sep._form_unchanged[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a10_sep._form_unchanged[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a10_sep._form_unchanged[3].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a10_sep._form_unchanged[4].integrals()[0].integrand()))

    assert ("sum_{i_{287}} ([v_0[0], v_0[1]])[i_{287}] * ({ A | A_{i_{280}} = sum_{i_{281}} ([{ A | A_{i_{296}}"
            + " = (grad(v_1))[0, i_{296}] }, { A | A_{i_{295}} = (grad(v_1))[1, i_{295}] }])[i_{280}, i_{281}]"
            + " * f_10[i_{281}]  })[i_{287}] "
            == str(a10_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("f_9 * (sum_{i_{288}} ([v_0[0], v_0[1]])[i_{288}] * ([v_1[0], v_1[1]])[i_{288}] )"
            == str(a10_sep._form_unchanged[1].integrals()[0].integrand()))
    assert ("-1 * v_1[2] * (sum_{i_{289}} ({ A | A_{i_{282}, i_{283}} = sum_{i_{284}} ([{ A | A_{i_{298}}"
            + " = (grad(v_0))[0, i_{298}] }, { A | A_{i_{297}} = (grad(v_0))[1, i_{297}] }])[i_{284}, i_{283}]"
            + " * f_11[i_{282}, i_{284}]  })[i_{289}, i_{289}] )"
            == str(a10_sep._form_unchanged[2].integrals()[0].integrand()))
    assert ("v_0[2] * -1 * f_9 * (sum_{i_{290}} ([{ A | A_{i_{300}} = (grad(v_1))[0, i_{300}] }, { A | A_{i_{299}}"
            + " = (grad(v_1))[1, i_{299}] }])[i_{290}, i_{290}] )"
            == str(a10_sep._form_unchanged[3].integrals()[0].integrand()))
    assert "f_10[0] * v_0[2] * -1 * v_1[2]" == str(a10_sep._form_unchanged[4].integrals()[0].integrand())


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="11", depends=["10"])
def test_separated_parametrized_forms_mixed_11():
    a11 = (inner(expr12 * grad(u), grad(v)) * dx + inner(grad(u) * expr11, v) * dx + expr10 * inner(u, v) * dx
           - p * tr(expr12 * grad(v)) * dx - expr10 * q * div(u) * dx - expr11[0] * p * q * dx)
    a11_sep = SeparatedParametrizedForm(a11)
    test_logger.log(DEBUG, "*** ###              FORM 11             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 1, but each term is multiplied by a Function,"
                    + " which is the solution of a parametrized problem")
    a11_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a11_sep.coefficients)))

    assert 6 == len(a11_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a11_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a11_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a11_sep.coefficients[2])))
    test_logger.log(DEBUG, "\t\t" + str(len(a11_sep.coefficients[3])))
    test_logger.log(DEBUG, "\t\t" + str(len(a11_sep.coefficients[4])))
    test_logger.log(DEBUG, "\t\t" + str(len(a11_sep.coefficients[5])))

    assert 1 == len(a11_sep.coefficients[0])
    assert 1 == len(a11_sep.coefficients[1])
    assert 1 == len(a11_sep.coefficients[2])
    assert 1 == len(a11_sep.coefficients[3])
    assert 1 == len(a11_sep.coefficients[4])
    assert 1 == len(a11_sep.coefficients[5])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a11_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep.coefficients[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep.coefficients[3][0]))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep.coefficients[4][0]))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep.coefficients[5][0]))

    assert "f_26" == str(a11_sep.coefficients[0][0])
    assert "f_23" == str(a11_sep.coefficients[1][0])
    assert "f_20" == str(a11_sep.coefficients[2][0])
    assert "f_26" == str(a11_sep.coefficients[3][0])
    assert "-1 * f_20" == str(a11_sep.coefficients[4][0])
    assert "f_23[0]" == str(a11_sep.coefficients[5][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._placeholders[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._placeholders[3][0]))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._placeholders[4][0]))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._placeholders[5][0]))

    assert "f_93" == str(a11_sep._placeholders[0][0])
    assert "f_94" == str(a11_sep._placeholders[1][0])
    assert "f_95" == str(a11_sep._placeholders[2][0])
    assert "f_96" == str(a11_sep._placeholders[3][0])
    assert "f_97" == str(a11_sep._placeholders[4][0])
    assert "f_98" == str(a11_sep._placeholders[5][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._form_with_placeholders[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._form_with_placeholders[3].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._form_with_placeholders[4].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a11_sep._form_with_placeholders[5].integrals()[0].integrand()))

    assert ("sum_{i_{311}} sum_{i_{310}} ([{ A | A_{i_{317}} = (grad(v_0))[0, i_{317}] }, { A | A_{i_{316}}"
            + " = (grad(v_0))[1, i_{316}] }])[i_{310}, i_{311}] * ({ A | A_{i_{302}, i_{303}} = sum_{i_{304}}"
            + " ([{ A | A_{i_{319}} = (grad(v_1))[0, i_{319}] }, { A | A_{i_{318}} = (grad(v_1))[1, i_{318}] }])"
            + "[i_{304}, i_{303}] * f_93[i_{302}, i_{304}]  })[i_{310}, i_{311}]  "
            == str(a11_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("sum_{i_{312}} ([v_0[0], v_0[1]])[i_{312}] * ({ A | A_{i_{305}} = sum_{i_{306}} ([{ A | A_{i_{321}}"
            + " = (grad(v_1))[0, i_{321}] }, { A | A_{i_{320}} = (grad(v_1))[1, i_{320}] }])[i_{305}, i_{306}]"
            + " * f_94[i_{306}]  })[i_{312}] "
            == str(a11_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert ("f_95 * (sum_{i_{313}} ([v_0[0], v_0[1]])[i_{313}] * ([v_1[0], v_1[1]])[i_{313}] )"
            == str(a11_sep._form_with_placeholders[2].integrals()[0].integrand()))
    assert ("-1 * v_1[2] * (sum_{i_{314}} ({ A | A_{i_{307}, i_{308}} = sum_{i_{309}} ([{ A | A_{i_{323}}"
            + " = (grad(v_0))[0, i_{323}] }, { A | A_{i_{322}} = (grad(v_0))[1, i_{322}] }])[i_{309}, i_{308}]"
            + " * f_96[i_{307}, i_{309}]  })[i_{314}, i_{314}] )"
            == str(a11_sep._form_with_placeholders[3].integrals()[0].integrand()))
    assert ("v_0[2] * f_97 * (sum_{i_{315}} ([{ A | A_{i_{325}} = (grad(v_1))[0, i_{325}] }, { A | A_{i_{324}}"
            + " = (grad(v_1))[1, i_{324}] }])[i_{315}, i_{315}] )"
            == str(a11_sep._form_with_placeholders[4].integrals()[0].integrand()))
    assert "f_98 * v_0[2] * -1 * v_1[2]" == str(a11_sep._form_with_placeholders[5].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a11_sep._form_unchanged)))

    assert 0 == len(a11_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="12", depends=["11"])
def test_separated_parametrized_forms_mixed_12():
    a12 = (expr13_split_0_split[0] * scalar_trial * scalar_test * dx
           + expr13_split_0_split[1] * scalar_trial.dx(0) * scalar_test * dx)
    a12_sep = SeparatedParametrizedForm(a12)
    test_logger.log(DEBUG, "*** ###              FORM 12             ### ***")
    test_logger.log(DEBUG, "Test usage of Indexed components of a Function (solution of a parametrized problem)"
                    + " defined on a mixed function space in a form on a scalar function space.")
    a12_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a12_sep.coefficients)))

    assert 2 == len(a12_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a12_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a12_sep.coefficients[1])))

    assert 1 == len(a12_sep.coefficients[0])
    assert 1 == len(a12_sep.coefficients[1])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a12_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a12_sep.coefficients[1][0]))

    assert "f_29[0]" == str(a12_sep.coefficients[0][0])
    assert "f_29[1]" == str(a12_sep.coefficients[1][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a12_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a12_sep._placeholders[1][0]))

    assert "f_99" == str(a12_sep._placeholders[0][0])
    assert "f_100" == str(a12_sep._placeholders[1][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a12_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a12_sep._form_with_placeholders[1].integrals()[0].integrand()))

    assert "v_0 * v_1 * f_99" == str(a12_sep._form_with_placeholders[0].integrals()[0].integrand())
    assert "(grad(v_1))[0] * v_0 * f_100" == str(a12_sep._form_with_placeholders[1].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a12_sep._form_unchanged)))

    assert 0 == len(a12_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="13", depends=["12"])
def test_separated_parametrized_forms_mixed_13():
    a13 = inner(expr13_split[0], grad(scalar_trial)) * scalar_test * dx
    a13_sep = SeparatedParametrizedForm(a13)
    test_logger.log(DEBUG, "*** ###              FORM 13             ### ***")
    test_logger.log(DEBUG, "Test usage of ListTensor components of a Function (solution of a parametrized problem)"
                    + " defined on a mixed function space in a form on a scalar function space.")
    a13_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a13_sep.coefficients)))

    assert 1 == len(a13_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a13_sep.coefficients[0])))

    assert 1 == len(a13_sep.coefficients[0])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a13_sep.coefficients[0][0]))

    assert "[f_29[0], f_29[1]]" == str(a13_sep.coefficients[0][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a13_sep._placeholders[0][0]))

    assert "f_101" == str(a13_sep._placeholders[0][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a13_sep._form_with_placeholders[0].integrals()[0].integrand()))

    assert ("v_0 * (sum_{i_{327}} f_101[i_{327}] * (grad(v_1))[i_{327}] )"
            == str(a13_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a13_sep._form_unchanged)))

    assert 0 == len(a13_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="14", depends=["13"])
def test_separated_parametrized_forms_mixed_14():
    a14 = (inner(expr16 * grad(u), grad(v)) * dx + inner(grad(u) * expr15, v) * dx + expr14 * inner(u, v) * dx
           - p * tr(expr16 * grad(v)) * dx - expr14 * q * div(u) * dx - expr15[0] * p * q * dx)
    a14_sep = SeparatedParametrizedForm(a14)
    test_logger.log(DEBUG, "*** ###              FORM 14             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 11, but each term is multiplied by a Function,"
                    + " which is not the solution of a parametrized problem")
    a14_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a14_sep.coefficients)))

    assert 0 == len(a14_sep.coefficients)
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a14_sep._form_unchanged)))

    assert 6 == len(a14_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a14_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a14_sep._form_unchanged[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a14_sep._form_unchanged[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a14_sep._form_unchanged[3].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a14_sep._form_unchanged[4].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a14_sep._form_unchanged[5].integrals()[0].integrand()))

    assert ("sum_{i_{337}} sum_{i_{336}} ([{ A | A_{i_{343}} = (grad(v_0))[0, i_{343}] }, { A | A_{i_{342}}"
            + " = (grad(v_0))[1, i_{342}] }])[i_{336}, i_{337}] * ({ A | A_{i_{328}, i_{329}} = sum_{i_{330}}"
            + " ([{ A | A_{i_{345}} = (grad(v_1))[0, i_{345}] }, { A | A_{i_{344}} = (grad(v_1))[1, i_{344}] }])"
            + "[i_{330}, i_{329}] * f_38[i_{328}, i_{330}]  })[i_{336}, i_{337}]  "
            == str(a14_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("sum_{i_{338}} ([v_0[0], v_0[1]])[i_{338}] * ({ A | A_{i_{331}} = sum_{i_{332}} ([{ A | A_{i_{347}}"
            + " = (grad(v_1))[0, i_{347}] }, { A | A_{i_{346}} = (grad(v_1))[1, i_{346}] }])[i_{331}, i_{332}]"
            + " * f_35[i_{332}]  })[i_{338}] "
            == str(a14_sep._form_unchanged[1].integrals()[0].integrand()))
    assert ("f_32 * (sum_{i_{339}} ([v_0[0], v_0[1]])[i_{339}] * ([v_1[0], v_1[1]])[i_{339}] )"
            == str(a14_sep._form_unchanged[2].integrals()[0].integrand()))
    assert ("-1 * v_1[2] * (sum_{i_{340}} ({ A | A_{i_{333}, i_{334}} = sum_{i_{335}} ([{ A | A_{i_{349}}"
            + " = (grad(v_0))[0, i_{349}] }, { A | A_{i_{348}} = (grad(v_0))[1, i_{348}] }])[i_{335}, i_{334}]"
            + " * f_38[i_{333}, i_{335}]  })[i_{340}, i_{340}] )"
            == str(a14_sep._form_unchanged[3].integrals()[0].integrand()))
    assert ("v_0[2] * -1 * f_32 * (sum_{i_{341}} ([{ A | A_{i_{351}} = (grad(v_1))[0, i_{351}] },"
            + " { A | A_{i_{350}} = (grad(v_1))[1, i_{350}] }])[i_{341}, i_{341}] )"
            == str(a14_sep._form_unchanged[4].integrals()[0].integrand()))
    assert "f_35[0] * v_0[2] * -1 * v_1[2]" == str(a14_sep._form_unchanged[5].integrals()[0].integrand())


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="15", depends=["14"])
def test_separated_parametrized_forms_mixed_15():
    a15 = (expr17_split_0_split[0] * scalar_trial * scalar_test * dx
           + expr17_split_0_split[1] * scalar_trial.dx(0) * scalar_test * dx)
    a15_sep = SeparatedParametrizedForm(a15)
    test_logger.log(DEBUG, "*** ###              FORM 15             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 12, except that Indexed is obtained using a Function"
                    + " which is not the solution of a parametrized problem")
    a15_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a15_sep.coefficients)))

    assert 0 == len(a15_sep.coefficients)
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a15_sep._form_unchanged)))

    assert 2 == len(a15_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a15_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a15_sep._form_unchanged[1].integrals()[0].integrand()))

    assert "v_0 * f_41[0] * v_1" == str(a15_sep._form_unchanged[0].integrals()[0].integrand())
    assert "(grad(v_1))[0] * f_41[1] * v_0" == str(a15_sep._form_unchanged[1].integrals()[0].integrand())


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="16", depends=["15"])
def test_separated_parametrized_forms_mixed_16():
    a16 = inner(expr17_split[0], grad(scalar_trial)) * scalar_test * dx
    a16_sep = SeparatedParametrizedForm(a16)
    test_logger.log(DEBUG, "*** ###              FORM 16             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 13, except that ListTensor is obtained using a Function"
                    + " which is not the solution of a parametrized problem")
    a16_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a16_sep.coefficients)))

    assert 0 == len(a16_sep.coefficients)
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a16_sep._form_unchanged)))

    assert 1 == len(a16_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a16_sep._form_unchanged[0].integrals()[0].integrand()))

    assert ("v_0 * (sum_{i_{353}} ([f_41[0], f_41[1]])[i_{353}] * (grad(v_1))[i_{353}] )"
            == str(a16_sep._form_unchanged[0].integrals()[0].integrand()))


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="17", depends=["16"])
def test_separated_parametrized_forms_mixed_17():
    a17 = (inner(grad(expr11) * grad(u), grad(v)) * dx + inner(grad(u) * grad(expr10), v) * dx
           + expr10.dx(0) * inner(u, v) * dx - p * tr(grad(expr11) * grad(v)) * dx - div(expr11) * q * div(u) * dx
           - expr10.dx(1) * p * q * dx)
    a17_sep = SeparatedParametrizedForm(a17)
    test_logger.log(DEBUG, "*** ###              FORM 17             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 11, but each term is multiplied by the gradient/"
                    + "partial derivative of a Function, which is the solution of a parametrized problem")
    a17_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a17_sep.coefficients)))

    assert 6 == len(a17_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a17_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a17_sep.coefficients[1])))
    test_logger.log(DEBUG, "\t\t" + str(len(a17_sep.coefficients[2])))
    test_logger.log(DEBUG, "\t\t" + str(len(a17_sep.coefficients[3])))
    test_logger.log(DEBUG, "\t\t" + str(len(a17_sep.coefficients[4])))
    test_logger.log(DEBUG, "\t\t" + str(len(a17_sep.coefficients[5])))

    assert 1 == len(a17_sep.coefficients[0])
    assert 1 == len(a17_sep.coefficients[1])
    assert 1 == len(a17_sep.coefficients[2])
    assert 1 == len(a17_sep.coefficients[3])
    assert 1 == len(a17_sep.coefficients[4])
    assert 1 == len(a17_sep.coefficients[5])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a17_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep.coefficients[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep.coefficients[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep.coefficients[3][0]))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep.coefficients[4][0]))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep.coefficients[5][0]))

    assert "grad(f_23)" == str(a17_sep.coefficients[0][0])
    assert "grad(f_20)" == str(a17_sep.coefficients[1][0])
    assert "(grad(f_20))[0]" == str(a17_sep.coefficients[2][0])
    assert "grad(f_23)" == str(a17_sep.coefficients[3][0])
    assert "grad(f_23)" == str(a17_sep.coefficients[4][0])
    assert "(grad(f_20))[1]" == str(a17_sep.coefficients[5][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._placeholders[1][0]))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._placeholders[2][0]))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._placeholders[3][0]))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._placeholders[4][0]))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._placeholders[5][0]))

    assert "f_102" == str(a17_sep._placeholders[0][0])
    assert "f_103" == str(a17_sep._placeholders[1][0])
    assert "f_104" == str(a17_sep._placeholders[2][0])
    assert "f_105" == str(a17_sep._placeholders[3][0])
    assert "f_106" == str(a17_sep._placeholders[4][0])
    assert "f_107" == str(a17_sep._placeholders[5][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._form_with_placeholders[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._form_with_placeholders[3].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._form_with_placeholders[4].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a17_sep._form_with_placeholders[5].integrals()[0].integrand()))

    assert ("sum_{i_{363}} sum_{i_{362}} ([{ A | A_{i_{370}} = (grad(v_0))[0, i_{370}] }, { A | A_{i_{369}}"
            + " = (grad(v_0))[1, i_{369}] }])[i_{362}, i_{363}] * ({ A | A_{i_{354}, i_{355}} = sum_{i_{356}}"
            + " ([{ A | A_{i_{372}} = (grad(v_1))[0, i_{372}] }, { A | A_{i_{371}} = (grad(v_1))[1, i_{371}] }])"
            + "[i_{356}, i_{355}] * f_102[i_{354}, i_{356}]  })[i_{362}, i_{363}]  "
            == str(a17_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("sum_{i_{364}} ([v_0[0], v_0[1]])[i_{364}] * ({ A | A_{i_{357}} = sum_{i_{358}} ([{ A | A_{i_{374}}"
            + " = (grad(v_1))[0, i_{374}] }, { A | A_{i_{373}} = (grad(v_1))[1, i_{373}] }])[i_{357}, i_{358}]"
            + " * f_103[i_{358}]  })[i_{364}] "
            == str(a17_sep._form_with_placeholders[1].integrals()[0].integrand()))
    assert ("f_104 * (sum_{i_{365}} ([v_0[0], v_0[1]])[i_{365}] * ([v_1[0], v_1[1]])[i_{365}] )"
            == str(a17_sep._form_with_placeholders[2].integrals()[0].integrand()))
    assert ("-1 * v_1[2] * (sum_{i_{366}} ({ A | A_{i_{359}, i_{360}} = sum_{i_{361}} ([{ A | A_{i_{376}}"
            + " = (grad(v_0))[0, i_{376}] }, { A | A_{i_{375}} = (grad(v_0))[1, i_{375}] }])[i_{361}, i_{360}]"
            + " * f_105[i_{359}, i_{361}]  })[i_{366}, i_{366}] )"
            == str(a17_sep._form_with_placeholders[3].integrals()[0].integrand()))
    assert ("v_0[2] * -1 * (sum_{i_{368}} ([{ A | A_{i_{378}} = (grad(v_1))[0, i_{378}] }, { A | A_{i_{377}}"
            + " = (grad(v_1))[1, i_{377}] }])[i_{368}, i_{368}] ) * (sum_{i_{367}} f_106[i_{367}, i_{367}] )"
            == str(a17_sep._form_with_placeholders[4].integrals()[0].integrand()))
    assert "f_107 * v_0[2] * -1 * v_1[2]" == str(a17_sep._form_with_placeholders[5].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a17_sep._form_unchanged)))

    assert 0 == len(a17_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="18", depends=["17"])
def test_separated_parametrized_forms_mixed_18():
    a18 = (inner(grad(expr13_split_0_split[0]), grad(scalar_trial)) * scalar_test * dx
           + expr13_split_0_split[1].dx(0) * scalar_trial.dx(0) * scalar_test * dx)
    a18_sep = SeparatedParametrizedForm(a18)
    test_logger.log(DEBUG, "*** ###              FORM 18             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 12, but each term is multiplied by the gradient/"
                    + "partial derivative of a Function, which is the solution of a parametrized problem")
    a18_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a18_sep.coefficients)))

    assert 2 == len(a18_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a18_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a18_sep.coefficients[1])))

    assert 1 == len(a18_sep.coefficients[0])
    assert 1 == len(a18_sep.coefficients[1])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a18_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a18_sep.coefficients[1][0]))

    assert "grad(f_29)" == str(a18_sep.coefficients[0][0])
    assert "(grad(f_29))[1, 0]" == str(a18_sep.coefficients[1][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a18_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a18_sep._placeholders[1][0]))

    assert "f_108" == str(a18_sep._placeholders[0][0])
    assert "f_109" == str(a18_sep._placeholders[1][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a18_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a18_sep._form_with_placeholders[1].integrals()[0].integrand()))

    assert ("v_0 * (sum_{i_{381}} f_108[0, i_{381}] * (grad(v_1))[i_{381}] )"
            == str(a18_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert "v_0 * (grad(v_1))[0] * f_109" == str(a18_sep._form_with_placeholders[1].integrals()[0].integrand())
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a18_sep._form_unchanged)))

    assert 0 == len(a18_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="19", depends=["18"])
def test_separated_parametrized_forms_mixed_19():
    a19 = (inner(grad(expr13_split[0]) * grad(scalar_trial), grad(scalar_test)) * dx
           + inner(expr13_split[0].dx(0), grad(scalar_trial)) * scalar_test * dx)
    a19_sep = SeparatedParametrizedForm(a19)
    test_logger.log(DEBUG, "*** ###              FORM 19             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 13, but each term is multiplied by the gradient/"
                    + "partial derivative of a Function, which is the solution of a parametrized problem")
    a19_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a19_sep.coefficients)))

    assert 2 == len(a19_sep.coefficients)
    test_logger.log(DEBUG, "\tSublen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a19_sep.coefficients[0])))
    test_logger.log(DEBUG, "\t\t" + str(len(a19_sep.coefficients[1])))

    assert 1 == len(a19_sep.coefficients[0])
    assert 1 == len(a19_sep.coefficients[1])
    test_logger.log(DEBUG, "\tCoefficients:")
    test_logger.log(DEBUG, "\t\t" + str(a19_sep.coefficients[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a19_sep.coefficients[1][0]))

    assert "grad(f_29)" == str(a19_sep.coefficients[0][0])
    assert "grad(f_29)" == str(a19_sep.coefficients[1][0])
    test_logger.log(DEBUG, "\tPlaceholders:")
    test_logger.log(DEBUG, "\t\t" + str(a19_sep._placeholders[0][0]))
    test_logger.log(DEBUG, "\t\t" + str(a19_sep._placeholders[1][0]))

    assert "f_110" == str(a19_sep._placeholders[0][0])
    assert "f_111" == str(a19_sep._placeholders[1][0])
    test_logger.log(DEBUG, "\tForms with placeholders:")
    test_logger.log(DEBUG, "\t\t" + str(a19_sep._form_with_placeholders[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a19_sep._form_with_placeholders[1].integrals()[0].integrand()))

    assert ("sum_{i_{387}} ({ A | A_{i_{384}} = sum_{i_{385}} ([{ A | A_{i_{390}} = f_110[0, i_{390}] },"
            + " { A | A_{i_{389}} = f_110[1, i_{389}] }])[i_{384}, i_{385}] * (grad(v_1))[i_{385}]  })[i_{387}]"
            + " * (grad(v_0))[i_{387}] "
            == str(a19_sep._form_with_placeholders[0].integrals()[0].integrand()))
    assert ("v_0 * (sum_{i_{388}} ([{ A | A_{i_{392}} = f_111[0, i_{392}] }, { A | A_{i_{391}}"
            + " = f_111[1, i_{391}] }])[i_{388}, 0] * (grad(v_1))[i_{388}] )"
            == str(a19_sep._form_with_placeholders[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a19_sep._form_unchanged)))

    assert 0 == len(a19_sep._form_unchanged)


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="20", depends=["19"])
def test_separated_parametrized_forms_mixed_20():
    a20 = (inner(grad(expr15) * grad(u), grad(v)) * dx + inner(grad(u) * grad(expr14), v) * dx
           + expr14.dx(0) * inner(u, v) * dx - p * tr(grad(expr15) * grad(v)) * dx - div(expr15) * q * div(u) * dx
           - expr14.dx(1) * p * q * dx)
    a20_sep = SeparatedParametrizedForm(a20)
    test_logger.log(DEBUG, "*** ###              FORM 20             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 14, but each term is multiplied by the gradient/"
                    + "partial derivative of a Function, which is not the solution of a parametrized problem")
    a20_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a20_sep.coefficients)))

    assert 0 == len(a20_sep.coefficients)
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a20_sep._form_unchanged)))

    assert 6 == len(a20_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a20_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a20_sep._form_unchanged[1].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a20_sep._form_unchanged[2].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a20_sep._form_unchanged[3].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a20_sep._form_unchanged[4].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a20_sep._form_unchanged[5].integrals()[0].integrand()))

    assert ("sum_{i_{402}} sum_{i_{401}} ([{ A | A_{i_{409}} = (grad(v_0))[0, i_{409}] }, { A | A_{i_{408}}"
            + " = (grad(v_0))[1, i_{408}] }])[i_{401}, i_{402}] * ({ A | A_{i_{393}, i_{394}} = sum_{i_{395}}"
            + " ([{ A | A_{i_{411}} = (grad(v_1))[0, i_{411}] }, { A | A_{i_{410}} = (grad(v_1))[1, i_{410}] }])"
            + "[i_{395}, i_{394}] * (grad(f_35))[i_{393}, i_{395}]  })[i_{401}, i_{402}]  "
            == str(a20_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("sum_{i_{403}} ([v_0[0], v_0[1]])[i_{403}] * ({ A | A_{i_{396}} = sum_{i_{397}} ([{ A | A_{i_{413}}"
            + " = (grad(v_1))[0, i_{413}] }, { A | A_{i_{412}} = (grad(v_1))[1, i_{412}] }])[i_{396}, i_{397}]"
            + " * (grad(f_32))[i_{397}]  })[i_{403}] "
            == str(a20_sep._form_unchanged[1].integrals()[0].integrand()))
    assert ("(grad(f_32))[0] * (sum_{i_{404}} ([v_0[0], v_0[1]])[i_{404}] * ([v_1[0], v_1[1]])[i_{404}] )"
            == str(a20_sep._form_unchanged[2].integrals()[0].integrand()))
    assert ("-1 * v_1[2] * (sum_{i_{405}} ({ A | A_{i_{398}, i_{399}} = sum_{i_{400}} ([{ A | A_{i_{415}}"
            + " = (grad(v_0))[0, i_{415}] }, { A | A_{i_{414}} = (grad(v_0))[1, i_{414}] }])[i_{400}, i_{399}]"
            + " * (grad(f_35))[i_{398}, i_{400}]  })[i_{405}, i_{405}] )"
            == str(a20_sep._form_unchanged[3].integrals()[0].integrand()))
    assert ("v_0[2] * -1 * (sum_{i_{407}} ([{ A | A_{i_{417}} = (grad(v_1))[0, i_{417}] }, { A | A_{i_{416}}"
            + " = (grad(v_1))[1, i_{416}] }])[i_{407}, i_{407}] ) * (sum_{i_{406}} (grad(f_35))[i_{406}, i_{406}] )"
            == str(a20_sep._form_unchanged[4].integrals()[0].integrand()))
    assert "(grad(f_32))[1] * v_0[2] * -1 * v_1[2]" == str(a20_sep._form_unchanged[5].integrals()[0].integrand())


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="21", depends=["20"])
def test_separated_parametrized_forms_mixed_21():
    a21 = (inner(grad(expr17_split_0_split[0]), grad(scalar_trial)) * scalar_test * dx
           + expr17_split_0_split[1].dx(0) * scalar_trial.dx(0) * scalar_test * dx)
    a21_sep = SeparatedParametrizedForm(a21)
    test_logger.log(DEBUG, "*** ###              FORM 21             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 15, but each term is multiplied by the gradient/"
                    + "partial derivative of a Function, which is not the solution of a parametrized problem")
    a21_sep.separate()
    test_logger.log(DEBUG, "\tLen coefficients:")
    test_logger.log(DEBUG, "\t\t" + str(len(a21_sep.coefficients)))

    assert 0 == len(a21_sep.coefficients)
    test_logger.log(DEBUG, "\tLen unchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(len(a21_sep._form_unchanged)))

    assert 2 == len(a21_sep._form_unchanged)
    test_logger.log(DEBUG, "\tUnchanged forms:")
    test_logger.log(DEBUG, "\t\t" + str(a21_sep._form_unchanged[0].integrals()[0].integrand()))
    test_logger.log(DEBUG, "\t\t" + str(a21_sep._form_unchanged[1].integrals()[0].integrand()))

    assert ("v_0 * (sum_{i_{420}} (grad(f_41))[0, i_{420}] * (grad(v_1))[i_{420}] )"
            == str(a21_sep._form_unchanged[0].integrals()[0].integrand()))
    assert "v_0 * (grad(v_1))[0] * (grad(f_41))[1, 0]" == str(a21_sep._form_unchanged[1].integrals()[0].integrand())


@skip_in_parallel
@enable_separated_parametrized_form_logging
@pytest.mark.dependency(name="22", depends=["21"])
def test_separated_parametrized_forms_mixed_22():
    a22 = (inner(grad(expr17_split[0]) * grad(scalar_trial), grad(scalar_test)) * dx
           + inner(expr17_split[0].dx(0), grad(scalar_trial)) * scalar_test * dx)
    a22_sep = SeparatedParametrizedForm(a22)
    test_logger.log(DEBUG, "*** ###              FORM 22             ### ***")
    test_logger.log(DEBUG, "This form is similar to form 16, but each term is multiplied by the gradient/"
                    + "partial derivative of a Function, which is not the solution of a parametrized problem")
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

    assert ("sum_{i_{426}} ({ A | A_{i_{423}} = sum_{i_{424}} ([{ A | A_{i_{429}} = (grad(f_41))[0, i_{429}] },"
            + " { A | A_{i_{428}} = (grad(f_41))[1, i_{428}] }])[i_{423}, i_{424}] * (grad(v_1))[i_{424}]  })"
            + "[i_{426}] * (grad(v_0))[i_{426}] "
            == str(a22_sep._form_unchanged[0].integrals()[0].integrand()))
    assert ("v_0 * (sum_{i_{427}} ([{ A | A_{i_{431}} = (grad(f_41))[0, i_{431}] }, { A | A_{i_{430}}"
            + " = (grad(f_41))[1, i_{430}] }])[i_{427}, 0] * (grad(v_1))[i_{427}] )"
            == str(a22_sep._form_unchanged[1].integrals()[0].integrand()))
