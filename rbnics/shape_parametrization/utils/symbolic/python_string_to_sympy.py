# Copyright (C) 2015-2019 by the RBniCS authors
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

from sympy import ImmutableMatrix, Matrix, MatrixSymbol, symbols, sympify, zeros
from rbnics.problems.base import ParametrizedProblem
from rbnics.utils.decorators import overload, tuple_of
from rbnics.shape_parametrization.utils.symbolic.sympy_symbolic_coordinates import sympy_symbolic_coordinates

def MatrixListSymbol(prefix, dim, one):
    assert one == 1
    return Matrix([symbols(prefix + "[" + str(i) + "]") for i in range(dim)])

@overload
def python_string_to_sympy(string_expression: str, problem: ParametrizedProblem):
    """
    Convert a string (with math python syntax, e.g. **2 instead of pow(., 2)) to sympy
    """
    x_symb = sympy_symbolic_coordinates(problem.V.mesh().geometry().dim(), MatrixListSymbol)
    mu_symb = MatrixListSymbol("mu", len(problem.mu), 1)
    return python_string_to_sympy(string_expression, x_symb, mu_symb)
    
@overload
def python_string_to_sympy(string_expression: str, x_symb: (Matrix, MatrixSymbol, None), mu_symb: (Matrix, MatrixSymbol, None)):
    return sympify(string_expression, locals={"x": x_symb, "mu": mu_symb})

@overload
def python_string_to_sympy(string_expression: tuple_of(str), problem: ParametrizedProblem):
    """
    Convert a vector of strings (with math python syntax, e.g. **2 instead of pow(., 2)) to sympy
    """
    x_symb = sympy_symbolic_coordinates(problem.V.mesh().geometry().dim(), MatrixListSymbol)
    mu_symb = MatrixListSymbol("mu", len(problem.mu), 1)
    return python_string_to_sympy(string_expression, x_symb, mu_symb)
    
@overload
def python_string_to_sympy(string_expression: tuple_of(str), x_symb: (Matrix, MatrixSymbol, None), mu_symb: (Matrix, MatrixSymbol, None)):
    sympy_expression = zeros(len(string_expression), 1)
    for (i, si) in enumerate(string_expression):
        sympy_expression[i] = sympify(si, locals={"x": x_symb, "mu": mu_symb})
    return ImmutableMatrix(sympy_expression)
    
@overload
def python_string_to_sympy(string_expression: tuple_of(tuple_of(str)), problem: ParametrizedProblem):
    """
    Convert a matrix of strings (with math python syntax, e.g. **2 instead of pow(., 2)) to sympy
    """
    x_symb = sympy_symbolic_coordinates(problem.V.mesh().geometry().dim(), MatrixListSymbol)
    mu_symb = MatrixListSymbol("mu", len(problem.mu), 1)
    return python_string_to_sympy(string_expression, x_symb, mu_symb)
    
@overload
def python_string_to_sympy(string_expression: tuple_of(tuple_of(str)), x_symb: (Matrix, MatrixSymbol, None), mu_symb: (Matrix, MatrixSymbol, None)):
    assert all([len(si) == len(string_expression[0]) for si in string_expression[1:]])
    sympy_expression = zeros(len(string_expression), len(string_expression[0]))
    for (i, si) in enumerate(string_expression):
        for (j, sij) in enumerate(si):
            sympy_expression[i, j] = sympify(sij, locals={"x": x_symb, "mu": mu_symb})
    return ImmutableMatrix(sympy_expression)
