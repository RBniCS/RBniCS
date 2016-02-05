# Copyright (C) 2015-2016 by the RBniCS authors
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
from __future__ import print_function
import glpk

print("Check if glpk is correctly installed by solving a simple linear program.")

lp = glpk.glp_create_prob()
glpk.glp_set_obj_dir(lp, glpk.GLP_MIN)
glpk.glp_add_cols(lp, 2) # two unknowns: x and y
glpk.glp_set_col_bnds(lp, 1, glpk.GLP_DB, 0., 1.) # 0 <= x <= 1
glpk.glp_set_col_bnds(lp, 2, glpk.GLP_DB, 0., 1.) # 0 <= y <= 1
glpk.glp_add_rows(lp, 2) # 2 constraints
matrix_row_index = glpk.intArray(5)
matrix_column_index = glpk.intArray(5)
matrix_content = glpk.doubleArray(5)
# First constraint: x + y >= 1.
matrix_row_index[1] = 1; matrix_column_index[1] = 1; matrix_content[1] = 1.;
matrix_row_index[2] = 1; matrix_column_index[2] = 2; matrix_content[2] = 1.;
glpk.glp_set_row_bnds(lp, 1, glpk.GLP_LO, 1., 0.)
# Second constraint: - x + y >= -0.5.
matrix_row_index[3] = 2; matrix_column_index[3] = 1; matrix_content[3] = -1.;
matrix_row_index[4] = 2; matrix_column_index[4] = 2; matrix_content[4] = 1.;
glpk.glp_set_row_bnds(lp, 2, glpk.GLP_LO, -0.5, 0.)
# Load them
glpk.glp_load_matrix(lp, 4, matrix_row_index, matrix_column_index, matrix_content)
# Cost function: 0.5 x + y
glpk.glp_set_obj_coef(lp, 1, 0.5)
glpk.glp_set_obj_coef(lp, 2, 1.)
# Solve the linear programming problem
options = glpk.glp_smcp()
glpk.glp_init_smcp(options)
options.msg_lev = glpk.GLP_MSG_ERR
options.meth = glpk.GLP_DUAL
glpk.glp_simplex(lp, options)
print("Computed optimum: x =", glpk.glp_get_col_prim(lp, 1), ", y =", glpk.glp_get_col_prim(lp, 2) ,", obj =", glpk.glp_get_obj_val(lp))
print("Expected optimum: x = 0.75, y = 0.25, obj = 0.625")
glpk.glp_delete_prob(lp)

