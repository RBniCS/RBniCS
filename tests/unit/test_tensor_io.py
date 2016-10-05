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
## @file 
#  @brief 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import os
from numpy import isclose
from dolfin import *
from RBniCS.backends.fenics import evaluate, ProjectedParametrizedTensor
from RBniCS.backends.fenics.wrapping import tensor_load, tensor_save

# Keep an override of tensor save handy that disables saving.
# This override can be used to check import with different
# number of processors
"""
def tensor_save(tensor, directory, filename):
    pass
"""

# Make output directory, if necessary
try: 
    os.makedirs("test_tensor_io.output_dir")
except OSError:
    if not os.path.isdir("test_tensor_io.output_dir"):
        raise

mesh = UnitSquareMesh(10, 10)

# ~~~ Elliptic case ~~~ #
V = FunctionSpace(mesh, "Lagrange", 2)
u = TrialFunction(V)
v = TestFunction(V)

form_1 = v*dx
projected_tensor_1 = ProjectedParametrizedTensor(form_1, V)
evaluate_1 = evaluate(projected_tensor_1)

tensor_save(evaluate_1, "test_tensor_io.output_dir", "test_tensor_io_1_elliptic")
evaluate_1_read = tensor_load("test_tensor_io.output_dir", "test_tensor_io_1_elliptic", V)

assert isclose(evaluate_1.array(), evaluate_1_read.array()).all()

form_2 = u*v*dx
projected_tensor_2 = ProjectedParametrizedTensor(form_2, V)
evaluate_2 = evaluate(projected_tensor_2)

tensor_save(evaluate_2, "test_tensor_io.output_dir", "test_tensor_io_2_elliptic")
evaluate_2_read = tensor_load("test_tensor_io.output_dir", "test_tensor_io_2_elliptic", V)

assert isclose(evaluate_2.array(), evaluate_2_read.array()).all()

# ~~~ Mixed case ~~~ #
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element)
u = TrialFunction(V)
v = TestFunction(V)
(u_0, u_1) = split(u)
(v_0, v_1) = split(v)

form_1 = v_0[0]*dx + v_0[1]*dx + v_1*dx
projected_tensor_1 = ProjectedParametrizedTensor(form_1, V)
evaluate_1 = evaluate(projected_tensor_1)

tensor_save(evaluate_1, "test_tensor_io.output_dir", "test_tensor_io_1_mixed")
evaluate_1_read = tensor_load("test_tensor_io.output_dir", "test_tensor_io_1_mixed", V)

assert isclose(evaluate_1.array(), evaluate_1_read.array()).all()

form_2 = inner(u_0, v_0)*dx + u_1*v_1*dx + u_0[0]*v_1*dx + u_1*v_0[1]*dx
projected_tensor_2 = ProjectedParametrizedTensor(form_2, V)
evaluate_2 = evaluate(projected_tensor_2)

tensor_save(evaluate_2, "test_tensor_io.output_dir", "test_tensor_io_2_mixed")
evaluate_2_read = tensor_load("test_tensor_io.output_dir", "test_tensor_io_2_mixed", V)

assert isclose(evaluate_2.array(), evaluate_2_read.array()).all()
