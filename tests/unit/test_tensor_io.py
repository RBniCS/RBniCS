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
## @file 
#  @brief 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import os
from numpy import isclose
from dolfin import *
from rbnics.backends.fenics import evaluate, ParametrizedTensorFactory
from rbnics.backends.fenics.wrapping import tensor_load, tensor_save

# Possibly disable saving to file. This bool can be used to 
# check loading with a number of processors different from
# the one originally used when saving.
skip_tensor_save = False

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
tensor_1 = ParametrizedTensorFactory(form_1)
evaluate_1 = evaluate(tensor_1)

if not skip_tensor_save:
    tensor_save(evaluate_1, "test_tensor_io.output_dir", "test_tensor_io_1_elliptic")
evaluate_1_read = tensor_load("test_tensor_io.output_dir", "test_tensor_io_1_elliptic", (V, ))

assert isclose(evaluate_1.array(), evaluate_1_read.array()).all()

form_2 = u*v*dx
tensor_2 = ParametrizedTensorFactory(form_2)
evaluate_2 = evaluate(tensor_2)

if not skip_tensor_save:
    tensor_save(evaluate_2, "test_tensor_io.output_dir", "test_tensor_io_2_elliptic")
evaluate_2_read = tensor_load("test_tensor_io.output_dir", "test_tensor_io_2_elliptic", (V, V))

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
tensor_1 = ParametrizedTensorFactory(form_1)
evaluate_1 = evaluate(tensor_1)

if not skip_tensor_save:
    tensor_save(evaluate_1, "test_tensor_io.output_dir", "test_tensor_io_1_mixed")
evaluate_1_read = tensor_load("test_tensor_io.output_dir", "test_tensor_io_1_mixed", (V, ))

assert isclose(evaluate_1.array(), evaluate_1_read.array()).all()

form_2 = inner(u_0, v_0)*dx + u_1*v_1*dx + u_0[0]*v_1*dx + u_1*v_0[1]*dx
tensor_2 = ParametrizedTensorFactory(form_2)
evaluate_2 = evaluate(tensor_2)

if not skip_tensor_save:
    tensor_save(evaluate_2, "test_tensor_io.output_dir", "test_tensor_io_2_mixed")
evaluate_2_read = tensor_load("test_tensor_io.output_dir", "test_tensor_io_2_mixed", (V, V))

assert isclose(evaluate_2.array(), evaluate_2_read.array()).all()

# ~~~ Collapsed case ~~~ #
U = FunctionSpace(mesh, element)
V = U.sub(0).collapse()
u = TrialFunction(U)
(u_0, u_1) = split(u)
v = TestFunction(V)

form_1 = v[0]*dx + v[1]*dx
tensor_1 = ParametrizedTensorFactory(form_1)
evaluate_1 = evaluate(tensor_1)

if not skip_tensor_save:
    tensor_save(evaluate_1, "test_tensor_io.output_dir", "test_tensor_io_1_collapsed")
evaluate_1_read = tensor_load("test_tensor_io.output_dir", "test_tensor_io_1_collapsed", (V, ))

assert isclose(evaluate_1.array(), evaluate_1_read.array()).all()

form_2 = inner(u_0, v)*dx + u_1*v[0]*dx
tensor_2 = ParametrizedTensorFactory(form_2)
evaluate_2 = evaluate(tensor_2)

if not skip_tensor_save:
    tensor_save(evaluate_2, "test_tensor_io.output_dir", "test_tensor_io_2_collapsed")
evaluate_2_read = tensor_load("test_tensor_io.output_dir", "test_tensor_io_2_collapsed", (V, U))

assert isclose(evaluate_2.array(), evaluate_2_read.array()).all()

