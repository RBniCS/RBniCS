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

from dolfin import *
set_log_level(PROGRESS)
from fenicstools import DofMapPlotter
from RBniCS.backends.fenics import ReducedMesh
from RBniCS.backends.fenics.evaluate import evaluate_and_vectorize_sparse_matrix_at_dofs

mesh = UnitSquareMesh(3, 3)

V = FunctionSpace(mesh, "CG", 2)

"""
dmp = DofMapPlotter(V)
dmp.plot()
dmp.show()
"""

reduced_mesh = ReducedMesh(V)
dofs = [(1, 2), (11, 12), (48, 12), (41, 41)]

for pair in dofs:
    log(PROGRESS, "Adding " + str(pair))
    reduced_mesh.append(pair)
    
    """
    plot(reduced_mesh.reduced_mesh_cells_marker)
    interactive()
    if reduced_mesh.get_reduced_mesh() is not None:
        plot(reduced_mesh.get_reduced_mesh())
        interactive()
    """
    """
    if reduced_mesh.get_reduced_function_space() is not None:
        dmp = DofMapPlotter(reduced_mesh.get_reduced_function_space())
        dmp.plot()
        dmp.show()
    """

reduced_V = reduced_mesh.get_reduced_function_space()
reduced_dofs = reduced_mesh.get_reduced_dofs_list()


u = TrialFunction(V)
v = TestFunction(V)

u_N = TrialFunction(reduced_V)
v_N = TestFunction(reduced_V)

A = assemble((u.dx(0)*v + u*v)*dx)
A_N = assemble((u_N.dx(0)*v_N + u_N*v_N)*dx)

A_dofs = evaluate_and_vectorize_sparse_matrix_at_dofs(A, dofs)
A_N_reduced_dofs = evaluate_and_vectorize_sparse_matrix_at_dofs(A_N, reduced_dofs)

log(PROGRESS, "A at dofs:\n" + str(A_dofs))
log(PROGRESS, "A_N at reduced dofs:\n" + str(A_N_reduced_dofs))
log(PROGRESS, "Error:\n" + str(A_dofs - A_N_reduced_dofs))

