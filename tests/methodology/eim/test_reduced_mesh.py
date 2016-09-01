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
from fenicstools import DofMapPlotter
from RBniCS.backends.fenics import ReducedMesh

mesh = UnitSquareMesh(3, 3)

V = FunctionSpace(mesh, "CG", 2)

dmp = DofMapPlotter(V)
dmp.plot()
dmp.show()

reduced_mesh = ReducedMesh(V)
dofs = [(1, 2), (11, 12)]

reduced_mesh.add_dofs(dofs[0])
reduced_mesh.add_dofs(dofs[1])

(reduced_V, reduced_dofs) = reduced_mesh[:1]
dmp = DofMapPlotter(reduced_V)
dmp.plot()
dmp.show()

u = TrialFunction(V)
v = TestFunction(V)

u_N = TrialFunction(reduced_V)
v_N = TestFunction(reduced_V)

A = assemble(u*v*dx)
A_N = assemble(u_N*v_N*dx)

print A.array().shape
for dof in dofs:
    print A.array()[dof],
print 

print A_N.array().shape
for reduced_dof in reduced_dofs:
    print A_N.array()[reduced_dof],
print
