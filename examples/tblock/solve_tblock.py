# Copyright (C) 2015 SISSA mathLab
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
from elliptic_coercive_base import *

class Tblock(EllipticCoerciveBase):

    def get_alpha_lb(self):
        return 0.1

    def compute_theta_a(self):
        mu1 = self.mu[0]
        mu2 = self.mu[1]
        theta_a0 = mu1
        theta_a1 = 1.
        self.theta_a = (theta_a0, theta_a1)

    def compute_theta_f(self):
        self.theta_f = (self.mu[1],)


parameters.linear_algebra_backend = 'PETSc'

# Create mesh and define function space
mesh = Mesh("tblock.xml")
subd = MeshFunction("size_t", mesh, "tblock_physical_region.xml")
bound = MeshFunction("size_t", mesh, "tblock_facet_region.xml")



V = FunctionSpace(mesh, "Lagrange", 1)

tb = Tblock(V)

# 
# Define new measures associated with the interior domains and
# exterior boundaries
dx = Measure("dx")[subd]
ds = Measure("ds")[bound]
# 
# mu1 = 0.3
# mu2 = 2.0
# mur = 2.0
# mu3 = 5.0
# 
# 
# 
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

a0 = inner(grad(u),grad(v))*dx(1) +1e-15*inner(u,v)*dx
A0 = assemble(a0)

a1 = inner(grad(u),grad(v))*dx(2) +1e-15*inner(u,v)*dx
A1 = assemble(a1)



f0 = v*ds(1) + 1e-15*v*dx
F0 = assemble(f0)
out0 = v*ds(1)
Out0 = assemble(out0)

bc = DirichletBC(V, 0.0, bound, 3)

A_vec_no_bc = (A0, A1)
bc.apply(A0)
bc.apply(A1)
bc.zero(A1)
bc.apply(F0)


A_vec = (A0, A1)
F_vec = (F0,)

tb.setA_vec(A_vec)
tb.setF_vec(F_vec)

mu_range = [(0.1, 10.0), (-1.0, 1.0)]
tb.setmu_range(mu_range)
tb.settheta_train(1000)
tb.setNmax(4)
first_mu = (0.5,1.0)
tb.setmu(first_mu)
#tb.offline()
mu = (0.3,-1.0)
tb.online_solve(mu)
