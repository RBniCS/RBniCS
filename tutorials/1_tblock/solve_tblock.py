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
## @file solve_tblock.py
#  @brief Example 1: thermal block test case
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
from elliptic_coercive_base import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 1: THERMAL BLOCK CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
class Tblock(EllipticCoerciveBase):
    
    ## Default initialization of members
    def __init__(self, V, subd, bound):
        EllipticCoerciveBase.__init__(self, V)
        self.dx = Measure("dx")[subd]
        self.ds = Measure("ds")[bound]
        self.bc = DirichletBC(V, 0.0, bound, 3)
    
    ## Return the alpha_lower bound.
    def get_alpha_lb(self):
        return 0.1
    
    ## Set theta multiplicative terms of the affine expansion of a.
    def compute_theta_a(self):
        mu1 = self.mu[0]
        mu2 = self.mu[1]
        theta_a0 = mu1
        theta_a1 = 1.
        self.theta_a = (theta_a0, theta_a1)
    
    ## Set theta multiplicative terms of the affine expansion of f.
    def compute_theta_f(self):
        self.theta_f = (self.mu[1],)
    
    ## Set matrices resulting from the truth discretization of a.
    def assemble_truth_a(self):
        u = self.u
        v = self.v
        dx = self.dx
        # Assemble A0
        a0 = inner(grad(u),grad(v))*dx(1) +1e-15*inner(u,v)*dx
        A0 = assemble(a0)
        # Assemble A1
        a1 = inner(grad(u),grad(v))*dx(2) +1e-15*inner(u,v)*dx
        A1 = assemble(a1)
        # Apply BCs
        self.bc.apply(A0)
        self.bc.apply(A1)
        self.bc.zero(A1)
        # Save
        self.A_vec = (A0, A1)
    
    ## Set vectors resulting from the truth discretization of f.
    def assemble_truth_f(self):
        v = self.v
        dx = self.dx
        ds = self.ds
        # Assemble F0
        f0 = v*ds(1) + 1e-15*v*dx
        F0 = assemble(f0)
        # Apply BCs
        self.bc.apply(F0)
        # Save
        self.F_vec = (F0,)

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 1: MAIN PROGRAM     ~~~~~~~~~~~~~~~~~~~~~~~~~# 

# 1. Read the mesh for this problem
mesh = Mesh("data/tblock.xml")
subd = MeshFunction("size_t", mesh, "data/tblock_physical_region.xml")
bound = MeshFunction("size_t", mesh, "data/tblock_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the Thermal Block class
tb = Tblock(V, subd, bound)

# 4. Choose PETSc solvers as linear algebra backend
parameters.linear_algebra_backend = 'PETSc'

# 5. Set mu range, eta_train and Nmax
mu_range = [(0.1, 10.0), (-1.0, 1.0)]
tb.setmu_range(mu_range)
tb.seteta_train(1000)
tb.setNmax(4)

# 6. Perform the offline phase
first_mu = (0.5,1.0)
tb.setmu(first_mu)
tb.offline()

# 7. Perform an online solve
mu = (0.3,-1.0)
tb.online_solve(mu)
