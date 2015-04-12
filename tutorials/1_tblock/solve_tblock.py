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
from elliptic_coercive_rb_base import *
#from elliptic_coercive_pod_base import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 1: THERMAL BLOCK CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
class Tblock(EllipticCoerciveRBBase):
#class Tblock(EllipticCoercivePODBase):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, subd, bound):
        # Call the standard initialization
        EllipticCoerciveRBBase.__init__(self, V)
#        EllipticCoercivePODBase.__init__(self, V)
        # ... and also store FEniCS data structures for assembly
        self.dx = Measure("dx")[subd]
        self.ds = Measure("ds")[bound]
        self.bc = DirichletBC(V, 0.0, bound, 3)
        # Use the H^1 seminorm on V as norm, instead of the H^1 norm
        u = self.u
        v = self.v
        dx = self.dx
        scalar = inner(grad(u),grad(v))*dx
        self.S = assemble(scalar)
        self.bc.apply(self.S)
    
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return the alpha_lower bound.
    def get_alpha_lb(self):
        return min(self.compute_theta_a())
    
    ## Set theta multiplicative terms of the affine expansion of a.
    def compute_theta_a(self):
        mu1 = self.mu[0]
        mu2 = self.mu[1]
        theta_a0 = mu1
        theta_a1 = 1.
        return (theta_a0, theta_a1)
    
    ## Set theta multiplicative terms of the affine expansion of f.
    def compute_theta_f(self):
        return (self.mu[1],)
    
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
        # Return
        return (A0, A1)
    
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
        # Return
        return (F0,)
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 

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

# 5. Set mu range, xi_train and Nmax
mu_range = [(0.1, 10.0), (-1.0, 1.0)]
tb.setmu_range(mu_range)
tb.setxi_train(500)
tb.setNmax(6)

# 6. Perform the offline phase
first_mu = (0.5,1.0)
tb.setmu(first_mu)
tb.offline()

# 7. Perform an online solve
online_mu = (8.,-1.0)
tb.setmu(online_mu)
tb.online_solve()

# 8. Perform an error analysis
tb.error_analysis()
