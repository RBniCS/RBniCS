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
## @file solve_elast.py
#  @brief Example 2: elastic block test case
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
from RBniCS import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 2: ELASTIC BLOCK CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
class Eblock(EllipticCoercivePODBase):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, subd, bound):
        bc = DirichletBC(V, (0.0, 0.0), bound, 6)
        # Call the standard initialization
        super(Eblock, self).__init__(V, [bc])
        # ... and also store FEniCS data structures for assembly
        self.dx = Measure("dx")(subdomain_data=subd)
        self.ds = Measure("ds")(subdomain_data=bound)
        # ...
        self.f = Constant((1.0, 0.0))
        self.E  = 1.0
        self.nu = 0.3
        self.lambda_1 = self.E*self.nu / ((1.0 + self.nu)*(1.0 - 2.0*self.nu))
        self.lambda_2 = self.E / (2.0*(1.0 + self.nu))
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Set theta multiplicative terms of the affine expansion of a.
    def compute_theta_a(self):
        mu = self.mu
        mu1 = mu[0]
        mu2 = mu[1]
        mu3 = mu[2]
        mu4 = mu[3]
        mu5 = mu[4]
        mu6 = mu[5]
        mu7 = mu[6]
        mu8 = mu[7]
        theta_a0 = mu1
        theta_a1 = mu2
        theta_a2 = mu3
        theta_a3 = mu4
        theta_a4 = mu5
        theta_a5 = mu6
        theta_a6 = mu7
        theta_a7 = mu8
        theta_a8 = 1.
        return (theta_a0, theta_a1 ,theta_a2 ,theta_a3 ,theta_a4 ,theta_a5 ,theta_a6 ,theta_a7 ,theta_a8)
    
    ## Set theta multiplicative terms of the affine expansion of f.
    def compute_theta_f(self):
        mu = self.mu
        mu9 = mu[8]
        mu10 = mu[9]
        mu11 = mu[10]
        theta_f0 = mu9
        theta_f1 = mu10
        theta_f2 = mu11
        return (theta_f0, theta_f1, theta_f2)
    
    ## Set matrices resulting from the truth discretization of a.
    def assemble_truth_a(self):
        u = self.u
        v = self.v
        dx = self.dx
        # Define
        a0 = self.elasticity(u,v)*dx(1) +1e-15*inner(u,v)*dx
        a1 = self.elasticity(u,v)*dx(2) +1e-15*inner(u,v)*dx
        a2 = self.elasticity(u,v)*dx(3) +1e-15*inner(u,v)*dx
        a3 = self.elasticity(u,v)*dx(4) +1e-15*inner(u,v)*dx
        a4 = self.elasticity(u,v)*dx(5) +1e-15*inner(u,v)*dx
        a5 = self.elasticity(u,v)*dx(6) +1e-15*inner(u,v)*dx
        a6 = self.elasticity(u,v)*dx(7) +1e-15*inner(u,v)*dx
        a7 = self.elasticity(u,v)*dx(8) +1e-15*inner(u,v)*dx
        a8 = self.elasticity(u,v)*dx(9) +1e-15*inner(u,v)*dx
        # Assemble
        A0 = assemble(a0)
        A1 = assemble(a1)
        A2 = assemble(a2)
        A3 = assemble(a3)
        A4 = assemble(a4)
        A5 = assemble(a5)
        A6 = assemble(a6)
        A7 = assemble(a7)
        A8 = assemble(a8)
        # Return
        return (A0, A1, A2, A3, A4, A5, A6, A7, A8)
    
    ## Set vectors resulting from the truth discretization of f.
    def assemble_truth_f(self):
        v = self.v
        dx = self.dx
        ds = self.ds
        l = Constant((1e-11, 1e-11))
        f = self.f
        # Define
        f0 = inner(f,v)*ds(2) + inner(l,v)*dx
        f1 = inner(f,v)*ds(3) + inner(l,v)*dx 
        f2 = inner(f,v)*ds(4) + inner(l,v)*dx
        # Assemble
        F0 = assemble(f0)
        F1 = assemble(f1)
        F2 = assemble(f2)
        # Return
        return (F0,F1,F2)
    
    ## Auxiliary function to compute the elasticity bilinear form    
    def elasticity(self, u, v):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        return 2.0*lambda_2*inner(sym(grad(u)),sym(grad(v))) + lambda_1*tr(sym(grad(u)))*tr(sym(grad(v)))
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 2: MAIN PROGRAM     ~~~~~~~~~~~~~~~~~~~~~~~~~# 

# 1. Read the mesh for this problem
mesh = Mesh("data/elastic.xml")
subd = MeshFunction("size_t", mesh, "data/elastic_physical_region.xml")
bound = MeshFunction("size_t", mesh, "data/elastic_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1, two components)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the Elastic Block class
eb = Eblock(V, subd, bound)

# 4. Choose PETSc solvers as linear algebra backend
parameters.linear_algebra_backend = 'PETSc'

# 5. Set mu range, xi_train and Nmax
mu_range = [ \
    (1.0, 100.0), \
    (1.0, 100.0), \
    (1.0, 100.0), \
    (1.0, 100.0), \
    (1.0, 100.0), \
    (1.0, 100.0), \
    (1.0, 100.0), \
    (1.0, 100.0), \
    (-1.0, 1.0), \
    (-1.0, 1.0), \
    (-1.0, 1.0), \
]
eb.setmu_range(mu_range)
eb.setxi_train(500)
eb.setNmax(20)

# 6. Perform the offline phase
first_mu = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0)
eb.setmu(first_mu)
eb.offline()

# 7. Perform an online solve
online_mu = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0)
eb.setmu(online_mu)
eb.online_solve()

# 8. Perform an error analysis
eb.setxi_test(500)
eb.error_analysis()
