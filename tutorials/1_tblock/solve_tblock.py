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
## @file solve_tblock.py
#  @brief Example 1: thermal block test case
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
from RBniCS import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 1: THERMAL BLOCK CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
class Tblock(EllipticCoerciveProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, subd, bound):
        # Call the standard initialization
        super(Tblock, self).__init__(V)
        # ... and also store FEniCS data structures for assembly
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=subd)
        self.ds = Measure("ds")(subdomain_data=bound)
    
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return the alpha_lower bound.
    def get_alpha_lb(self):
        return min(self.compute_theta("a"))
    
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu1 = self.mu[0]
        mu2 = self.mu[1]
        if term == "a":
            theta_a0 = mu1
            theta_a1 = 1.
            return (theta_a0, theta_a1)
        elif term == "f":
            theta_f0 = mu2
            return (theta_f0,)
        elif term == "dirichlet_bc":
            return (0.,)
        else:
            raise RuntimeError("Invalid term for compute_theta().")
    
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            a0 = inner(grad(u),grad(v))*dx(1) + 1e-15*inner(u,v)*dx
            a1 = inner(grad(u),grad(v))*dx(2) + 1e-15*inner(u,v)*dx
            return (a0, a1)
        elif term == "f":
            ds = self.ds
            f0 = v*ds(1) + 1e-15*v*dx
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = [(self.V, Constant(0.0), self.bound, 3)]
            return (bc0,)
        elif term == "inner_product":
            x0 = inner(grad(u),grad(v))*dx
            return (x0,)
        else:
            raise RuntimeError("Invalid term for assemble_operator().")
        
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
tb.setxi_train(100)
tb.setNmax(4)

# 6. Perform the offline phase
first_mu = (0.5,1.0)
tb.setmu(first_mu)
tb.offline()

# 7. Perform an online solve
online_mu = (8.,-1.0)
tb.setmu(online_mu)
tb.online_solve()

# 8. Perform an error analysis
tb.setxi_test(500)
tb.error_analysis()
