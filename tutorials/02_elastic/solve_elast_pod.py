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
class ElasticBlock(EllipticCoerciveProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        EllipticCoerciveProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
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
    
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        mu1 = mu[0]
        mu2 = mu[1]
        mu3 = mu[2]
        mu4 = mu[3]
        mu5 = mu[4]
        mu6 = mu[5]
        mu7 = mu[6]
        mu8 = mu[7]
        mu9 = mu[8]
        mu10 = mu[9]
        mu11 = mu[10]
        if term == "a":
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
        elif term == "f":
            theta_f0 = mu9
            theta_f1 = mu10
            theta_f2 = mu11
            return (theta_f0, theta_f1, theta_f2)
        else:
            raise ValueError("Invalid term for compute_theta().")
                
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            a0 = self.elasticity(u,v)*dx(1)
            a1 = self.elasticity(u,v)*dx(2)
            a2 = self.elasticity(u,v)*dx(3)
            a3 = self.elasticity(u,v)*dx(4)
            a4 = self.elasticity(u,v)*dx(5)
            a5 = self.elasticity(u,v)*dx(6)
            a6 = self.elasticity(u,v)*dx(7)
            a7 = self.elasticity(u,v)*dx(8)
            a8 = self.elasticity(u,v)*dx(9)
            return (a0, a1, a2, a3, a4, a5, a6, a7, a8)
        elif term == "f":
            ds = self.ds
            f = self.f
            f0 = inner(f,v)*ds(2)
            f1 = inner(f,v)*ds(3)
            f2 = inner(f,v)*ds(4)
            return (f0,f1,f2)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant((0.0, 0.0)), self.boundaries, 6)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(u, v)*dx + inner(grad(u),grad(v))*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
    
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
subdomains = MeshFunction("size_t", mesh, "data/elastic_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/elastic_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1, two components)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the Elastic Block class
elastic_block_problem = ElasticBlock(V, subdomains=subdomains, boundaries=boundaries)
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
elastic_block_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a POD-Galerkin method
pod_galerkin_method = PODGalerkin(elastic_block_problem)
pod_galerkin_method.set_Nmax(20)

# 5. Perform the offline phase
pod_galerkin_method.set_xi_train(500)
reduced_elastic_block_problem = pod_galerkin_method.offline()

# 6. Perform an online solve
online_mu = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0)
reduced_elastic_block_problem.set_mu(online_mu)
reduced_elastic_block_problem.solve()

# 7. Perform an error analysis
pod_galerkin_method.set_xi_test(500)
pod_galerkin_method.error_analysis()
