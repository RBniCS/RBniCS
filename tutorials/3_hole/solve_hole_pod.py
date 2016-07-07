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
## @file solve_hole.py
#  @brief Example 3: geometrical parametrization
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
from RBniCS import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 3: GEOMETRICAL PARAMETRIZATION CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
@ShapeParametrization(
    ("2.0 - 2.0*mu[0] + mu[0]*x[0] +(2.0-2.0*mu[0])*x[1]", "2.0 -2.0*mu[1] + (2.0-mu[1])*x[1]"), # subdomain 1
    ("2.0*mu[0]-2.0 +x[0] +(mu[0]-1.0)*x[1]", "2.0 -2.0*mu[1] + (2.0-mu[1])*x[1]"), # subdomain 2
    ("2.0 - 2.0*mu[0] + (2.0-mu[0])*x[0]", "2.0 -2.0*mu[1] + (2.0-2.0*mu[1])*x[0] + mu[1]*x[1]"), # subdomain 3
    ("2.0 - 2.0*mu[0] + (2.0-mu[0])*x[0]", "2.0*mu[1] -2.0 + (mu[1]-1.0)*x[0] + x[1]"), # subdomain 4
    ("2.0*mu[0] -2.0 + (2.0-mu[0])*x[0]", "2.0 -2.0*mu[1] + (2.0*mu[1]-2.0)*x[0] + mu[1]*x[1]"), # subdomain 5
    ("2.0*mu[0] -2.0 + (2.0-mu[0])*x[0]", "2.0*mu[1] -2.0 + (1.0 - mu[1])*x[0] + x[1]"), # subdomain 6
    ("2.0 -2.0*mu[0] + mu[0]*x[0] + (2.0*mu[0]-2.0)*x[1]", "2.0*mu[1] -2.0 + (2.0 - mu[1])*x[1]"), # subdomain 7
    ("2.0*mu[0] -2.0 + x[0] + (1.0-mu[0])*x[1]", "2.0*mu[1] -2.0 + (2.0 - mu[1])*x[1]"), # subdomain 8
)
class Hole(EllipticCoerciveProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        EllipticCoerciveProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs and "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=subdomains)
        self.ds = Measure("ds")(subdomain_data=boundaries)
        self.subdomains = subdomains
        self.boundaries = boundaries
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        m1 = self.mu[0]
        m2 = self.mu[1]
        m3 = self.mu[2]
        if term == "a":
            # subdomains 1 and 7
            theta_a0 = - (m2 - 2)/m1 - (2*(2*m1 - 2)*(m1 - 1))/(m1*(m2 - 2)) #K11
            theta_a1 = -m1/(m2 - 2) #K22
            theta_a2 = -(2*(m1 - 1))/(m2 - 2) #K12 and K21
            # subdomains 2 and 8
            theta_a3 = 2 - (m1 - 1)*(m1 - 1)/(m2 - 2) - m2
            theta_a4 = -1/(m2 - 2)
            theta_a5 = (m1 - 1)/(m2 - 2)
            # subdomains 3 and 5
            theta_a6 = -m2/(m1 - 2)
            theta_a7 = - (m1 - 2)/m2 - (2*(2*m2 - 2)*(m2 - 1))/(m2*(m1 - 2))
            theta_a8 = -(2*(m2 - 1))/(m1 - 2)
            # subdomains 4 and 6
            theta_a9 = -1/(m1 - 2)
            theta_a10 = 2 - (m2 - 1)*(m2 - 1)/(m1 - 2) - m1
            theta_a11 = (m2 - 1)/(m1 - 2)
            # boundaries 5, 6, 7 and 8
            theta_a12 = m3
            # Return
            return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7, theta_a8, theta_a9, theta_a10, theta_a11, theta_a12)
        elif term == "f":
            theta_f0 = - m1*(m2 - 2.0) # boundary 1
            theta_f1 = - m2*(m1 - 2.0) # boundary 2
            theta_f2 = - m1*(m2 - 2.0) # boundary 3
            theta_f3 = - m2*(m1 - 2.0) # boundary 4
            # Return
            return (theta_f0, theta_f1, theta_f2, theta_f3)
        else:
            raise RuntimeError("Invalid term for compute_theta().")
                
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        u = self.u
        v = self.v
        dx = self.dx
        ds = self.ds
        if term == "a":
            # subdomains 1 and 7
            a0 = inner(u.dx(0), v.dx(0))*dx(1) +  inner(u.dx(0), v.dx(0))*dx(7)
            a1 = inner(u.dx(1), v.dx(1))*dx(1) +  inner(u.dx(1), v.dx(1))*dx(7)
            a2 = inner(u.dx(0), v.dx(1))*dx(1) +  inner(u.dx(1), v.dx(0))*dx(1) - (inner(u.dx(0), v.dx(1))*dx(7) +  inner(u.dx(1), v.dx(0))*dx(7))
            # subdomains 2 and 8
            a3 = inner(u.dx(0), v.dx(0))*dx(2) +  inner(u.dx(0), v.dx(0))*dx(8)
            a4 = inner(u.dx(1), v.dx(1))*dx(2) +  inner(u.dx(1), v.dx(1))*dx(8)
            a5 = inner(u.dx(0), v.dx(1))*dx(2) +  inner(u.dx(1), v.dx(0))*dx(2) - (inner(u.dx(0), v.dx(1))*dx(8) +  inner(u.dx(1), v.dx(0))*dx(8))
            # subdomains 3 and 5
            a6 = inner(u.dx(0), v.dx(0))*dx(3) +  inner(u.dx(0), v.dx(0))*dx(5)
            a7 = inner(u.dx(1), v.dx(1))*dx(3) +  inner(u.dx(1), v.dx(1))*dx(5)
            a8 = inner(u.dx(0), v.dx(1))*dx(3) +  inner(u.dx(1), v.dx(0))*dx(3) - (inner(u.dx(0), v.dx(1))*dx(5) +  inner(u.dx(1), v.dx(0))*dx(5))
            # subdomains 4 and 6
            a9 = inner(u.dx(0), v.dx(0))*dx(4) +  inner(u.dx(0), v.dx(0))*dx(6)
            a10 = inner(u.dx(1), v.dx(1))*dx(4) +  inner(u.dx(1), v.dx(1))*dx(6)
            a11 = inner(u.dx(0), v.dx(1))*dx(4) +  inner(u.dx(1), v.dx(0))*dx(4) - (inner(u.dx(0), v.dx(1))*dx(6) +  inner(u.dx(1), v.dx(0))*dx(6))
            # boundaries 5, 6, 7 and 8
            a12 = inner(u,v)*ds(5) + inner(u,v)*ds(6) + inner(u,v)*ds(7) + inner(u,v)*ds(8)
            # Return
            return (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)
        elif term == "f":
            f0 = v*ds(1) # boundary 1
            f1 = v*ds(2) # boundary 2
            f2 = v*ds(3) # boundary 3
            f3 = v*ds(4) # boundary 4
            # Return
            return (f0, f1, f2, f3)
        elif term == "inner_product":
            x0 = u*v*dx + inner(grad(u),grad(v))*dx
            return (x0,)
        else:
            raise RuntimeError("Invalid term for assemble_operator().")
                    
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
    
#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 3: MAIN PROGRAM     ~~~~~~~~~~~~~~~~~~~~~~~~~# 

# 1. Read the mesh for this problem
mesh = Mesh("data/hole.xml")
subdomains = MeshFunction("size_t", mesh, "data/hole_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/hole_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the Hole class
hole_problem = Hole(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.5, 1.5), (0.5, 1.5), (0.01, 1.0)]
hole_problem.set_mu_range(mu_range)

# 4. Choose PETSc solvers as linear algebra backend
parameters.linear_algebra_backend = 'PETSc'

# 5. Prepare reduction with a POD-Galerkin method
pod_galerkin_method = PODGalerkin(hole_problem)
pod_galerkin_method.set_Nmax(20)

# 6. Perform the offline phase
pod_galerkin_method.set_xi_train(500)
reduced_hole_problem = pod_galerkin_method.offline()

# 7. Perform an online solve
online_mu = (0.5, 0.5, 0.01)
reduced_hole_problem.set_mu(online_mu)
reduced_hole_problem.solve()

# 8. Perform an error analysis
pod_galerkin_method.set_xi_test(500)
pod_galerkin_method.error_analysis()
