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
## @file solve_graetz.py
#  @brief Example 4: Graetz test case
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
from RBniCS import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 18: OPTIMAL CONTROL CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
@ShapeParametrization(
    ("x[0]", "x[1]"), # subdomain 1
    ("mu[0]*(x[0] - 1) + 1", "x[1]"), # subdomain 2
)
class EllipticOptimalControl(EllipticOptimalControlProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        EllipticOptimalControlProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        yup = TrialFunction(V)
        (self.y, self.u, self.p) = split(yup)
        zvq = TestFunction(V)
        (self.z, self.v, self.q) = split(zvq)
        self.dx = Measure("dx")(subdomain_data=subdomains)
        self.ds = Measure("ds")(subdomain_data=boundaries)
        # Regularization coefficient
        self.alpha = 0.01
                
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
        
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu1 = self.mu[0]
        mu2 = self.mu[1]
        if term == "a" or term == "a*":
            theta_a0 = 1.0
            theta_a1 = 1.0/mu1
            theta_a2 = mu1
            return (theta_a0, theta_a1, theta_a2)
        elif term == "c" or term == "c*":
            theta_c0 = 1.0
            theta_c1 = mu1
            return (theta_c0, theta_c1)
        elif term == "m":
            theta_m0 = 1.0
            theta_m1 = mu1
            return (theta_m0, theta_m1)
        elif term == "n":
            alpha = self.alpha
            theta_n0 = alpha
            theta_n1 = alpha*mu1
            return (theta_n0, theta_n1)
        elif term == "f":
            theta_f0 = 1.0
            return (theta_f0,)
        elif term == "g":
            theta_g0 = 1.0
            theta_g1 = mu1*mu2
            return (theta_g0, theta_g1)
        elif term == "h":
            theta_h0 = 1.0 + mu1*mu2**2
            return (theta_h0,)
        elif term == "dirichlet_bc_y":
            theta_bc0 = 1.
            return (theta_bc0,)
        else:
            raise ValueError("Invalid term for compute_theta().")
                    
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        dx = self.dx
        if term == "a":
            y = self.y
            q = self.q
            a0 = inner(grad(y),grad(q))*dx(1)
            a1 = y.dx(0)*q.dx(0)*dx(2)
            a2 = y.dx(1)*q.dx(1)*dx(2)
            return (a0, a1, a2)
        elif term == "a*":
            z = self.z
            p = self.p
            as0 = inner(grad(z),grad(p))*dx(1)
            as1 = z.dx(0)*p.dx(0)*dx(2)
            as2 = z.dx(1)*p.dx(1)*dx(2)
            return (as0, as1, as2)
        elif term == "c":
            u = self.u
            q = self.q
            c0 = u*q*dx(1)
            c1 = u*q*dx(2)
            return (c0, c1)
        elif term == "c*":
            v = self.v
            p = self.p
            cs0 = v*p*dx(1)
            cs1 = v*p*dx(2)
            return (cs0, cs1)
        elif term == "m":
            y = self.y
            z = self.z
            m0 = y*z*dx(1)
            m1 = y*z*dx(2)
            return (m0, m1)
        elif term == "n":
            u = self.u
            v = self.v
            n0 = u*v*dx(1)
            n1 = u*v*dx(2)
            return (n0, n1)
        elif term == "f":
            q = self.q
            f0 = Constant(0.0)*q*dx
            return (f0,)
        elif term == "g":
            z = self.z
            g0 = z*dx(1)
            g1 = z*dx(2)
            return (g0, g1)
        elif term == "h":
            h0 = 1.0
            return (h0,)
        elif term == "dirichlet_bc_y":
            bc0 = [DirichletBC(self.V.sub(0), Constant(1.0), self.boundaries, i) for i in range(1, 9)]
            return (bc0,)
        elif term == "dirichlet_bc_p":
            bc0 = [DirichletBC(self.V.sub(2), Constant(0.0), self.boundaries, i) for i in range(1, 9)]
            return (bc0,)
        elif term == "inner_product_y":
            y = self.y
            z = self.z
            x0 = inner(grad(y), grad(z))*dx
            return (x0,)
        elif term == "inner_product_u":
            u = self.u
            v = self.v
            x0 = u*v*dx
            return (x0,)
        elif term == "inner_product_p":
            p = self.p
            q = self.q
            x0 = inner(grad(p), grad(q))*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
    
#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 18: MAIN PROGRAM     ~~~~~~~~~~~~~~~~~~~~~~~~~# 

# 1. Read the mesh for this problem
mesh = Mesh("data/mesh1.xml")
subdomains = MeshFunction("size_t", mesh, "data/mesh1_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/mesh1_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
scalar_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = MixedElement(scalar_element, scalar_element, scalar_element)
V = FunctionSpace(mesh, element, components=["y", "u", "p"])

# 3. Allocate an object of the EllipticOptimalControl class
elliptic_optimal_control = EllipticOptimalControl(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(1.0, 3.5), (0.5, 2.5)]
elliptic_optimal_control.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
pod_galerkin_method = PODGalerkin(elliptic_optimal_control)
pod_galerkin_method.set_Nmax(10)

# 5. Perform the offline phase
first_mu = (1.0, 1.0)
elliptic_optimal_control.set_mu(first_mu)
pod_galerkin_method.initialize_training_set(100)
reduced_elliptic_optimal_control = pod_galerkin_method.offline()

# 6. Perform an online solve
online_mu = (3.0, 0.6)
reduced_elliptic_optimal_control.set_mu(online_mu)
reduced_elliptic_optimal_control.solve()
reduced_elliptic_optimal_control.export_solution("EllipticOptimalControl", "online_solution")
print "Reduced output for mu =", online_mu, "is", reduced_elliptic_optimal_control.output()

# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(100)
pod_galerkin_method.error_analysis()

