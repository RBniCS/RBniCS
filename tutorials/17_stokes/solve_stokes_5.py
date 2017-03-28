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
## @file solve_elast.py
#  @brief Example 2: elastic block test case
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
from RBniCS import *
from sampling import LinearlyDependentUniformDistribution

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 17: STOKES CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
@ShapeParametrization(
    ("mu[4]*x[0] + mu[1] - mu[4]", "tan(mu[5])*x[0] + mu[0]*x[1] + mu[2] - tan(mu[5]) - mu[0]"), # subdomain 1
    ("mu[1]*x[0]", "mu[3]*x[1] + mu[2] + mu[0] - 2*mu[3]"), # subdomain 2
    ("mu[1]*x[0]", "mu[0]*x[1] + mu[2] - mu[0]"), # subdomain 3
    ("mu[1]*x[0]", "mu[2]*x[1]"), # subdomain 4
)
class Stokes(StokesProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        StokesProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        up = TrialFunction(V)
        (self.u, self.p) = split(up)
        vq = TestFunction(V)
        (self.v, self.q) = split(vq)
        self.s = TrialFunction(V.sub(0).collapse())
        self.r = TestFunction(V.sub(0).collapse())
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        #
        self.f = Constant((0.0, -10.0))
        self.g = Constant(0.0)
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return the lower bound for inf-sup constant.
    def get_stability_factor(self):
        return 1.
    
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        mu1 = mu[0]
        mu2 = mu[1]
        mu3 = mu[2]
        mu4 = mu[3]
        mu5 = mu[4]
        mu6 = mu[5]
        if term == "a":
            theta_a0 = mu1/mu5
            theta_a1 = -tan(mu6)/mu5
            theta_a2 = (tan(mu6)**2 + mu5**2)/(mu5*mu1)
            theta_a3 = mu4/mu2
            theta_a4 = mu2/mu4
            theta_a5 = mu1/mu2
            theta_a6 = mu2/mu1
            theta_a7 = mu3/mu2
            theta_a8 = mu2/mu3
            return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7, theta_a8)
        elif term == "b" or term == "bt" or term == "bt_restricted":
            theta_b0 = mu1
            theta_b1 = -tan(mu6)
            theta_b2 = mu5
            theta_b3 = mu4
            theta_b4 = mu2
            theta_b5 = mu1
            theta_b6 = mu2
            theta_b7 = mu3
            theta_b8 = mu2
            return (theta_b0, theta_b1, theta_b2, theta_b3, theta_b4, theta_b5, theta_b6, theta_b7, theta_b8)
        elif term == "f":
            theta_f0 = mu[0]*mu[4]
            theta_f1 = mu[1]*mu[3]
            theta_f2 = mu[0]*mu[1]
            theta_f3 = mu[1]*mu[2]
            return (theta_f0, theta_f1, theta_f2, theta_f3)
        elif term == "g":
            theta_g0 = mu[0]*mu[4]
            theta_g1 = mu[1]*mu[3]
            theta_g2 = mu[0]*mu[1]
            theta_g3 = mu[1]*mu[2]
            return (theta_g0, theta_g1, theta_g2, theta_g3)
        else:
            raise ValueError("Invalid term for compute_theta().")
                
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        dx = self.dx
        if term == "a":
            u = self.u
            v = self.v
            a0 = (u[0].dx(0)*v[0].dx(0) + u[1].dx(0)*v[1].dx(0))*dx(1)
            a1 = (u[0].dx(0)*v[0].dx(1) + u[0].dx(1)*v[0].dx(0) + u[1].dx(0)*v[1].dx(1) + u[1].dx(1)*v[1].dx(0))*dx(1)
            a2 = (u[0].dx(1)*v[0].dx(1) + u[1].dx(1)*v[1].dx(1))*dx(1)
            a3 = (u[0].dx(0)*v[0].dx(0) + u[1].dx(0)*v[1].dx(0))*dx(2)
            a4 = (u[0].dx(1)*v[0].dx(1) + u[1].dx(1)*v[1].dx(1))*dx(2)
            a5 = (u[0].dx(0)*v[0].dx(0) + u[1].dx(0)*v[1].dx(0))*dx(3)
            a6 = (u[0].dx(1)*v[0].dx(1) + u[1].dx(1)*v[1].dx(1))*dx(3)
            a7 = (u[0].dx(0)*v[0].dx(0) + u[1].dx(0)*v[1].dx(0))*dx(4)
            a8 = (u[0].dx(1)*v[0].dx(1) + u[1].dx(1)*v[1].dx(1))*dx(4)
            return (a0, a1, a2, a3, a4, a5, a6, a7, a8)
        elif term == "b":
            u = self.u
            q = self.q
            b0 = - q*u[0].dx(0)*dx(1)
            b1 = - q*u[0].dx(1)*dx(1)
            b2 = - q*u[1].dx(1)*dx(1)
            b3 = - q*u[0].dx(0)*dx(2)
            b4 = - q*u[1].dx(1)*dx(2)
            b5 = - q*u[0].dx(0)*dx(3)
            b6 = - q*u[1].dx(1)*dx(3)
            b7 = - q*u[0].dx(0)*dx(4)
            b8 = - q*u[1].dx(1)*dx(4)
            return (b0, b1, b2, b3, b4, b5, b6, b7, b8)
        elif term == "bt" or term == "bt_restricted":
            p = self.p
            if term == "bt":
                v = self.v
            elif term == "bt_restricted":
                v = self.r
            bt0 = - p*v[0].dx(0)*dx(1)
            bt1 = - p*v[0].dx(1)*dx(1)
            bt2 = - p*v[1].dx(1)*dx(1)
            bt3 = - p*v[0].dx(0)*dx(2)
            bt4 = - p*v[1].dx(1)*dx(2)
            bt5 = - p*v[0].dx(0)*dx(3)
            bt6 = - p*v[1].dx(1)*dx(3)
            bt7 = - p*v[0].dx(0)*dx(4)
            bt8 = - p*v[1].dx(1)*dx(4)
            return (bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8)
        elif term == "f":
            v = self.v
            f0 = inner(self.f, v)*dx(0)
            f1 = inner(self.f, v)*dx(1)
            f2 = inner(self.f, v)*dx(2)
            f3 = inner(self.f, v)*dx(3)
            return (f0, f1, f2, f3)
        elif term == "g":
            q = self.q
            g0 = self.g*q*dx(0)
            g1 = self.g*q*dx(1)
            g2 = self.g*q*dx(2)
            g3 = self.g*q*dx(3)
            return (g0, g1, g2, g3)
        elif term == "dirichlet_bc_u" or term == "dirichlet_bc_s":
            if term == "dirichlet_bc_u":
                V_s = self.V.sub(0)
            elif term == "dirichlet_bc_s":
                V_s = self.V.sub(0).collapse()
            bc0 = [DirichletBC(V_s, Constant((0.0, 0.0)), self.boundaries, 3)]
            return (bc0,)
        elif term == "inner_product_u" or term == "inner_product_s":
            if term == "inner_product_u":
                u = self.u
                v = self.v
            elif term == "inner_product_s":
                u = self.s
                v = self.r
            x0 = inner(grad(u),grad(v))*dx
            return (x0,)
        elif term == "inner_product_p":
            p = self.p
            q = self.q
            x0 = inner(p, q)*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 17: MAIN PROGRAM     ~~~~~~~~~~~~~~~~~~~~~~~~~# 

# 1. Read the mesh for this problem
mesh = Mesh("data/t_bypass.xml")
subdomains = MeshFunction("size_t", mesh, "data/t_bypass_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/t_bypass_facet_region.xml")

# 2. Create Finite Element space (Taylor-Hood P2-P1)
element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_u, element_p)
V = FunctionSpace(mesh, element, components=[["u", "s"], "p"])

# 3. Allocate an object of the Elastic Block class
stokes_problem = Stokes(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [ \
    (0.5, 1.5), \
    (0.5, 1.5), \
    (0.5, 1.5), \
    (0.5, 1.5), \
    (0.5, 1.5), \
    (0., pi/6.) \
]
stokes_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a POD-Galerkin method
reduced_basis_method = ReducedBasis(stokes_problem)
reduced_basis_method.set_Nmax(25)
reduced_basis_method.set_tolerance(1e-6)

# 5. Perform the offline phase
first_mu = (0.5, 0.5, 0.5, 0.5, 0.5, 0.)
stokes_problem.set_mu(first_mu)
reduced_basis_method.initialize_training_set(100, sampling=LinearlyDependentUniformDistribution())
reduced_stokes_problem = reduced_basis_method.offline()

# 6. Perform an online solve
online_mu = (1.0, 1.0, 1.0, 1.0, 1.0, pi/6.)
reduced_stokes_problem.set_mu(online_mu)
reduced_stokes_problem.solve()
reduced_stokes_problem.export_solution("Stokes", "online_solution")

# 7. Perform an error analysis
reduced_basis_method.initialize_testing_set(100, sampling=LinearlyDependentUniformDistribution())
reduced_basis_method.error_analysis()

# 8. Perform a speedup analysis
reduced_basis_method.speedup_analysis()
