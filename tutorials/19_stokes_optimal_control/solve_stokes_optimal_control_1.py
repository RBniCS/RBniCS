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
    ("x[0]", "mu[0]*x[1]"), # subdomain 1
)
class StokesOptimalControl(StokesOptimalControlProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        StokesOptimalControlProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        trial = TrialFunction(V)
        (self.v, self.p, self.u, self.w, self.q) = split(trial)
        test = TestFunction(V)
        (self.psi, self.pi, self.tau, self.phi, self.xi) = split(test)
        self.s, self.r = TrialFunction(V.sub("s").collapse()), TrialFunction(V.sub("r").collapse())
        self.sigma, self.rho = TestFunction(V.sub("s").collapse()), TestFunction(V.sub("r").collapse())
        self.dx = Measure("dx")(subdomain_data=subdomains)
        self.ds = Measure("ds")(subdomain_data=boundaries)
        # Regularization coefficient
        self.alpha = 0.008
        # Constant viscosity
        self.nu = 0.1
        # Desired velocity
        self.vx_d = Expression("x[1]", degree=1)
                
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
            nu = self.nu
            theta_a0 = nu*mu1
            theta_a1 = nu/mu1
            return (theta_a0, theta_a1)
        elif (
            term == "b" or term == "b*"
                or 
            term == "bt" or term == "bt*"
                or
            term == "bt_restricted" or term == "bt*_restricted"
        ):
            theta_b0 = mu1
            theta_b1 = 1.0
            return (theta_b0, theta_b1)
        elif term == "c" or term == "c*":
            theta_c0 = mu1
            return (theta_c0,)
        elif term == "m":
            theta_m0 = mu1
            return (theta_m0,)
        elif term == "n":
            theta_n0 = self.alpha*mu1
            return (theta_n0,)
        elif term == "f":
            theta_f0 = - mu1*mu2
            return (theta_f0,)
        elif term == "g":
            theta_g0 = mu1**2
            return (theta_g0,)
        elif term == "h":
            theta_h0 = mu1**3/3.
            return (theta_h0,)
        elif term == "l":
            theta_l0 = mu1
            return (theta_l0,)
        elif term == "dirichlet_bc_v":
            theta_bc0 = mu1
            return (theta_bc0,)
        else:
            raise ValueError("Invalid term for compute_theta().")
                    
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        dx = self.dx
        if term == "a":
            v = self.v
            phi = self.phi
            a0 = v[0].dx(0)*phi[0].dx(0)*dx + v[1].dx(0)*phi[1].dx(0)*dx
            a1 = v[0].dx(1)*phi[0].dx(1)*dx + v[1].dx(1)*phi[1].dx(1)*dx
            return (a0, a1)
        elif term == "a*":
            psi = self.psi
            w = self.w
            as0 = psi[0].dx(0)*w[0].dx(0)*dx + psi[1].dx(0)*w[1].dx(0)*dx
            as1 = psi[0].dx(1)*w[0].dx(1)*dx + psi[1].dx(1)*w[1].dx(1)*dx
            return (as0, as1)
        elif term == "b":
            xi = self.xi
            v = self.v
            b0 = - xi*v[0].dx(0)*dx
            b1 = - xi*v[1].dx(1)*dx
            return (b0, b1)
        elif term == "bt" or term == "bt_restricted":
            p = self.p
            if term == "bt":
                phi = self.phi
            elif term == "bt_restricted":
                phi = self.sigma
            bt0 = - p*phi[0].dx(0)*dx
            bt1 = - p*phi[1].dx(1)*dx
            return (bt0, bt1)
        elif term == "b*":
            pi = self.pi
            w = self.w
            bs0 = - pi*w[0].dx(0)*dx
            bs1 = - pi*w[1].dx(1)*dx
            return (bs0, bs1)
        elif term == "bt*" or term == "bt*_restricted":
            q = self.q
            if term == "bt*":
                psi = self.psi
            elif term == "bt*_restricted":
                psi = self.rho
            bts0 = - q*psi[0].dx(0)*dx
            bts1 = - q*psi[1].dx(1)*dx
            return (bts0, bts1)
        elif term == "c":
            u = self.u
            phi = self.phi
            c0 = inner(u, phi)*dx
            return (c0,)
        elif term == "c*":
            tau = self.tau
            w = self.w
            cs0 = inner(tau, w)*dx
            return (cs0,)
        elif term == "m":
            v = self.v
            psi = self.psi
            m0 = v[0]*psi[0]*dx
            return (m0,)
        elif term == "n":
            u = self.u
            tau = self.tau
            n0 = inner(u, tau)*dx
            return (n0,)
        elif term == "f":
            phi = self.phi
            f0 = phi[1]*dx
            return (f0,)
        elif term == "g":
            psi = self.psi
            g0 = self.vx_d*psi[0]*dx
            return (g0,)
        elif term == "h":
            h0 = 1.0
            return (h0,)
        elif term == "l":
            xi = self.xi
            l0 = Constant(0.0)*xi*dx
            return (l0,)
        elif term == "dirichlet_bc_v":
            bc0 = [DirichletBC(self.V.sub("v").sub(0), self.vx_d    , self.boundaries, 1),
                   DirichletBC(self.V.sub("v").sub(1), Constant(0.0), self.boundaries, 1)]
            return (bc0,)
        elif term == "dirichlet_bc_w":
            bc0 = [DirichletBC(self.V.sub("w"), Constant((0.0, 0.0)), self.boundaries, 1)]
            return (bc0,)
        elif term == "dirichlet_bc_s":
            bc0 = [DirichletBC(self.V.sub("s").collapse(), Constant((0.0, 0.0)), self.boundaries, 1)]
            return (bc0,)
        elif term == "dirichlet_bc_r":
            bc0 = [DirichletBC(self.V.sub("r").collapse(), Constant((0.0, 0.0)), self.boundaries, 1)]
            return (bc0,)
        elif term == "inner_product_v" or term == "inner_product_s":
            if term == "inner_product_v":
                v = self.v
                psi = self.psi
            elif term == "inner_product_s":
                v = self.s
                psi = self.sigma
            x0 = inner(grad(v), grad(psi))*dx
            return (x0,)
        elif term == "inner_product_p":
            p = self.p
            pi = self.pi
            x0 = p*pi*dx
            return (x0,)
        elif term == "inner_product_u":
            u = self.u
            tau = self.tau
            x0 = inner(u, tau)*dx
            return (x0,)
        elif term == "inner_product_w" or term == "inner_product_r":
            if term == "inner_product_w":
                w = self.w
                phi = self.phi
            elif term == "inner_product_r":
                w = self.r
                phi = self.rho
            x0 = inner(grad(w), grad(phi))*dx
            return (x0,)
        elif term == "inner_product_q":
            q = self.q
            xi = self.xi
            x0 = q*xi*dx
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
velocity_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
pressure_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = MixedElement(velocity_element, pressure_element, velocity_element, velocity_element, pressure_element)
V = FunctionSpace(mesh, element, components=[["v", "s"], "p", "u", ["w", "r"], "q"])

# 3. Allocate an object of the StokesOptimalControl class
stokes_optimal_control = StokesOptimalControl(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.5, 2.0), (0.5, 1.5)]
stokes_optimal_control.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
pod_galerkin_method = PODGalerkin(stokes_optimal_control)
pod_galerkin_method.set_Nmax(10)

# 5. Perform the offline phase
first_mu = (1.0, 1.0)
stokes_optimal_control.set_mu(first_mu)
pod_galerkin_method.initialize_training_set(100)
reduced_stokes_optimal_control = pod_galerkin_method.offline()

# 6. Perform an online solve
online_mu = (1.7, 1.5)
reduced_stokes_optimal_control.set_mu(online_mu)
reduced_stokes_optimal_control.solve()
reduced_stokes_optimal_control.export_solution("StokesOptimalControl", "online_solution")
print "Reduced output for mu =", online_mu, "is", reduced_stokes_optimal_control.compute_output()

# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(100)
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.speedup_analysis()
