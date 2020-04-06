# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *
from problems import *
from reduction_methods import *


class Geostrophic(GeostrophicProblem):
    def __init__(self, W, **kwargs):
        GeostrophicProblem.__init__(self, W, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        psiq = TrialFunction(W)
        (self.psi, self.q) = split(psiq)
        phip = TestFunction(W)
        (self.phi, self.p) = split(phip)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.f = Expression("-sin(pi*x[1])", element=W.sub(0).ufl_element())

    def compute_theta(self, term):
        mu = self.mu
        if term == "a":
            theta_a1 = 1.
            theta_a2 = mu[0]
            theta_a3 = mu[1]
            theta_a4 = 1.
            theta_a5 = 1.
            return (theta_a1, theta_a2, theta_a3, theta_a4, theta_a5)
        elif term == "f":
            theta_f0 = 1.
            return (theta_f0,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    def assemble_operator(self, term):
        dx = self.dx
        phi = self.phi
        p = self.p
        if term == "a":
            psi = self.psi
            q = self.q
            a1 = inner(psi.dx(0), p) * dx
            a2 = inner(q, p) * dx
            a3 = inner(grad(q), grad(p)) * dx
            a4 = inner(q, phi) * dx
            a5 = inner(grad(psi), grad(phi)) * dx
            return (a1, a2, a3, a4, a5)
        elif term == "f":
            f = self.f
            f0 = inner(f, p) * dx
            return (f0,)
        elif term == "dirichlet_bc_psi":
            bc0 = [DirichletBC(W.sub(0), Constant(0.0), boundaries, idx) for idx in [1, 2, 3, 4]]
            return (bc0,)
        elif term == "dirichlet_bc_q":
            bc0 = [DirichletBC(W.sub(1), Constant(0.0), boundaries, idx) for idx in [1, 2, 3, 4]]
            return (bc0,)
        elif term == "inner_product_psi":
            psi = self.psi
            phi = self.phi
            x0 = inner(grad(phi), grad(psi)) * dx
            return (x0,)
        elif term == "inner_product_q":
            q = self.q
            p = self.p
            x0 = inner(grad(p), grad(q)) * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")


# 1. Read the mesh for this problem
mesh = Mesh("data/square.xml")
subdomains = MeshFunction("size_t", mesh, "data/square_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/square_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1, two components)
V_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W_element = MixedElement(V_element, Q_element)
W = FunctionSpace(mesh, W_element, components=["psi", "q"])

# 3. Allocate an object of the Geostrophic class
geostrophic_problem = Geostrophic(W, subdomains=subdomains, boundaries=boundaries)
mu_range = [(1e-4, 1.0), (1e-4, 1.0)]
geostrophic_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a POD-Galerkin method
pod_galerkin_method = PODGalerkin(geostrophic_problem)
pod_galerkin_method.set_Nmax(20)

# 5. Perform the offline phase
pod_galerkin_method.initialize_training_set(100, sampling=LogUniformDistribution())
reduced_geostrophic_problem = pod_galerkin_method.offline()

# 6. Perform an online solve
online_mu = (1e-4, (7e4 / 1e6)**3)
reduced_geostrophic_problem.set_mu(online_mu)
reduced_geostrophic_problem.solve()
reduced_geostrophic_problem.export_solution(filename="online_solution")

# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(100, sampling=LogUniformDistribution())
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.speedup_analysis()
