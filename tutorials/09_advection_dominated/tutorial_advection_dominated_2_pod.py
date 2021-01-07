# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *
from problems import *
from reduction_methods import *


@OnlineStabilization()
class AdvectionDominated(EllipticCoerciveProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        EllipticCoerciveProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=subdomains)
        self.ds = Measure("ds")(subdomain_data=boundaries)
        # Store forcing expression and boundary conditions
        self.f = Constant(0.)
        self.bc1 = Constant(1.0)
        self.bc2 = Expression("0.0 + 1.0*(x[0] == 0.0)*(x[1] == 0.25)", element=self.V.ufl_element())
        # Store terms related to stabilization
        self.delta = 1.0
        self.h = CellDiameter(V.mesh())

    # Return custom problem name
    def name(self):
        return "AdvectionDominated2POD"

    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term == "a":
            theta_a0 = 10**(- mu[1])
            theta_a1 = cos(mu[0])
            theta_a2 = sin(mu[0])
            if self.stabilized:
                delta = self.delta
                theta_a3 = - delta * 10**(- mu[1]) * cos(mu[0])
                theta_a4 = - delta * 10**(- mu[1]) * sin(mu[0])
                theta_a5 = delta * cos(mu[0])**2
                theta_a6 = delta * cos(mu[0]) * sin(mu[0])
                theta_a7 = delta * sin(mu[0])**2
            else:
                theta_a3 = 0.0
                theta_a4 = 0.0
                theta_a5 = 0.0
                theta_a6 = 0.0
                theta_a7 = 0.0
            return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7)
        elif term == "f":
            theta_f0 = 1.0
            if self.stabilized:
                delta = self.delta
                theta_f1 = delta * cos(mu[0])
                theta_f2 = delta * sin(mu[0])
            else:
                theta_f1 = 0.0
                theta_f2 = 0.0
            return (theta_f0, theta_f1, theta_f2)
        elif term == "dirichlet_bc":
            theta_bc0 = 1.0
            return (theta_bc0,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            h = self.h
            a0 = inner(grad(u), grad(v)) * dx
            a1 = u.dx(0) * v * dx
            a2 = u.dx(1) * v * dx
            a3 = inner(div(grad(u)), h * v.dx(0)) * dx
            a4 = inner(div(grad(u)), h * v.dx(1)) * dx
            a5 = h * u.dx(0) * v.dx(0) * dx
            a6 = h * u.dx(0) * v.dx(1) * dx + h * u.dx(1) * v.dx(0) * dx
            a7 = h * u.dx(1) * v.dx(1) * dx
            return (a0, a1, a2, a3, a4, a5, a6, a7)
        elif term == "f":
            f = self.f
            h = self.h
            f0 = f * v * dx
            f1 = f * h * v.dx(0) * dx
            f2 = f * h * v.dx(1) * dx
            return (f0, f1, f2)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, self.bc1, self.boundaries, 1),
                   DirichletBC(self.V, self.bc2, self.boundaries, 2)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(grad(u), grad(v)) * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")


# 1. Read the mesh for this problem
mesh = Mesh("data/square.xml")
subdomains = MeshFunction("size_t", mesh, "data/square_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/square_facet_region.xml")

# 2. Create Finite Element space (Lagrange P2)
V = FunctionSpace(mesh, "Lagrange", 2)

# 3. Allocate an object of the AdvectionDominated class
problem = AdvectionDominated(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.5, 1.0), (4.0, 5.0)]
problem.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
reduction_method = PODGalerkin(problem)
reduction_method.set_Nmax(50)
reduction_method.set_tolerance(1e-7)

# 5. Perform the offline phase
lifting_mu = (1.0, 4.0)
problem.set_mu(lifting_mu)
reduction_method.initialize_training_set(200)
reduced_problem = reduction_method.offline()

# 6. Perform an online solve
online_mu = (0.5, 5.0)
reduced_problem.set_mu(online_mu)
reduced_problem.solve(online_stabilization=True)
reduced_problem.export_solution(filename="online_solution_with_stabilization")
reduced_problem.solve(online_stabilization=False)
reduced_problem.export_solution(filename="online_solution_without_stabilization")

# 7. Perform an error analysis
reduction_method.initialize_testing_set(100)
reduction_method.error_analysis(online_stabilization=True, filename="error_analysis_with_stabilization")
reduction_method.error_analysis(online_stabilization=False, filename="error_analysis_without_stabilization")

# 8. Perform a speedup analysis
reduction_method.speedup_analysis(online_stabilization=True, filename="speedup_analysis_with_stabilization")
reduction_method.speedup_analysis(online_stabilization=False, filename="speedup_analysis_without_stabilization")
