# Copyright (C) 2015-2022 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *
from problems import *
from reduction_methods import *


@PullBackFormsToReferenceDomain()
@ShapeParametrization(
    ("x[0]", "x[1]"),  # subdomain 1
    ("mu[0]*(x[0] - 1) + 1", "x[1]"),  # subdomain 2
)
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
        # Store advection and forcing expressions
        self.vel = Expression("x[1]*(1-x[1])", element=self.V.ufl_element())
        self.f = Constant(0.)
        # Store terms related to stabilization
        self.delta = 1.0
        self.h = CellDiameter(V.mesh())

    # Return custom problem name
    def name(self):
        return "AdvectionDominated3POD"

    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term == "a":
            theta_a0 = 10.0**(- mu[1])
            theta_a1 = 1.0
            if self.stabilized:
                delta = self.delta
                theta_a2 = - delta * 10.0**(- mu[1])
                theta_a3 = delta
            else:
                theta_a2 = 0.0
                theta_a3 = 0.0
            return (theta_a0, theta_a1, theta_a2, theta_a3)
        elif term == "f":
            theta_f0 = 1.0
            if self.stabilized:
                delta = self.delta
                theta_f1 = delta
            else:
                theta_f1 = 0.0
            return (theta_f0, theta_f1)
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
            vel = self.vel
            h = self.h
            a0 = inner(grad(u), grad(v)) * dx
            a1 = vel * u.dx(0) * v * dx
            a2 = inner(div(grad(u)), h * vel * v.dx(0)) * dx
            a3 = inner(vel * u.dx(0), h * vel * v.dx(0)) * dx
            return (a0, a1, a2, a3)
        elif term == "f":
            f = self.f
            vel = self.vel
            h = self.h
            f0 = f * v * dx
            f1 = f * h * vel * v.dx(0) * dx
            return (f0, f1)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant(0.0), self.boundaries, 1),
                   DirichletBC(self.V, Constant(1.0), self.boundaries, 2),
                   DirichletBC(self.V, Constant(1.0), self.boundaries, 3),
                   DirichletBC(self.V, Constant(1.0), self.boundaries, 5),
                   DirichletBC(self.V, Constant(1.0), self.boundaries, 6),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 7),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 8)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(grad(u), grad(v)) * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")


# 1. Read the mesh for this problem
mesh = Mesh("data/graetz.xml")
subdomains = MeshFunction("size_t", mesh, "data/graetz_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/graetz_facet_region.xml")

# 2. Create Finite Element space (Lagrange P2)
V = FunctionSpace(mesh, "Lagrange", 2)

# 3. Allocate an object of the AdvectionDominated class
problem = AdvectionDominated(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.5, 4.0), (0.0, 6.0)]
problem.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
reduction_method = PODGalerkin(problem)
reduction_method.set_Nmax(50)
reduction_method.set_tolerance(1e-7)

# 5. Perform the offline phase
lifting_mu = (1.0, 1.0)
problem.set_mu(lifting_mu)
reduction_method.initialize_training_set(200)
reduced_problem = reduction_method.offline()

# 6. Perform an online solve
online_mu = (3.0, 5.0)
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
