# Copyright (C) 2015-2022 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *
from problems import *
from reduction_methods import *


@OnlineRectification()
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
        self.beta = Constant((1.0, 1.0))
        self.f = Constant(1.0)
        # Store terms related to stabilization
        self.delta = 0.5
        self.h = CellDiameter(V.mesh())

    # Return custom problem name
    def name(self):
        return "AdvectionDominated1RBRectification"

    # Return stability factor
    def get_stability_factor_lower_bound(self):
        return 1.

    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term == "a":
            theta_a0 = 10.0**(- mu[0])
            theta_a1 = 1.0
            if self.stabilized:
                delta = self.delta
                theta_a2 = - delta * 10.0**(- mu[0])
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
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            beta = self.beta
            h = self.h
            a0 = inner(grad(u), grad(v)) * dx
            a1 = inner(beta, grad(u)) * v * dx
            a2 = inner(div(grad(u)), h * inner(beta, grad(v))) * dx
            a3 = inner(inner(beta, grad(u)), h * inner(beta, grad(v))) * dx
            return (a0, a1, a2, a3)
        elif term == "f":
            f = self.f
            beta = self.beta
            h = self.h
            f0 = f * v * dx
            f1 = inner(f, h * inner(beta, grad(v))) * dx
            return (f0, f1)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant(0.0), self.boundaries, 1),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 2)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(grad(u), grad(v)) * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")


def bool_to_string(bool_):
    return "with" if bool_ else "without"


# 1. Read the mesh for this problem
mesh = Mesh("data/square.xml")
subdomains = MeshFunction("size_t", mesh, "data/square_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/square_facet_region.xml")

# 2. Create Finite Element space (Lagrange P2)
V = FunctionSpace(mesh, "Lagrange", 2)

# 3. Allocate an object of the AdvectionDominated class
problem = AdvectionDominated(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.0, 6.0)]
problem.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
reduction_method = ReducedBasis(problem)
reduction_method.set_Nmax(15)

# 5. Perform the offline phase
reduction_method.initialize_training_set(100)
reduced_problem = reduction_method.offline()

# 6. Perform an online solve
online_mu = (6.0, )
reduced_problem.set_mu(online_mu)
for online_stabilization in (True, False):
    for online_rectification in (True, False):
        def filename(prefix):
            return ("online_" + prefix + "_" + bool_to_string(online_stabilization) + "_stabilization_and_"
                    + bool_to_string(online_rectification) + "_rectification")
        reduced_problem.solve(
            online_stabilization=online_stabilization, online_rectification=online_rectification)
        reduced_problem.export_solution(filename=filename("solution"))
        reduced_problem.export_error(filename=filename("error"))

# 7. Perform an error analysis
reduction_method.initialize_testing_set(100)
for online_stabilization in (True, False):
    for online_rectification in (True, False):
        filename = ("error_analysis_" + bool_to_string(online_stabilization) + "_stabilization_and_"
                    + bool_to_string(online_rectification) + "_rectification")
        reduction_method.error_analysis(
            online_stabilization=online_stabilization, online_rectification=online_rectification, filename=filename)

# 8. Perform a speedup analysis
for online_stabilization in (True, False):
    for online_rectification in (True, False):
        filename = ("speedup_analysis_" + bool_to_string(online_stabilization) + "_stabilization_and_"
                    + bool_to_string(online_rectification) + "_rectification")
        reduction_method.speedup_analysis(
            online_stabilization=online_stabilization, online_rectification=online_rectification, filename=filename)
