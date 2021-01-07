# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *


@DEIM("online")
@ExactParametrizedFunctions("offline")
class Gaussian(EllipticCoerciveProblem):

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
        self.f = ParametrizedExpression(
            self, "exp( - 2*pow(x[0]-mu[0], 2) - 2*pow(x[1]-mu[1], 2) )", mu=(0., 0.), element=V.ufl_element())
        # note that we cannot use self.mu in the initialization of self.f, because self.mu has not been initialized yet

    # Return custom problem name
    def name(self):
        return "GaussianDEIMOnlineExactOffline"

    # Return the alpha_lower bound.
    def get_stability_factor_lower_bound(self):
        return 1.

    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        if term == "a":
            return (1., )
        elif term == "f":
            return (1., )
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            a0 = inner(grad(u), grad(v)) * dx
            return (a0,)
        elif term == "f":
            f = self.f
            f0 = f * v * dx
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant(0.0), self.boundaries, 1),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 2),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 3)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(grad(u), grad(v)) * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")


# 1. Read the mesh for this problem
mesh = Mesh("data/gaussian.xml")
subdomains = MeshFunction("size_t", mesh, "data/gaussian_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/gaussian_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the Gaussian class
problem = Gaussian(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(-1.0, 1.0), (-1.0, 1.0)]
problem.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
reduction_method = ReducedBasis(problem)
reduction_method.set_Nmax(20, DEIM=21)
reduction_method.set_tolerance(1e-4, DEIM=1e-8)

# 5. Perform the offline phase
reduction_method.initialize_training_set(50, DEIM=60)
reduced_problem = reduction_method.offline()

# 6. Perform an online solve
online_mu = (0.3, -1.0)
reduced_problem.set_mu(online_mu)
reduced_problem.solve()
reduced_problem.export_solution(filename="online_solution")
reduced_problem.solve(DEIM=11)
reduced_problem.export_solution(filename="online_solution__DEIM_11")
reduced_problem.solve(DEIM=1)
reduced_problem.export_solution(filename="online_solution__DEIM_1")

# 7. Perform an error analysis
reduction_method.initialize_testing_set(50, DEIM=60)
reduction_method.error_analysis(filename="error_analysis")

# 8. Perform a speedup analysis
reduction_method.speedup_analysis(filename="speedup_analysis")

# 9. Perform an error analysis, but employing a smaller number of DEIM basis functions
reduction_method.error_analysis(DEIM=11, filename="error_analysis__DEIM_11")

# 10. Perform a speedup analysis, but employing a smaller number of DEIM basis functions
reduction_method.speedup_analysis(DEIM=11, filename="speedup_analysis__DEIM_11")
