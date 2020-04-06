# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *


@SCM()
@PullBackFormsToReferenceDomain()
@ShapeParametrization(
    ("x[0]", "x[1]"),  # subdomain 1
    ("mu[0]*(x[0] - 1) + 1", "x[1]"),  # subdomain 2
)
class Graetz(EllipticCoerciveProblem):

    # Default initialization of members
    @generate_function_space_for_stability_factor
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
        # Store the velocity expression
        self.vel = Expression("x[1]*(1-x[1])", element=self.V.ufl_element())
        # Customize eigen solver parameters
        self._eigen_solver_parameters.update({
            "bounding_box_minimum": {
                "problem_type": "gen_hermitian", "spectral_transform": "shift-and-invert",
                "spectral_shift": 1.e-5, "linear_solver": "mumps"
            },
            "bounding_box_maximum": {
                "problem_type": "gen_hermitian", "spectral_transform": "shift-and-invert",
                "spectral_shift": 1.e5, "linear_solver": "mumps"
            },
            "stability_factor": {
                "problem_type": "gen_hermitian", "spectral_transform": "shift-and-invert",
                "spectral_shift": 1.e-5, "linear_solver": "mumps"
            }
        })

    # Return custom problem name
    def name(self):
        return "Graetz2"

    # Return theta multiplicative terms of the affine expansion of the problem.
    @compute_theta_for_stability_factor
    def compute_theta(self, term):
        mu = self.mu
        if term == "a":
            theta_a0 = mu[1]
            theta_a1 = 1.0
            return (theta_a0, theta_a1)
        elif term == "f":
            theta_f0 = 1.0
            return (theta_f0, )
        elif term == "dirichlet_bc":
            theta_bc0 = mu[2]
            theta_bc1 = mu[3]
            return (theta_bc0, theta_bc1)
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    @assemble_operator_for_stability_factor
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            vel = self.vel
            a0 = inner(grad(u), grad(v)) * dx
            a1 = vel * u.dx(0) * v * dx
            return (a0, a1)
        elif term == "f":
            f0 = Constant(0.0) * v * dx
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant(0.0), self.boundaries, 1),
                   DirichletBC(self.V, Constant(1.0), self.boundaries, 2),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 3),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 5),
                   DirichletBC(self.V, Constant(1.0), self.boundaries, 6),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 7),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 8)]
            bc1 = [DirichletBC(self.V, Constant(0.0), self.boundaries, 1),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 2),
                   DirichletBC(self.V, Constant(1.0), self.boundaries, 3),
                   DirichletBC(self.V, Constant(1.0), self.boundaries, 5),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 6),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 7),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 8)]
            return (bc0, bc1)
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

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the Graetz class
graetz_problem = Graetz(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.1, 10.0), (0.01, 10.0), (0.5, 1.5), (0.5, 1.5)]
graetz_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
reduced_basis_method = ReducedBasis(graetz_problem)
reduced_basis_method.set_Nmax(30, SCM=50)
reduced_basis_method.set_tolerance(1e-5, SCM=1e-3)

# 5. Perform the offline phase
lifting_mu = (1.0, 1.0, 1.0, 1.0)
graetz_problem.set_mu(lifting_mu)
reduced_basis_method.initialize_training_set(500, SCM=250)
reduced_graetz_problem = reduced_basis_method.offline()

# 6. Perform an online solve
online_mu = (10.0, 0.01, 1.0, 1.0)
reduced_graetz_problem.set_mu(online_mu)
reduced_graetz_problem.solve()
reduced_graetz_problem.export_solution(filename="online_solution")

# 7. Perform an error analysis
reduced_basis_method.initialize_testing_set(100, SCM=100)
reduced_basis_method.error_analysis(filename="error_analysis")

# 8. Perform a speedup analysis
reduced_basis_method.speedup_analysis(filename="speedup_analysis")

# 9. Perform an error analysis employing a smaller number of SCM constraints
reduced_basis_method.error_analysis(SCM=5, filename="error_analysis__SCM_5")

# 10. Perform a speedup analysis employing a smaller number of SCM constraints
reduced_basis_method.speedup_analysis(SCM=5, filename="speedup_analysis__SCM_5")

# 11. Perform an error analysis with respect to the exact problem,
#     for which SCM is replaced by ExactCoercivityConstant
reduced_basis_method.error_analysis(with_respect_to=exact_problem, filename="error_analysis__with_respect_to_exact")

# 12. Perform a speedup analysis with respect to the exact problem
reduced_basis_method.speedup_analysis(with_respect_to=exact_problem, filename="speedup_analysis__with_respect_to_exact")
