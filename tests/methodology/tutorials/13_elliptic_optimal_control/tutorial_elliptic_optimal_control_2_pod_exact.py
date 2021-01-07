# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *


@ExactParametrizedFunctions()
class EllipticOptimalControl(EllipticOptimalControlProblem):

    # Default initialization of members
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
        # Store the velocity expression
        self.vel = Expression("x[1]*(1-x[1])", element=self.V.sub(0).ufl_element())
        # Customize linear solver parameters
        self._linear_solver_parameters.update({
            "linear_solver": "mumps"
        })

    # Return custom problem name
    def name(self):
        return "EllipticOptimalControl2PODExact"

    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term in ("a", "a*"):
            theta_a0 = 1.0 / mu[0]
            theta_a1 = 1.0
            return (theta_a0, theta_a1)
        elif term in ("c", "c*"):
            theta_c0 = 1.0
            return (theta_c0,)
        elif term == "m":
            theta_m0 = 1.0
            return (theta_m0,)
        elif term == "n":
            theta_n0 = self.alpha
            return (theta_n0,)
        elif term == "f":
            theta_f0 = 1.0
            return (theta_f0,)
        elif term == "g":
            theta_g0 = mu[1]
            theta_g1 = mu[2]
            return (theta_g0, theta_g1)
        elif term == "h":
            theta_h0 = 0.24 * mu[1]**2 + 0.52 * mu[2]**2
            return (theta_h0,)
        elif term == "dirichlet_bc_y":
            theta_bc0 = 1.
            return (theta_bc0,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        dx = self.dx
        if term == "a":
            y = self.y
            q = self.q
            vel = self.vel
            a0 = inner(grad(y), grad(q)) * dx
            a1 = vel * y.dx(0) * q * dx
            return (a0, a1)
        elif term == "a*":
            z = self.z
            p = self.p
            vel = self.vel
            as0 = inner(grad(z), grad(p)) * dx
            as1 = - vel * p.dx(0) * z * dx
            return (as0, as1)
        elif term == "c":
            u = self.u
            q = self.q
            c0 = u * q * dx
            return (c0,)
        elif term == "c*":
            v = self.v
            p = self.p
            cs0 = v * p * dx
            return (cs0,)
        elif term == "m":
            y = self.y
            z = self.z
            m0 = y * z * dx(1) + y * z * dx(2)
            return (m0,)
        elif term == "n":
            u = self.u
            v = self.v
            n0 = u * v * dx
            return (n0,)
        elif term == "f":
            q = self.q
            f0 = Constant(0.0) * q * dx
            return (f0,)
        elif term == "g":
            z = self.z
            g0 = z * dx(1)
            g1 = z * dx(2)
            return (g0, g1)
        elif term == "h":
            h0 = 1.0
            return (h0,)
        elif term == "dirichlet_bc_y":
            bc0 = [DirichletBC(self.V.sub(0), Constant(i), self.boundaries, i) for i in (1, 2)]
            return (bc0,)
        elif term == "dirichlet_bc_p":
            bc0 = [DirichletBC(self.V.sub(2), Constant(0.0), self.boundaries, i) for i in (1, 2)]
            return (bc0,)
        elif term == "inner_product_y":
            y = self.y
            z = self.z
            x0 = inner(grad(y), grad(z)) * dx
            return (x0,)
        elif term == "inner_product_u":
            u = self.u
            v = self.v
            x0 = u * v * dx
            return (x0,)
        elif term == "inner_product_p":
            p = self.p
            q = self.q
            x0 = inner(grad(p), grad(q)) * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")


# 1. Read the mesh for this problem
mesh = Mesh("data/mesh2.xml")
subdomains = MeshFunction("size_t", mesh, "data/mesh2_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/mesh2_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
scalar_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = MixedElement(scalar_element, scalar_element, scalar_element)
V = FunctionSpace(mesh, element, components=["y", "u", "p"])

# 3. Allocate an object of the EllipticOptimalControl class
problem = EllipticOptimalControl(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(3.0, 20.0), (0.5, 1.5), (1.5, 2.5)]
problem.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
reduction_method = PODGalerkin(problem)
reduction_method.set_Nmax(20)

# 5. Perform the offline phase
lifting_mu = (3.0, 1.0, 2.0)
problem.set_mu(lifting_mu)
reduction_method.initialize_training_set(100)
reduced_problem = reduction_method.offline()

# 6. Perform an online solve
online_mu = (15.0, 0.6, 1.8)
reduced_problem.set_mu(online_mu)
reduced_problem.solve()
reduced_problem.export_solution(filename="online_solution")
print("Reduced output for mu =", online_mu, "is", reduced_problem.compute_output())

# 7. Perform an error analysis
reduction_method.initialize_testing_set(100)
reduction_method.error_analysis()

# 8. Perform a speedup analysis
reduction_method.speedup_analysis()
