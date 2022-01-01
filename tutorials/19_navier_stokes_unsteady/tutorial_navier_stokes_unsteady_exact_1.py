# Copyright (C) 2015-2022 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *


@ExactParametrizedFunctions()
class NavierStokesUnsteady(NavierStokesUnsteadyProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        NavierStokesUnsteadyProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.dup = TrialFunction(V)
        (self.du, self.dp) = split(self.dup)
        (self.u, _) = split(self._solution)
        vq = TestFunction(V)
        (self.v, self.q) = split(vq)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        # ... as well as forcing terms and inlet velocity
        self.inlet = Expression(("1./0.042025*x[1]*(0.41 - x[1])", "0."), degree=2)
        self.f = Constant((0.0, 0.0))
        self.g = Constant(0.0)
        self._time_stepping_parameters.update({
            "report": True,
            "snes_solver": {
                "linear_solver": "mumps",
                "maximum_iterations": 20,
                "report": True
            }
        })

    # Return custom problem name
    def name(self):
        return "NavierStokesUnsteadyExact1"

    # Return theta multiplicative terms of the affine expansion of the problem.
    @compute_theta_for_derivatives
    @compute_theta_for_supremizers
    def compute_theta(self, term):
        mu = self.mu
        if term == "a":
            theta_a0 = mu[0]
            return (theta_a0,)
        elif term in ("b", "bt"):
            theta_b0 = 1.
            return (theta_b0,)
        elif term == "c":
            theta_c0 = 1.
            return (theta_c0,)
        elif term == "f":
            theta_f0 = 1.
            return (theta_f0,)
        elif term == "g":
            theta_g0 = 1.
            return (theta_g0,)
        elif term == "m":
            theta_m0 = 1.
            return (theta_m0, )
        elif term == "dirichlet_bc_u":
            theta_bc0 = 1.
            return (theta_bc0,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    @assemble_operator_for_derivatives
    @assemble_operator_for_supremizers
    def assemble_operator(self, term):
        dx = self.dx
        if term == "a":
            u = self.du
            v = self.v
            a0 = inner(grad(u), grad(v)) * dx
            return (a0,)
        elif term == "b":
            u = self.du
            q = self.q
            b0 = - q * div(u) * dx
            return (b0,)
        elif term == "bt":
            p = self.dp
            v = self.v
            bt0 = - p * div(v) * dx
            return (bt0,)
        elif term == "c":
            u = self.u
            v = self.v
            c0 = inner(grad(u) * u, v) * dx
            return (c0,)
        elif term == "f":
            v = self.v
            f0 = inner(self.f, v) * dx
            return (f0,)
        elif term == "g":
            q = self.q
            g0 = self.g * q * dx
            return (g0,)
        elif term == "m":
            u = self.du
            v = self.v
            m0 = inner(u, v) * dx
            return (m0,)
        elif term == "dirichlet_bc_u":
            bc0 = [DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), self.boundaries, 1),
                   DirichletBC(self.V.sub(0), self.inlet, self.boundaries, 3),
                   DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), self.boundaries, 4)]
            return (bc0,)
        elif term == "inner_product_u":
            u = self.du
            v = self.v
            x0 = inner(grad(u), grad(v)) * dx
            return (x0,)
        elif term == "inner_product_p":
            p = self.dp
            q = self.q
            x0 = inner(p, q) * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")


# Customize the resulting reduced problem
@CustomizeReducedProblemFor(NavierStokesUnsteadyProblem)
def CustomizeReducedNavierStokesUnsteady(ReducedNavierStokesUnsteady_Base):
    class ReducedNavierStokesUnsteady(ReducedNavierStokesUnsteady_Base):
        def __init__(self, truth_problem, **kwargs):
            ReducedNavierStokesUnsteady_Base.__init__(self, truth_problem, **kwargs)
            self._time_stepping_parameters.update({
                "report": True,
                "nonlinear_solver": {
                    "report": True,
                    "line_search": "wolfe"
                }
            })

    return ReducedNavierStokesUnsteady


# 1. Read the mesh for this problem
mesh = Mesh("data/cylinder.xml")
subdomains = MeshFunction("size_t", mesh, "data/cylinder_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/cylinder_facet_region.xml")

# 2. Create Finite Element space for Stokes problem (Taylor-Hood P2-P1)
element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = MixedElement(element_u, element_p)
V = FunctionSpace(mesh, element, components=[["u", "s"], "p"])

# 3. Allocate an object of the NavierStokesUnsteady class
problem = NavierStokesUnsteady(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(1e-2, 1e-1)]
problem.set_mu_range(mu_range)
problem.set_time_step_size(0.01)
problem.set_final_time(1.0)

# 4. Prepare reduction with a POD-Galerkin method
reduction_method = PODGalerkin(problem)
reduction_method.set_Nmax(15, nested_POD=3)
reduction_method.set_tolerance(0.0, nested_POD=1e-3)

# 5. Perform the offline phase
lifting_mu = (1e-1, )
problem.set_mu(lifting_mu)
reduction_method.initialize_training_set(10)
reduced_problem = reduction_method.offline()

# 6. Perform an online solve
online_mu = (1e-2, )
reduced_problem.set_mu(online_mu)
reduced_problem.solve()
reduced_problem.export_solution(filename="online_solution")

# 7. Perform an error analysis
reduction_method.initialize_testing_set(10)
reduction_method.error_analysis()

# 8. Perform a speedup analysis
reduction_method.speedup_analysis()
