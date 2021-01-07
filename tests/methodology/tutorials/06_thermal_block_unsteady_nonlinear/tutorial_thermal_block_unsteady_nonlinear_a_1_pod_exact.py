# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *


@ExactParametrizedFunctions()
class UnsteadyThermalBlock(NonlinearParabolicProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        NonlinearParabolicProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        # Customize time stepping parameters
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
        return "UnsteadyThermalBlockNonlinearA1PODExact"

    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term == "m":
            theta_m0 = 1.
            return (theta_m0, )
        elif term in ("a", "c", "dc"):
            theta_a0 = 0.5 * mu[0]
            theta_a1 = 0.5
            return (theta_a0, theta_a1)
        elif term == "f":
            theta_f0 = mu[1]
            return (theta_f0,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "m":
            u = self.u
            m0 = u * v * dx
            return (m0, )
        elif term in ("a", "dc"):
            u = self.u
            a0 = inner(grad(u), grad(v)) * dx(1)
            a1 = inner(grad(u), grad(v)) * dx(2)
            return (a0, a1)
        elif term == "c":
            u = self._solution
            c0 = inner(grad(u), grad(v)) * dx(1)
            c1 = inner(grad(u), grad(v)) * dx(2)
            return (c0, c1)
        elif term == "f":
            ds = self.ds
            f0 = v * ds(1)
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant(0.0), self.boundaries, 3)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(grad(u), grad(v)) * dx
            return (x0,)
        elif term == "projection_inner_product":
            u = self.u
            x0 = u * v * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")


# Customize the resulting reduced problem
@CustomizeReducedProblemFor(NonlinearParabolicProblem)
def CustomizeReducedNonlinearParabolic(ReducedNonlinearParabolic_Base):
    class ReducedNonlinearParabolic(ReducedNonlinearParabolic_Base):
        def __init__(self, truth_problem, **kwargs):
            ReducedNonlinearParabolic_Base.__init__(self, truth_problem, **kwargs)
            self._time_stepping_parameters.update({
                "report": True,
                "nonlinear_solver": {
                    "report": True
                }
            })

    return ReducedNonlinearParabolic


# 1. Read the mesh for this problem
mesh = Mesh("data/thermal_block.xml")
subdomains = MeshFunction("size_t", mesh, "data/thermal_block_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/thermal_block_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1, two components)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the UnsteadyThermalBlock class
problem = UnsteadyThermalBlock(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.1, 10.0), (-1.0, 1.0)]
problem.set_mu_range(mu_range)
problem.set_time_step_size(0.05)
problem.set_final_time(3)

# 4. Prepare reduction with a POD-Galerkin method
reduction_method = PODGalerkin(problem)
reduction_method.set_Nmax(20, nested_POD=4)
reduction_method.set_tolerance(1e-8, nested_POD=1e-4)

# 5. Perform the offline phase
reduction_method.initialize_training_set(100)
reduced_problem = reduction_method.offline()

# 6. Perform an online solve
online_mu = (8.0, -1.0)
reduced_problem.set_mu(online_mu)
reduced_problem.solve()
reduced_problem.export_solution(filename="online_solution")

# 7. Perform an error analysis
reduction_method.initialize_testing_set(10)
reduction_method.error_analysis()

# 8. Perform a speedup analysis
reduction_method.speedup_analysis()
