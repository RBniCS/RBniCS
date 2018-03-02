# Copyright (C) 2015-2018 by the RBniCS authors
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

from dolfin import *
from rbnics import *
from problems import *
from reduction_methods import *
from sampling.distributions import *
from sampling.weights import *

@WeightedUncertaintyQuantification()
class WeightedThermalBlock(EllipticCoerciveProblem):
    
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
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        
    # Return custom problem name
    def name(self):
        return "WeightedThermalBlock1POD"
    
    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu1 = self.mu[0]
        mu2 = self.mu[1]
        mu3 = self.mu[2]
        mu4 = self.mu[3]
        mu5 = self.mu[4]
        mu6 = self.mu[5]
        mu7 = self.mu[6]
        mu8 = self.mu[7]
        mu9 = self.mu[8]
        if term == "a":
            theta_a0 = mu1
            theta_a1 = mu2
            theta_a2 = mu3
            theta_a3 = mu4
            theta_a4 = mu5
            theta_a5 = mu6
            theta_a6 = mu7
            theta_a7 = mu8
            theta_a8 = mu9
            return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7, theta_a8)
        elif term == "f":
            theta_f0 = 1.0
            return (theta_f0,)
        else:
            raise ValueError("Invalid term for compute_theta().")
    
    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            a0 = inner(grad(u), grad(v))*dx(1)
            a1 = inner(grad(u), grad(v))*dx(2)
            a2 = inner(grad(u), grad(v))*dx(3)
            a3 = inner(grad(u), grad(v))*dx(4)
            a4 = inner(grad(u), grad(v))*dx(5)
            a5 = inner(grad(u), grad(v))*dx(6)
            a6 = inner(grad(u), grad(v))*dx(7)
            a7 = inner(grad(u), grad(v))*dx(8)
            a8 = inner(grad(u), grad(v))*dx(9)
            return (a0, a1, a2, a3, a4, a5, a6, a7, a8)
        elif term == "f":
            f0 = v*dx
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant(0.0), self.boundaries, i) for i in range(1, 7)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(grad(u), grad(v))*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
        
# 1. Read the mesh for this problem
mesh = Mesh("data/tblock.xml")
subdomains = MeshFunction("size_t", mesh, "data/tblock_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/tblock_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the WeightedThermalBlock class
thermal_block_problem = WeightedThermalBlock(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(1.0, 3.0) for _ in range(9)]
thermal_block_problem.set_mu_range(mu_range)
beta_a = [75.0 for _ in range(9)]
beta_b = [75.0 for _ in range(9)]

# 4. Prepare reduction with a POD-Galerkin method
pod_galerkin_method = PODGalerkin(thermal_block_problem)
pod_galerkin_method.set_Nmax(15)
pod_galerkin_method.set_tolerance(1e-5)

# 5. Perform the offline phase
first_mu = tuple([2.0 for _ in range(9)])
thermal_block_problem.set_mu(first_mu)
pod_galerkin_method.initialize_training_set(100, sampling=BetaDistribution(beta_a, beta_b), weight=BetaWeight(beta_a, beta_b))
reduced_thermal_block_problem = pod_galerkin_method.offline()

# 6. Perform an online solve
online_mu = tuple([1.0 for _ in range(9)])
reduced_thermal_block_problem.set_mu(online_mu)
reduced_thermal_block_problem.solve()
reduced_thermal_block_problem.export_solution(filename="online_solution")

# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(100, sampling=BetaDistribution(beta_a, beta_b))
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.speedup_analysis()
