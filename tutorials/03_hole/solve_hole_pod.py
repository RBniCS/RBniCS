# Copyright (C) 2015-2017 by the RBniCS authors
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

@PullBackFormsToReferenceDomain("a", "f")
@AffineShapeParametrization("data/hole_vertices_mapping.pkl")
class Hole(EllipticCoerciveProblem):
    
    ## Default initialization of members
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
        self.subdomains = subdomains
        self.boundaries = boundaries
        
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term == "a":
            theta_a0 = 1.0
            theta_a1 = mu[2]
            # Return
            return (theta_a0, theta_a1)
        elif term == "f":
            theta_f0 = 1.0
            # Return
            return (theta_f0, )
        else:
            raise ValueError("Invalid term for compute_theta().")
                
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        u = self.u
        v = self.v
        dx = self.dx
        ds = self.ds
        if term == "a":
            a0 = inner(grad(u), grad(v))*dx
            a1 = inner(u,v)*ds(5) + inner(u,v)*ds(6) + inner(u,v)*ds(7) + inner(u,v)*ds(8)
            # Return
            return (a0, a1)
        elif term == "f":
            f0 = v*ds(1) + v*ds(2) + v*ds(3) + v*ds(4)
            # Return
            return (f0, )
        elif term == "inner_product":
            x0 = u*v*dx + inner(grad(u),grad(v))*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
                    
# 1. Read the mesh for this problem
mesh = Mesh("data/hole.xml")
subdomains = MeshFunction("size_t", mesh, "data/hole_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/hole_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the Hole class
hole_problem = Hole(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.5, 1.5), (0.5, 1.5), (0.01, 1.0)]
hole_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a POD-Galerkin method
pod_galerkin_method = PODGalerkin(hole_problem)
pod_galerkin_method.set_Nmax(20)
pod_galerkin_method.set_tolerance(1e-6)

# 5. Perform the offline phase
pod_galerkin_method.initialize_training_set(500)
reduced_hole_problem = pod_galerkin_method.offline()

# 6. Perform an online solve
online_mu = (0.5, 0.5, 0.01)
reduced_hole_problem.set_mu(online_mu)
reduced_hole_problem.solve()
reduced_hole_problem.export_solution(filename="online_solution")

# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(500)
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.initialize_testing_set(100)
pod_galerkin_method.speedup_analysis()
