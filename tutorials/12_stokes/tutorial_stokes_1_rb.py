# Copyright (C) 2015-2019 by the RBniCS authors
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
from sampling import LinearlyDependentUniformDistribution

@PullBackFormsToReferenceDomain()
@AffineShapeParametrization("data/t_bypass_vertices_mapping.vmp")
class Stokes(StokesProblem):
    
    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        StokesProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        up = TrialFunction(V)
        (self.u, self.p) = split(up)
        vq = TestFunction(V)
        (self.v, self.q) = split(vq)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        # ...
        self.f = Constant((0.0, -10.0))
        self.g = Constant(0.0)
        
    # Return custom problem name
    def name(self):
        return "Stokes1RB"
        
    # Return the lower bound for inf-sup constant.
    def get_stability_factor_lower_bound(self):
        return 1.
    
    # Return theta multiplicative terms of the affine expansion of the problem.
    @compute_theta_for_supremizers
    def compute_theta(self, term):
        if term == "a":
            theta_a0 = 1.0
            return (theta_a0, )
        elif term in ("b", "bt"):
            theta_b0 = 1.0
            return (theta_b0, )
        elif term == "f":
            theta_f0 = 1.0
            return (theta_f0, )
        elif term == "g":
            theta_g0 = 1.0
            return (theta_g0, )
        else:
            raise ValueError("Invalid term for compute_theta().")
                
    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    @assemble_operator_for_supremizers
    def assemble_operator(self, term):
        dx = self.dx
        if term == "a":
            u = self.u
            v = self.v
            a0 = inner(grad(u), grad(v))*dx
            return (a0, )
        elif term == "b":
            u = self.u
            q = self.q
            b0 = - q*div(u)*dx
            return (b0, )
        elif term == "bt":
            p = self.p
            v = self.v
            bt0 = - p*div(v)*dx
            return (bt0, )
        elif term == "f":
            v = self.v
            f0 = inner(self.f, v)*dx
            return (f0, )
        elif term == "g":
            q = self.q
            g0 = self.g*q*dx
            return (g0, )
        elif term == "dirichlet_bc_u":
            bc0 = [DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), self.boundaries, 3)]
            return (bc0,)
        elif term == "inner_product_u":
            u = self.u
            v = self.v
            x0 = inner(grad(u), grad(v))*dx
            return (x0, )
        elif term == "inner_product_p":
            p = self.p
            q = self.q
            x0 = inner(p, q)*dx
            return (x0, )
        else:
            raise ValueError("Invalid term for assemble_operator().")
        
# 1. Read the mesh for this problem
mesh = Mesh("data/t_bypass.xml")
subdomains = MeshFunction("size_t", mesh, "data/t_bypass_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/t_bypass_facet_region.xml")

# 2. Create Finite Element space (Taylor-Hood P2-P1)
element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = MixedElement(element_u, element_p)
V = FunctionSpace(mesh, element, components=[["u", "s"], "p"])

# 3. Allocate an object of the Stokes class
stokes_problem = Stokes(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [
    (0.5, 1.5),
    (0.5, 1.5),
    (0.5, 1.5),
    (0.5, 1.5),
    (0.5, 1.5),
    (0., pi/6.)
]
stokes_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a POD-Galerkin method
reduced_basis_method = ReducedBasis(stokes_problem)
reduced_basis_method.set_Nmax(25)
reduced_basis_method.set_tolerance(1e-6)

# 5. Perform the offline phase
reduced_basis_method.initialize_training_set(100, sampling=LinearlyDependentUniformDistribution())
reduced_stokes_problem = reduced_basis_method.offline()

# 6. Perform an online solve
online_mu = (1.0, 1.0, 1.0, 1.0, 1.0, pi/6.)
reduced_stokes_problem.set_mu(online_mu)
reduced_stokes_problem.solve()
reduced_stokes_problem.export_solution(filename="online_solution")

# 7. Perform an error analysis
reduced_basis_method.initialize_testing_set(100, sampling=LinearlyDependentUniformDistribution())
reduced_basis_method.error_analysis()

# 8. Perform a speedup analysis
reduced_basis_method.speedup_analysis()
