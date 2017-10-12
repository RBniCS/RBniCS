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

@ShapeParametrization(
    ("mu[0]*x[0]", "x[1]"),
)
class StokesUnsteady(StokesUnsteadyProblem):
    
    ## Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        StokesUnsteadyProblem.__init__(self, V, **kwargs)
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
        #
        self.bc1 = Constant((1.0, 0.0))
        self.bc2 = Expression(("0.0 + 1.0*(x[1] == 1.0)", "0.0"), element=self.V.sub(0).ufl_element())
        self.f = Constant((0.0, 0.0))
        self.g = Constant(0.0)
        
    ## Return custom problem name
    def name(self):
        return "StokesUnsteady1"
        
    ## Return theta multiplicative terms of the affine expansion of the problem.
    @compute_theta_for_restriction({"bt_restricted": "bt"})
    def compute_theta(self, term):
        mu = self.mu
        mu1 = mu[0]
        if term == "a":
            theta_a0 = 1./mu1
            theta_a1 = mu1
            return (theta_a0, theta_a1)
        elif term in ("b", "bt"):
            theta_b0 = 1.
            theta_b1 = mu1
            return (theta_b0, theta_b1)
        elif term == "f":
            theta_f0 = mu1
            return (theta_f0, )
        elif term == "g":
            theta_g0 = mu1
            return (theta_g0, )
        elif term == "m":
            theta_m0 = mu1
            return (theta_m0, )
        elif term == "dirichlet_bc_u":
            theta_bc0 = 1.
            return (theta_bc0,)
        else:
            raise ValueError("Invalid term for compute_theta().")
                
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    @assemble_operator_for_restriction({"bt_restricted": "bt"}, test="s")
    @assemble_operator_for_restriction({"dirichlet_bc_s": "dirichlet_bc_u"}, trial="s")
    @assemble_operator_for_restriction({"inner_product_s": "inner_product_u"}, test="s", trial="s")
    def assemble_operator(self, term):
        dx = self.dx
        if term == "a":
            u = self.u
            v = self.v
            a0 = (u[0].dx(0)*v[0].dx(0) + u[1].dx(0)*v[1].dx(0))*dx
            a1 = (u[0].dx(1)*v[0].dx(1) + u[1].dx(1)*v[1].dx(1))*dx
            return (a0, a1)
        elif term == "b":
            u = self.u
            q = self.q
            b0 = - q*u[0].dx(0)*dx
            b1 = - q*u[1].dx(1)*dx
            return (b0, b1)
        elif term == "bt":
            p = self.p
            v = self.v
            bt0 = - p*v[0].dx(0)*dx
            bt1 = - p*v[1].dx(1)*dx
            return (bt0, bt1)
        elif term == "f":
            v = self.v
            f0 = inner(self.f, v)*dx
            return (f0, )
        elif term == "g":
            q = self.q
            g0 = self.g*q*dx
            return (g0, )
        elif term == "m":
            u = self.u
            v = self.v
            m0 = inner(u, v)*dx
            return (m0, )
        elif term == "dirichlet_bc_u":
            bc0 = [DirichletBC(self.V.sub(0), self.bc1, self.boundaries, 1),
                   DirichletBC(self.V.sub(0), self.bc2, self.boundaries, 2)]
            return (bc0,)
        elif term == "dirichlet_bc_p":
            class CenterDomain(SubDomain):
                def inside(self, x, on_boundary):
                    return near(x[0], 0.5, DOLFIN_EPS)
            center_domain = CenterDomain()
            bc0 = [DirichletBC(self.V.sub(1), Constant(0.), center_domain, method="pointwise")]
            return (bc0, )
        elif term == "inner_product_u":
            u = self.u
            v = self.v
            x0 = inner(grad(u),grad(v))*dx
            return (x0,)
        elif term == "inner_product_p":
            p = self.p
            q = self.q
            x0 = inner(p, q)*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")

# 1. Read the mesh for this problem
mesh = Mesh("data/cavity.xml")
subdomains = MeshFunction("size_t", mesh, "data/cavity_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/cavity_facet_region.xml")

# 2. Create Finite Element space (Taylor-Hood P2-P1)
element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_u, element_p)
V = FunctionSpace(mesh, element, components=[["u", "s"], "p"])

# 3. Allocate an object of the StokesUnsteady class
stokes_unsteady_problem = StokesUnsteady(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.5, 2.5)]
stokes_unsteady_problem.set_mu_range(mu_range)
stokes_unsteady_problem.set_time_step_size(0.01)
stokes_unsteady_problem.set_final_time(0.15)

# 4. Prepare reduction with a POD-Galerkin method
pod_galerkin_method = PODGalerkin(stokes_unsteady_problem)
pod_galerkin_method.set_Nmax(15, nested_POD=3)
pod_galerkin_method.set_tolerance(0.0, nested_POD=1e-3)

# 5. Perform the offline phase
lifting_mu = (1.0, )
stokes_unsteady_problem.set_mu(lifting_mu)
pod_galerkin_method.initialize_training_set(30)
reduced_stokes_unsteady_problem = pod_galerkin_method.offline()

# 6. Perform an online solve
online_mu = (1.0, )
reduced_stokes_unsteady_problem.set_mu(online_mu)
reduced_stokes_unsteady_problem.solve()
reduced_stokes_unsteady_problem.export_solution(filename="online_solution")

# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(30)
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.speedup_analysis()
