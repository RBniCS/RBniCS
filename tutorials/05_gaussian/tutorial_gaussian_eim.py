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

@EIM()
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
        self.f = ParametrizedExpression(self, "exp( - 2*pow(x[0]-mu[0], 2) - 2*pow(x[1]-mu[1], 2) )", mu=(0., 0.), element=V.ufl_element())
        # note that we cannot use self.mu in the initialization of self.f, because self.mu has not been initialized yet
        
    # Return custom problem name
    def name(self):
        return "GaussianEIM"
        
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
            a0 = inner(grad(u), grad(v))*dx
            return (a0,)
        elif term == "f":
            f = self.f
            f0 = f*v*dx
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant(0.0), self.boundaries, 1),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 2),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 3)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(grad(u), grad(v))*dx
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
gaussian_problem = Gaussian(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(-1.0, 1.0), (-1.0, 1.0)]
gaussian_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
reduced_basis_method = ReducedBasis(gaussian_problem)
reduced_basis_method.set_Nmax(20, EIM=21)
reduced_basis_method.set_tolerance(1e-4, EIM=1e-3)

# 5. Perform the offline phase
reduced_basis_method.initialize_training_set(50, EIM=60)
reduced_gaussian_problem = reduced_basis_method.offline()

# 6. Perform an online solve
online_mu = (0.3, -1.0)
reduced_gaussian_problem.set_mu(online_mu)
reduced_gaussian_problem.solve()
reduced_gaussian_problem.export_solution(filename="online_solution")
reduced_gaussian_problem.solve(EIM=11)
reduced_gaussian_problem.export_solution(filename="online_solution__EIM_11")
reduced_gaussian_problem.solve(EIM=1)
reduced_gaussian_problem.export_solution(filename="online_solution__EIM_1")

# 7. Perform an error analysis
reduced_basis_method.initialize_testing_set(50, EIM=60)
reduced_basis_method.error_analysis(filename="error_analysis")

# 8. Perform a speedup analysis
reduced_basis_method.speedup_analysis(filename="speedup_analysis")

# 9. Perform an error analysis with respect to the exact problem,
#    for which EIM is replaced by ExactParametrizedFunctions
reduced_basis_method.error_analysis(with_respect_to=exact_problem, filename="error_analysis__with_respect_to_exact")

# 10. Perform a speedup analysis with respect to the exact problem
reduced_basis_method.speedup_analysis(with_respect_to=exact_problem, filename="speedup_analysis__with_respect_to_exact")

# 11. Perform an error analysis with respect to the exact problem, but
#     employing a smaller number of EIM basis functions
reduced_basis_method.error_analysis(with_respect_to=exact_problem, EIM=11, filename="error_analysis__with_respect_to_exact__EIM_11")

# 12. Perform a speedup analysis with respect to the exact problem, but
#     employing a smaller number of EIM basis functions
reduced_basis_method.speedup_analysis(with_respect_to=exact_problem, EIM=11, filename="speedup_analysis__with_respect_to_exact__DEIM_11")
