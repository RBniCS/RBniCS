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
from shape_parametrization.problems import *
from shape_parametrization.reduction_methods import *

@PyGeM("FFD", "data/naca0012.prm", {
    ((1, 1, 0), "y"): 0,
    ((1, 2, 0), "y"): 1,
    ((2, 1, 0), "y"): 2,
    ((2, 2, 0), "y"): 3,
    ((3, 1, 0), "y"): 4,
    ((3, 2, 0), "y"): 5,
    ((4, 1, 0), "y"): 6,
    ((4, 2, 0), "y"): 7,
})
@ExactParametrizedFunctions()
class NACA0012(EllipticCoerciveProblem):
    
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
        
    # Return the alpha_lower bound.
    def get_stability_factor(self):
        return 1.
    
    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        if term == "a":
            return (1.,)
        elif term == "f":
            return (1.,)
        else:
            raise ValueError("Invalid term for compute_theta().")
                
    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        ds = self.ds
        if term == "a":
            u = self.u
            a0 = inner(grad(u), grad(v))*dx
            return (a0,)
        elif term == "f":
            f0 = v*ds(3)
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant(0.0), self.boundaries, 5)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(grad(u), grad(v))*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
    
    # Also compute pressure using Bernoulli equation
    def export_solution(self, folder=None, filename=None, solution=None, component=None, suffix=None):
        if filename is None:
            filename = "solution"
        # Export solution
        EllipticCoerciveProblem.export_solution(self, folder, filename, solution, component, suffix)
        # Export pressure
        if solution is None:
            solution = self._solution
        pressure = project(-0.5*grad(solution)**2)
        EllipticCoerciveProblem.export_solution(self, folder, filename + "_p", pressure)
        
        
# 1. Read the mesh for this problem
mesh = Mesh("data/naca0012.xml")
subdomains = MeshFunction("size_t", mesh, "data/naca0012_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/naca0012_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the naca0012 class
naca0012_problem = NACA0012(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(-0.1, 0.1)]*8
naca0012_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
reduced_basis_method = ReducedBasis(naca0012_problem)
reduced_basis_method.set_Nmax(70, DEIM=200)
reduced_basis_method.set_tolerance(1e-4, DEIM=1e-3)

# 5. Perform the offline phase
first_mu = (0., )*8
naca0012_problem.set_mu(first_mu)
reduced_basis_method.initialize_training_set(200, DEIM=500)
reduced_naca0012_problem = reduced_basis_method.offline()

# 6. Perform an online solve
online_mu = (0.1, )*8
reduced_naca0012_problem.set_mu(online_mu)
reduced_naca0012_problem.solve()
reduced_naca0012_problem.export_solution(filename="online_solution")

# 7. Perform an error analysis
reduced_basis_method.initialize_testing_set(100, DEIM=300)
reduced_basis_method.error_analysis()

# 8. Perform a speedup analysis
reduced_basis_method.speedup_analysis()
