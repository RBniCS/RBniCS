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

@SCM()
@ShapeParametrization(
    ("x[0]", "x[1]"), # subdomain 1
    ("mu[0]*(x[0] - 1) + 1", "x[1]"), # subdomain 2
)
class Graetz(EllipticCoerciveProblem):
    
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
        self.lifting = self.solve_lifting()
        # Store the velocity expression
        self.vel = Expression("x[1]*(1-x[1])", element=self.V.ufl_element())
        
    ## Return custom problem name
    def name(self):
        return "Graetz1"
        
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu1 = self.mu[0]
        mu2 = self.mu[1]
        if term == "a":
            theta_a0 = mu2
            theta_a1 = mu2/mu1
            theta_a2 = mu1*mu2
            theta_a3 = 1.0
            return (theta_a0, theta_a1, theta_a2, theta_a3)
        elif term == "f":
            theta_f0 = - mu2
            theta_f1 = - mu2/mu1
            theta_f2 = - mu1*mu2
            theta_f3 = - 1.0
            return (theta_f0, theta_f1, theta_f2, theta_f3)
        elif term == "s":
            return (1.0,)
        else:
            raise ValueError("Invalid term for compute_theta().")
                    
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            vel = self.vel
            a0 = inner(grad(u),grad(v))*dx(1)
            a1 = u.dx(0)*v.dx(0)*dx(2)
            a2 = u.dx(1)*v.dx(1)*dx(2)
            a3 = vel*u.dx(0)*v*dx(1) + vel*u.dx(0)*v*dx(2)
            return (a0, a1, a2, a3)
        elif term == "f":
            lifting = self.lifting
            vel = self.vel
            f0 = inner(grad(lifting),grad(v))*dx(1)
            f1 = lifting.dx(0)*v.dx(0)*dx(2)
            f2 = lifting.dx(1)*v.dx(1)*dx(2)
            f3 = vel*lifting.dx(0)*v*dx(1) + vel*lifting.dx(0)*v*dx(2)
            return (f0, f1, f2, f3)
        elif term == "s":
            ds = self.ds
            s0 = v*ds(4)
            return (s0,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant(0.0), self.boundaries, 1),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 2),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 3),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 5),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 6),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 7),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 8)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(grad(u),grad(v))*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
        
    def solve_lifting(self):
        # We will consider non-homogeneous Dirichlet BCs with a lifting.
        # First of all, assemble a suitable lifting function
        lifting_bc = [DirichletBC(self.V, Constant(0.0), self.boundaries, 1), # homog. bcs
                      DirichletBC(self.V, Constant(1.0), self.boundaries, 2), # non-homog. bcs
                      DirichletBC(self.V, Constant(1.0), self.boundaries, 3), # non-homog. bcs
                      DirichletBC(self.V, Constant(1.0), self.boundaries, 5), # non-homog. bcs
                      DirichletBC(self.V, Constant(1.0), self.boundaries, 6), # non-homog. bcs
                      DirichletBC(self.V, Constant(0.0), self.boundaries, 7), # homog. bcs
                      DirichletBC(self.V, Constant(0.0), self.boundaries, 8)] # homog. bcs
        u = self.u
        v = self.v
        dx = self.dx
        lifting_a = inner(grad(u),grad(v))*dx
        lifting_A = assemble(lifting_a)
        lifting_f = Constant(0.)*v*dx
        lifting_F = assemble(lifting_f)
        [bc.apply(lifting_A) for bc in lifting_bc] # Apply BCs on LHS
        [bc.apply(lifting_F) for bc in lifting_bc] # Apply BCs on RHS
        lifting = Function(V)
        solve(lifting_A, lifting.vector(), lifting_F)
        return lifting
        
    ## Preprocess the solution before export to add lifting
    def export_solution(self, folder=None, filename=None, solution=None, component=None, suffix=None):
        assert component is None
        assert suffix is None
        if solution is None:
            solution = self._solution
        solution_with_lifting = Function(self.V)
        solution_with_lifting.vector()[:] = solution.vector()[:] + self.lifting.vector()[:]
        EllipticCoerciveProblem.export_solution(self, folder, filename, solution_with_lifting)
        
    ## Preprocess the solution after import to remove lifting
    def import_solution(self, folder=None, filename=None, solution=None, suffix=None):
        assert suffix is None
        solution_with_lifting = Function(self.V)
        EllipticCoerciveProblem.import_solution(self, folder, filename, solution_with_lifting)
        if solution is None:
            solution = self._solution
        assign(solution, solution_with_lifting)
        solution.vector()[:] = solution_with_lifting.vector()[:] - self.lifting.vector()[:]
        
# 1. Read the mesh for this problem
mesh = Mesh("data/graetz.xml")
subdomains = MeshFunction("size_t", mesh, "data/graetz_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/graetz_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the Graetz class
graetz_problem = Graetz(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.1, 10.0), (0.01, 10.0)]
graetz_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
reduced_basis_method = ReducedBasis(graetz_problem)
reduced_basis_method.set_Nmax(30, dual=30, SCM=50)
reduced_basis_method.set_tolerance(1e-5, dual=1e-5, SCM=1e-3)

# 5. Perform the offline phase
first_mu = (1.0, 1.0)
graetz_problem.set_mu(first_mu)
reduced_basis_method.initialize_training_set(200, dual=200, SCM=250)
reduced_graetz_problem = reduced_basis_method.offline()

# 6. Perform an online solve
online_mu = (10.0, 0.01)
reduced_graetz_problem.set_mu(online_mu)
reduced_graetz_problem.solve()
reduced_graetz_problem.export_solution(filename="online_solution")

# 7. Perform an error analysis
reduced_basis_method.initialize_testing_set(100, dual=100, SCM=100)
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
