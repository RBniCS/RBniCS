# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file solve_graetz.py
#  @brief Example 4: Graetz test case
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
from RBniCS import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 4: GRAETZ CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
@SCM(
    constrain_minimum_eigenvalue = 1.e4,
    constrain_maximum_eigenvalue = 1.e-4,
    bounding_box_minimum_eigensolver_parameters = dict(spectral_transform="shift-and-invert", spectral_shift=1.e-5),
    bounding_box_maximum_eigensolver_parameters = dict(spectral_transform="shift-and-invert", spectral_shift=1.),
    coercivity_eigensolver_parameters = dict(spectral_transform="shift-and-invert", spectral_shift=1.e-5)
)
@ShapeParametrization(
    ("x[0]", "x[1]"), # subdomain 1
    ("mu[0]*(x[0] - 1) + 1", "x[1]"), # subdomain 2
)
class Graetz(EllipticCoerciveProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        EllipticCoerciveProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs and "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=subdomains)
        self.ds = Measure("ds")(subdomain_data=boundaries)
        self.lifting = self.solve_lifting()
        # Store the velocity expression
        self.vel = Expression("x[1]*(1-x[1])", element=self.V.ufl_element())
                
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
        
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
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
            raise RuntimeError("Invalid term for compute_theta().")
                    
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            vel = self.vel
            a0 = inner(grad(u),grad(v))*dx(1) + 1e-15*u*v*dx
            a1 = u.dx(0)*v.dx(0)*dx(2) + 1e-15*u*v*dx
            a2 = u.dx(1)*v.dx(1)*dx(2) + 1e-15*u*v*dx
            a3 = vel*u.dx(0)*v*dx(1) + vel*u.dx(0)*v*dx(2) + 1e-15*u*v*dx
            return (a0, a1, a2, a3)
        elif term == "f":
            lifting = self.lifting
            vel = self.vel
            f0 = inner(grad(lifting),grad(v))*dx(1) + 1e-15*lifting*v*dx
            f1 = lifting.dx(0)*v.dx(0)*dx(2) + 1e-15*lifting*v*dx
            f2 = lifting.dx(1)*v.dx(1)*dx(2) + 1e-15*lifting*v*dx
            f3 = vel*lifting.dx(0)*v*dx(1) + vel*lifting.dx(0)*v*dx(2) + 1e-15*lifting*v*dx
            return (f0, f1, f2, f3)
        elif term == "s":
            s0 = v*dx(2)
            return (s0,)
        elif term == "dirichlet_bc":
            bc0 = [(self.V, Constant(0.0), self.boundaries, 1),
                   (self.V, Constant(0.0), self.boundaries, 5),
                   (self.V, Constant(0.0), self.boundaries, 6),
                   (self.V, Constant(0.0), self.boundaries, 2),
                   (self.V, Constant(0.0), self.boundaries, 4)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = u*v*dx + inner(grad(u),grad(v))*dx
            return (x0,)
        else:
            raise RuntimeError("Invalid term for assemble_operator().")
        
    def solve_lifting(self):
        # We will consider non-homogeneous Dirichlet BCs with a lifting.
        # First of all, assemble a suitable lifting function
        lifting_bc = [ 
            DirichletBC(self.V, Constant(0.0), self.boundaries, 1), # homog. bcs
            DirichletBC(self.V, Constant(0.0), self.boundaries, 5), # homog. bcs
            DirichletBC(self.V, Constant(0.0), self.boundaries, 6), # homog. bcs
            DirichletBC(self.V, Constant(1.0), self.boundaries, 2), # non-homog. bcs
            DirichletBC(self.V, Constant(1.0), self.boundaries, 4)  # non-homog. bcs
        ]
        u = self.u
        v = self.v
        dx = self.dx
        lifting_a = inner(grad(u),grad(v))*dx
        lifting_A = assemble(lifting_a)
        lifting_f = 1e-15*v*dx
        lifting_F = assemble(lifting_f)
        [bc.apply(lifting_A) for bc in lifting_bc] # Apply BCs on LHS
        [bc.apply(lifting_F) for bc in lifting_bc] # Apply BCs on RHS
        lifting = Function(V)
        solve(lifting_A, lifting.vector(), lifting_F)
        return lifting
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    ## Preprocess the solution before plotting to add a lifting
    def preprocess_solution_for_plot(self, solution):
        solution_with_lifting = Function(self.V)
        solution_with_lifting.vector()[:] = solution.vector()[:] + self.lifting.vector()[:]
        return solution_with_lifting
        
    #  @}
    ########################### end - I/O - end ########################### 
    
#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 4: MAIN PROGRAM     ~~~~~~~~~~~~~~~~~~~~~~~~~# 

# 1. Read the mesh for this problem
mesh = Mesh("data/graetz.xml")
subdomains = MeshFunction("size_t", mesh, "data/graetz_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/graetz_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the Graetz class
graetz_problem = Graetz(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.01, 10.0), (0.01, 10.0)]
graetz_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
reduced_basis_method = ReducedBasis(graetz_problem)
reduced_basis_method.set_Nmax(10, SCM=10)

# 5. Perform the offline phase
first_mu = (1.0, 1.0)
graetz_problem.set_mu(first_mu)
reduced_basis_method.set_xi_train(100)
reduced_graetz_problem = reduced_basis_method.offline()

# 6. Perform an online solve
online_mu = (10.0, 0.01)
reduced_graetz_problem.set_mu(online_mu)
reduced_graetz_problem.solve()

# 7. Perform an error analysis
reduced_basis_method.set_xi_test(100)
reduced_basis_method.error_analysis()
