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
## @file solve_gaussian.py
#  @brief Example 5: gaussian EIM test case
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
from RBniCS import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 5: GAUSSIAN DEIM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
@DEIM()
class Gaussian(EllipticCoerciveProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
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
        self.f = ParametrizedExpression(self, "exp( - 2*pow(x[0]-mu[0], 2) - 2*pow(x[1]-mu[1], 2) )", mu=(0., 0.), element=V.ufl_element())
        # note that we cannot use self.mu in the initialization of self.f, because self.mu has not been initialized yet
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return the alpha_lower bound.
    def get_stability_factor(self):
        return 1.
    
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        if term == "a":
            return (1., )
        elif term == "f":
            return (1., )
        else:
            raise ValueError("Invalid term for compute_theta().")
                
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            a0 = inner(grad(u),grad(v))*dx
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
            x0 = inner(grad(u),grad(v))*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
            
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
    
#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 5: MAIN PROGRAM     ~~~~~~~~~~~~~~~~~~~~~~~~~# 

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
reduced_basis_method.set_Nmax(20, DEIM=21)

# 5. Perform the offline phase
first_mu = (0.5,1.0)
gaussian_problem.set_mu(first_mu)
reduced_basis_method.initialize_training_set(50, DEIM=60)
reduced_gaussian_problem = reduced_basis_method.offline()

# 6. Perform an online solve
online_mu = (0.3,-1.0)
reduced_gaussian_problem.set_mu(online_mu)
reduced_gaussian_problem.solve()
reduced_gaussian_problem.export_solution("Gaussian", "online_solution")
reduced_gaussian_problem.solve(DEIM=11)
reduced_gaussian_problem.export_solution("Gaussian", "online_solution__DEIM_11")
reduced_gaussian_problem.solve(DEIM=1)
reduced_gaussian_problem.export_solution("Gaussian", "online_solution__DEIM_1")

# 7. Perform an error analysis
reduced_basis_method.initialize_testing_set(50, DEIM=60)
reduced_basis_method.error_analysis()

# 8. Perform a speedup analysis
reduced_basis_method.speedup_analysis()

# 9. Define a new class corresponding to the exact version of Gaussian,
#    for which DEIM is replaced by ExactParametrizedFunctions
ExactGaussian = ExactProblem(Gaussian)

# 10. Allocate an object of the ExactGaussian class
exact_gaussian_problem = ExactGaussian(V, subdomains=subdomains, boundaries=boundaries)
exact_gaussian_problem.set_mu_range(mu_range)

# 11. Perform an error analysis with respect to the exact problem
reduced_basis_method.error_analysis(with_respect_to=exact_gaussian_problem)

# 12. Perform a speedup analysis with respect to the exact problem
reduced_basis_method.speedup_analysis(with_respect_to=exact_graetz_problem)

# 13. Perform an error analysis with respect to the exact problem, but
#     employing a smaller number of DEIM basis functions
reduced_basis_method.error_analysis(with_respect_to=exact_gaussian_problem, DEIM=11)

# 14. Perform a speedup analysis with respect to the exact problem, but
#     employing a smaller number of DEIM basis functions
reduced_basis_method.speedup_analysis(with_respect_to=exact_graetz_problem, DEIM=11)
