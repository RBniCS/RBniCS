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
## @file solve_gaussian.py
#  @brief Example 5: gaussian EIM test case
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
from RBniCS import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 5: GAUSSIAN EIM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
@EIM(
    "exp( - 2*pow(x[0]-mu[0], 2) - 2*pow(x[1]-mu[1], 2) )"
)
class Gaussian(EllipticCoerciveProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, subd, bound):
        # Call the standard initialization
        super(Gaussian, self).__init__(V, bc_list)
        # ... and also store FEniCS data structures for assembly
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=subd)
        self.ds = Measure("ds")(subdomain_data=bound)
        # Finally, initialize an EIM object for the interpolation of the forcing term
        self.EIM_N = None # if None, use the maximum number of EIM basis functions, otherwise use EIM_N
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return the alpha_lower bound.
    def get_alpha_lb(self):
        return 1.
    
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        if term == "a":
            return (1., )
        elif term == "f":
            self.EIM[0].setmu(self.mu)
            return self.EIM[0].compute_interpolated_theta(self.EIM_N)
        elif term == "dirichlet_bc":
            return (0.,)
        else:
            raise RuntimeError("Invalid term for compute_theta().")
                
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        if term == "a":
            return (self.S,)
        elif term == "f":
            v = self.v
            dx = self.dx
            # Call EIM
            self.EIM[0].setmu(self.mu)
            interpolated_gaussian = self.EIM[0].assemble_mu_independent_interpolated_function()
            # Assemble
            all_f = ()
            for q in range(len(interpolated_gaussian)):
                all_f += (interpolated_gaussian[q]*v*dx,)
            # Return
            return all_f
        elif term == "dirichlet_bc":
            bc0 = [(self.V, Constant(0.0), self.bound, 1),
                   (self.V, Constant(0.0), self.bound, 2),
                   (self.V, Constant(0.0), self.bound, 3)]
            return (bc0,)
        elif term == "inner_product":
            x0 = inner(grad(u),grad(v))*dx
            return (x0,)
        else:
            raise RuntimeError("Invalid term for assemble_operator().")
            
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
    
#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 5: MAIN PROGRAM     ~~~~~~~~~~~~~~~~~~~~~~~~~# 

# 1. Read the mesh for this problem
mesh = Mesh("data/gaussian.xml")
subd = MeshFunction("size_t", mesh, "data/gaussian_physical_region.xml")
bound = MeshFunction("size_t", mesh, "data/gaussian_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the Gaussian class
gaussian_problem = Gaussian(V, subd, bound)
mu_range = [(-1.0, 1.0), (-1.0, 1.0)]
gaussian_problem.setmu_range(mu_range)

# 4. Choose PETSc solvers as linear algebra backend
parameters.linear_algebra_backend = 'PETSc'

# 5. Prepare reduction with a reduced basis method
reduced_basis_method = ReducedBasis(graetz_problem)
reduced_basis_method.setNmax(20)

# 6. Perform the offline phase
first_mu = (0.5,1.0)
gaussian_problem.setmu(first_mu)
reduced_basis_method.setxi_train(50)
reduced_gaussian_problem = reduced_basis_method.offline()

# 7. Perform an online solve
online_mu = (0.3,-1.0)
reduced_gaussian_problem.setmu(online_mu)
reduced_gaussian_problem.online_solve()

# 8. Perform an error analysis
reduced_basis_method.setxi_test(50)
reduced_basis_method.error_analysis()
