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
## @file eim.py
#  @brief Implementation of the empirical interpolation method for the interpolation of parametrized functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.problems.base import ParametrizedProblem
from RBniCS.backends import abs, copy, evaluate, export, max
from RBniCS.backends.online import OnlineAffineExpansionStorage, OnlineLinearSolver, OnlineVector, OnlineFunction
from RBniCS.utils.decorators import sync_setters, Extends, override

#~~~~~~~~~~~~~~~~~~~~~~~~~     EIM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EIM
#
# Empirical interpolation method for the interpolation of parametrized functions
@Extends(ParametrizedProblem)
class EIMApproximation(ParametrizedProblem):

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the EIM object
    #  @{

    ## Default initialization of members
    @override
    @sync_setters("truth_problem", "set_mu", "mu")
    @sync_setters("truth_problem", "set_mu_range", "mu_range")
    def __init__(self, truth_problem, parametrized_expression, folder_prefix, basis_generation):        
        # Call the parent initialization
        ParametrizedProblem.__init__(self, folder_prefix)
        # Store the parametrized expression
        self.parametrized_expression = parametrized_expression
        self.truth_problem = truth_problem
        assert basis_generation in ("Greedy", "POD")
        self.basis_generation = basis_generation
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # Online reduced space dimension
        self.N = 0
        # Define additional storage for EIM
        self.interpolation_locations = parametrized_expression.create_interpolation_locations_container() # interpolation locations selected by the greedy (either a ReducedVertices or ReducedMesh)
        self.interpolation_matrix = OnlineAffineExpansionStorage(1) # interpolation matrix
        # Solution
        self._interpolation_coefficients = OnlineFunction()
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        self.snapshot = None # will be filled in by Function, Vector or Matrix as appropriate in the EIM preprocessing
        # Basis functions container
        self.Z = parametrized_expression.create_basis_container()
        # I/O. Since we are decorating the parametrized problem we do not want to change the name of the
        # basis function/reduced operator folder, but rather add a new one. For this reason we use
        # the __eim suffix in the variable name.
        self.folder["basis"] = self.folder_prefix + "/" + "basis"
        self.folder["reduced_operators"] = self.folder_prefix + "/" + "reduced_operators"
        
        # Avoid useless linear system solves
        self._solve__previous_mu = None
        self._solve__previous_N = None
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################

    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{

    ## Initialize data structures required for the online phase
    def init(self, current_stage="online"):
        assert current_stage in ("online", "offline")
        # Read/Initialize reduced order data structures
        if current_stage == "online":
            self.interpolation_locations.load(self.folder["reduced_operators"], "interpolation_locations")
            self.interpolation_matrix.load(self.folder["reduced_operators"], "interpolation_matrix")
            self.Z.load(self.folder["basis"], "basis")
            self.N = len(self.Z)
        elif current_stage == "offline":
            # Nothing to be done
            pass
        else:
            raise AssertionError("Invalid stage in init().")

    # Perform an online solve.
    def solve(self, N=None, for_rhs=None):
        if N is None:
            N = self.N
        
        if self._solve__previous_mu != self.mu or self._solve__previous_N != N:
            self._solve(self.parametrized_expression, N)
            
            # Store to avoid repeated computations
            self._solve__previous_mu = self.mu
            self._solve__previous_N = N
        
        return self._interpolation_coefficients
        
    def _solve(self, rhs_, N=None):
        if N is None:
            N = self.N
            
        if N == 0:
            self._interpolation_coefficients = OnlineFunction()
            return
            
        # Evaluate the parametrized expression at interpolation locations
        rhs = evaluate(rhs_, self.interpolation_locations[:N])
        
        # Extract the interpolation matrix
        lhs = self.interpolation_matrix[0][:N,:N]
        
        # Solve the interpolation problem
        self._interpolation_coefficients = OnlineFunction(N)
        
        solver = OnlineLinearSolver(lhs, self._interpolation_coefficients, rhs)
        solver.solve()
        
        
    ## Call online_solve and then convert the result of online solve from OnlineVector to a tuple
    def compute_interpolated_theta(self, N=None):
        interpolated_theta = self.solve(N)
        interpolated_theta_list = list()
        for theta in interpolated_theta.vector():
            interpolated_theta_list.append(float(theta))
        if N is not None:
            # Make sure to append a 0 coefficient for each basis function
            # which has not been requested
            for n in range(N, self.N):
                interpolated_theta_list.append(0.0)
        return tuple(interpolated_theta_list)

    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    # Compute the interpolation error and/or its maximum location
    def compute_maximum_interpolation_error(self, N=None):
        if N is None:
            N = self.N
        
        # Compute the error (difference with the eim approximation)
        if N > 0:
            error = self.snapshot - self.Z[:N]*self._interpolation_coefficients
        else:
            error = copy(self.snapshot) # need a copy because it will be rescaled
        
        # Get the location of the maximum error
        (maximum_error, maximum_location) = max(abs(error))
        
        # Return
        return (error, maximum_error, maximum_location)

    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{

    ## Export solution to file
    def export_solution(self, folder, filename, solution=None):
        assert solution is not None
        export(solution, folder, filename)
        
    #  @}
    ########################### end - I/O - end ########################### 
    
