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

from dolfin import project, vertices
from RBniCS.problems.base import ParametrizedProblem
from RBniCS.linear_algebra import OnlineVector, OnlineFunction, BasisFunctionsMatrix, FunctionsList, solve, AffineExpansionOnlineStorage, TruthFunction
from RBniCS.utils.decorators import sync_setters, Extends, override
from RBniCS.utils.mpi import mpi_comm
from RBniCS.eim.utils.io import PointsList

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
    def __init__(self, truth_problem, parametrized_expression, folder_prefix):        
        # Call the parent initialization
        ParametrizedProblem.__init__(self, folder_prefix)
        # Store the parametrized expression
        self.parametrized_expression = parametrized_expression
        self.truth_problem = truth_problem
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # Online reduced space dimension
        self.N = 0
        # Define additional storage for EIM
        self.interpolation_locations = InterpolationLocationsList(parametrized_expression.space) # list of interpolation locations selected by the greedy
        self.interpolation_matrix = AffineExpansionOnlineStorage(1) # interpolation matrix
        # Solution
        self._interpolation_coefficients = OnlineFunction()
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        self.snapshot = Function(parametrized_expression.space)
        # Basis functions matrix
        self.Z = BasisFunctionsMatrix(parametrized_expression.space) # 
        # I/O. Since we are decorating the parametrized problem we do not want to change the name of the
        # basis function/reduced operator folder, but rather add a new one. For this reason we use
        # the __eim suffix in the variable name.
        self.folder["basis"] = self.folder_prefix + "/" + "basis"
        self.folder["reduced_operators"] = self.folder_prefix + "/" + "reduced_operators"
        
        # Avoid useless linear system solves
        self.solve.__func__.previous_mu = None
        self.solve.__func__.previous_N = None
        
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
    def solve(self, N=None):
        if N is None:
            N = self.N
        
        if self.solve.__func__.previous_mu != self.mu or self.solve.__func__.previous_N != N:
            # Evaluate the function at interpolation points
            rhs = OnlineVector(N)
            for p in range(N):
                rhs[p] = evaluate(self.parametrized_expression, self.interpolation_locations[p])
            
            # Extract the interpolation matrix
            lhs = self.interpolation_matrix[0][:N,:N]
            
            # Solve the interpolation problem
            self._interpolation_coefficients = OnlineFunction(N)
            solve(lhs, self._interpolation_coefficients, rhs)
            
            # Store to avoid repeated computations
            self.solve.__func__.previous_mu = self.mu
            self.solve.__func__.previous_N = N
        
        return self._interpolation_coefficients
        
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
            error = difference(self.snapshot, self.Z[:N]*self._interpolation_coefficients)
        else:
            error = self.snapshot
        
        # Get the location of the maximum error
        (maximum_error, maximum_location) = max(abs(error))
        
        # Return
        return (maximum_error, maximum_error, maximum_location)

    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{

    ## Export solution in VTK format
    def export_solution(self, solution, folder, filename):
        self._export_vtk(solution, folder, filename, with_mesh_motion=True, with_preprocessing=True)
        
    #  @}
    ########################### end - I/O - end ########################### 
    
