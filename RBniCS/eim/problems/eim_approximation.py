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

from RBniCS.problems import ParametrizedProblem
from RBniCS.linear_algebra import OnlineVector, BasisFunctionsMatrix, solve, AffineExpansionOnlineStorage
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
    def __init__(self, V, truth_problem, parametrized_expression, folder_prefix):
        # Call the parent initialization
        ParametrizedProblem.__init__(self, folder_prefix)
        # Store the parametrized expression
        self.parametrized_expression = parametrized_expression
        self.truth_problem = truth_problem
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # Online reduced space dimension
        self.N = 0
        # Define additional storage for EIM
        self.interpolation_points = PointsList(V.mesh()) # list of interpolation points selected by the greedy
        self.interpolation_matrix = AffineExpansionOnlineStorage(1) # interpolation matrix
        # Solution
        self._interpolation_coefficients = OnlineVector()
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        self.V = V
        # Basis functions matrix
        self.Z = BasisFunctionsMatrix()
        # I/O. Since we are decorating the parametrized problem we do not want to change the name of the
        # basis function/reduced operator folder, but rather add a new one. For this reason we use
        # the __eim suffix in the variable name.
        self.folder["basis"] = self.folder_prefix + "/" + "basis"
        self.folder["reduced_operators"] = self.folder_prefix + "/" + "reduced_operators"
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################

    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{

    ## Initialize data structures required for the online phase
    def init(self, current_stage="online"):
        # Read/Initialize reduced order data structures
        if current_stage == "online":
            self.interpolation_points.load(self.folder["reduced_operators"], "interpolation_points")
            self.interpolation_matrix.load(self.folder["reduced_operators"], "interpolation_matrix")
            self.Z.load(self.folder["basis"], "basis", self.truth_problem.V)
            self.N = len(self.Z)
        elif current_stage == "offline":
            # Nothing to be done
            pass
        else:
            raise ValueError("Invalid stage in init().")

    # Perform an online solve.
    def solve(self, N=None):
        if N is None:
            N = self.N
        
        # Evaluate the function at interpolation points
        rhs = OnlineVector(N)
        for p in range(N):
            rhs[p] = self.evaluate_parametrized_expression_at_x(*self.interpolation_points[p])
        
        # Extract the interpolation matrix
        lhs = self.interpolation_matrix[0][:N,:N]
        
        # Solve the interpolation problem
        self._interpolation_coefficients = OnlineVector(N)
        solve(lhs, self._interpolation_coefficients, rhs)
        
        return self._interpolation_coefficients
        
    ## Call online_solve and then convert the result of online solve from OnlineVector to a tuple
    def compute_interpolated_theta(self, N=None):
        interpolated_theta = self.solve(N)
        interpolated_theta_list = list()
        for n in range(len(interpolated_theta)):
            interpolated_theta_list.append(float(interpolated_theta[n]))
        if N is not None:
            # Make sure to append a 0 coefficient for each basis function
            # which has not been requested
            for n in range(N, self.N):
                interpolated_theta_list.append(0.0)
        return tuple(interpolated_theta_list)
        
    ## Evaluate the parametrized function f(x; mu) for the current value of mu
    def evaluate_parametrized_expression_at_x(self, x, processor_id):
        from numpy import zeros as EvalOutputType
        from mpi4py.MPI import FLOAT
        out = EvalOutputType(self.parametrized_expression.value_size())
        if mpi_comm.rank == processor_id:
            self.parametrized_expression.eval(out, x)
        mpi_comm.Bcast([out, FLOAT], root=processor_id)
        return out

    #  @}
    ########################### end - ONLINE STAGE - end ########################### 

    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{

    ## Export solution in VTK format
    def export_solution(self, solution, folder, filename):
        self._export_vtk(solution, folder, filename, with_mesh_motion=True, with_preprocessing=True)
        
    #  @}
    ########################### end - I/O - end ########################### 
    
