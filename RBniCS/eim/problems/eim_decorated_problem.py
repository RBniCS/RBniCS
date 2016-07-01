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

from __future__ import print_function
from numpy import log, exp, mean, sqrt # for error analysis
import os # for path and makedir
import shutil # for rm
import random # to randomize selection in case of equal error bound
from RBniCS.problems.parametrized_problem import ParametrizedProblem

def EIMDecoratedProblem(*parametrized_expressions):
    def EIMDecoratedProblem_Decorator(ParametrizedProblem_DerivedClass):
    
        class EIMDecoratedProblem_Class(ParametrizedProblem_DerivedClass):
            
            ## Default initialization of members
            def __init__(self, V, *args):
                # Call the parent initialization
                ParametrizedProblem_DerivedClass.__init__(self, V, *args)
                # Attach EIM reduced problems
                self.EIM_approximation = []
                for i in range(len(parametrized_expressions)):
                    self.EIM_approximation.append(_EIMApproximation(V, parametrized_expressions[i], ParametrizedProblem_DerivedClass.__name__ + "/eim/" + str(i)))
                # Signal to the factory that this problem has been decorated
                if not hasattr(self, "_problem_decorators"):
                    self._problem_decorators = dict() # string to bool
                self._problem_decorators["EIM"] = True
                    
            ###########################     SETTERS     ########################### 
            ## @defgroup Setters Set properties of the reduced order approximation
            #  @{
        
            # Propagate the values of all setters also to the EIM object
            
            ## OFFLINE: set the range of the parameters    
            def set_mu_range(self, mu_range):
                ParametrizedProblem_DerivedClass.set_mu_range(self, mu_range)
                for i in range(len(self.EIM_approximation)):
                    self.EIM_approximation[i].set_mu_range(mu_range)
                    
            ## OFFLINE/ONLINE: set the current value of the parameter
            def set_mu(self, mu):
                ParametrizedProblem_DerivedClass.set_mu(self, mu)
                for i in range(len(self.EIM_approximation)):
                    self.EIM_approximation[i].set_mu(mu)
                
            #  @}
            ########################### end - SETTERS - end ########################### 
            
        #~~~~~~~~~~~~~~~~~~~~~~~~~     EIM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
        ## @class EIM
        #
        # Empirical interpolation method for the interpolation of parametrized functions
        class _EIMApproximation(ParametrizedProblem):

            ###########################     CONSTRUCTORS     ########################### 
            ## @defgroup Constructors Methods related to the construction of the EIM object
            #  @{
        
            ## Default initialization of members
            def __init__(self, V, parametrized_expression__as_string, folder_prefix):
                # Call the parent initialization
                ParametrizedProblem.__init__(self, folder_prefix)
                # Store the parametrized expression
                self.parametrized_expression__as_string = parametrized_expression__as_string
                self.parametrized_expression = ParametrizedExpression()
                
                # $$ ONLINE DATA STRUCTURES $$ #
                # Define additional storage for EIM
                self.interpolation_points = PointsList() # list of interpolation points selected by the greedy
                self.interpolation_points_dof = DofsList() # list of dofs corresponding to interpolation points selected by the greedy
                self.interpolation_matrix = OnlineMatrix() # interpolation matrix
                # Solution
                self._interpolation_coefficients = OnlineVector()
                
                # $$ OFFLINE DATA STRUCTURES $$ #
                # 6. Basis functions matrix
                self.Z = BasisFunctionsMatrix()
                # 9. I/O. Since we are decorating the parametrized problem we do not want to change the name of the
                # basis function/reduced operator folder, but rather add a new one. For this reason we use
                # the __eim suffix in the variable name.
                self.folder["basis"] = self.folder_prefix + "/" + "basis"
                self.folder["reduced_operators"] = self.folder_prefix + "/" + "reduced_operators"
                
            #  @}
            ########################### end - CONSTRUCTORS - end ###########################
            
            ###########################     SETTERS     ########################### 
            ## @defgroup Setters Set properties of the reduced order approximation
            #  @{
        
            # Propagate the values of all setters also to the parametrized expression object
                                
            ## OFFLINE/ONLINE: set the current value of the parameter
            def set_mu(self, mu):
                ParametrizedProblem.set_mu(self, mu)
                self.parametrized_expression.set_mu(mu)
                
            #  @}
            ########################### end - SETTERS - end ########################### 
        
            ###########################     ONLINE STAGE     ########################### 
            ## @defgroup OnlineStage Methods related to the online stage
            #  @{
            
            ## Initialize data structures required for the online phase
            def init(self, current_stage="online"):
                self.current_stage = current_stage
                # Initialize the parametrized expression
                self.parametrized_expression = ParametrizedExpression(self.parametrized_expression__as_string, mu=self.mu, element=self.V.ufl_element())
                # Read/Initialize reduced order data structures
                if current_stage == "online":
                    self.interpolation_points.load(self.folder["reduced_operators"], "interpolation_points")
                    self.interpolation_points_dof.load(self.folder["reduced_operators"], "interpolation_points_dof")
                    self.interpolation_matrix.load(self.folder["reduced_operators"], "interpolation_matrix")
                    self.Z = np.load(self.folder["basis"], "basis")
                    self.N = len(self.Z)
                elif current_stage == "offline":
                    # Create empty files
                    self.interpolation_points.save(self.folder["reduced_operators"], "interpolation_points")
                    self.interpolation_points_dof.save(self.folder["reduced_operators"], "interpolation_points_dof")
                    self.interpolation_matrix.save(self.folder["reduced_operators"], "interpolation_matrix")
                    self.Z.save(self.folder["basis"], "basis")
                else:
                    raise RuntimeError("Invalid stage in init().")
        
            # Perform an online solve.
            def solve(self, N=None):
                self.init()
                if N is None:
                    N = self.N
                
                if N == 0:
                    return # nothing to be done
                
                # Evaluate the function at interpolation points
                rhs = OnlineVector(N)
                for p in range(N):
                    rhs[p] = self.evaluate_parametrized_expression_at_x(self.interpolation_points[p])
                
                # Extract the interpolation matrix
                lhs = self.interpolation_matrix[:N,:N]
                
                # Solve the interpolation problem
                solve(lhs == rhs, self._interpolation_coefficients)
                
            ## Call online_solve and then convert the result of online solve from OnlineVector to a tuple
            def compute_interpolated_theta(self, N=None):
                self.solve(N)
                interpolated_theta = tuple(self._interpolation_coefficients)
                if N is not None:
                    # Make sure to append a 0 coefficient for each basis function
                    # which has not been requested
                    for n in range(N, self.N):
                        interpolated_theta += (0.0,)
                return interpolated_theta
                
            ## Evaluate the parametrized function f(x; mu) for the current value of mu
            def evaluate_parametrized_expression_at_x(self, x):
                from numpy import array as EvalOutputType
                out = EvalOutputType([0.])
                self.parametrized_expression.eval(out, x)
                return out[0]
        
            #  @}
            ########################### end - ONLINE STAGE - end ########################### 
        
            ###########################     OFFLINE STAGE     ########################### 
            ## @defgroup OfflineStage Methods related to the offline stage
            #  @{
                            
            ## Assemble the interpolation matrix
            def update_interpolation_matrix(self):
                for j in range(self.N):
                    self.interpolation_matrix[self.N - 1, j] = self.evaluate_basis_function_at_dof(j, self.interpolation_points_dof[self.N - 1])
                self.interpolation_matrix.save(self.folder["reduced_operators"], "interpolation_matrix")
                
            ## Return the basis functions as tuples of functions
            def assemble_mu_independent_interpolated_function(self):
                output = ()
                for n in range(self.N):
                    fun = Function(self.V)
                    fun.vector()[:] = np.array(self.Z[:, n], dtype=np.float)
                    output += (fun,)
                return output
                                
            ## Evaluate the b-th basis function at the point corresponding to dof d
            def evaluate_basis_function_at_dof(self, b, d):
                return self.Z[d, b]
                
            #  @}
            ########################### end - OFFLINE STAGE - end ########################### 
        
            ###########################     I/O     ########################### 
            ## @defgroup IO Input/output methods
            #  @{

            ## Export solution in VTK format
            def export_solution(self, solution, filename):
                self._export_vtk(solution, filename, with_mesh_motion=True, with_preprocessing=True)
                
            #  @}
            ########################### end - I/O - end ########################### 
            
        # return value (a class) for the decorator
        return EIMDecoratedProblem_Class
    
    # return the decorator itself
    return EIMDecoratedProblem_Decorator
    
# For the sake of the user, since this is the only class that he/she needs to use, rename it to an easier name
EIM = EIMDecoratedProblem
