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

from dolfin import Function
from RBniCS.problems import ParametrizedProblem
from RBniCS.linear_algebra import OnlineVector, BasisFunctionsMatrix, solve, AffineExpansionOnlineStorage
from RBniCS.io_utils import KeepClassName, SyncSetters
from RBniCS.eim.io_utils import PointsList

def EIMDecoratedProblem():
    def EIMDecoratedProblem_Decorator(ParametrizedProblem_DerivedClass):

        #~~~~~~~~~~~~~~~~~~~~~~~~~     EIM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
        ## @class EIM
        #
        # Empirical interpolation method for the interpolation of parametrized functions
        @SyncSetters("truth_problem", "set_mu", "mu")
        @SyncSetters("truth_problem", "set_mu_range", "mu_range")
        class _EIMApproximation(ParametrizedProblem):

            ###########################     CONSTRUCTORS     ########################### 
            ## @defgroup Constructors Methods related to the construction of the EIM object
            #  @{

            ## Default initialization of members
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
                self.interpolation_points = PointsList() # list of interpolation points selected by the greedy
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
                    raise RuntimeError("Invalid stage in init().")

            # Perform an online solve.
            def solve(self, N=None):
                if N is None:
                    N = self.N
                
                # Evaluate the function at interpolation points
                rhs = OnlineVector(N)
                for p in range(N):
                    rhs[p] = self.evaluate_parametrized_expression_at_x(self.interpolation_points[p])
                
                # Extract the interpolation matrix
                lhs = self.interpolation_matrix[0][:N,:N]
                
                # Solve the interpolation problem
                self._interpolation_coefficients = OnlineVector(N)
                solve(lhs, self._interpolation_coefficients, rhs)
                
                return self._interpolation_coefficients
                
            ## Call online_solve and then convert the result of online solve from OnlineVector to a tuple
            def compute_interpolated_theta(self, N=None):
                interpolated_theta = list(self.solve(N))
                if N is not None:
                    # Make sure to append a 0 coefficient for each basis function
                    # which has not been requested
                    for n in range(N, self.N):
                        interpolated_theta.append(0.0)
                return tuple(interpolated_theta)
                
            ## Evaluate the parametrized function f(x; mu) for the current value of mu
            def evaluate_parametrized_expression_at_x(self, x):
                from numpy import zeros as EvalOutputType
                out = EvalOutputType(self.parametrized_expression.value_size())
                self.parametrized_expression.eval(out, x)
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
        
        class EIMDecoratedProblem_Class(
            KeepClassName(ParametrizedProblem_DerivedClass)
        ):
            
            ## Default initialization of members
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                ParametrizedProblem_DerivedClass.__init__(self, V, **kwargs)
                # Storage for EIM reduced problems
                self.EIMApproximation = _EIMApproximation
                self.EIM_approximations = dict() # from terms to AffineExpansionEIMStorage
                
                # Signal to the factory that this problem has been decorated
                if not hasattr(self, "_problem_decorators"):
                    self._problem_decorators = dict() # string to bool
                self._problem_decorators["EIM"] = True
                
            ###########################     PROBLEM SPECIFIC     ########################### 
            ## @defgroup ProblemSpecific Problem specific methods
            #  @{
            
            def assemble_operator(self, term, exact_evaluation=False):
                original_forms = ParametrizedProblem_DerivedClass.assemble_operator(self, term) # may raise an error
                if term in self.terms and not exact_evaluation:
                    eim_forms = []
                    for q in range(len(original_forms)):
                        EIM_approximation_q = self.EIM_approximations[term][q]
                        if EIM_approximation_q is not None:
                            assert len(original_forms[q].coefficients()) == 1
                            coef = original_forms[q].coefficients()[0]
                            interpolated_functions_q = EIM_approximation_q.Z
                            replacement_q = dict()
                            for a in range(len(interpolated_functions_q)): # over EIM addends
                                replacement_q[coef] = interpolated_functions_q[a]
                                eim_forms.append(replace(original_forms[q], replacement_q))
                        else:
                            eim_forms.append(original_forms[q])
                    return tuple(eim_forms)
                else:
                    return original_forms
                    
            def compute_theta(self, term, exact_evaluation=False):
                original_theta = ParametrizedProblem_DerivedClass.compute_theta(self, term) # may raise an error
                if term in self.terms and not exact_evaluation:
                    eim_thetas = []
                    for q in range(len(original_thetas)):
                        EIM_approximation_q = self.EIM_approximations[term][q]
                        if EIM_approximation_q is not None:
                            interpolated_theta_q = EIM_approximation_q.compute_interpolated_theta()
                            for a in range(len(interpolated_theta)): # over EIM addends
                                eim_thetas.append(original_theta[q]*interpolated_theta_q[a])
                        else:
                            eim_thetas.append(original_thetas[q])
                    return tuple(eim_thetas)
                else:
                    return original_thetas
                        
            #  @}
            ########################### end - PROBLEM SPECIFIC - end ########################### 
            
        # return value (a class) for the decorator
        return EIMDecoratedProblem_Class
        
    # return the decorator itself
    return EIMDecoratedProblem_Decorator
    
# For the sake of the user, since this is the only class that he/she needs to use, rename it to an easier name
EIM = EIMDecoratedProblem
