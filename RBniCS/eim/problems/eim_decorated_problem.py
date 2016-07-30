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

from itertools import product as cartesian_product
from dolfin import Function
from RBniCS.problems import ParametrizedProblem
from RBniCS.linear_algebra import OnlineVector, BasisFunctionsMatrix, solve, AffineExpansionOnlineStorage
from RBniCS.utils.decorators import SyncSetters, extends, override
from RBniCS.utils.mpi import mpi_comm
from RBniCS.eim.utils.io import AffineExpansionSeparatedFormsStorage, PointsList
from RBniCS.eim.utils.ufl import SeparatedParametrizedForm

def EIMDecoratedProblem():
    def EIMDecoratedProblem_Decorator(ParametrizedProblem_DerivedClass):

        #~~~~~~~~~~~~~~~~~~~~~~~~~     EIM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
        ## @class EIM
        #
        # Empirical interpolation method for the interpolation of parametrized functions
        @extends(ParametrizedProblem) # needs to be first in order to override for last the methods
        @SyncSetters("truth_problem", "set_mu", "mu")
        @SyncSetters("truth_problem", "set_mu_range", "mu_range")
        class _EIMApproximation(ParametrizedProblem):

            ###########################     CONSTRUCTORS     ########################### 
            ## @defgroup Constructors Methods related to the construction of the EIM object
            #  @{

            ## Default initialization of members
            @override
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
        
        @extends(ParametrizedProblem_DerivedClass, preserve_class_name=True)
        class EIMDecoratedProblem_Class(ParametrizedProblem_DerivedClass):
            
            ## Default initialization of members
            @override
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                ParametrizedProblem_DerivedClass.__init__(self, V, **kwargs)
                # Storage for EIM reduced problems
                self.separated_forms = dict() # from terms to AffineExpansionSeparatedFormsStorage
                self.EIM_approximations = dict() # from coefficients to _EIMApproximation
                
                # Preprocess each term in the affine expansions
                for term in self.terms:
                    forms = ParametrizedProblem_DerivedClass.assemble_operator(self, term)
                    Q = len(forms)
                    self.separated_forms[term] = AffineExpansionSeparatedFormsStorage(Q)
                    for q in range(Q):
                        self.separated_forms[term][q] = SeparatedParametrizedForm(forms[q])
                        self.separated_forms[term][q].separate()
                        # All parametrized coefficients should be approximated by EIM
                        for i in range(len(self.separated_forms[term][q].coefficients)):
                            for coeff in self.separated_forms[term][q].coefficients[i]:
                                if coeff not in self.EIM_approximations:
                                    self.EIM_approximations[coeff] = _EIMApproximation(self.V, self, coeff, type(self).__name__ + "/eim/" + str(coeff.hash_code))
                                    
                # Signal to the factory that this problem has been decorated
                if not hasattr(self, "_problem_decorators"):
                    self._problem_decorators = dict() # string to bool
                self._problem_decorators["EIM"] = True
                
            ###########################     PROBLEM SPECIFIC     ########################### 
            ## @defgroup ProblemSpecific Problem specific methods
            #  @{
            
            @override
            def assemble_operator(self, term):
                if term in self.terms:
                    eim_forms = list()
                    for q in range(len(self.separated_forms[term])):
                        # Append forms computed with EIM, if applicable
                        for i in range(len(self.separated_forms[term][q].coefficients)):
                            eim_forms_coefficients_q_i = self.separated_forms[term][q].coefficients[i]
                            eim_forms_replacements_q_i__list = list()
                            for coeff in eim_forms_coefficients_q_i:
                                eim_forms_replacements_q_i__list.append(self.EIM_approximations[coeff].Z)
                            eim_forms_replacements_q_i__cartesian_product = cartesian_product(*eim_forms_replacements_q_i__list)
                            for t in eim_forms_replacements_q_i__cartesian_product:
                                new_coeffs = [Function(self.EIM_approximations[coeff].V, new_coeff) for new_coeff in t]
                                eim_forms.append(
                                    self.separated_forms[term][q].replace_placeholders(i, new_coeffs)
                                )
                        # Append forms which did not require EIM, if applicable
                        for unchanged_form in self.separated_forms[term][q]._form_unchanged:
                            eim_forms.append(unchanged_form)
                    return tuple(eim_forms)
                else:
                    return ParametrizedProblem_DerivedClass.assemble_operator(self, term) # may raise an exception
                    
            @override
            def compute_theta(self, term):
                original_thetas = ParametrizedProblem_DerivedClass.compute_theta(self, term) # may raise an exception
                if term in self.terms:
                    eim_thetas = list()
                    for q in range(len(original_thetas)):
                        # Append coefficients computed with EIM, if applicable
                        for i in range(len(self.separated_forms[term][q].coefficients)):
                            eim_thetas_q_i__list = list()
                            for coeff in self.separated_forms[term][q].coefficients[i]:
                                eim_thetas_q_i__list.append(self.EIM_approximations[coeff].compute_interpolated_theta())
                            eim_thetas_q_i__cartesian_product = cartesian_product(*eim_thetas_q_i__list)
                            for t in eim_thetas_q_i__cartesian_product:
                                eim_thetas_q_i_t = original_thetas[q]
                                for r in t:
                                    eim_thetas_q_i_t *= r
                                eim_thetas.append(eim_thetas_q_i_t)
                        # Append coefficients which did not require EIM, if applicable
                        for i in range(len(self.separated_forms[term][q]._form_unchanged)):
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
