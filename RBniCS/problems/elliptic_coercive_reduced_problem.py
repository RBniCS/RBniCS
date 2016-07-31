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
## @file elliptic_coercive_reduced_problem.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
import types
from dolfin import Function
from math import sqrt
from RBniCS.problems.parametrized_problem import ParametrizedProblem
from RBniCS.problems.elliptic_coercive_problem import EllipticCoerciveProblem
from RBniCS.linear_algebra import AffineExpansionOnlineStorage, BasisFunctionsMatrix, OnlineVector, product, transpose, solve, sum
from RBniCS.utils.decorators import SyncSetters, Extends, override, ReducedProblemFor
from RBniCS.utils.mpi import print
from RBniCS.problems.elliptic_coercive_problem import EllipticCoerciveProblem
from RBniCS.reduction_methods import EllipticCoerciveReductionMethod

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveReducedOrderModelBase
#
# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
@Extends(ParametrizedProblem) # needs to be first in order to override for last the methods
@ReducedProblemFor(EllipticCoerciveProblem, EllipticCoerciveReductionMethod)
@SyncSetters("truth_problem", "set_mu", "mu")
@SyncSetters("truth_problem", "set_mu_range", "mu_range")
class EllipticCoerciveReducedProblem(ParametrizedProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members.
    @override
    def __init__(self, truth_problem):
        # Call to parent
        ParametrizedProblem.__init__(self, type(truth_problem).__name__)
        
        # Consistency check
        assert isinstance(truth_problem, EllipticCoerciveProblem)
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # Online reduced space dimension
        self.N = 0
        self.N_bc = 0
        # Number of terms in the affine expansion
        self.terms = truth_problem.terms
        self.Q = dict() # from string to integer
        # Reduced order operators
        self.operator = dict() # from string to AffineExpansionOnlineStorage
        # Solution
        self._solution = OnlineVector()
        self._output = 0
        self.compute_error.__func__.previous_mu = None
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # High fidelity problem
        self.truth_problem = truth_problem
        # Basis functions matrix
        self.Z = BasisFunctionsMatrix()
        # I/O
        self.folder["basis"] = self.folder_prefix + "/" + "basis"
        self.folder["reduced_operators"] = self.folder_prefix + "/" + "reduced_operators"
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    ## Initialize data structures required for the online phase
    def init(self, current_stage="online"):
        self._init_operators(current_stage)
        self._init_basis_functions(current_stage)
            
    def _init_operators(self, current_stage="online"):
        if current_stage == "online":
            for term in self.terms:
                self.operator[term] = self.assemble_operator(term, "online")
                self.Q[term] = len(self.operator[term])
        elif current_stage == "offline":
            for term in self.terms:
                self.Q[term] = self.truth_problem.Q[term]
                self.operator[term] = AffineExpansionOnlineStorage(self.Q[term])
        else:
            raise ValueError("Invalid stage in _init_operators().")
        
    def _init_basis_functions(self, current_stage="online"):
        if current_stage == "online":
            self.Z.load(self.folder["basis"], "basis", self.truth_problem.V)
            # To properly initialize N and N_bc, detect how many theta terms
            # are related to boundary conditions
            try:
                theta_bc = self.compute_theta("dirichlet_bc")
            except ValueError: # there were no Dirichlet BCs to be imposed by lifting
                self.N = len(self.Z)
            else: # there were Dirichlet BCs to be imposed by lifting
                self.N = len(self.Z) - len(theta_bc)
                self.N_bc = len(theta_bc)
        elif current_stage == "offline":
            # Store the lifting functions in self.Z
            self.assemble_operator("dirichlet_bc", "offline") # no return value from assemble_operator in this case
        else:
            raise ValueError("Invalid stage in _init_basis_functions().")
            
    # Perform an online solve. self.N will be used as matrix dimension if the default value is provided for N.
    def solve(self, N=None, with_plot=True):
        if N is None:
            N = self.N
        uN = self._solve(N)
        if with_plot:
            self._plot(uN, title = "Reduced solution. mu = " + str(self.mu), interactive = True)
        return uN
    
    # Perform an online solve (internal)
    def _solve(self, N):
        N += self.N_bc
        assembled_operator = dict()
        assembled_operator["a"] = sum(product(self.compute_theta("a"), self.operator["a"][:N, :N]))
        assembled_operator["f"] = sum(product(self.compute_theta("f"), self.operator["f"][:N]))
        try:
            theta_bc = self.compute_theta("dirichlet_bc")
        except ValueError: # there were no Dirichlet BCs to be imposed by lifting
            theta_bc = None
        self._solution = OnlineVector(N)
        solve(assembled_operator["a"], self._solution, assembled_operator["f"], theta_bc)
        return self._solution
        
    # Perform an online evaluation of the (compliant) output
    def output(self):
        N = self._solution.size
        assembled_output_operator = sum(product(self.compute_theta("f"), self.operator["f"][:N]))
        self._output = transpose(assembled_output_operator)*self._solution
        return self._output
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 

    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
        
    ## Assemble the reduced order affine expansion.
    def build_reduced_operators(self):
        for term in self.terms:
            self.operator[term] = self.assemble_operator(term, "offline")
        
    ## Postprocess a snapshot before adding it to the basis/snapshot matrix, for instance removing
    # non-homogeneous Dirichlet boundary conditions
    def postprocess_snapshot(self, snapshot):
        try:
            theta_bc = self.compute_theta("dirichlet_bc")
        except ValueError: # there were no Dirichlet BCs to be imposed by lifting
            pass # nothing to be done
        else: # there were Dirichlet BCs
            assert self.N_bc == len(theta_bc)
            snapshot -= self.Z[:self.N_bc]*theta_bc
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # for the current value of mu
    def compute_error(self, N=None):
        if N is None:
            N = self.N
        if self.compute_error.__func__.previous_mu != self.mu:
            self.truth_problem.solve()
            self.truth_problem.output()
            # Do not carry out truth solves anymore for the same parameter
            self.compute_error.__func__.previous_mu = self.mu
        # Compute the error on the solution
        uN = self.solve(N, with_plot=False)
        error = Function(self.truth_problem.V, self.Z[:N]*uN)
        error.vector().add_local(- self.truth_problem._solution.vector().array())
        error.vector().apply("") # store the error as a function in the reduced solution
        error_norm_squared = transpose(error.vector())*self._error_inner_product_matrix()*error.vector() # norm of the error
        # Compute the error on the output
        error_output = abs(self.truth_problem._output - self.output())
        return (sqrt(error_norm_squared), error_output)
        
    # Internal method for error computation: returns the inner product matrix to be used.
    def _error_inner_product_matrix(self):
        assembled_error_inner_product_operator = sum(product(self.truth_problem.compute_theta("a"), self.truth_problem.operator["a"])) # use the energy norm (skew part will discarded by the scalar product)
        return assembled_error_inner_product_operator
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{

    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        return self.truth_problem.compute_theta(term)
        
    ## Assemble the reduced order affine expansion
    def assemble_operator(self, term, current_stage="online"):
        if current_stage == "online": # load from file
            if not term in self.operator:
                self.operator[term] = AffineExpansionOnlineStorage()
            # Note that it would not be needed to return the loaded operator in 
            # init(), since it has been already modified in-place. We do this, however,
            # because we want this interface to be compatible with the one in 
            # EllipticCoerciveProblem, i.e. we would like to be able to use a reduced 
            # problem also as a truth problem for a nested reduction
            if term == "a":
                self.operator["a"].load(self.folder["reduced_operators"], "operator_a")
                return self.operator["a"]
            elif term == "f":
                self.operator["f"].load(self.folder["reduced_operators"], "operator_f")
                return self.operator["f"]
            elif term == "dirichlet_bc":
                raise ValueError("There should be no need to assemble Dirichlet BCs when querying online reduced problems.")
            else:
                raise ValueError("Invalid term for assemble_operator().")
        elif current_stage == "offline":
            # As in the previous case, there is no need to return anything because 
            # we are still training the reduced order model, so the previous remark 
            # (on the usage of a reduced problem as a truth one) cannot hold here.
            # However, in order to have a consistent interface we return the assembled
            # operator
            if term == "a":
                for q in range(self.Q["a"]):
                    self.operator["a"][q] = transpose(self.Z)*self.truth_problem.operator["a"][q]*self.Z
                self.operator["a"].save(self.folder["reduced_operators"], "operator_a")
                return self.operator["a"]
            elif term == "f":
                for q in range(self.Q["f"]):
                    self.operator["f"][q] = transpose(self.Z)*self.truth_problem.operator["f"][q]
                self.operator["f"].save(self.folder["reduced_operators"], "operator_f")
                return self.operator["f"]
            elif term == "dirichlet_bc":
                try:
                    theta_bc = self.compute_theta("dirichlet_bc")
                except ValueError: # there were no Dirichlet BCs to be imposed by lifting
                    return
                Q_dirichlet_bcs = len(theta_bc)
                # Temporarily override compute_theta method to return only one nonzero 
                # theta term related to boundary conditions
                standard_compute_theta = self.truth_problem.compute_theta
                for i in range(Q_dirichlet_bcs):
                    def modified_compute_theta(self, term):
                        if term == "dirichlet_bc":
                            modified_theta_bc = list()
                            for j in range(Q_dirichlet_bcs):
                                if j != i:
                                    modified_theta_bc.append(0.)
                                else:
                                    modified_theta_bc.append(theta_bc[i])
                            return tuple(modified_theta)
                        else:
                            return standard_compute_theta()
                    self.truth_problem.compute_theta = types.MethodType(modified_compute_theta, self.truth_problem)
                    # ... and store the solution of the truth problem corresponding to that boundary condition
                    # as lifting function
                    print("Computing and storing lifting function n.", i, " in the basis matrix")
                    lifting = self.truth_problem.solve()
                    lifting.vector()[:] /= theta_bc[i]
                    self.Z.enrich(lifting)
                # Restore the standard compute_theta method
                self.truth_problem.compute_theta = standard_compute_theta
                # Save basis functions matrix, that contains up to now only lifting functions
                self.Z.save(self.folder["basis"], "basis")
                self.N_bc = Q_dirichlet_bcs
                # Note that, however, self.N is not increased, so it will actually contain the number
                # of basis functions without the lifting ones
            else:
                raise ValueError("Invalid term for assemble_operator().")
        else:
            raise ValueError("Invalid stage in assemble_operator().")
    
    ## Return a lower bound for the coercivity constant
    def get_stability_factor(self):
        return self.truth_problem.get_stability_factor()
                    
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    ## Interactive plot
    @override
    def _plot(self, solution, *args, **kwargs):
        N = solution.size
        solution_on_high_fidelity = Function(self.truth_problem.V, self.Z[:N]*solution)
        self.truth_problem._plot(solution_on_high_fidelity, *args, **kwargs)
        
    ## Export in VTK format
    @override
    def _export_vtk(self, solution, folder, filename, **output_options):
        N = solution.size
        solution_on_high_fidelity = Function(self.truth_problem.V, self.Z[:N]*solution)
        self.truth_problem._export_vtk(solution_on_high_fidelity, folder, filename, **output_options)
            
    #  @}
    ########################### end - I/O - end ########################### 

