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
from abc import ABCMeta, abstractmethod
import types
from RBniCS.problems.base.parametrized_problem import ParametrizedProblem
from RBniCS.backends import BasisFunctionsMatrix, transpose
from RBniCS.backends.online import OnlineAffineExpansionStorage, OnlineFunction
from RBniCS.utils.decorators import sync_setters, Extends, override, MultiLevelReducedProblem
from RBniCS.utils.mpi import print

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveReducedOrderModelBase
#
# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
@Extends(ParametrizedProblem, assert_recursion_level=1) # needs to be first in order to override for last the methods. assert_recursion_level is set because MultiLevelReducedProblem introduces an additional level of inheritance
@MultiLevelReducedProblem
class ParametrizedReducedDifferentialProblem(ParametrizedProblem):
    __metaclass__ = ABCMeta
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members.
    @override
    @sync_setters("truth_problem", "set_mu", "mu")
    @sync_setters("truth_problem", "set_mu_range", "mu_range")
    def __init__(self, truth_problem):
        # Call to parent
        ParametrizedProblem.__init__(self, type(truth_problem).__name__)
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # Online reduced space dimension
        self.N = 0
        self.N_bc = 0
        self.dirichlet_bc = False
        self.dirichlet_bc_are_homogeneous = False
        # Number of terms in the affine expansion
        self.terms = truth_problem.terms
        self.terms_order = truth_problem.terms_order
        self.Q = dict() # from string to integer
        # Reduced order operators
        self.operator = dict() # from string to OnlineAffineExpansionStorage
        # Solution
        self._solution = OnlineFunction()
        self._output = 0
        self._compute_error__previous_mu = None
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # High fidelity problem
        self.truth_problem = truth_problem
        # Basis functions matrix
        self.Z = BasisFunctionsMatrix(truth_problem.V)
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
        assert current_stage in ("online", "offline")
        if current_stage == "online":
            for term in self.terms:
                self.operator[term] = self.assemble_operator(term, "online")
                self.Q[term] = len(self.operator[term])
        elif current_stage == "offline":
            for term in self.terms:
                self.Q[term] = self.truth_problem.Q[term]
                self.operator[term] = OnlineAffineExpansionStorage(self.Q[term])
        else:
            raise AssertionError("Invalid stage in _init_operators().")
        
    def _init_basis_functions(self, current_stage="online"):
        assert current_stage in ("online", "offline")
        if current_stage == "online":
            self.Z.load(self.folder["basis"], "basis")
            # To properly initialize N and N_bc, detect how many theta terms
            # are related to boundary conditions
            try:
                theta_bc = self.compute_theta("dirichlet_bc")
            except ValueError: # there were no Dirichlet BCs to be imposed by lifting
                self.dirichlet_bc = False
                self.N = len(self.Z)
            else:
                self.dirichlet_bc = True
                if self.truth_problem.dirichlet_bc_are_homogeneous:
                    self.N = len(self.Z)
                    self.dirichlet_bc_are_homogeneous = True
                else:
                    self.N = len(self.Z) - len(theta_bc)
                    self.N_bc = len(theta_bc)
                    self.dirichlet_bc_are_homogeneous = False
        elif current_stage == "offline":
            # Store the lifting functions in self.Z
            self.assemble_operator("dirichlet_bc", "offline") # no return value from assemble_operator in this case
        else:
            raise AssertionError("Invalid stage in _init_basis_functions().")
            
    # Perform an online solve.
    @abstractmethod
    def solve(self, N=None, **kwargs):
        raise NotImplementedError("The method solve() is problem-specific and needs to be overridden.")
    
    # Perform an online evaluation of the output
    @abstractmethod
    def output(self):
        raise NotImplementedError("The method output() is problem-specific and needs to be overridden.")
        
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
        if self.dirichlet_bc and not self.dirichlet_bc_are_homogeneous:
            assert self.N_bc == len(theta_bc)
            snapshot -= self.Z[:self.N_bc]*theta_bc
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # for the current value of mu
    @abstractmethod
    def compute_error(self, N=None, **kwargs):
        raise NotImplementedError("The method compute_error() is problem-specific and needs to be overridden.")
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
        
    ## Export solution to file
    @override
    def export_solution(self, solution, folder, filename):
        N = solution.vector().size
        self.truth_problem.export_solution(self.Z[:N]*solution, folder, filename)
            
    #  @}
    ########################### end - I/O - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{

    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        return self.truth_problem.compute_theta(term)
        
    ## Assemble the reduced order affine expansion
    def assemble_operator(self, term, current_stage="online"):
        assert current_stage in ("online", "offline")
        if current_stage == "online": # load from file
            if not term in self.operator:
                self.operator[term] = OnlineAffineExpansionStorage(0) # it will be resized by load
            # Note that it would not be needed to return the loaded operator in 
            # init(), since it has been already modified in-place. We do this, however,
            # because we want this interface to be compatible with the one in 
            # EllipticCoerciveProblem, i.e. we would like to be able to use a reduced 
            # problem also as a truth problem for a nested reduction
            if term in self.terms:
                self.operator[term].load(self.folder["reduced_operators"], "operator_" + term)
                return self.operator[term]
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
            if term in self.terms:
                for q in range(self.Q[term]):
                    assert self.terms_order[term] in (1, 2)
                    if self.terms_order[term] == 2:
                        self.operator[term][q] = transpose(self.Z)*self.truth_problem.operator[term][q]*self.Z
                    elif self.terms_order[term] == 1:
                        self.operator[term][q] = transpose(self.Z)*self.truth_problem.operator[term][q]
                    else:
                        raise AssertionError("Invalid value for order of term " + term)
                self.operator[term].save(self.folder["reduced_operators"], "operator_" + term)
                return self.operator[term]
            elif term == "dirichlet_bc":
                if self.dirichlet_bc and not self.dirichlet_bc_are_homogeneous:
                    # Otherwise, compute lifting functions
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
                                return tuple(modified_theta_bc)
                            else:
                                return standard_compute_theta(term)
                        self.truth_problem.compute_theta = types.MethodType(modified_compute_theta, self.truth_problem)
                        # ... and store the solution of the truth problem corresponding to that boundary condition
                        # as lifting function
                        print("Computing and storing lifting function n.", i, "in the basis matrix")
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
            raise AssertionError("Invalid stage in assemble_operator().")
    
    ## Return a lower bound for the coercivity constant
    def get_stability_factor(self):
        return self.truth_problem.get_stability_factor()
                    
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
    
