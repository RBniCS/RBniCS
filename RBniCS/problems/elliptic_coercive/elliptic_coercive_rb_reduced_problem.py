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

from math import sqrt
from numpy import isclose
from RBniCS.problems.elliptic_coercive.elliptic_coercive_reduced_problem import EllipticCoerciveReducedProblem
from RBniCS.backends import Function, FunctionsList, product, transpose, LinearSolver, sum
from RBniCS.backends.online import OnlineAffineExpansionStorage
from RBniCS.utils.decorators import Extends, override, ReducedProblemFor
from RBniCS.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from RBniCS.reduction_methods.elliptic_coercive import EllipticCoerciveRBReduction

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveReducedOrderModelBase
#
# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
@Extends(EllipticCoerciveReducedProblem) # needs to be first in order to override for last the methods
@ReducedProblemFor(EllipticCoerciveProblem, EllipticCoerciveRBReduction)
class EllipticCoerciveRBReducedProblem(EllipticCoerciveReducedProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members.
    @override
    def __init__(self, truth_problem):
        # Call to parent
        EllipticCoerciveReducedProblem.__init__(self, truth_problem)
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # Residual terms
        self._riesz_solve_storage = Function(self.truth_problem.V)
        self.riesz = dict() # from string to FunctionsList
        self.riesz_product = dict() # from string to OnlineAffineExpansionStorage
        self.build_error_estimation_operators__initialized = False
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # I/O
        self.folder["error_estimation"] = self.folder_prefix + "/" + "error_estimation"

        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    ## Initialize data structures required for the online phase
    @override
    def init(self, current_stage="online"):
        EllipticCoerciveReducedProblem.init(self, current_stage)
        self._init_error_estimation_operators(current_stage)
        
    def _init_error_estimation_operators(self, current_stage="online"):
        # Also initialize data structures related to error estimation
        assert current_stage in ("online", "offline")
        if current_stage == "online":
            self.riesz_product["aa"] = self.assemble_error_estimation_operators("riesz_product_aa", "online")
            self.riesz_product["af"] = self.assemble_error_estimation_operators("riesz_product_af", "online")
            self.riesz_product["ff"] = self.assemble_error_estimation_operators("riesz_product_ff", "online")
        elif current_stage == "offline":
            self.riesz["a"] = OnlineAffineExpansionStorage(self.Q["a"])
            for qa in range(self.Q["a"]):
                self.riesz["a"][qa] = FunctionsList(self.truth_problem.V)
            self.riesz["f"] = OnlineAffineExpansionStorage(self.Q["f"])
            for qf in range(self.Q["f"]):
                self.riesz["f"][qf] = FunctionsList(self.truth_problem.V) # even though it will be composed of only one function
            self.riesz_product["aa"] = OnlineAffineExpansionStorage(self.Q["a"], self.Q["a"])
            self.riesz_product["af"] = OnlineAffineExpansionStorage(self.Q["a"], self.Q["f"])
            self.riesz_product["ff"] = OnlineAffineExpansionStorage(self.Q["f"], self.Q["f"])
        else:
            raise AssertionError("Invalid stage in _init_error_estimation_operators().")
    
    ## Return an error bound for the current solution
    def estimate_error(self):
        eps2 = self.get_residual_norm_squared()
        alpha = self.get_stability_factor()
        assert eps2 >= 0. or isclose(eps2, 0.)
        assert alpha >= 0.
        return sqrt(abs(eps2)/alpha)
    
    ## Return an error bound for the current output
    def estimate_error_output(self):
        eps2 = self.get_residual_norm_squared()
        alpha = self.get_stability_factor()
        assert eps2 >= 0. or isclose(eps2, 0.)
        assert alpha >= 0.
        return abs(eps2)/alpha
        
    ## Return the numerator of the error bound for the current solution
    def get_residual_norm_squared(self):
        N = self._solution.N
        theta_a = self.compute_theta("a")
        theta_f = self.compute_theta("f")
        return (
              sum(product(theta_f, self.riesz_product["ff"], theta_f))
            + 2.0*(transpose(self._solution)*sum(product(theta_a, self.riesz_product["af"][:N], theta_f)))
            + transpose(self._solution)*sum(product(theta_a, self.riesz_product["aa"][:N, :N], theta_a))*self._solution
        )
            
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Build operators for error estimation
    def build_error_estimation_operators(self):
        if not self.build_error_estimation_operators__initialized: # this part does not depend on N, so we compute it only once
            # Compute the Riesz representation of f
            self.compute_riesz_f()
            # Compute the (f, f) Riesz representors product
            self.assemble_error_estimation_operators("riesz_product_ff", "offline")
            #
            self.build_error_estimation_operators__initialized = True
            
        # Update the Riesz representation of -A*Z with the new basis function(s)
        self.update_riesz_a()
        # Update the (a, f) Riesz representors product with the new basis function
        self.assemble_error_estimation_operators("riesz_product_af", "offline")
        # Update the (a, a) Riesz representors product with the new basis function
        self.assemble_error_estimation_operators("riesz_product_aa", "offline")
            
    ## Compute the Riesz representation of a
    def update_riesz_a(self):
        assert len(self.truth_problem.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
        inner_product = self.truth_problem.inner_product[0]
        for qa in range(self.Q["a"]):
            for n in range(len(self.riesz["a"][qa]), self.N + self.N_bc):
                if self.truth_problem.dirichlet_bc is not None:
                    theta_bc = (0.,)*len(self.truth_problem.dirichlet_bc)
                    homogeneous_dirichlet_bc = sum(product(theta_bc, self.truth_problem.dirichlet_bc))
                else:
                    homogeneous_dirichlet_bc = None
                solver = LinearSolver(inner_product, self._riesz_solve_storage, -1.*self.truth_problem.operator["a"][qa]*self.Z[n], homogeneous_dirichlet_bc)
                solver.solve()
                self.riesz["a"][qa].enrich(self._riesz_solve_storage)
    
    ## Compute the Riesz representation of f
    def compute_riesz_f(self):
        assert len(self.truth_problem.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
        inner_product = self.truth_problem.inner_product[0]
        for qf in range(self.Q["f"]):
            if self.truth_problem.dirichlet_bc is not None:
                theta_bc = (0.,)*len(self.truth_problem.dirichlet_bc)
                homogeneous_dirichlet_bc = sum(product(theta_bc, self.truth_problem.dirichlet_bc))
            else:
                homogeneous_dirichlet_bc = None
            solver = LinearSolver(inner_product, self._riesz_solve_storage, self.truth_problem.operator["f"][qf], homogeneous_dirichlet_bc)
            solver.solve()
            self.riesz["f"][qf].enrich(self._riesz_solve_storage)
            
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Assemble operators for error estimation
    def assemble_error_estimation_operators(self, term, current_stage="online"):
        assert current_stage in ("online", "offline")
        short_term = term.replace("riesz_product_", "")
        if current_stage == "online": # load from file
            if not short_term in self.riesz_product:
                self.riesz_product[short_term] = OnlineAffineExpansionStorage(0, 0) # it will be resized by load
            if term == "riesz_product_aa":
                self.riesz_product["aa"].load(self.folder["error_estimation"], "riesz_product_aa")
            elif term == "riesz_product_af":
                self.riesz_product["af"].load(self.folder["error_estimation"], "riesz_product_af")
            elif term == "riesz_product_ff":
                self.riesz_product["ff"].load(self.folder["error_estimation"], "riesz_product_ff")
            else:
                raise ValueError("Invalid term for assemble_error_estimation_operators().")
            return self.riesz_product[short_term]
        elif current_stage == "offline":
            assert len(self.truth_problem.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
            inner_product = self.truth_problem.inner_product[0]
            if term == "riesz_product_aa":
                for qa in range(self.Q["a"]):
                    assert len(self.riesz["a"][qa]) == self.N + self.N_bc
                    for qap in range(qa, self.Q["a"]):
                        assert len(self.riesz["a"][qap]) == self.N + self.N_bc
                        self.riesz_product["aa"][qa, qap] = transpose(self.riesz["a"][qa])*inner_product*self.riesz["a"][qap]
                        if qa != qap:
                            self.riesz_product["aa"][qap, qa] = self.riesz_product["aa"][qa, qap]
                self.riesz_product["aa"].save(self.folder["error_estimation"], "riesz_product_aa")
            elif term == "riesz_product_af":
                for qa in range(self.Q["a"]):
                    assert len(self.riesz["a"][qa]) == self.N + self.N_bc
                    for qf in range(0, self.Q["f"]):
                        assert len(self.riesz["f"][qf]) == 1
                        self.riesz_product["af"][qa, qf] = transpose(self.riesz["a"][qa])*inner_product*self.riesz["f"][qf][0]
                self.riesz_product["af"].save(self.folder["error_estimation"], "riesz_product_af")
            elif term == "riesz_product_ff":
                for qf in range(self.Q["f"]):
                    assert len(self.riesz["f"][qf]) == 1
                    for qfp in range(qf, self.Q["f"]):
                        assert len(self.riesz["f"][qfp]) == 1
                        self.riesz_product["ff"][qf, qfp] = transpose(self.riesz["f"][qf][0])*inner_product*self.riesz["f"][qfp][0]
                        if qf != qfp:
                            self.riesz_product["ff"][qfp, qf] = self.riesz_product["ff"][qf, qfp]
                self.riesz_product["ff"].save(self.folder["error_estimation"], "riesz_product_ff")
            else:
                raise ValueError("Invalid term for assemble_error_estimation_operators().")
            return self.riesz_product[short_term]
        else:
            raise AssertionError("Invalid stage in assemble_error_estimation_operators().")
            
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
