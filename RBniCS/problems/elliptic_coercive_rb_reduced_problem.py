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
from RBniCS.elliptic_coercive_reduced_problem import EllipticCoerciveReducedProblem

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveReducedOrderModelBase
#
# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
class EllipticCoerciveRBReducedProblem(EllipticCoerciveReducedProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members.
    def __init__(self, truth_problem):
        # Call to parent
        EllipticCoerciveReducedProblem.__init__(self)
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # 5. Residual terms
        self._riesz = Function(self.truth_problem.V)
        self.riesz_a = AffineExpansionOnlineStorage()
        self.riesz_f = AffineExpansionOnlineStorage()
        self.riesz_aa_product = AffineExpansionOnlineStorage()
        self.riesz_af_product = AffineExpansionOnlineStorage()
        self.riesz_ff_product = AffineExpansionOnlineStorage()
        self.build_error_estimation_matrices.__func__.initialized = False
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 9. I/O
        self.error_estimation_folder = "error_estimation"

        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    ## Initialize data structures required for the online phase
    def init(self, current_stage="online"):
        super(EllipticCoerciveRBReducedProblem, self).init(current_stage)
        # Also initialize data structures related to error estimation
        if current_stage == "online":
            self.assemble_error_estimation_operators("riesz_aa_product")
            self.assemble_error_estimation_operators("riesz_af_product")
            self.assemble_error_estimation_operators("riesz_ff_product")
        elif current_stage == "offline":
            self.riesz_a = AffineExpansionOnlineStorage(self.Qa)
            for qa in range(self.Qa):
                self.riesz_a[qa] = FunctionsList()
            self.riesz_f = AffineExpansionOnlineStorage(self.Qf)
            for qf in range(self.Qf):
                self.riesz_f[qf] = FunctionsList() # even though it will be composed of only one function
            self.riesz_aa_product = AffineExpansionOnlineStorage(self.Qa, self.Qa)
            self.riesz_af_product = AffineExpansionOnlineStorage(self.Qa, self.Qf)
            self.riesz_ff_product = AffineExpansionOnlineStorage(self.Qf, self.Qf)
        else:
            raise RuntimeError("Invalid stage in init().")
    
    ## Return an error bound for the current solution
    def get_delta(self):
        eps2 = self.get_eps2()
        alpha = self.get_alpha_lb()
        return np.sqrt(np.abs(eps2)/alpha)
    
    ## Return an error bound for the current output
    def get_delta_output(self):
        eps2 = self.get_eps2()
        alpha = self.get_alpha_lb()
        return np.abs(eps2)/alpha
        
    ## Return the numerator of the error bound for the current solution
    def get_eps2(self):
        eps2 = 0.0
        
        # Add the (F, F) product part
        for qf in range(self.Qf):
            for qfp in range(self.Qf):
                eps2 += self.theta_f[qf]*self.theta_f[qfp]*self.riesz_ff_product[qf, qfp]
        
        # Add the (A, F) product part
        for qa in range(self.Qa):
            for qf in range(self.Qf):
                eps2 += 2.0*self.theta_a[qa]*self.theta_f[qf]*self.riesz_af_product[qa, qf]*self.uN

        # Add the (A, A) product part
        for qa in range(self.Qa):
            for qap in range(self.Qa):
                eps2 += self.theta_a[qa]*self.theta_a[qap]*transpose(self.uN)*self.riesz_aa_product[qa, qap]*self.uN
        
        return eps2
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Build matrices for error estimation
    def build_error_estimation_matrices(self):
        if not self.build_error_estimation_matrices.__func__.initialized: # this part does not depend on N, so we compute it only once
            # Compute the Riesz representation of f
            self.compute_riesz_f()
            # Compute the (f, f) Riesz representors product
            self.assemble_error_estimation_operators("riesz_ff_product")
            #
            self.build_error_estimation_matrices.__func__.initialized = True
            
        # Update the Riesz representation of -A*Z with the new basis function(s)
        self.update_riesz_a()
        # Update the (a, f) Riesz representors product with the new basis function
        self.assemble_error_estimation_operators("riesz_af_product")
        # Update the (a, a) Riesz representors product with the new basis function
        self.assemble_error_estimation_operators("riesz_aa_product")
            
    ## Compute the Riesz representation of a
    def update_riesz_a(self):
        for qa in range(self.Qa):
            for n in range(len(self.riesz_a[qa]), self.N):
                theta_bc = (0.,)*len(self.truth_problem.dirichet_bc)
                homogeneous_dirichlet_bc = sum(product(theta_bc, self.truth_problem.dirichet_bc))
                solve(self.S, self._riesz.vector(), -1.*self.truth_A[qa]*self.Z[n], homogeneous_dirichlet_bc)
                self.riesz_a[qa].enrich(self._riesz)
    
    ## Compute the Riesz representation of f
    def compute_riesz_f(self):
        for qf in range(self.Qf):
            theta_bc = (0.,)*len(self.truth_problem.dirichet_bc)
            homogeneous_dirichlet_bc = sum(product(theta_bc, self.truth_problem.dirichet_bc))
            solve(self.S, self._riesz.vector(), self.truth_F[qf], homogeneous_dirichlet_bc)
            self.riesz_f[qf].enrich(self._riesz)
            
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Assemble the reduced order affine expansion
    def assemble_error_estimation_operators(self, term):
        if self.current_stage == "online": # load from file
            if term == "riesz_aa_product":
                self.riesz_aa_product.load(self.error_estimation_folder, "riesz_aa_product")
            elif term == "riesz_af_product":
                self.riesz_af_product.load(self.error_estimation_folder, "riesz_af_product")
            elif term == "riesz_ff_product":
                self.riesz_ff_product.load(self.error_estimation_folder, "riesz_ff_product")
            else:
                raise RuntimeError("Invalid term for assemble_error_estimation_operators().")
        elif self.current_stage == "offline":
            if term == "riesz_aa_product":
                for qa in range(0, self.Qa):
                    for qap in range(qa, self.Qa):
                        self.riesz_aa_product[qa, qap] = transpose(self.riesz_a[qa])*self.S*self.riesz_a[qap]
                        if qa != qap:
                            self.riesz_aa_product[qap, qa] = self.riesz_aa_product[qa, qap]
                self.riesz_aa_product.save(self.error_estimation_folder, "riesz_aa_product")
            elif term == "riesz_af_product":
                for qa in range(0, self.Qa):
                    for qf in range(0, self.Qf):
                        self.riesz_af_product[qa, qf] = transpose(self.riesz_a[qa])*self.S*self.riesz_f[qf]
                self.riesz_af_product.save(self.error_estimation_folder, "riesz_af_product")
            elif term == "riesz_ff_product":
                for qf in range(0, self.Qf):
                    for qfp in range(qf, self.Qf):
                        self.riesz_ff_product[qf, qfp] = transpose(self.riesz_f[qf])*self.S*self.riesz_f[qfp]
                        if qf != qfp:
                            self.riesz_ff_product[qfp, qf] = self.riesz_ff_product[qf, qfp]
                self.riesz_ff_product.save(self.error_estimation_folder, "riesz_ff_product")
            else:
                raise RuntimeError("Invalid term for assemble_error_estimation_operators().")
        else:
            raise RuntimeError("Invalid stage in assemble_error_estimation_operators().")
            
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
