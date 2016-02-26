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
## @file elliptic_coercive_rom_base.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from RBniCS.elliptic_coercive_problem import EllipticCoerciveProblem

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE ROM BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveROMBase
#
# Base class containing the interface of a projection based ROM
# for elliptic coercive problems. This class takes in input the high-fidelity
# elliptic problem, and contains the reduced order version of the provided elliptic problem
class EllipticCoerciveROMBase(EllipticCoerciveProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, truth_problem):
        # Call to parent
        EllipticCoerciveProblem.__init__(self)
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # 3a. Number of terms in the affine expansion
        self.Qa = 0
        self.Qf = 0
        # 3b. Theta multiplicative factors of the affine expansion
        self.theta_a = tuple()
        self.theta_f = tuple()
        # 3c. Reduced order operators
        self.reduced_A = AffineExpansionOnlineStorage()
        self.reduced_F = AffineExpansionOnlineStorage()
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 3. High fidelity problem
        self.truth_problem = truth_problem
        # 6. Basis functions matrix
        self.Z = BasisFunctionsMatrix()
            
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    ## Initialize data structures required for the online phase
    def init(self):
        self.load_reduced_data_structures()
        
    # Perform an online solve. self.N will be used as matrix dimension if the default value is provided for N.
    def solve(self, N=None, with_plot=True):
        self.init()
        if N is None:
            N = self.N
        uN = self._solve(N)
        reduced_solution = self.Z*uN
        if with_plot == True:
            self._plot(reduced_solution, title = "Reduced solution. mu = " + str(self.mu), interactive = True)
        return reduced_solution
    
    # Perform an online solve (internal)
    def _solve(self, N):
        self.theta_a = self.compute_theta("a")
        self.theta_f = self.compute_theta("f")
        assembled_reduced_A = sum(product(self.theta_a, self.reduced_A[:N, :N]))
        assembled_reduced_F = sum(product(self.theta_f, self.reduced_F[:N]))
        uN = OnlineVector(N)
        solve(assembled_reduced_A, uN, assembled_reduced_F)
        return uN
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Initialize data structures required for the offline phase
    def _init_offline(self):
        # Initialize the affine expansion in the truth problem
        self.truth_problem.init()
        
        # Initialize reduced order data structures in the reduced problem
        self.Qa = self.truth_problem.Qa
        self.Qf = self.truth_problem.Qf
        self.operator_a = AffineExpansionOnlineStorage(self.Qa)
        self.operator_f = AffineExpansionOnlineStorage(self.Qf)
    
    ## Assemble the reduced order affine expansion
    def build_reduced_operators(self):
        # a
        for qa in range(self.Qa):
            self.operator_a[qa] = transpose(self.Z)*self.truth_problem.operator_a[qa]*self.Z
        self.operator_a.save(self.reduced_operators_folder, "operator_a")
        # f
        for qf in range(self.Qf):
            self.operator_f[qf] = transpose(self.Z)*self.truth_problem.operator_f[qf]
        self.operator_f.save(self.reduced_operators_folder, "operator_f")
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    ## Initialize data structures required for the error analysis phase
    def _init_error_analysis(self):
        # Initialize the affine expansion in the truth problem
        self.truth_problem.init()
        
        # Initialize reduced order data structures in the reduced problem
        self.init() # reading from file
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # for the current value of mu
    def compute_error(self, truth_solution, N=None):
        reduced_solution = self.online_solve(N, False)
        reduced_solution -= truth_solution # store the error as a function in the reduced solution
        error = reduced_solution
        self.theta_a = self.compute_theta("a") # not really necessary, for symmetry with the parabolic case
        assembled_operator_a = sum(product(self.theta_a, self.operator_a)) # use the energy norm (skew part will discarded by the scalar product)
        error_norm_squared = self.compute_scalar_product(error, assembled_operator_a, error) # norm of the error
        return sqrt(error_norm_squared)
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    ## Load reduced order data structures
    def load_reduced_data_structures(self):
        was_operator_a_loaded = self.operator_a.load(self.reduced_operators_folder, "operator_a")
        was_operator_f_loaded = self.operator_f.load(self.reduced_operators_folder, "operator_f")
        was_Z_loaded = self.Z.load(self.basis_folder, "basis")
        if was_operator_a_loaded:
            self.Qa = len(self.operator_a)
        if was_operator_f_loaded:
            self.Qf = len(self.operator_f)
        if was_Z_loaded:
            self.N = len(self.Z)

    ## Export solution in VTK format
    def export_solution(self, solution, filename):
        self._export_vtk(solution, filename, with_mesh_motion=True, with_preprocessing=True)
                
    #  @}
    ########################### end - I/O - end ###########################

