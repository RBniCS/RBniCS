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
## @file reduction_method_base.py
#  @brief Implementation of a class containing an offline/online decomposition for ROM for parametrized problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import os # for path and makedir

#~~~~~~~~~~~~~~~~~~~~~~~~~     REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ReductionMethodBase
#
# Implementation of a class containing an offline/online decomposition of ROM for parametrized problems
class ReductionMethodBase(object):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self):
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 1. Maximum reduced order space dimension or tolerance to be used for the stopping criterion in the basis selection
        self.Nmax = 10
        # 2. Parameter ranges and training set
        self.xi_train = ParameterSpaceSubset()
        # 9. I/O
        self.xi_train_folder = "xi_train/"
        self.xi_test_folder = "xi_test/"
        
        # $$ ERROR ANALYSIS DATA STRUCTURES $$ #
        # 2. Test set
        self.xi_test = ParameterSpaceSubset()
    
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     SETTERS     ########################### 
    ## @defgroup Setters Set properties of the reduced order approximation
    #  @{
    
    ## OFFLINE: set maximum reduced space dimension (stopping criterion)
    def setNmax(self, nmax):
        self.Nmax = nmax

    ## OFFLINE: set the elements in the training set \xi_train.
    # See the documentation of generate_train_or_test_set for more details
    def setxi_train(self, ntrain, enable_import=False, sampling="random"):
        # Create I/O folder
        if not os.path.exists(self.xi_train_folder):
            os.makedirs(self.xi_train_folder)
        # Test if can import
        import_successful = False
        if enable_import:
            import_successful = self.xi_train.load(self.xi_train_folder, "xi_train") \
                and  (len(self.xi_train) == ntrain)
        if not import_successful:
            self.xi_train.generate(self.mu_range, ntrain, sampling)
            # Export 
            self.xi_train.save(self.xi_train_folder, "xi_train")
        # Prepare for the offline phase
        self._init_offline()
        
    ## ERROR ANALYSIS: set the elements in the test set \xi_test.
    # See the documentation of generate_train_or_test_set for more details
    def setxi_test(self, ntest, enable_import=False, sampling="random"):
        # Create I/O folder
        if not os.path.exists(self.xi_test_folder):
            os.makedirs(self.xi_test_folder)
        # Test if can import
        import_successful = False
        if enable_import:
            import_successful = self.xi_test.load(self.xi_test_folder, "xi_test") \
                and  (len(self.xi_test) == ntest)
        if not import_successful:
            self.xi_test.generate(self.mu_range, ntest, sampling)
            # Export 
            self.xi_test.save(self.xi_test_folder, "xi_test")
        # Prepare for the error analysis
        self._init_error_analysis()
            
    #  @}
    ########################### end - SETTERS - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Perform the offline phase of the reduced order model
    def offline(self):
        raise RuntimeError("Please implement the offline phase of the reduced order model.")
        
    ## Initialize data structures required for the offline phase
    def _init_offline(self):
        pass
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
        
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the test set
    def error_analysis(self, N=None):
        raise RuntimeError("Please implement the error analysis of the reduced order model.")
        
    ## Initialize data structures required for the error analysis phase
    def _init_error_analysis(self):
        pass
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    

