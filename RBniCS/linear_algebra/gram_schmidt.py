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
## @file gram_schmidt.py
#  @brief Implementation of the Gram Schmidt process
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

#~~~~~~~~~~~~~~~~~~~~~~~~~     GRAM SCHMIDT CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class GramSchmidt
#
# Class containing the implementation of Gram Schmidt
class GramSchmidt():

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of a GS object
    #  @{
    
    ## Default initialization of members
    def __init__(self, compute_scalar_product_method, X):
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 7. Inner product
        self.compute_scalar_product = compute_scalar_product_method
        self.X = X
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
        
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
            
    ## Apply Gram Schmidt on the basis functions matrix Z
    def apply(self, Z):
        n_basis = len(Z) # basis are store as a list of FE vectors
        b = Z[n_basis - 1] # reference to the last basis
        for i in range(n_basis - 1):
            b -= (transpose(b)*self.X*Z[i]) * Z[i]
        b /= transpose(b)*self.X*b
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
