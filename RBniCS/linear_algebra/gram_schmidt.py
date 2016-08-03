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

from math import sqrt
from RBniCS.linear_algebra.transpose import transpose
from RBniCS.linear_algebra.truth_function import TruthFunction
from RBniCS.linear_algebra.online_function import OnlineFunction

# Class containing the implementation of Gram Schmidt
class GramSchmidt(object):

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of a GS object
    #  @{
    
    ## Default initialization of members
    def __init__(self, X):
        # $$ OFFLINE DATA STRUCTURES $$ #
        # Inner product
        self.X = X
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
        
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
            
    ## Apply Gram Schmidt on the basis functions matrix Z
    def apply(self, Z):
        assert len(self.X) == 1 # note that we cannot move this assert in __init__ because
                                # self.X has not been assembled yet there
        X = self.X[0]
        n_basis = len(Z) # basis are store as a list of FE vectors
        b = Z[n_basis - 1] # reference to the last basis
        assert isinstance(b, TruthFunction) or isinstance(b, OnlineFunction)
        if isinstance(b, TruthFunction):
            for i in range(n_basis - 1):
                b.vector().add_local( - (transpose(b)*X*Z[i]) * Z[i].vector().array() )
                b.vector().apply("add")
            b.vector()[:] /= sqrt(transpose(b)*X*b)
        elif isinstance(b, OnlineFunction):
            for i in range(n_basis - 1):
                b.vector()[:] -= (transpose(b)*X*Z[i]) * Z[i].vector()
            b.vector()[:] /= sqrt(transpose(b)*X*b)
        else: # impossible to arrive here anyway, thanks to the assert
            raise TypeError("Invalid arguments in GramSchmidt.apply().")
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
