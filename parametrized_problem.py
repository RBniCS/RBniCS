# Copyright (C) 2015 SISSA mathLab
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
## @file parametrized_problem.py
#  @brief Implementation of a class containing an offline/online decomposition of parametrized problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ParametrizedProblem
#
# Base class containing an offline/online decomposition of parametrized problems
class ParametrizedProblem:
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self):
        # $$ ONLINE DATA STRUCTURES $$ #
        # 1. Online reduced space dimension
        self.N = 0
        # 2. Current parameter
        self.mu = []
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 1. Maximum reduced order space dimension or tolerance to be used for the stopping criterion in the basis selection
        self.Nmax = 10
        self.tol = 1.e-15
        # 2. Parameter ranges and training set
        self.mu_range = []
        self.mu = []
        self.xi_train = []
    
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     SETTERS     ########################### 
    ## @defgroup Setters Set properties of the reduced order approximation
    #  @{
    
    ## OFFLINE: set maximum reduced space dimension (stopping criterion)
    def setNmax(self, nmax):
        self.Nmax = nmax
        
    ## OFFLINE: set tolerance of the offline phase (stopping criterion)
    def settol(self, tol):
        self.tol = tol
    
    ## OFFLINE: set the range of the parameters
    def setmu_range(self, mu_range):
        self.mu_range = mu_range
    
    ## OFFLINE: set the elements in the training set \xi_train, from a random uniform distribution
    # If the optional argument is equal to "random", ntrain parameters are drawn from a random uniform distribution
    # Else, if the optional argument is equal to "linspace", (approximately) ntrain parameters are obtained from a cartesian grid
    def setxi_train(self, ntrain, sampling="random"):
        if sampling == "random":
            ss = "[("
            for i in range(len(self.mu_range)):
                ss += "np.random.uniform(self.mu_range[" + str(i) + "][0],self.mu_range[" + str(i) + "][1])"
                if i < len(self.mu_range)-1:
                    ss += ", "
                else:
                    ss += ") for _ in range(" + str(ntrain) +")]"
            self.xi_train = eval(ss)
        elif sampling == "linspace":
            ntrain_P_root = ceil(ntrain**(1./len(self.mu_range)))
            ss = "itertools.product("
            for i in range(len(self.mu_range)):
                ss += "[np.linspace(self.mu_range[" + str(i) + "][0],self.mu_range[" + str(i) + "][1]]"
                if i < len(self.mu_range)-1:
                    ss += ", "
                else:
                    ss += ")"
            self.xi_train = eval(ss)
        else:
            sys.exit("Invalid sampling mode.")

    ## OFFLINE/ONLINE: set the current value of the parameter
    def setmu(self, mu):
        self.mu = mu
    
    #  @}
    ########################### end - SETTERS - end ########################### 

