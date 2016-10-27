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
## @file parametrized_problem.py
#  @brief Implementation of a class containing basic definitions of parametrized problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.io import Folders

#~~~~~~~~~~~~~~~~~~~~~~~~~     PARAMETRIZED PROBLEM BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ParametrizedProblem
#
# Implementation of a class containing basic definitions of parametrized problems
class ParametrizedProblem(object):
    """This is the base class, which is inherited by all other
    classes. It defines the base interface with variables and
    functions that the derived classes have to set and/or
    overwrite. The end user should not care about the implementation
    of this very class but he/she should derive one of the Elliptic or
    Parabolic class for solving an actual problem.

    The following functions are implemented:

    ## Set properties of the reduced order approximation
    - set_Nmax()
    - settol()
    - set_mu_range()
    - set_mu()

    """
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, folder_prefix):
        # Current parameters value
        self.mu = tuple() # tuple of real numbers
        # Parameter ranges
        self.mu_range = list() # list of (min, max) pairs, such that len(self.mu) == len(self.mu_range)
        #
        self.folder_prefix = folder_prefix
        self.folder = Folders()
    
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     SETTERS     ########################### 
    ## @defgroup Setters Set properties of the reduced order approximation
    #  @{
    
    ## OFFLINE: set the range of the parameters
    def set_mu_range(self, mu_range):
        self.mu_range = mu_range
        # Initialize mu so that it has the correct length
        self.mu = tuple([r[0] for r in self.mu_range])
    
    ## OFFLINE/ONLINE: set the current value of the parameter
    def set_mu(self, mu):
        assert len(mu) == len(self.mu_range), "mu and mu_range must have the same length"
        self.mu = mu
    
    #  @}
    ########################### end - SETTERS - end ########################### 
    

