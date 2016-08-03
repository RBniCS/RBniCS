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
## @file online_function.py
#  @brief Type of online function
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.linear_algebra.online_vector import OnlineVector, OnlineVector_Type

###########################     ONLINE STAGE     ########################### 
## @defgroup OnlineStage Methods related to the online stage
#  @{

class OnlineFunction(object):
    def __init__(self, arg=None):
        assert arg is None or isinstance(arg, int) or isinstance(arg, OnlineVector_Type)
        if arg is None:
            self._v = None
        elif isinstance(arg, int):
            self._v = OnlineVector(arg)
        elif isinstance(arg, OnlineVector_Type):
            self._v = v
        else:
            raise TypeError("Invalid arguments in OnlineFunction")
    
    def vector(self):
        return self._v
        
    def copy(deepcopy=False):
        assert deepcopy is True # we have only this use in the library, the default argument is provided
                                # for compatibility with FEniCS
        v = OnlineVector(self._v.size)
        v[:] = self._v
        return OnlineFunction(v)
        
    
#  @}
########################### end - ONLINE STAGE - end ########################### 

