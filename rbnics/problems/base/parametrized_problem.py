# Copyright (C) 2015-2017 by the RBniCS authors
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

from rbnics.utils.io import Folders

class ParametrizedProblem(object):
    """This is the base class, which is inherited by all other
    classes. It defines the base interface with variables and
    functions that the derived classes have to set and/or
    overwrite.
    """
    
    def __init__(self, folder_prefix):
        """
        Initialization of current parameter mu and its range
        """
        # Current parameters value
        self.mu = tuple() # tuple of real numbers
        # Parameter ranges
        self.mu_range = list() # list of (min, max) pairs, such that len(self.mu) == len(self.mu_range)
        #
        self.folder_prefix = folder_prefix
        self.folder = Folders()
    
    def set_mu_range(self, mu_range):
        """
        Set the range of the parameters.
        
        :param mu_range: the range into which the parameter changes.
        :type mu_range: list of (min, max) pairs, such that len(self.mu) == len(self.mu_range)
        """
        self.mu_range = mu_range
        # Initialize mu so that it has the correct length
        self.set_mu(tuple([r[0] for r in self.mu_range]))
    
    def set_mu(self, mu):
        """
        Set the current value of the parameter
        
        :param mu: the value of the current parameter.
        :type mu: tuple of real numbers
        """
        assert len(mu) == len(self.mu_range), "mu and mu_range must have the same length"
        self.mu = mu
        
