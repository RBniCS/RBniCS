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
## @file io.py
#  @brief I/O helper functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

###########################     I/O     ########################### 
## @defgroup IO Input/output methods
#  @{

import pickle
import numpy

class utils(object):
    
    ## Load a variable from file using pickle
    @staticmethod
    def load_pickle_file(directory, filename):
        with open(directory + "/" + filename + ".pkl", "rb") as infile:
            return pickle.load(infile)
    
    ## Save a variable to file using pickle
    @staticmethod
    def save_pickle_file(subset, directory, filename):
        with open(directory + "/" + filename + ".pkl", "wb") as outfile:
            pickle.dump(subset, outfile, protocol=pickle.HIGHEST_PROTOCOL)
            
    ## Check if a pickle file exists
    @staticmethod
    def exists_pickle_file(directory, filename):
        return os.path.exists(directory + "/" + filename + ".pkl")
        
    ## Load a variable from file using numpy
    @staticmethod
    def load_numpy_file(directory, filename):
        return numpy.load(directory + "/" + filename + ".npy")
    
    ## Save a variable to file using numpy
    @staticmethod
    def save_numpy_file(subset, directory, filename):
        np.save(directory + "/" + filename, subset)
            
    ## Check if a numpy file exists
    @staticmethod
    def exists_numpy_file(directory, filename):
        return os.path.exists(directory + "/" + filename + ".npy")

#  @}
########################### end - I/O - end ########################### 

