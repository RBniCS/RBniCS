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
## @file numpy_io.py
#  @brief I/O helper functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from rbnics.utils.mpi import log, DEBUG

def ForDecoratorsStore(Key, storage, data, go_to_next_level):
    # We store the data organized by levels. This serves as additional values
    # for replaces (stored in data[2]), such that higher levels will automatically
    # replace lower levels
    log(DEBUG, "Looking for level of " + str(Key))
    level = 0
    for dict_ in storage:
        for StoredKey in dict_:
            if go_to_next_level(Key, StoredKey):
                log(DEBUG, "\tlevel " + str(level) + " has been discarded becuase it contains " 
                    + str(StoredKey) + " which is a parent of " + str(Key))
                break # go on to the next level of storage
        else: # for loop was not broken: this is the correct level
            break
        level += 1 # if not broken by the previous else, go on and increment level
    log(DEBUG, "\tlevel " + str(level) + " has been selected\n")
    log(DEBUG, "")
    assert level <= len(storage)
    if level == len(storage):
        storage.append(dict()) # make room for a new level
    if Key not in storage[level]:
        storage[level][Key] = list() # make room for a new Key
    storage[level][Key].append(data)
    
def ForDecoratorsLogging(storage, key_string, data_0_string, data_1_string):
    for (level, dict_) in enumerate(storage):
        log(DEBUG, "\tLevel " + str(level) + ":")
        for Key in dict_:
            log(DEBUG, "\t\t" + key_string + " " + str(Key) + ":")
            for (i, data) in enumerate(dict_[Key]):
                log(DEBUG, "\t\t\tItem " + str(i) + ":")
                log(DEBUG, "\t\t\t\t" + data_0_string + ": " + str(data[0]))
                if data_1_string is not None:
                    log(DEBUG, "\t\t\t\t" + data_1_string + ": " + str(data[1]))
                log(DEBUG, "\t\t\t\t" + "replaces" + ": " + str(data[2]))
                log(DEBUG, "\t\t\t\t" + "replaces_if" + ": " + str(data[3]))
                
            
    
