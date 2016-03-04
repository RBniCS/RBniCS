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
## @file affine_expansion_offline_storage.py
#  @brief Type for storing offline quantities related to an affine expansion
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

###########################     OFFLINE STAGE     ########################### 
## @defgroup OfflineStage Methods related to the offline stage
#  @{

# Hide the implementation of the storage of offline data structures, with respect to the affine expansion index.
# For instance, if FEniCS forms are provided to the constructor, then they are assembled and stored,
# while if Dirichlet BCs are provided they are just stored in this class, as no assembly is needed.
# Requires: access with operator[]
from RBniCS.linear_algebra.affine_expansion_online_storage import AffineExpansionOnlineStorage
class AffineExpansionOfflineStorage(object):
    def __init__(self, args):
        self._content = None
        if args:
            if isinstance(args, AffineExpansionOnlineStorage):
                self._content = args
            else:
                if isinstance(args[0], list): # for Dirichlet boundary conditions
                    self._content = _DirichletBCsAffineExpansionOfflineStorageContent(args)
                else: # FEniCS forms
                    self._content = _AssembledFormsAffineExpansionOfflineStorageContent()
                    for i in range(len(args)):
                        from dolfin import assemble as dolfin_assemble
                        self._content.append(dolfin_assemble(args[i]))
    
    def __getitem__(self, key):
        return self._content[key]
                
    def __len__(self):
        return len(self._content)

# Auxiliary class employed to properly differentiate cases in product(): copy of a list of Dirichlet boundary conditions
class _DirichletBCsAffineExpansionOfflineStorageContent(list):
    def __init__(self, other_list):
        self.extend(other_list) # this will create a copy, because other_list contains tuple
        
# Auxiliary class employed to properly differentiate cases in product(): list of assembled forms
class _AssembledFormsAffineExpansionOfflineStorageContent(list):
    pass
    
#  @}
########################### end - OFFLINE STAGE - end ########################### 

