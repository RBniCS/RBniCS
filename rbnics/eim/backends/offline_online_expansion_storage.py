# Copyright (C) 2015-2019 by the RBniCS authors
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

import os
from rbnics.backends import AffineExpansionStorage, NonAffineExpansionStorage
from rbnics.eim.backends.offline_online_switch import OfflineOnlineSwitch
from rbnics.utils.cache import cache
from rbnics.utils.io import Folders
from rbnics.utils.test import PatchInstanceMethod

@cache
def OfflineOnlineExpansionStorage(problem_name):
    _OfflineOnlineExpansionStorage_Base = OfflineOnlineSwitch(problem_name)
    class _OfflineOnlineExpansionStorage(_OfflineOnlineExpansionStorage_Base):
        
        def __init__(self, problem, expansion_storage_type_attribute):
            _OfflineOnlineExpansionStorage_Base.__init__(self)
            self._content = {
                "offline": dict(),
                "online": dict()
            }
            self._problem = problem
            self._expansion_storage_type_attribute = expansion_storage_type_attribute
            setattr(problem, expansion_storage_type_attribute, None)
        
        def set_is_affine(self, is_affine):
            assert isinstance(is_affine, bool)
            if is_affine:
                setattr(self._problem, self._expansion_storage_type_attribute, AffineExpansionStorage)
            else:
                setattr(self._problem, self._expansion_storage_type_attribute, NonAffineExpansionStorage)
            
        def unset_is_affine(self):
            setattr(self._problem, self._expansion_storage_type_attribute, None)
            
        def __getitem__(self, term):
            return self._content[_OfflineOnlineExpansionStorage_Base._current_stage][term]
            
        def __setitem__(self, term, expansion_storage):
            def patch_save_load(expansion_storage):
                def _patch_save_load(expansion_storage, method):
                    if not hasattr(expansion_storage, method + "_patched"):
                        original_method = getattr(expansion_storage, method)
                        def patched_method(self, directory, filename):
                            # Get full directory name
                            full_directory = Folders.Folder(os.path.join(str(directory), _OfflineOnlineExpansionStorage_Base._current_stage))
                            full_directory.create()
                            # Call original implementation
                            return original_method(full_directory, filename)
                        PatchInstanceMethod(expansion_storage, method, patched_method).patch()
                        setattr(expansion_storage, method + "_patched", True)
                
                assert (
                    hasattr(expansion_storage, "save")
                        ==
                    hasattr(expansion_storage, "load")
                )
                if hasattr(expansion_storage, "save"):
                    for method in ("save", "load"):
                        _patch_save_load(expansion_storage, method)
                    
            patch_save_load(expansion_storage)
            self._content[_OfflineOnlineExpansionStorage_Base._current_stage][term] = expansion_storage
                
        def __contains__(self, term):
            return term in self._content[_OfflineOnlineExpansionStorage_Base._current_stage]
            
        def __len__(self):
            return len(self._content[_OfflineOnlineExpansionStorage_Base._current_stage])
            
        def items(self):
            return self._content[_OfflineOnlineExpansionStorage_Base._current_stage].items()
            
    return _OfflineOnlineExpansionStorage
