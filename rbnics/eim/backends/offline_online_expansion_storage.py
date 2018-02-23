# Copyright (C) 2015-2018 by the RBniCS authors
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
from rbnics.backends import NonAffineExpansionStorage
from rbnics.backends.abstract import NonAffineExpansionStorage as AbstractNonAffineExpansionStorage
from rbnics.eim.backends.offline_online_switch import OfflineOnlineSwitch
from rbnics.utils.io import Folders
from rbnics.utils.test import PatchInstanceMethod

def OfflineOnlineExpansionStorage(problem_name):
    if problem_name not in _offline_online_expansion_storage_cache:
        _OfflineOnlineExpansionStorage_Base = OfflineOnlineSwitch(problem_name)
        class _OfflineOnlineExpansionStorage(_OfflineOnlineExpansionStorage_Base):
            _is_affine = None
            
            def __init__(self):
                _OfflineOnlineExpansionStorage_Base.__init__(self)
                self._content = {
                    "offline": dict(),
                    "online": dict()
                }
            
            @classmethod
            def set_is_affine(cls, is_affine):
                assert isinstance(is_affine, bool)
                cls._is_affine = is_affine
                
            @classmethod
            def unset_is_affine(cls):
                cls._is_affine = None
                
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
                        
                assert _OfflineOnlineExpansionStorage._is_affine is not None
                if not _OfflineOnlineExpansionStorage._is_affine:
                    if not isinstance(expansion_storage, AbstractNonAffineExpansionStorage):
                        if expansion_storage is not None:
                            expansion_storage = NonAffineExpansionStorage(expansion_storage)
                        else:
                            expansion_storage = None
                        patch_save_load(expansion_storage)
                        self._content[_OfflineOnlineExpansionStorage_Base._current_stage][term] = expansion_storage
                    else:
                        assert expansion_storage is self._content[_OfflineOnlineExpansionStorage_Base._current_stage][term]
                else:
                    patch_save_load(expansion_storage)
                    self._content[_OfflineOnlineExpansionStorage_Base._current_stage][term] = expansion_storage
                    
            def __contains__(self, term):
                return term in self._content[_OfflineOnlineExpansionStorage_Base._current_stage]
                
            def __len__(self):
                return len(self._content[_OfflineOnlineExpansionStorage_Base._current_stage])
                
            def items(self):
                return self._content[_OfflineOnlineExpansionStorage_Base._current_stage].items()
                
        _offline_online_expansion_storage_cache[problem_name] = _OfflineOnlineExpansionStorage
    
    return _offline_online_expansion_storage_cache[problem_name]
        
_offline_online_expansion_storage_cache = dict()
