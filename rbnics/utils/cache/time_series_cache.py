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

from rbnics.utils.cache.cache import Cache

class TimeSeriesCache(Cache):
    def __setitem__(self, key, value):
        """
        Set key in both RAM and disk storage.
        """
        from rbnics.backends.abstract import TimeSeries
        from rbnics.utils.test import PatchInstanceMethod
        assert isinstance(value, TimeSeries)
        if self._filename_generator is not None:
            # Patch value's append method to save to file
            (args, kwargs, storage_key) = self._compute_storage_key(key)
            storage_filename = self._filename_generator(*args, **kwargs)
            original_append = value.append
            def patched_append(self_, item):
                self._export(storage_filename, item, len(self_))
                original_append(item)
            PatchInstanceMethod(value, "append", patched_append).patch()
        # Call standard setitem, disabling export
        bak_filename_generator = self._filename_generator
        self._filename_generator = None
        Cache.__setitem__(self, key, value)
        self._filename_generator = bak_filename_generator
