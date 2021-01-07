# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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
