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

from collections import MutableMapping
from functools import wraps
from logging import DEBUG, getLogger
from pylru import lrucache

logger = getLogger("rbnics/utils/cache/cache.py")

class Cache(object):
    def __init__(self, config_section=None, key_generator=None, import_=None, export=None, filename_generator=None):
        self._config_section = config_section
        if self._config_section is None:
            self._storage = dict()
            self._key_generator = None
            self._import = None
            self._export = None
            self._filename_generator = None
        else:
            from rbnics.utils.config import config # cannot import at global scope
            cache_options = config.get(self._config_section, "cache")
            assert isinstance(cache_options, set)
            if "RAM" in cache_options:
                cache_size = config.get(self._config_section, "RAM cache limit")
                assert isinstance(cache_size, str)
                if cache_size == "unlimited":
                    self._storage = dict()
                else:
                    assert cache_size.isdigit()
                    cache_size = int(cache_size)
                    assert cache_size > 0
                    self._storage = lrucache(cache_size)
                assert key_generator is not None
                self._key_generator = key_generator
            else:
                self._storage = DisabledStorage()
                self._key_generator = key_generator
            if "disk" in cache_options:
                cache_size = config.get(self._config_section, "disk cache limit")
                assert isinstance(cache_size, str)
                assert cache_size == "unlimited"
                assert import_ is not None
                self._import = import_
                assert export is not None
                self._export = export
                assert filename_generator is not None
                self._filename_generator = filename_generator
            else:
                self._import = None
                self._export = None
                self._filename_generator = None
        
    def __len__(self):
        """
        Returns the size of RAM cache.
        """
        return len(self._storage)
        
    def clear(self):
        """
        Clears RAM cache, but not disk one.
        """
        self._storage.clear()
        
    def __contains__(self, key):
        """
        Checks if key is in current RAM cache.
        """
        (_, _, storage_key) = self._compute_storage_key(key)
        return storage_key in self._storage
        
    def __getitem__(self, key):
        """
        Get key from either RAM or disk storage, if possible.
        """
        (args, kwargs, storage_key) = self._compute_storage_key(key)
        try:
            storage_value = self._storage[storage_key]
        except KeyError as key_error:
            if self._filename_generator is not None:
                storage_filename = self._filename_generator(*args, **kwargs)
                try:
                    self._storage[storage_key] = self._import(storage_filename)
                except OSError:
                    logger.log(DEBUG, "Could not load key " + str(storage_key) + " (corresponding to args = " + str(args) + " and kwargs = " + str(kwargs) + ") from cache or disk")
                    raise key_error
                else:
                    logger.log(DEBUG, "Loaded key " + str(storage_key) + " (corresponding to args = " + str(args) + " and kwargs = " + str(kwargs) + ") from disk")
                    return self._storage[storage_key]
            else:
                logger.log(DEBUG, "Could not load key " + str(storage_key) + " (corresponding to args = " + str(args) + " and kwargs = " + str(kwargs) + ") from cache")
                raise key_error
        else:
            logger.log(DEBUG, "Loaded key " + str(storage_key) + " (corresponding to args = " + str(args) + " and kwargs = " + str(kwargs) + ") from cache")
            return storage_value
        
    def __setitem__(self, key, value):
        """
        Set key in both RAM and disk storage.
        """
        (args, kwargs, storage_key) = self._compute_storage_key(key)
        self._storage[storage_key] = value
        if self._filename_generator is not None:
            storage_filename = self._filename_generator(*args, **kwargs)
            self._export(storage_filename)
        
    def __delitem__(self, key):
        """
        Remove key from RAM cache (but not from disk storage).
        """
        (_, _, storage_key) = self._compute_storage_key(key)
        del self._storage[storage_key]
        
    def _compute_storage_key(self, key):
        from rbnics.utils.io import OnlineSizeDict # cannot import at global scope
        if isinstance(key, tuple):
            if len(key) > 0 and isinstance(key[-1], dict) and not isinstance(key[-1], OnlineSizeDict):
                kwargs = key[-1]
                args = key[:-1]
            else:
                kwargs = {}
                args = key
        else:
            kwargs = {}
            args = (key, )
        if self._key_generator is not None:
            storage_key = self._key_generator(*args, **kwargs)
        else:
            assert len(kwargs) == 0
            if len(args) == 0:
                storage_key = args
            elif len(args) == 1:
                storage_key = args[0]
            else:
                storage_key = args
        return (args, kwargs, storage_key)
        
    def __iter__(self):
        """
        Iterate over current RAM cache.
        """
        return iter(self._storage)
        
    def items(self):
        """
        Returns items in current RAM cache.
        """
        return self._storage.items()
        
    def keys(self):
        """
        Returns keys in current RAM cache.
        """
        return self._storage.keys()
        
    def values(self):
        """
        Returns values in current RAM cache.
        """
        return self._storage.values()
        
def cache(fun):
    storage = Cache()
    
    @wraps(fun)
    def wrapper(*args):
        try:
            return storage[args]
        except KeyError:
            storage[args] = fun(*args)
            return storage[args]
        
    return wrapper
    
class DisabledStorage(MutableMapping):
    def __getitem__(self, key):
        raise KeyError

    def __setitem__(self, key, value):
        pass

    def __delitem__(self, key):
        pass

    def __iter__(self):
        yield from ()

    def __len__(self):
        return 0

    def __keytransform__(self, key):
        return key
