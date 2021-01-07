# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import types


class PatchInstanceMethod(object):
    def __init__(self, instance, method_name, patched_method):
        self._instance = instance
        self._method_name = method_name
        self._patched_method = patched_method
        self._unpatched_method = getattr(self._instance, self._method_name)

    def patch(self):
        setattr(self._instance, self._method_name, types.MethodType(self._patched_method, self._instance))

    def unpatch(self):
        setattr(self._instance, self._method_name, self._unpatched_method)
