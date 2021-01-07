# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
from rbnics.utils.decorators import BackendFor
from rbnics.backends.abstract import SeparatedParametrizedForm as AbstractSeparatedParametrizedForm


@BackendFor("common", inputs=(Number, ))
class SeparatedParametrizedForm(AbstractSeparatedParametrizedForm):
    def __init__(self, form):
        AbstractSeparatedParametrizedForm.__init__(self, form)
        self._form = form
        self._coefficients = list()  # empty
        self._form_unchanged = list()  # will contain a single number
        self._form_unchanged.append(form)

    def separate(self):
        pass

    @property
    def coefficients(self):
        return self._coefficients

    @property
    def unchanged_forms(self):
        return self._form_unchanged

    def replace_placeholders(self, i, new_coefficients):
        raise RuntimeError("This method should have never been called.")

    def placeholders_names(self, i):
        raise RuntimeError("This method should have never been called.")
