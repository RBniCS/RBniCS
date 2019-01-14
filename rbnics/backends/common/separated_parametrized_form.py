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

from numbers import Number
from rbnics.utils.decorators import BackendFor
from rbnics.backends.abstract import SeparatedParametrizedForm as AbstractSeparatedParametrizedForm

@BackendFor("common", inputs=(Number, ))
class SeparatedParametrizedForm(AbstractSeparatedParametrizedForm):
    def __init__(self, form):
        AbstractSeparatedParametrizedForm.__init__(self, form)
        self._form = form
        self._coefficients = list() # empty
        self._form_unchanged = list() # will contain a single number
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
