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

from rbnics.utils.decorators import BackendFor, Extends, override
from rbnics.backends.abstract import SeparatedParametrizedForm as AbstractSeparatedParametrizedForm

@Extends(AbstractSeparatedParametrizedForm)
@BackendFor("common", inputs=((float, int), ))
class SeparatedParametrizedForm(AbstractSeparatedParametrizedForm):
    def __init__(self, form):
        AbstractSeparatedParametrizedForm.__init__(self, form)
        self._form = form
        self._coefficients = list() # empty
        self._form_unchanged = list() # will contain a single float or int
        self._form_unchanged.append(form)
    
    @override
    def separate(self):
        pass

    @override        
    @property
    def coefficients(self):
        return self._coefficients
        
    @override
    @property
    def unchanged_forms(self):
        return self._form_unchanged

    @override        
    def replace_placeholders(self, i, new_coefficients):
        raise RuntimeError("This method should have never been called.")
        
    @override
    def placeholders_names(self, i):
        raise RuntimeError("This method should have never been called.")

