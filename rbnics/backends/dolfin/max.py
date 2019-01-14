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

from rbnics.backends.dolfin.abs import AbsOutput
from rbnics.utils.decorators import backend_for

# max function to compute the maximum absolute value of entries in EIM. To be used in combination with abs,
# even though abs actually carries out both the max and the abs!
@backend_for("dolfin", inputs=(AbsOutput, ))
def max(abs_output):
    return (abs_output.max_abs_return_value, abs_output.max_abs_return_location)
