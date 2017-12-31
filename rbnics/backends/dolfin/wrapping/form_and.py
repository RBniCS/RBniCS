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

from ufl import Form
from dolfin import assemble
from rbnics.backends.dolfin.wrapping.dirichlet_bc import InvertProductOutputDirichletBC

def custom__and__(self, other):
    if isinstance(other, InvertProductOutputDirichletBC):
        output = assemble(self, keep_diagonal=True)
        return output & other
    else:
        return NotImplemented
setattr(Form, "__and__", custom__and__)
