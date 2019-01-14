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

from ufl import Form
from rbnics.backends.basic import NonAffineExpansionStorage as BasicNonAffineExpansionStorage
from rbnics.backends.dolfin.parametrized_tensor_factory import ParametrizedTensorFactory
from rbnics.utils.decorators import BackendFor, ModuleWrapper, tuple_of

backend = ModuleWrapper(ParametrizedTensorFactory)
wrapping = ModuleWrapper()
NonAffineExpansionStorage_Base = BasicNonAffineExpansionStorage(backend, wrapping)

@BackendFor("dolfin", inputs=(tuple_of(Form), ))
class NonAffineExpansionStorage(NonAffineExpansionStorage_Base):
    pass
