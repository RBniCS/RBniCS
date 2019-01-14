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

from rbnics.backends.abstract import FunctionsList as AbstractFunctionsList
from rbnics.backends.basic import GramSchmidt as BasicGramSchmidt
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.transpose import transpose
from rbnics.backends.online.numpy.wrapping import gram_schmidt_projection_step
from rbnics.utils.decorators import BackendFor, ModuleWrapper

backend = ModuleWrapper(Function, transpose)
wrapping = ModuleWrapper(gram_schmidt_projection_step)
GramSchmidt_Base = BasicGramSchmidt(backend, wrapping)

@BackendFor("numpy", inputs=(AbstractFunctionsList, Matrix.Type(), (str, None)))
class GramSchmidt(GramSchmidt_Base):
    pass
