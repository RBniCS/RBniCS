# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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
