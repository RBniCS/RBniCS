# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl import Form
from ufl.core.operator import Operator
from dolfin import FunctionSpace
from rbnics.backends.basic import GramSchmidt as BasicGramSchmidt
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.transpose import transpose
from rbnics.backends.dolfin.wrapping import (function_extend_or_restrict, function_from_ufl_operators,
                                             get_function_subspace, gram_schmidt_projection_step)
from rbnics.utils.decorators import BackendFor, dict_of, ModuleWrapper, overload

backend = ModuleWrapper(Function, transpose)
wrapping = ModuleWrapper(function_extend_or_restrict, get_function_subspace, gram_schmidt_projection_step)
GramSchmidt_Base = BasicGramSchmidt(backend, wrapping)


@BackendFor("dolfin", inputs=(FunctionSpace, (Form, Matrix.Type()), (str, None)))
class GramSchmidt(GramSchmidt_Base):
    @overload(Operator, (None, str, dict_of(str, str)))
    def _extend_or_restrict_if_needed(self, function, component):
        function = function_from_ufl_operators(function)
        return GramSchmidt_Base._extend_or_restrict_if_needed(self, function, component)
