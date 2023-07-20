# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

try:
    from ufl_legacy import Form
except ImportError:
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
