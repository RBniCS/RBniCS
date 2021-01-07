# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
from rbnics.utils.decorators import BackendFor
from rbnics.backends.abstract import ParametrizedTensorFactory as AbstractParametrizedTensorFactory


@BackendFor("common", inputs=(Number, ))
class ParametrizedTensorFactory(AbstractParametrizedTensorFactory):
    def __init__(self, scalar):
        AbstractParametrizedTensorFactory.__init__(self, scalar)

    def create_interpolation_locations_container(self):
        raise RuntimeError("This method should have never been called.")

    def create_snapshots_container(self):
        raise RuntimeError("This method should have never been called.")

    def create_empty_snapshot(self):
        raise RuntimeError("This method should have never been called.")

    def create_basis_container(self):
        raise RuntimeError("This method should have never been called.")

    def create_POD_container(self):
        raise RuntimeError("This method should have never been called.")

    def name(self):
        raise RuntimeError("This method should have never been called.")

    def description(self):
        raise RuntimeError("This method should have never been called.")

    def is_parametrized(self):
        return False

    def is_time_dependent(self):
        return False
