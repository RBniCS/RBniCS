# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod


@AbstractBackend
class MeshMotion(object, metaclass=ABCMeta):
    def __init__(self, space, subdomains, shape_parametrization_expression):
        pass

    @abstractmethod
    def init(self, problem):
        pass

    @abstractmethod
    def move_mesh(self):
        pass

    @abstractmethod
    def reset_reference(self):
        pass
