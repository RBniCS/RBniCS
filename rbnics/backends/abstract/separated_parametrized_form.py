# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod, abstractproperty


@AbstractBackend
class SeparatedParametrizedForm(object, metaclass=ABCMeta):
    def __init__(self, form):
        pass

    @abstractmethod
    def separate(self):
        pass

    @abstractproperty
    def coefficients(self):
        pass

    @abstractproperty
    def unchanged_forms(self):
        pass

    @abstractmethod
    def replace_placeholders(self, i, new_coefficients):
        pass

    @abstractmethod
    def placeholders_names(self, i):
        pass
