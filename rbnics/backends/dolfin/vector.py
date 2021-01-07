# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin.cpp.la import GenericVector


def Vector():
    raise NotImplementedError("This is dummy function (not required by the interface) just store the Type")


# Attach a Type() function
def Type():
    return GenericVector


Vector.Type = Type


# pybind11 wrappers do not implement __neg__ unary operator
def custom__neg__(self):
    return -1. * self


setattr(GenericVector, "__neg__", custom__neg__)


# Preserve generator attribute in algebraic operators, as required by DEIM
def preserve_generator_attribute(operator):
    original_operator = getattr(GenericVector, operator)

    def custom_operator(self, other):
        if hasattr(self, "generator"):
            output = original_operator(self, other)
            output.generator = self.generator
            return output
        else:
            return original_operator(self, other)

    setattr(GenericVector, operator, custom_operator)


for operator in ("__add__", "__radd__", "__iadd__", "__sub__", "__rsub__", "__isub__",
                 "__mul__", "__rmul__", "__imul__", "__truediv__", "__itruediv__"):
    preserve_generator_attribute(operator)
