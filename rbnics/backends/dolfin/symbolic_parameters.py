# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import FunctionSpace
from rbnics.backends.abstract import SymbolicParameters as AbstractSymbolicParameters
from rbnics.backends.dolfin.wrapping import ParametrizedConstant
from rbnics.utils.decorators import BackendFor, ParametersType


@BackendFor("dolfin", inputs=(object, FunctionSpace, ParametersType))
class SymbolicParameters(AbstractSymbolicParameters, tuple):
    def __new__(cls, problem, V, mu):
        return tuple.__new__(cls, [
            ParametrizedConstant(problem, "mu[" + str(idx) + "]", mu=mu) for (idx, _) in enumerate(mu)])

    def __str__(self):
        if len(self) == 0:
            return "()"
        elif len(self) == 1:
            return "(" + str(float(self[0])) + ",)"
        else:
            output = "("
            for mu_p in self:
                output += str(float(mu_p)) + ", "
            output = output[:-2]
            output += ")"
            return output
