# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin.function.argument import Argument


def form_argument_replace(argument, reduced_V):
    return Argument(reduced_V[argument.number()], argument.number(), argument.part())
