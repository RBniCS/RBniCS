# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import assemble as dolfin_assemble


def assemble(form, tensor=None):
    return dolfin_assemble(form, keep_diagonal=True, tensor=tensor)
