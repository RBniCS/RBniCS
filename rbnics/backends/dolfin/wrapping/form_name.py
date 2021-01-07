# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import hashlib
from rbnics.backends.dolfin.wrapping.expression_name import expression_name
from rbnics.utils.decorators import ModuleWrapper


def basic_form_name(backend, wrapping):

    def _basic_form_name(form):
        str_repr = ""
        for integral in form.integrals():
            str_repr += wrapping.expression_name(integral.integrand())
            str_repr += "measure(" + integral.integral_type() + ")[" + str(integral.subdomain_id()) + "]"
        hash_code = hashlib.sha1(str_repr.encode("utf-8")).hexdigest()
        return hash_code

    return _basic_form_name


backend = ModuleWrapper()
wrapping = ModuleWrapper(expression_name=expression_name)
form_name = basic_form_name(backend, wrapping)
