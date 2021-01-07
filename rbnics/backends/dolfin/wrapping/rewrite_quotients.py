# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.corealg.multifunction import MultiFunction
from rbnics.backends.dolfin.wrapping.remove_complex_nodes import remove_complex_nodes


class RewriteQuotientsReplacer(MultiFunction):
    expr = MultiFunction.reuse_if_untouched

    def division(self, o, n, d):
        # We need to rewrite quotients in this way so that expression like
        #   expr1*v/expr2
        # get factorized by SeparatedParametrizedForm as
        #   coefficient1 = expr1
        #   coefficient2 = 1/expr2
        # and not
        #   coefficient1 = expr1
        #   coefficient2 = expr2
        return n * (1. / d)


def rewrite_quotients(form):
    # TODO support forms in the complex field. This is currently needed otherwise conj(a/b) does not get rewritten.
    form = remove_complex_nodes(form)
    return map_integrand_dags(RewriteQuotientsReplacer(), form)
