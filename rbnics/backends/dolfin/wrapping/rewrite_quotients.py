# Copyright (C) 2015-2019 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

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
        return n*(1./d)
        
def rewrite_quotients(form):
    form = remove_complex_nodes(form) # TODO support forms in the complex field. This is currently needed otherwise conj(a/b) does not get rewritten.
    return map_integrand_dags(RewriteQuotientsReplacer(), form)
