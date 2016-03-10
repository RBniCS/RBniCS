# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file parametrized_expression.py
#  @brief Extension of FEniCS expression to handle parameters.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import Expression

class ParametrizedExpression(Expression):
    def __init__(self, cppcode=None, *args, **kwargs):
        assert mu in kwargs
        self.mu = kwargs["mu"]
        assert self.mu is not None
        assert isinstance(self.mu, tuple)
        for p in range(len(self.mu)):
            parametrized_expression_code = parametrized_expression_code.replace("mu[" + str(p) + "]", "mu_" + str(p))
        mu_dict = {}
        for p in range(len(self.mu)):
            mu_dict[ "mu_" + str(p) ] = self.mu[p]
        del kwargs["mu"]
        kwargs.update(mu_dict)
        Expression.__init__(self, cppcode=parametrized_expression_code, *args, **kwargs)
    

