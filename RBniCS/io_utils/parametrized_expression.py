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
import types

# This ideally should be a subclass of Expression. However, dolfin manual
# states that subclassing Expression may be significantly slower than using 
# JIT-compiled expressions. To this end we havoid subclassing expression and
# just add the set_mu method using the types library
def ParametrizedExpression(parametrized_expression_code=None, *args, **kwargs):    
    if parametrized_expression_code is None:
        return None
    
    assert "mu" in kwargs
    mu = kwargs["mu"]
    assert mu is not None
    assert isinstance(mu, tuple)
    for p in range(len(mu)):
        if isinstance(parametrized_expression_code, tuple):
            new_parametrized_expression_code = list()
            for i in range(len(parametrized_expression_code)):
                assert isinstance(parametrized_expression_code[i], str)
                new_parametrized_expression_code.append(parametrized_expression_code[i].replace("mu[" + str(p) + "]", "mu_" + str(p)))
            parametrized_expression_code = tuple(new_parametrized_expression_code)
        elif isinstance(parametrized_expression_code, str):
            parametrized_expression_code = parametrized_expression_code.replace("mu[" + str(p) + "]", "mu_" + str(p))
        else:
            raise RuntimeError("Invalid expression type in ParametrizedExpression")
    
    mu_dict = {}
    for p in range(len(mu)):
        mu_dict[ "mu_" + str(p) ] = mu[p]
    del kwargs["mu"]
    kwargs.update(mu_dict)
            
    expression = Expression(parametrized_expression_code, *args, **kwargs)
    expression.len_mu = len(mu)
    
    def set_mu(self, mu):
        assert isinstance(mu, tuple)
        assert len(mu) == self.len_mu
        for p in range(len(mu)):
            setattr(self, "mu_" + str(p), mu[p])
    expression.set_mu = types.MethodType(set_mu, expression)
    
    return expression

