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
# JIT-compiled expressions. To this end we avoid subclassing expression and
# just add the set_mu method using the types library
def ParametrizedExpression(truth_problem, parametrized_expression_code=None, *args, **kwargs):    
    if parametrized_expression_code is None:
        return None
    
    assert "mu" in kwargs
    mu = kwargs["mu"]
    assert mu is not None
    assert isinstance(mu, tuple)
    assert len(mu) > 0
    for p in range(len(mu)):
        if isinstance(parametrized_expression_code, tuple):
            if isinstance(parametrized_expression_code[0], tuple):
                new_parametrized_expression_code = list()
                for i in range(len(parametrized_expression_code)):
                    assert isinstance(parametrized_expression_code[i], tuple)
                    new_parametrized_expression_code_i = list()
                    for j in range(len(parametrized_expression_code[i])):
                        assert isinstance(parametrized_expression_code[i][j], str)
                        new_parametrized_expression_code_i.append(parametrized_expression_code[i][j].replace("mu[" + str(p) + "]", "mu_" + str(p)))
                    parametrized_expression_code_i = tuple(new_parametrized_expression_code_i)
                    new_parametrized_expression_code.append(parametrized_expression_code_i)
                parametrized_expression_code = tuple(new_parametrized_expression_code)
            else:
                new_parametrized_expression_code = list()
                for i in range(len(parametrized_expression_code)):
                    assert isinstance(parametrized_expression_code[i], str)
                    new_parametrized_expression_code.append(parametrized_expression_code[i].replace("mu[" + str(p) + "]", "mu_" + str(p)))
                parametrized_expression_code = tuple(new_parametrized_expression_code)
        elif isinstance(parametrized_expression_code, str):
            parametrized_expression_code = parametrized_expression_code.replace("mu[" + str(p) + "]", "mu_" + str(p))
        else:
            raise TypeError("Invalid expression type in ParametrizedExpression")
    
    mu_dict = {}
    for p in range(len(mu)):
        mu_dict[ "mu_" + str(p) ] = mu[p]
    del kwargs["mu"]
    kwargs.update(mu_dict)
            
    expression = Expression(parametrized_expression_code, *args, **kwargs)
    expression.mu = mu # to avoid repeated assignments
        
    standard_set_mu = truth_problem.set_mu
    def overridden_set_mu(self, mu):
        standard_set_mu(mu)
        if expression.mu is not mu:
            assert isinstance(mu, tuple)
            assert len(mu) == len(expression.mu)
            for p in range(len(mu)):
                setattr(expression, "mu_" + str(p), mu[p])
            expression.mu = mu
    truth_problem.set_mu = types.MethodType(overridden_set_mu, truth_problem)
    # Note that this override is different from the one that we use in decorated problems,
    # since (1) we do not want to define a new child class, (2) we have to execute some preprocessing
    # on the data, (3) it is a one-way propagation rather than a sync. 
    # For these reasons, the decorator @SyncSetters is not used but we partially duplicate some code
    
    return expression

