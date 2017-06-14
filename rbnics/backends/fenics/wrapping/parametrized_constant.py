# Copyright (C) 2015-2017 by the RBniCS authors
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

import types
import re
from dolfin import Expression

def ParametrizedConstant(truth_problem, parametrized_constant_code=None, *args, **kwargs):
    from rbnics.backends.fenics.wrapping.parametrized_expression import ParametrizedExpression
    if "element" not in kwargs:
        element = truth_problem.V.ufl_element()
        while element.num_sub_elements() > 0:
            element = element.sub_elements()[0]
        kwargs["element"] = element
    return ParametrizedExpression(truth_problem, parametrized_constant_code, *args, **kwargs)
    
def is_parametrized_constant(expr):
    return isinstance(expr, Expression) and bool(is_parametrized_constant.regex.match(expr.cppcode))
is_parametrized_constant.regex = re.compile("^mu_[0-9]+$")

def parametrized_constant_to_float(expr, point=None):
    if point is None:
        point = expr._mesh.coordinates()[0]
    return expr(point)
    
def expression_float(self):
    if is_parametrized_constant(self):
        return parametrized_constant_to_float(self)
    else:
        return NotImplemented
Expression.__float__ = expression_float

class ParametrizedConstantTuple(tuple):
    def __new__(cls, truth_problem, mu):
        return tuple.__new__(cls, [ParametrizedConstant(truth_problem, "mu[" + str(idx) + "]", mu=mu) for (idx, _) in enumerate(mu)])
        
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
    
import types
import rbnics.utils.decorators
from rbnics.utils.decorators import Extends, override, ProblemDecoratorFor as ProblemDecoratorFor_Base, ReducedProblemDecoratorFor as ReducedProblemDecoratorFor_Base, StoreProblemDecoratorsForFactories

def ProblemDecoratorFor(Algorithm, ExactAlgorithm=None, replaces=None, replaces_if=None, **kwargs):
    from rbnics.eim.problems import DEIM, EIM, ExactParametrizedFunctions
    if Algorithm in (DEIM, EIM):
        # Change ProblemDecoratorFor to override DEIMDecoratedProblem.set_mu_range so that querying self.mu
        # actually returns a ParametrizedConstant rather than a float
        def ProblemDecoratorFor_Decorator(ProblemDecorator):
            def ProblemDecoratorFor_DecoratedProblemGenerator(Problem):
                DecoratedProblem_Base = ProblemDecoratorFor_Base(Algorithm, ExactAlgorithm, replaces, replaces_if, **kwargs)(ProblemDecorator)(Problem)
                
                @Extends(DecoratedProblem_Base, preserve_class_name=True)
                @StoreProblemDecoratorsForFactories(DecoratedProblem_Base, Algorithm, ExactAlgorithm, replaces, replaces_if, **kwargs)
                class DecoratedProblem(DecoratedProblem_Base):
                    @override
                    def set_mu_range(self, mu_range):
                        # Storage for backup of parameter assigned while calling set_mu
                        self._mu_ParametrizedConstant_override = None
                        # Hack set_mu method to convert tuple to parametrized constants
                        # only in this call (i.e. when _mu_ParametrizedConstant_override is set)
                        original_set_mu = self.set_mu
                        def modified_set_mu(self, mu):
                            if hasattr(self, "_mu_ParametrizedConstant_override"):
                                self.mu = ParametrizedConstantTuple(self, mu)
                                self._mu_ParametrizedConstant_override = mu
                            else:
                                original_set_mu(mu)
                        self.set_mu = types.MethodType(modified_set_mu, self)
                        # Call parent, which in turns assembles operators using the hacked set_mu
                        DecoratedProblem_Base.set_mu_range(self, mu_range)
                        # Restore self.mu
                        assert self._mu_ParametrizedConstant_override is not None
                        self.mu = self._mu_ParametrizedConstant_override
                        del self._mu_ParametrizedConstant_override
                                        
                return DecoratedProblem
            return ProblemDecoratorFor_DecoratedProblemGenerator
        return ProblemDecoratorFor_Decorator
    elif Algorithm is ExactParametrizedFunctions:
        # Change ProblemDecoratorFor to override ExactParametrizedFunctionsDecoratedProblem.set_mu_range so that querying self.mu
        # actually returns a ParametrizedConstant rather than a float
        def ProblemDecoratorFor_Decorator(ProblemDecorator):
            def ProblemDecoratorFor_DecoratedProblemGenerator(Problem):
                DecoratedProblem_Base = ProblemDecoratorFor_Base(Algorithm, ExactAlgorithm, replaces, replaces_if, **kwargs)(ProblemDecorator)(Problem)
                
                @Extends(DecoratedProblem_Base, preserve_class_name=True)
                @StoreProblemDecoratorsForFactories(DecoratedProblem_Base, Algorithm, ExactAlgorithm, replaces, replaces_if, **kwargs)
                class DecoratedProblem(DecoratedProblem_Base):
                    @override
                    def assemble_operator(self, term):
                        # Storage for backup of parameter assigned while calling set_mu
                        self._mu_ParametrizedConstant_override = self.mu
                        # Temporarily replace float by a parametrized constant
                        self.mu = ParametrizedConstantTuple(self, self.mu)
                        # Call parent
                        output = DecoratedProblem_Base.assemble_operator(self, term)
                        # Restore self.mu
                        assert self._mu_ParametrizedConstant_override is not None
                        self.mu = self._mu_ParametrizedConstant_override
                        del self._mu_ParametrizedConstant_override
                        # Return
                        return output
                                        
                return DecoratedProblem
            return ProblemDecoratorFor_DecoratedProblemGenerator
        return ProblemDecoratorFor_Decorator

    else:
        return ProblemDecoratorFor_Base(Algorithm, ExactAlgorithm, replaces, replaces_if, **kwargs)
    
rbnics.utils.decorators.ProblemDecoratorFor = ProblemDecoratorFor

