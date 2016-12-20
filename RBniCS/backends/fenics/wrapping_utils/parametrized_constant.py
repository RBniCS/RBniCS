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

from RBniCS.backends.fenics.wrapping_utils.parametrized_expression import ParametrizedExpression

def ParametrizedConstant(truth_problem, parametrized_constant_code=None, *args, **kwargs):
    if "element" not in kwargs:
        kwargs["element"] = truth_problem.V.ufl_element()
    return ParametrizedExpression(truth_problem, parametrized_constant_code, *args, **kwargs)
    
import types
from dolfin import Constant
import RBniCS.utils.decorators
from RBniCS.utils.decorators import Extends, override, ProblemDecoratorFor as ProblemDecoratorFor_Base

def ProblemDecoratorFor(Algorithm, ExactAlgorithm=None, replaces=None, replaces_if=None, **kwargs):
    from RBniCS.eim.problems import DEIM, ExactParametrizedFunctions
    if Algorithm is DEIM:
        # Change ProblemDecoratorFor to override DEIMDecoratedProblem.set_mu_range so that querying self.mu
        # actually returns a ParametrizedConstant rather than a float
        def ProblemDecoratorFor_Decorator(ProblemDecorator):
            def ProblemDecoratorFor_DecoratedProblemGenerator(Problem):
                DecoratedProblem_Base = ProblemDecoratorFor_Base(Algorithm, ExactAlgorithm, replaces, replaces_if, **kwargs)(ProblemDecorator)(Problem)
                @Extends(DecoratedProblem_Base, preserve_class_name=True)
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
                                self.mu = [ParametrizedConstant(self, "mu[" + str(idx) + "]", mu=mu) for (idx, _) in enumerate(mu)]
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
        # actually returns a Constant(float) rather than a float
        def ProblemDecoratorFor_Decorator(ProblemDecorator):
            def ProblemDecoratorFor_DecoratedProblemGenerator(Problem):
                DecoratedProblem_Base = ProblemDecoratorFor_Base(Algorithm, ExactAlgorithm, replaces, replaces_if, **kwargs)(ProblemDecorator)(Problem)
                @Extends(DecoratedProblem_Base, preserve_class_name=True)
                class DecoratedProblem(DecoratedProblem_Base):
                    @override
                    def assemble_operator(self, term):
                        # Storage for backup of parameter assigned while calling set_mu
                        self._mu_Constant_override = self.mu
                        # Temporarily replace float by Constant(float)
                        self.mu = [Constant(mu_i) for mu_i in self._mu_Constant_override]
                        # Call parent
                        output = DecoratedProblem_Base.assemble_operator(self, term)
                        # Restore self.mu
                        self.mu = self._mu_Constant_override
                        # Return
                        return output
                                        
                return DecoratedProblem
            return ProblemDecoratorFor_DecoratedProblemGenerator
        return ProblemDecoratorFor_Decorator
    else:
        return ProblemDecoratorFor_Base(Algorithm, ExactAlgorithm, replaces, replaces_if, **kwargs)
    
RBniCS.utils.decorators.ProblemDecoratorFor = ProblemDecoratorFor
