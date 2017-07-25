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

from rbnics.backends import SymbolicParameters
from rbnics.utils.decorators import Extends, override, ProblemDecoratorFor

def ExactParametrizedFunctions_OfflineAndOnline(**kwargs):
    # Enable exact parametrized functions evaluation both offline and online
    from rbnics.eim.problems.exact_parametrized_functions import ExactParametrizedFunctions
    kwargs["stages"] = ("offline", "online")
    return ExactParametrizedFunctions(**kwargs)

def ExactParametrizedFunctionsDecoratedProblem(
    stages=("offline", "online"),
    **decorator_kwargs
):

    from rbnics.eim.problems.deim import DEIM
    from rbnics.eim.problems.eim import EIM
    from rbnics.eim.problems.exact_parametrized_functions import ExactParametrizedFunctions
    
    @ProblemDecoratorFor(ExactParametrizedFunctions, ExactAlgorithm=ExactParametrizedFunctions_OfflineAndOnline,
        stages=stages
    )
    def ExactParametrizedFunctionsDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):
        
        @Extends(ParametrizedDifferentialProblem_DerivedClass, preserve_class_name=True)
        class ExactParametrizedFunctionsDecoratedProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
            
            ## Default initialization of members
            @override
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
                # Storage for symbolic parameters
                self.mu_symbolic = None
                
                # Store values passed to decorator
                assert isinstance(stages, (str, tuple))
                if isinstance(stages, tuple):
                    assert len(stages) in (1, 2)
                    assert stages[0] in ("offline", "online")
                    if len(stages) > 1:
                        assert stages[1] in ("offline", "online")
                        assert stages[0] != stages[1]
                    self._apply_exact_approximation_at_stages = stages
                elif isinstance(stages, str):
                    assert stages != "online", "This choice does not make any sense because it requires an EIM/DEIM offline stage which then is not used online"
                    assert stages == "offline"
                    self._apply_exact_approximation_at_stages = (stages, )
                else:
                    raise AssertionError("Invalid value for stages")
            
            @override
            def init(self):
                # Initialize symbolic parameters only once
                if self.mu_symbolic is None:
                    self.mu_symbolic = SymbolicParameters(self, self.V, self.mu)
                # Temporarily replace float parameters with symbols, so that the forms do not hardcode
                # the current value of the parameter while assemblying.
                mu_float = self.mu
                self.mu = self.mu_symbolic
                # Call parent
                ParametrizedDifferentialProblem_DerivedClass.init(self)
                # Restore float parameters
                self.mu = mu_float
            
        # return value (a class) for the decorator
        return ExactParametrizedFunctionsDecoratedProblem_Class
        
    # return the decorator itself
    return ExactParametrizedFunctionsDecoratedProblem_Decorator
