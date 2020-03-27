# Copyright (C) 2015-2020 by the RBniCS authors
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

from rbnics.utils.decorators import PreserveClassName

def DefineSymbolicParameters(ParametrizedDifferentialProblem_DerivedClass):
    from rbnics.backends import SymbolicParameters # cannot import at global scope

    if not hasattr(ParametrizedDifferentialProblem_DerivedClass, "attach_symbolic_parameters"):
        @PreserveClassName
        class DefineSymbolicParameters_Class(ParametrizedDifferentialProblem_DerivedClass):

            # Default initialization of members
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
                # Storage for symbolic parameters
                self.mu_float = None
                self.mu_symbolic = None
                self.attach_symbolic_parameters__calls = 0

            def attach_symbolic_parameters(self):
                # Initialize symbolic parameters only once (may be shared between DEIM/EIM and exact evaluation)
                if self.mu_symbolic is None:
                    self.mu_symbolic = SymbolicParameters(self, self.V, self.mu)
                # Swap storage
                if self.attach_symbolic_parameters__calls == 0:
                    self.mu_float = self.mu
                    self.mu = self.mu_symbolic
                self.attach_symbolic_parameters__calls += 1

            def detach_symbolic_parameters(self):
                self.attach_symbolic_parameters__calls -= 1
                assert self.attach_symbolic_parameters__calls >= 0
                # Restore original storage
                if self.attach_symbolic_parameters__calls == 0:
                    self.mu = self.mu_float
                    self.mu_float = None
    else:
        DefineSymbolicParameters_Class = ParametrizedDifferentialProblem_DerivedClass

    # return value (a class) for the decorator
    return DefineSymbolicParameters_Class
