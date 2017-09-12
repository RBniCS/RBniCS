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

from rbnics.eim.problems import DEIM, EIM, ExactParametrizedFunctions
from rbnics.utils.decorators import PreserveClassName, ProblemDecoratorFor
from shape_parametrization.utils import PyGeMWrapper
import backends.dolfin # make sure that dolfin backend is overridden

def PyGeMDecoratedProblem(pygem_morphing_type, pygem_parameters_filename, pygem_index_and_component_to_mu_index_map, **decorator_kwargs):
    assert pygem_morphing_type in ("FFD", "RBF")
    
    @ProblemDecoratorFor(PyGeM,
        pygem_morphing_type=pygem_morphing_type,
        pygem_parameters_filename=pygem_parameters_filename,
        pygem_index_and_component_to_mu_index_map=pygem_index_and_component_to_mu_index_map
    )
    def PyGeMDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):
        assert (
            hasattr(ParametrizedDifferentialProblem_DerivedClass, "ProblemDecorators")
        ), \
        """DEIM or ExactParametrizedFunctions are required when using PyGeM for mesh motion, while apparently your problem does not have any decorator!
Please make sure that you decorate your problem as
    @PyGeM(...)
    @DEIM(...)
rather than
    @DEIM(...)
    @PyGeM(...)
because DEIM (or ExactParametrizedFunctions) have to be applied first."""
        
        assert EIM not in ParametrizedDifferentialProblem_DerivedClass.ProblemDecorators, "EIM is not supported for mesh motion by PyGeM"
        
        assert (
            DEIM in ParametrizedDifferentialProblem_DerivedClass.ProblemDecorators
                or
            ExactParametrizedFunctions in ParametrizedDifferentialProblem_DerivedClass.ProblemDecorators
        ), \
        """DEIM or ExactParametrizedFunctions are required when using PyGeM for mesh motion. 
Please make sure that you decorate your problem as
    @PyGeM(...)
    @DEIM(...)
rather than
    @DEIM(...)
    @PyGeM(...)
because DEIM (or ExactParametrizedFunctions) have to be applied first."""
        
        @PreserveClassName
        class PyGeMDecoratedProblem_BaseClass(ParametrizedDifferentialProblem_DerivedClass):
        
            def __init__(self, V, **kwargs):
                # Call the standard initialization
                ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
                # Initialize a PyGeM wrapper
                self.pygem_wrapper = PyGeMWrapper(
                    pygem_morphing_type=pygem_morphing_type,
                    pygem_parameters_filename=pygem_parameters_filename,
                    pygem_index_and_component_to_mu_index_map=pygem_index_and_component_to_mu_index_map
                )
                self.pygem_wrapper.init(V.mesh())

            ## Initialize data structures required for the offline phase
            def init(self):
                ParametrizedDifferentialProblem_DerivedClass.init(self)
                # Check consistency between self.mu and parameters related to deformation.
                # Cannot do that in the initialization becase self.mu is not available yet.
                assert len(pygem_index_and_component_to_mu_index_map) <= len(self.mu)
                assert min(pygem_index_and_component_to_mu_index_map.values()) >= 0
                assert max(pygem_index_and_component_to_mu_index_map.values()) < len(self.mu)
                                
            def set_mu(self, mu):
                ParametrizedDifferentialProblem_DerivedClass.set_mu(self, mu)
                # Update pygem parameters and data structures
                self.pygem_wrapper.update(mu)
                 
        if ExactParametrizedFunctions in ParametrizedDifferentialProblem_DerivedClass.ProblemDecorators:
            @PreserveClassName
            class PyGeMDecoratedProblem_ExactParametrizedFunctionsClass(PyGeMDecoratedProblem_BaseClass):
                def set_mu(self, mu):
                    PyGeMDecoratedProblem_BaseClass.set_mu(self, mu)
                    # Deform mesh
                    self.pygem_wrapper.move_mesh()
                    
            # return value (a class) for the decorator
            return PyGeMDecoratedProblem_ExactParametrizedFunctionsClass
            
        elif DEIM in ParametrizedDifferentialProblem_DerivedClass.ProblemDecorators:
            @PreserveClassName
            class PyGeMDecoratedProblem_DEIMClass(PyGeMDecoratedProblem_BaseClass):
                ## Deform the mesh as a function of the geometrical parameters and then export solution to file
                def export_solution(self, folder, filename, solution=None, component=None, suffix=None):
                    self.pygem_wrapper.move_mesh()
                    PyGeMDecoratedProblem_BaseClass.export_solution(self, folder, filename, solution, component, suffix)
                    self.pygem_wrapper.reset_reference()
                    
            # return value (a class) for the decorator
            return PyGeMDecoratedProblem_DEIMClass
        else:
            raise AssertionError("DEIM or ExactParametrizedFunctions are required when using PyGeM for mesh motion.")

    # return the decorator itself
    return PyGeMDecoratedProblem_Decorator
    
# For the sake of the user, since this is the only class that he/she needs to use, rename it to an easier name
PyGeM = PyGeMDecoratedProblem
