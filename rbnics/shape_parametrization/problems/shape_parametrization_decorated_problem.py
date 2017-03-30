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

from rbnics.backends import MeshMotion
from rbnics.utils.decorators import Extends, override, ProblemDecoratorFor

def ShapeParametrizationDecoratedProblem(*shape_parametrization_expression, **decorator_kwargs):
    @ProblemDecoratorFor(ShapeParametrization,
        shape_parametrization_expression=shape_parametrization_expression
    )
    def ShapeParametrizationDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):
        
        # A decorator class that allows to overload methods related to shape parametrization and mesh motion
        @Extends(ParametrizedDifferentialProblem_DerivedClass, preserve_class_name=True)
        class ShapeParametrizationDecoratedProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
        
            ## Default initialization of members
            # The shape parametrization expression is a list of tuples. The i-th list element
            # corresponds to shape parametrization of the i-th subdomain, the j-th tuple element
            # corresponds to the expression of the j-th component of the shape parametrization
            @override
            def __init__(self, V, **kwargs):
                # Call the standard initialization
                ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
                # Store mesh motion class
                if len(shape_parametrization_expression) == 0:
                    shape_parametrization_expression__from_decorator = decorator_kwargs["shape_parametrization_expression"]
                else:
                    shape_parametrization_expression__from_decorator = shape_parametrization_expression
                assert "subdomains" in kwargs
                self.mesh_motion = MeshMotion(V, kwargs["subdomains"], shape_parametrization_expression__from_decorator)
            
            ## Initialize data structures required for the offline phase
            @override
            def init(self):
                ParametrizedDifferentialProblem_DerivedClass.init(self)
                # Also init mesh motion object
                self.mesh_motion.init(self)
                
            ## Deform the mesh as a function of the geometrical parameters and then export solution to file
            @override
            def export_solution(self, folder, filename, solution=None, component=None, suffix=None):
                self.mesh_motion.move_mesh()
                ParametrizedDifferentialProblem_DerivedClass.export_solution(self, folder, filename, solution, component, suffix)
                self.mesh_motion.reset_reference()
        
        # return value (a class) for the decorator
        return ShapeParametrizationDecoratedProblem_Class
    
    # return the decorator itself
    return ShapeParametrizationDecoratedProblem_Decorator
    
# For the sake of the user, since this is the only class that he/she needs to use, rename it to an easier name
ShapeParametrization = ShapeParametrizationDecoratedProblem
