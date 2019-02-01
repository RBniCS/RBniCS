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

from rbnics.shape_parametrization.problems.shape_parametrization_decorated_problem import ShapeParametrizationDecoratedProblem
from rbnics.shape_parametrization.utils.symbolic import affine_shape_parametrization_from_vertices_mapping, VerticesMappingIO
from rbnics.utils.decorators import ProblemDecoratorFor

def AffineShapeParametrizationDecoratedProblem(*shape_parametrization_vertices_mappings, **decorator_kwargs):
    
    if "shape_parametrization_vertices_mappings" in decorator_kwargs:
        assert len(shape_parametrization_vertices_mappings) == 0
        shape_parametrization_vertices_mappings = decorator_kwargs["shape_parametrization_vertices_mappings"]
    
    # Possibly read vertices mappings from file
    if (
        len(shape_parametrization_vertices_mappings) == 1
            and
        isinstance(shape_parametrization_vertices_mappings[0], str)
    ):
        filename = shape_parametrization_vertices_mappings[0]
        assert filename != "identity", "It does not make any sense to use this if you only have one subdomain without parametrization"
        assert VerticesMappingIO.exists_file("", filename)
        shape_parametrization_vertices_mappings = VerticesMappingIO.load_file("", filename)
        
    # Detect the mesh dimension based on the number of vertices to be mapped
    dim = None
    for vertices_mapping in shape_parametrization_vertices_mappings:
        if isinstance(vertices_mapping, str):
            assert vertices_mapping.lower() == "identity"
            continue
        else:
            assert isinstance(vertices_mapping, dict)
            assert len(vertices_mapping) in (3, 4)
            if len(vertices_mapping) == 3:
                if dim is None:
                    dim = 2
                else:
                    assert dim == 2
            elif len(vertices_mapping) == 4:
                if dim is None:
                    dim = 3
                else:
                    assert dim == 3
    assert dim is not None, "It does not make any sense to use this of all your subdomains are not parametrized"
        
    # Get the shape parametrization expression from vertices mappings
    shape_parametrization_expression = [affine_shape_parametrization_from_vertices_mapping(dim, vertices_mapping) for vertices_mapping in shape_parametrization_vertices_mappings]
    if decorator_kwargs.get("debug", False):
        print("=== DEBUGGING AFFINE SHAPE PARAMETRIZATION ===")
        for (subdomain, (vertices_mapping, expression)) in enumerate(zip(shape_parametrization_vertices_mappings, shape_parametrization_expression)):
            print("Subdomain", subdomain + 1)
            print("\tvertices mapping =", vertices_mapping)
            print("\tshape parametrization expression =", expression)
    decorator_kwargs.pop("debug", None)
    
    # Apply the parent decorator
    AffineShapeParametrizationDecoratedProblem_Decorator_Base = ShapeParametrizationDecoratedProblem(*shape_parametrization_expression, **decorator_kwargs)
    
    # Further decorate the resulting class
    from rbnics.shape_parametrization.problems.affine_shape_parametrization import AffineShapeParametrization
    
    @ProblemDecoratorFor(AffineShapeParametrization, shape_parametrization_vertices_mappings=shape_parametrization_vertices_mappings)
    def AffineShapeParametrizationDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):
        
        AffineShapeParametrizationDecoratedProblem_Class = AffineShapeParametrizationDecoratedProblem_Decorator_Base(ParametrizedDifferentialProblem_DerivedClass)
        
        # return value (a class) for the decorator
        return AffineShapeParametrizationDecoratedProblem_Class
    
    # return the decorator itself
    return AffineShapeParametrizationDecoratedProblem_Decorator
