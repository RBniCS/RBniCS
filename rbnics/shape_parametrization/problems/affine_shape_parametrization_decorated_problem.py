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

import re
from sympy import Inverse, Matrix, MatrixSymbol, symbols, sympify, Transpose, zeros
from rbnics.shape_parametrization.problems.shape_parametrization_decorated_problem import ShapeParametrizationDecoratedProblem
from rbnics.utils.decorators import PreserveClassName, ProblemDecoratorFor
from rbnics.utils.io import PickleIO

def AffineShapeParametrizationDecoratedProblem(*shape_parametrization_vertices_mappings, **decorator_kwargs):
    
    # Possibly read vertices mappings from file
    if (
        len(shape_parametrization_vertices_mappings) is 1
            and
        isinstance(shape_parametrization_vertices_mappings[0], str)
    ):
        filename = shape_parametrization_vertices_mappings[0]
        assert filename != "identity", "It does not make any sense to use this if you only have one subdomain without parametrization"
        assert PickleIO.exists_file("", filename)
        shape_parametrization_vertices_mappings = PickleIO.load_file("", filename)
        
    # Detect the mesh dimension based on the number of vertices to be mapped
    dim = None
    for vertices_mapping in shape_parametrization_vertices_mappings:
        if isinstance(vertices_mapping, str) and vertices_mapping.lower() == "identity":
            continue
        else:
            assert isinstance(vertices_mapping, dict)
            assert len(vertices_mapping) in (3, 4)
            if len(vertices_mapping) is 3:
                if dim is None:
                    dim = 2
                else:
                    assert dim is 2
            elif len(vertices_mapping) is 4:
                if dim is None:
                    dim = 3
                else:
                    assert dim is 3
    assert dim is not None, "It does not make any sens to use this of all your subdomains are not parametrized"
        
    # Get the shape parametrization expression from vertices mappings
    shape_parametrization_expression = [affine_shape_parametrization_from_vertices_mapping(dim, vertices_mapping) for vertices_mapping in shape_parametrization_vertices_mappings]
    
    # Apply the parent decorator
    AffineShapeParametrizationDecoratedProblem_Decorator_Base = ShapeParametrizationDecoratedProblem(*shape_parametrization_expression, **decorator_kwargs)
    
    # Further decorate the resulting class
    from rbnics.shape_parametrization.problems.affine_shape_parametrization import AffineShapeParametrization
    
    @ProblemDecoratorFor(AffineShapeParametrization, shape_parametrization_vertices_mappings=shape_parametrization_vertices_mappings)
    def AffineShapeParametrizationDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):
        
        AffineShapeParametrizationDecoratedProblem_Class_Base = AffineShapeParametrizationDecoratedProblem_Decorator_Base(ParametrizedDifferentialProblem_DerivedClass)
        
        @PreserveClassName
        class AffineShapeParametrizationDecoratedProblem_Class(AffineShapeParametrizationDecoratedProblem_Class_Base):
        
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                AffineShapeParametrizationDecoratedProblem_Class_Base.__init__(self, V, **kwargs)
                # Check mesh consistency
                assert V.mesh().topology().dim() is dim
                
        # return value (a class) for the decorator
        return AffineShapeParametrizationDecoratedProblem_Class
    
    # return the decorator itself
    return AffineShapeParametrizationDecoratedProblem_Decorator
    
# Auxiliary function
def affine_shape_parametrization_from_vertices_mapping(dim, vertices_mapping):
    # Detect how many parameters are used
    P = -1
    for (reference_vertex, deformed_vertex) in vertices_mapping.items():
        for i in range(dim):
            for match in mu_regex.findall(deformed_vertex[i]):
                P = max(P, int(match))
    if P >= 0:
        mu = MatrixSymbol("mu", P + 1, 1)
    else:
        mu = None
    # Convert vertices from string to symbols
    vertices_mapping_symbolic = dict()
    for (reference_vertex, deformed_vertex) in vertices_mapping.items():
        reference_vertex_symbolic = dim*[None]
        deformed_vertex_symbolic = dim*[None]
        for i in range(dim):
            reference_vertex_symbolic[i] = sympify(reference_vertex[i])
            deformed_vertex_symbolic[i] = sympify(deformed_vertex[i], locals={"mu": mu})
        reference_vertex_symbolic = tuple(reference_vertex_symbolic)
        deformed_vertex_symbolic = tuple(deformed_vertex_symbolic)
        assert reference_vertex_symbolic not in vertices_mapping_symbolic
        vertices_mapping_symbolic[reference_vertex_symbolic] = deformed_vertex_symbolic
    # Find A and b such that x_o = A x + b for all (x, x_o) in vertices_mapping
    lhs = zeros(dim + dim**2, dim + dim**2)
    rhs = zeros(dim + dim**2, 1)
    for (offset, (reference_vertex, deformed_vertex)) in enumerate(vertices_mapping_symbolic.items()):
        for i in range(dim):
            rhs[offset*dim + i] = deformed_vertex[i]
            lhs[offset*dim + i, i] = 1
            for j in range(dim):
                lhs[offset*dim + i, (i + 1)*dim + j] = reference_vertex[j]
    solution = Inverse(lhs)*rhs
    b = zeros(dim, 1)
    for i in range(dim):
        b[i] = solution[i]
    A = zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            A[i, j] = solution[dim + i*dim + j]
    # Convert into an expression
    x = Matrix([symbols("x[" + str(i) + "]") for i in range(dim)])
    x_o = A*x + b
    return tuple([str(x_o[i]).replace(", 0]", "]") for i in range(dim)])
    
mu_regex = re.compile("mu\[([0-9]+)\]")
