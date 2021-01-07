# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from logging import DEBUG, getLogger
from rbnics.shape_parametrization.problems.shape_parametrization_decorated_problem import (
    ShapeParametrizationDecoratedProblem)
from rbnics.shape_parametrization.utils.symbolic import (
    affine_shape_parametrization_from_vertices_mapping, VerticesMappingIO)
from rbnics.utils.decorators import ProblemDecoratorFor

logger = getLogger("rbnics/shape_parametrization/problems/affine_shape_parametrization_decorated_problem.py")


def AffineShapeParametrizationDecoratedProblem(*shape_parametrization_vertices_mappings, **decorator_kwargs):

    if "shape_parametrization_vertices_mappings" in decorator_kwargs:
        assert len(shape_parametrization_vertices_mappings) == 0
        shape_parametrization_vertices_mappings = decorator_kwargs["shape_parametrization_vertices_mappings"]

    # Possibly read vertices mappings from file
    if (len(shape_parametrization_vertices_mappings) == 1
            and isinstance(shape_parametrization_vertices_mappings[0], str)):
        filename = shape_parametrization_vertices_mappings[0]
        assert filename != "identity", (
            "It does not make any sense to use this if you only have one subdomain without parametrization")
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
    shape_parametrization_expression = [affine_shape_parametrization_from_vertices_mapping(dim, vertices_mapping)
                                        for vertices_mapping in shape_parametrization_vertices_mappings]
    if logger.isEnabledFor(DEBUG):
        logger.log(DEBUG, "=== DEBUGGING AFFINE SHAPE PARAMETRIZATION ===")
        for (subdomain, (vertices_mapping, expression)) in enumerate(zip(
                shape_parametrization_vertices_mappings, shape_parametrization_expression)):
            logger.log(DEBUG, "Subdomain " + str(subdomain + 1))
            logger.log(DEBUG, "\tvertices mapping = " + str(vertices_mapping))
            logger.log(DEBUG, "\tshape parametrization expression = " + str(expression))

    # Apply the parent decorator
    AffineShapeParametrizationDecoratedProblem_Decorator_Base = ShapeParametrizationDecoratedProblem(
        *shape_parametrization_expression, **decorator_kwargs)

    # Further decorate the resulting class
    from rbnics.shape_parametrization.problems.affine_shape_parametrization import AffineShapeParametrization

    @ProblemDecoratorFor(
        AffineShapeParametrization, shape_parametrization_vertices_mappings=shape_parametrization_vertices_mappings)
    def AffineShapeParametrizationDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):

        AffineShapeParametrizationDecoratedProblem_Class = AffineShapeParametrizationDecoratedProblem_Decorator_Base(
            ParametrizedDifferentialProblem_DerivedClass)

        # return value (a class) for the decorator
        return AffineShapeParametrizationDecoratedProblem_Class

    # return the decorator itself
    return AffineShapeParametrizationDecoratedProblem_Decorator
