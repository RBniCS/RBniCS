# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends import MeshMotion
from rbnics.utils.decorators import PreserveClassName, ProblemDecoratorFor


def ShapeParametrizationDecoratedProblem(*shape_parametrization_expression, **decorator_kwargs):
    from rbnics.shape_parametrization.problems.shape_parametrization import ShapeParametrization

    @ProblemDecoratorFor(ShapeParametrization, shape_parametrization_expression=shape_parametrization_expression)
    def ShapeParametrizationDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):

        # A decorator class that allows to overload methods related to shape parametrization and mesh motion
        @PreserveClassName
        class ShapeParametrizationDecoratedProblem_Class(ParametrizedDifferentialProblem_DerivedClass):

            # Default initialization of members
            # The shape parametrization expression is a list of tuples. The i-th list element
            # corresponds to shape parametrization of the i-th subdomain, the j-th tuple element
            # corresponds to the expression of the j-th component of the shape parametrization
            def __init__(self, V, **kwargs):
                # Call the standard initialization
                ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)

                # Get shape paramatrization expression
                if len(shape_parametrization_expression) == 0:
                    shape_parametrization_expression__from_decorator = decorator_kwargs[
                        "shape_parametrization_expression"]
                else:
                    shape_parametrization_expression__from_decorator = shape_parametrization_expression

                # Store mesh motion class
                assert "subdomains" in kwargs
                self.mesh_motion = MeshMotion(V, kwargs["subdomains"],
                                              shape_parametrization_expression__from_decorator)

                # Store the shape parametrization expression
                self.shape_parametrization_expression = shape_parametrization_expression__from_decorator

            # Initialize data structures required for the offline phase
            def init(self):
                ParametrizedDifferentialProblem_DerivedClass.init(self)
                # Also init mesh motion object
                self.mesh_motion.init(self)

            # Deform the mesh as a function of the geometrical parameters and then export solution to file
            def export_solution(self, folder=None, filename=None, solution=None, component=None, suffix=None):
                self.mesh_motion.move_mesh()
                ParametrizedDifferentialProblem_DerivedClass.export_solution(
                    self, folder, filename, solution, component, suffix)
                self.mesh_motion.reset_reference()

        # return value (a class) for the decorator
        return ShapeParametrizationDecoratedProblem_Class

    # return the decorator itself
    return ShapeParametrizationDecoratedProblem_Decorator
