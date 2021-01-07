# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
import os
import itertools
import functools
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from logging import DEBUG
from dolfin import (assign, CellDiameter, Constant, cos, div, exp, Expression, FiniteElement, Function, FunctionSpace,
                    grad, inner, Measure, Mesh, MeshFunction, MixedElement, pi, project, sin, split, sqrt, tan,
                    TestFunction, TrialFunction, VectorElement)
from rbnics import ShapeParametrization
from rbnics.backends.dolfin.wrapping import (assemble_operator_for_derivative, compute_theta_for_derivative,
                                             ParametrizedExpression, PullBackFormsToReferenceDomain,
                                             PushForwardToDeformedDomain)
from rbnics.backends.dolfin.wrapping.pull_back_to_reference_domain import (
    forms_are_close, logger as pull_back_to_reference_domain_logger)
from rbnics.eim.problems import DEIM, EIM, ExactParametrizedFunctions
from rbnics.problems.base import ParametrizedProblem
from rbnics.utils.decorators import StoreMapFromSolutionToProblem
from rbnics.utils.test import enable_logging

enable_pull_back_to_reference_domain_logging = enable_logging({pull_back_to_reference_domain_logger: DEBUG})

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "test_pull_back_to_reference_domain")


def theta_times_operator(problem, term):
    return sum([Constant(theta) * operator for (theta, operator) in zip(
        problem.compute_theta(term), problem.assemble_operator(term))])


def keep_shape_parametrization_affine(shape_parametrization_expression):
    return shape_parametrization_expression


def make_shape_parametrization_non_affine(shape_parametrization_expression):
    non_affine_shape_parametrization_expression = list()
    for shape_parametrization_expression_on_subdomain in shape_parametrization_expression:
        non_affine_shape_parametrization_expression_on_subdomain = list()
        for (coord, expression_coord) in enumerate(shape_parametrization_expression_on_subdomain):
            non_affine_shape_parametrization_expression_on_subdomain.append(
                expression_coord + " + 1e-16*mu[0]*x[" + str(coord) + "]**2"
            )
        non_affine_shape_parametrization_expression.append(tuple(
            non_affine_shape_parametrization_expression_on_subdomain))
    return non_affine_shape_parametrization_expression


def NoDecorator():
    def NoDecorator_decorator(Class):
        return Class
    return NoDecorator_decorator


def raises(ExceptionType):
    """
        Wrapper around pytest.raises to support None.
        Credits: https://github.com/pytest-dev/pytest/issues/1830
    """
    if ExceptionType is not None:
        return pytest.raises(ExceptionType)
    else:
        @contextmanager
        def not_raises():
            try:
                yield
            except Exception as e:
                raise e
        return not_raises()


def check_affine_and_non_affine_shape_parametrizations(*decorator_args, **decorator_kwargs):

    def generate_default_decorator_args(is_affine):
        header = "shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message"
        if is_affine:
            return (
                header,
                [
                    (keep_shape_parametrization_affine, NoDecorator, None, None),
                    (make_shape_parametrization_non_affine, NoDecorator, AssertionError, (
                        "Non affine parametric dependence detected. Please use one among DEIM, EIM"
                        + " and ExactParametrizedFunctions")),
                    (make_shape_parametrization_non_affine, DEIM, None, None),
                    (make_shape_parametrization_non_affine, EIM, None, None),
                    (make_shape_parametrization_non_affine, ExactParametrizedFunctions, None, None)
                ]
            )
        else:
            return (
                header,
                [
                    (keep_shape_parametrization_affine, NoDecorator, AssertionError, (
                        "Non affine parametric dependence detected. Please use one among DEIM, EIM"
                        + " and ExactParametrizedFunctions")),
                    (keep_shape_parametrization_affine, DEIM, None, None),
                    (keep_shape_parametrization_affine, EIM, None, None),
                    (keep_shape_parametrization_affine, ExactParametrizedFunctions, None, None)
                ]
            )
    global_is_affine = decorator_kwargs.get("is_affine", True)
    decorator_args_change_affinity = None
    for (decorator_arg_id, decorator_arg) in enumerate(decorator_args):
        assert len(decorator_arg) in (2, 3)
        if len(decorator_arg) == 3:
            assert decorator_args_change_affinity is None
            decorator_args_change_affinity = decorator_arg_id
    if decorator_args_change_affinity is None:
        decorator_args_list = [generate_default_decorator_args(is_affine=global_is_affine)]
        decorator_args_list.extend(decorator_args)
    else:
        default_decorator_args = dict()
        default_decorator_args[True] = generate_default_decorator_args(is_affine=True)
        default_decorator_args[False] = generate_default_decorator_args(is_affine=False)
        combined_default_decorator_args = (
            default_decorator_args[True][0] + ", " + decorator_args[decorator_arg_id][0],
            []
        )
        assert len(decorator_args[decorator_arg_id][1]) == len(decorator_args[decorator_arg_id][2])
        for (decorator_arg_1, decorator_arg_2) in zip(
                decorator_args[decorator_arg_id][1], decorator_args[decorator_arg_id][2]):
            for default_decorator_arg_1 in default_decorator_args[decorator_arg_2][1]:
                combined_default_decorator_args[1].append(default_decorator_arg_1 + decorator_arg_1)
        combined_default_decorator_args = tuple(combined_default_decorator_args)
        decorator_args_list = [combined_default_decorator_args]
        for (decorator_arg_id, decorator_arg) in enumerate(decorator_args):
            if decorator_arg_id != decorator_args_change_affinity:
                decorator_args_list.append(decorator_arg)
    decorators = [pytest.mark.parametrize(decorator_arg[0], decorator_arg[1])
                  for decorator_arg in decorator_args_list]

    def check_affine_and_non_affine_shape_parametrizations_decorator(original_test):
        @functools.wraps(original_test)
        def test_with_exception_check(
                shape_parametrization_preprocessing, AdditionalProblemDecorator,
                ExceptionType, exception_message, **kwargs):
            with raises(ExceptionType) as excinfo:
                original_test(shape_parametrization_preprocessing, AdditionalProblemDecorator,
                              ExceptionType, exception_message, **kwargs)
            if ExceptionType is not None:
                assert str(excinfo.value) == exception_message
        decorated_test = test_with_exception_check
        for decorator in decorators:
            decorated_test = decorator(decorated_test)
        return decorated_test

    return check_affine_and_non_affine_shape_parametrizations_decorator


# Test forms pull back to reference domain for tutorial 03
@enable_pull_back_to_reference_domain_logging
@check_affine_and_non_affine_shape_parametrizations()
def test_pull_back_to_reference_domain_hole(
        shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "hole.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "hole_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "hole_facet_region.xml"))

    # Define shape parametrization
    shape_parametrization_expression = [
        ("2 - 2 * mu[0] + mu[0] * x[0] + (2 - 2 * mu[0]) * x[1]", "2 - 2 * mu[1] + (2 - mu[1]) * x[1]"),  # subdomain 1
        ("2 * mu[0]- 2 + x[0] + (mu[0] - 1) * x[1]", "2 - 2 * mu[1] + (2 - mu[1]) * x[1]"),  # subdomain 2
        ("2 - 2 * mu[0] + (2 - mu[0]) * x[0]", "2 - 2 * mu[1] + (2- 2 * mu[1]) * x[0] + mu[1] * x[1]"),  # subdomain 3
        ("2 - 2 * mu[0] + (2 - mu[0]) * x[0]", "2 * mu[1] - 2 + (mu[1] - 1) * x[0] + x[1]"),  # subdomain 4
        ("2 * mu[0] - 2 + (2 - mu[0]) * x[0]", "2 - 2 * mu[1] + (2 * mu[1]- 2) * x[0] + mu[1] * x[1]"),  # subdomain 5
        ("2 * mu[0] - 2 + (2 - mu[0]) * x[0]", "2 * mu[1] - 2 + (1 - mu[1]) * x[0] + x[1]"),  # subdomain 6
        ("2 - 2 * mu[0] + mu[0] * x[0] + (2 * mu[0] - 2) * x[1]", "2 * mu[1] - 2 + (2 - mu[1]) * x[1]"),  # subdomain 7
        ("2 * mu[0] - 2 + x[0] + (1 - mu[0]) * x[1]", "2 * mu[1] - 2 + (2 - mu[1]) * x[1]")  # subdomain 8
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)

    # Define function space, test/trial functions, measures
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    dx = Measure("dx")(subdomain_data=subdomains)
    ds = Measure("ds")(subdomain_data=boundaries)

    # Define base problem
    class Hole(ParametrizedProblem, metaclass=ABCMeta):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1., 1., 0)
            self.mu_range = [(0.5, 1.5), (0.5, 1.5), (0.01, 1.0)]
            self.terms = ["a", "f"]
            self.operator = dict()
            self.Q = dict()

        def name(self):
            return "___".join([
                self.folder_prefix, shape_parametrization_preprocessing.__name__,
                AdditionalProblemDecorator.__name__])

        def init(self):
            self._init_operators()

        def _init_operators(self):
            pass

        @abstractmethod
        def compute_theta(self, term):
            pass

        @abstractmethod
        def assemble_operator(self, term):
            pass

    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class HoleOnReferenceDomain(Hole):
        def __init__(self, V, **kwargs):
            Hole.__init__(self, "HoleOnReferenceDomain")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            mu = self.mu
            if term == "a":
                # subdomains 1 and 7
                theta_a0 = - (mu[1] - 2) / mu[0] - (2 * (2 * mu[0] - 2) * (mu[0] - 1)) / (mu[0] * (mu[1] - 2))
                theta_a1 = - mu[0] / (mu[1] - 2)
                theta_a2 = - (2 * (mu[0] - 1)) / (mu[1] - 2)
                # subdomains 2 and 8
                theta_a3 = 2 - (mu[0] - 1) * (mu[0] - 1) / (mu[1] - 2) - mu[1]
                theta_a4 = - 1 / (mu[1] - 2)
                theta_a5 = (mu[0] - 1) / (mu[1] - 2)
                # subdomains 3 and 5
                theta_a6 = - mu[1] / (mu[0] - 2)
                theta_a7 = - (mu[0] - 2) / mu[1] - (2 * (2 * mu[1] - 2) * (mu[1] - 1)) / (mu[1] * (mu[0] - 2))
                theta_a8 = - (2 * (mu[1] - 1)) / (mu[0] - 2)
                # subdomains 4 and 6
                theta_a9 = -1 / (mu[0] - 2)
                theta_a10 = 2 - (mu[1] - 1) * (mu[1] - 1) / (mu[0] - 2) - mu[0]
                theta_a11 = (mu[1] - 1) / (mu[0] - 2)
                # boundaries 5, 6, 7 and 8
                theta_a12 = mu[2]
                # Return
                return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7, theta_a8,
                        theta_a9, theta_a10, theta_a11, theta_a12)
            elif term == "f":
                theta_f0 = mu[0]  # boundary 1
                theta_f1 = mu[1]  # boundary 2
                theta_f2 = mu[0]  # boundary 3
                theta_f3 = mu[1]  # boundary 4
                # Return
                return (theta_f0, theta_f1, theta_f2, theta_f3)
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                # subdomains 1 and 7
                a0 = inner(u.dx(0), v.dx(0)) * dx(1) + inner(u.dx(0), v.dx(0)) * dx(7)
                a1 = inner(u.dx(1), v.dx(1)) * dx(1) + inner(u.dx(1), v.dx(1)) * dx(7)
                a2 = (inner(u.dx(0), v.dx(1)) * dx(1) + inner(u.dx(1), v.dx(0)) * dx(1)
                      - inner(u.dx(0), v.dx(1)) * dx(7) - inner(u.dx(1), v.dx(0)) * dx(7))
                # subdomains 2 and 8
                a3 = inner(u.dx(0), v.dx(0)) * dx(2) + inner(u.dx(0), v.dx(0)) * dx(8)
                a4 = inner(u.dx(1), v.dx(1)) * dx(2) + inner(u.dx(1), v.dx(1)) * dx(8)
                a5 = (inner(u.dx(0), v.dx(1)) * dx(2) + inner(u.dx(1), v.dx(0)) * dx(2)
                      - inner(u.dx(0), v.dx(1)) * dx(8) - inner(u.dx(1), v.dx(0)) * dx(8))
                # subdomains 3 and 5
                a6 = inner(u.dx(0), v.dx(0)) * dx(3) + inner(u.dx(0), v.dx(0)) * dx(5)
                a7 = inner(u.dx(1), v.dx(1)) * dx(3) + inner(u.dx(1), v.dx(1)) * dx(5)
                a8 = (inner(u.dx(0), v.dx(1)) * dx(3) + inner(u.dx(1), v.dx(0)) * dx(3)
                      - inner(u.dx(0), v.dx(1)) * dx(5) - inner(u.dx(1), v.dx(0)) * dx(5))
                # subdomains 4 and 6
                a9 = inner(u.dx(0), v.dx(0)) * dx(4) + inner(u.dx(0), v.dx(0)) * dx(6)
                a10 = inner(u.dx(1), v.dx(1)) * dx(4) + inner(u.dx(1), v.dx(1)) * dx(6)
                a11 = (inner(u.dx(0), v.dx(1)) * dx(4) + inner(u.dx(1), v.dx(0)) * dx(4)
                       - inner(u.dx(0), v.dx(1)) * dx(6) - inner(u.dx(1), v.dx(0)) * dx(6))
                # boundaries 5, 6, 7 and 8
                a12 = inner(u, v) * ds(5) + inner(u, v) * ds(6) + inner(u, v) * ds(7) + inner(u, v) * ds(8)
                # Return
                return (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)
            elif term == "f":
                f0 = v * ds(1)  # boundary 1
                f1 = v * ds(2)  # boundary 2
                f2 = v * ds(3)  # boundary 3
                f3 = v * ds(4)  # boundary 4
                # Return
                return (f0, f1, f2, f3)
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain()
    @ShapeParametrization(*shape_parametrization_expression)
    class HolePullBack(Hole):
        def __init__(self, V, **kwargs):
            Hole.__init__(self, "HolePullBack")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            mu = self.mu
            if term == "a":
                theta_a0 = 1.0
                theta_a1 = mu[2]
                return (theta_a0, theta_a1)
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0, )
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v)) * dx
                a1 = inner(u, v) * ds(5) + inner(u, v) * ds(6) + inner(u, v) * ds(7) + inner(u, v) * ds(8)
                return (a0, a1)
            elif term == "f":
                f0 = v * ds(1) + v * ds(2) + v * ds(3) + v * ds(4)
                return (f0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Check forms
    problem_on_reference_domain = HoleOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = HolePullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)

        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)

        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)


# Test forms pull back to reference domain for tutorial 03 rotation
@enable_pull_back_to_reference_domain_logging
@check_affine_and_non_affine_shape_parametrizations()
def test_pull_back_to_reference_domain_hole_rotation(
        shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "hole.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "hole_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "hole_facet_region.xml"))

    # Define shape parametrization
    shape_parametrization_expression = [
        ("-2 * sqrt(2.0) * cos(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2)"
         + "+ x[1] * (-sqrt(2.0) * sin(mu[0]) / 2 - 3 * sqrt(2.0) * cos(mu[0]) / 2 + 2) + 2",
         "-2 * sqrt(2.0) * sin(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0]) / 2 - sqrt(2.0) * cos(mu[0]) / 2)"
         + "+ x[1] * (-3 * sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2 + 2) + 2"),  # subdomain 1
        ("2 * sqrt(2.0) * sin(mu[0]) + x[0] + x[1] * (sqrt(2.0) * sin(mu[0]) - 1) - 2",
         "-2 * sqrt(2.0) * cos(mu[0]) + x[1] * (-sqrt(2.0) * cos(mu[0]) + 2) + 2"),  # subdomain 2
        ("-2 * sqrt(2.0) * cos(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0])/2 - 3 * sqrt(2.0) * cos(mu[0]) / 2 + 2)"
         + "+ x[1] * (-sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2) + 2",
         "-2 * sqrt(2.0) * sin(mu[0]) + x[0] * (-3 * sqrt(2.0) * sin(mu[0]) / 2 - sqrt(2.0) * cos(mu[0]) / 2 + 2)"
         + "+ x[1] * (sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2) + 2"),  # subdomain 3
        ("-2 * sqrt(2.0) * sin(mu[0]) + x[0] * (-sqrt(2.0) * sin(mu[0]) + 2) + 2",
         "2 * sqrt(2.0) * cos(mu[0]) + x[0] * (sqrt(2.0) * cos(mu[0]) - 1) + x[1] - 2"),  # subdomain 4
        ("2 * sqrt(2.0) * sin(mu[0]) + x[0] * (-3 * sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2 + 2)"
         + "+ x[1] * (-sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2) - 2",
         "-2 * sqrt(2.0) * cos(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0]) / 2 + 3 * sqrt(2.0) * cos(mu[0]) / 2 - 2)"
         + "+ x[1] * (sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2) + 2"),  # subdomain 5
        ("2 * sqrt(2.0) * cos(mu[0]) + x[0] * (-sqrt(2.0) * cos(mu[0]) + 2) - 2",
         "2 * sqrt(2.0) * sin(mu[0]) + x[0] * (-sqrt(2.0) * sin(mu[0]) + 1) + x[1] - 2"),  # subdomain 6
        ("-2 * sqrt(2.0) * sin(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2)"
         + "+ x[1] * (3 * sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2 - 2) + 2",
         "2 * sqrt(2.0) * cos(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0]) / 2 - sqrt(2.0) * cos(mu[0]) / 2)"
         + "+ x[1] * (sqrt(2.0) * sin(mu[0]) / 2 - 3 * sqrt(2.0) * cos(mu[0]) / 2 + 2) - 2"),  # subdomain 7
        ("2 * sqrt(2.0) * cos(mu[0]) + x[0] + x[1] * (-sqrt(2.0) * cos(mu[0]) + 1) - 2",
         "2 * sqrt(2.0) * sin(mu[0]) + x[1] * (-sqrt(2.0) * sin(mu[0]) + 2) - 2")  # subdomain 8
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)

    # Define function space, test/trial functions, measures
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    dx = Measure("dx")(subdomain_data=subdomains)
    ds = Measure("ds")(subdomain_data=boundaries)

    # Define base problem
    class HoleRotation(ParametrizedProblem, metaclass=ABCMeta):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (pi / 4.0, 0.01)
            self.mu_range = [(pi / 4.0 - pi / 45.0, pi / 4.0 + pi / 45.0), (0.01, 1.0)]
            self.terms = ["a", "f"]
            self.operator = dict()
            self.Q = dict()

        def name(self):
            return "___".join([
                self.folder_prefix, shape_parametrization_preprocessing.__name__,
                AdditionalProblemDecorator.__name__])

        def init(self):
            self._init_operators()

        def _init_operators(self):
            pass

        @abstractmethod
        def compute_theta(self, term):
            pass

        @abstractmethod
        def assemble_operator(self, term):
            pass

    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class HoleRotationOnReferenceDomain(HoleRotation):
        def __init__(self, V, **kwargs):
            HoleRotation.__init__(self, "HoleRotationOnReferenceDomain")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            mu = self.mu
            if term == "a":
                theta_a0 = (-5 * sqrt(2.0)**2 + 16 * sqrt(2.0) * sin(mu[0]) + 8 * sqrt(2.0) * cos(mu[0]) - 16) / (
                    sqrt(2.0) * (sqrt(2.0) - 4 * cos(mu[0])))
                theta_a1 = -sqrt(2.0) / (sqrt(2.0) - 4 * cos(mu[0]))
                theta_a2 = (-2 * sqrt(2.0) + 4 * sin(mu[0])) / (sqrt(2.0) - 4 * cos(mu[0]))
                theta_a3 = (-sqrt(2.0)**2 + 2 * sqrt(2.0) * sin(mu[0]) + 4 * sqrt(2.0) * cos(mu[0]) - 5) / (
                    sqrt(2.0) * cos(mu[0]) - 2)
                theta_a4 = -1 / (sqrt(2.0) * cos(mu[0]) - 2)
                theta_a5 = (sqrt(2.0) * sin(mu[0]) - 1) / (sqrt(2.0) * cos(mu[0]) - 2)
                theta_a6 = -sqrt(2.0) / (sqrt(2.0) - 4 * sin(mu[0]))
                theta_a7 = (-5 * sqrt(2.0)**2 + 8 * sqrt(2.0) * sin(mu[0]) + 16 * sqrt(2.0) * cos(mu[0]) - 16) / (
                    sqrt(2.0) * (sqrt(2.0) - 4 * sin(mu[0])))
                theta_a8 = (-2 * sqrt(2.0) + 4 * cos(mu[0])) / (sqrt(2.0) - 4 * sin(mu[0]))
                theta_a9 = -1 / (sqrt(2.0) * sin(mu[0]) - 2)
                theta_a10 = (-sqrt(2.0)**2 + 4 * sqrt(2.0) * sin(mu[0]) + 2 * sqrt(2.0) * cos(mu[0]) - 5) / (
                    sqrt(2.0) * sin(mu[0]) - 2)
                theta_a11 = (sqrt(2.0) * cos(mu[0]) - 1) / (sqrt(2.0) * sin(mu[0]) - 2)
                theta_a12 = -sqrt(2.0) / (sqrt(2.0) - 4 * cos(mu[0]))
                theta_a13 = (-5 * sqrt(2.0)**2 + 16 * sqrt(2.0) * sin(mu[0]) + 8 * sqrt(2.0) * cos(mu[0]) - 16) / (
                    sqrt(2.0) * (sqrt(2.0) - 4 * cos(mu[0])))
                theta_a14 = 2 * (sqrt(2.0) - 2 * sin(mu[0])) / (sqrt(2.0) - 4 * cos(mu[0]))
                theta_a15 = - 1 / (sqrt(2.0) * cos(mu[0]) - 2)
                theta_a16 = (- sqrt(2.0)**2 + 2 * sqrt(2.0) * sin(mu[0]) + 4 * sqrt(2.0) * cos(mu[0]) - 5) / (
                    sqrt(2.0) * cos(mu[0]) - 2)
                theta_a17 = (- sqrt(2.0) * sin(mu[0]) + 1) / (sqrt(2.0) * cos(mu[0]) - 2)
                theta_a18 = (- 5 * sqrt(2.0)**2 + 8 * sqrt(2.0) * sin(mu[0]) + 16 * sqrt(2.0) * cos(mu[0]) - 16) / (
                    sqrt(2.0) * (sqrt(2.0) - 4 * sin(mu[0])))
                theta_a19 = - sqrt(2.0) / (sqrt(2.0) - 4 * sin(mu[0]))
                theta_a20 = 2 * (sqrt(2.0) - 2 * cos(mu[0])) / (sqrt(2.0) - 4 * sin(mu[0]))
                theta_a21 = (-sqrt(2.0)**2 + 4 * sqrt(2.0) * sin(mu[0]) + 2 * sqrt(2.0) * cos(mu[0]) - 5) / (
                    sqrt(2.0) * sin(mu[0]) - 2)
                theta_a22 = - 1 / (sqrt(2.0) * sin(mu[0]) - 2)
                theta_a23 = (- sqrt(2.0) * cos(mu[0]) + 1) / (sqrt(2.0) * sin(mu[0]) - 2)
                theta_a24 = mu[1]
                theta_a25 = mu[1]
                theta_a26 = mu[1]
                theta_a27 = mu[1]
                return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7, theta_a8,
                        theta_a9, theta_a10, theta_a11, theta_a12, theta_a13, theta_a14, theta_a15, theta_a16,
                        theta_a17, theta_a18, theta_a19, theta_a20, theta_a21, theta_a22, theta_a23, theta_a24,
                        theta_a25, theta_a26, theta_a27)
            elif term == "f":
                theta_f0 = 1.0
                theta_f1 = 1.0
                theta_f2 = 1.0
                theta_f3 = 1.0
                return (theta_f0, theta_f1, theta_f2, theta_f3)
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = u.dx(0) * v.dx(0) * dx(1)
                a1 = u.dx(1) * v.dx(1) * dx(1)
                a2 = u.dx(0) * v.dx(1) * dx(1) + u.dx(1) * v.dx(0) * dx(1)
                a3 = u.dx(0) * v.dx(0) * dx(2)
                a4 = u.dx(1) * v.dx(1) * dx(2)
                a5 = u.dx(0) * v.dx(1) * dx(2) + u.dx(1) * v.dx(0) * dx(2)
                a6 = u.dx(0) * v.dx(0) * dx(3)
                a7 = u.dx(1) * v.dx(1) * dx(3)
                a8 = u.dx(0) * v.dx(1) * dx(3) + u.dx(1) * v.dx(0) * dx(3)
                a9 = u.dx(0) * v.dx(0) * dx(4)
                a10 = u.dx(1) * v.dx(1) * dx(4)
                a11 = u.dx(0) * v.dx(1) * dx(4) + u.dx(1) * v.dx(0) * dx(4)
                a12 = u.dx(0) * v.dx(0) * dx(5)
                a13 = u.dx(1) * v.dx(1) * dx(5)
                a14 = u.dx(0) * v.dx(1) * dx(5) + u.dx(1) * v.dx(0) * dx(5)
                a15 = u.dx(0) * v.dx(0) * dx(6)
                a16 = u.dx(1) * v.dx(1) * dx(6)
                a17 = u.dx(0) * v.dx(1) * dx(6) + u.dx(1) * v.dx(0) * dx(6)
                a18 = u.dx(0) * v.dx(0) * dx(7)
                a19 = u.dx(1) * v.dx(1) * dx(7)
                a20 = u.dx(0) * v.dx(1) * dx(7) + u.dx(1) * v.dx(0) * dx(7)
                a21 = u.dx(0) * v.dx(0) * dx(8)
                a22 = u.dx(1) * v.dx(1) * dx(8)
                a23 = u.dx(0) * v.dx(1) * dx(8) + u.dx(1) * v.dx(0) * dx(8)
                a24 = u * v * ds(5)
                a25 = u * v * ds(6)
                a26 = u * v * ds(7)
                a27 = u * v * ds(8)
                return (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                        a20, a21, a22, a23, a24, a25, a26, a27)
            elif term == "f":
                f0 = v * ds(1)
                f1 = v * ds(2)
                f2 = v * ds(3)
                f3 = v * ds(4)
                return (f0, f1, f2, f3)
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain()
    @ShapeParametrization(*shape_parametrization_expression)
    class HoleRotationPullBack(HoleRotation):
        def __init__(self, V, **kwargs):
            HoleRotation.__init__(self, "HoleRotationPullBack")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            mu = self.mu
            if term == "a":
                theta_a0 = 1.0
                theta_a1 = mu[1]
                return (theta_a0, theta_a1)
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0,)
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v)) * dx
                a1 = u * v * ds(5) + u * v * ds(6) + u * v * ds(7) + u * v * ds(8)
                return (a0, a1)
            elif term == "f":
                f0 = v * ds(1) + v * ds(2) + v * ds(3) + v * ds(4)
                return (f0,)
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Check forms
    problem_on_reference_domain = HoleRotationOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = HoleRotationPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)

        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)

        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)


# Test forms pull back to reference domain for tutorial 04
@enable_pull_back_to_reference_domain_logging
@check_affine_and_non_affine_shape_parametrizations((
    "ExpressionOnDeformedDomainGenerator", [
        (lambda problem, cppcode, **kwargs: Expression(cppcode, element=kwargs["element"]), ),
        (lambda problem, cppcode, **kwargs: ParametrizedExpression(problem, cppcode, **kwargs), ),
        (lambda problem, cppcode, **kwargs: PushForwardToDeformedDomain(problem, Expression(
            cppcode, element=kwargs["element"])), ),
        (lambda problem, cppcode, **kwargs: PushForwardToDeformedDomain(problem, ParametrizedExpression(
            problem, cppcode, **kwargs)), )
    ], [  # is_affine:
        True,
        False,
        True,
        False
    ]
))
def test_pull_back_to_reference_domain_graetz(
        shape_parametrization_preprocessing, AdditionalProblemDecorator,
        ExceptionType, exception_message, ExpressionOnDeformedDomainGenerator):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "graetz.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "graetz_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "graetz_facet_region.xml"))

    # Define shape parametrization
    shape_parametrization_expression = [
        ("x[0]", "x[1]"),  # subdomain 1
        ("mu[0] * (x[0] - 1) + 1", "x[1]")  # subdomain 2
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)

    # Define function space, test/trial functions, measures, auxiliary expressions
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    dx = Measure("dx")(subdomain_data=subdomains)
    ff = Constant(1.)

    # Define base problem
    class Graetz(ParametrizedProblem, metaclass=ABCMeta):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1., 1.)
            self.mu_range = [(0.1, 10.0), (0.01, 10.0)]
            self.terms = ["a", "f"]
            self.operator = dict()
            self.Q = dict()

        def name(self):
            return "___".join([
                self.folder_prefix, shape_parametrization_preprocessing.__name__,
                AdditionalProblemDecorator.__name__])

        def init(self):
            self._init_operators()

        def _init_operators(self):
            pass

        @abstractmethod
        def compute_theta(self, term):
            pass

        @abstractmethod
        def assemble_operator(self, term):
            pass

    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class GraetzOnReferenceDomain(Graetz):
        def __init__(self, V, **kwargs):
            Graetz.__init__(self, "GraetzOnReferenceDomain")
            self.V = V
            self._solution = Function(V)
            self.vel = Expression("x[1] * (1 - x[1])", element=V.ufl_element())

        def compute_theta(self, term):
            mu = self.mu
            if term == "a":
                theta_a0 = mu[1]
                theta_a1 = mu[1] / mu[0]
                theta_a2 = mu[0] * mu[1]
                theta_a3 = 1.0
                return (theta_a0, theta_a1, theta_a2, theta_a3)
            elif term == "f":
                theta_f0 = 1.0
                theta_f1 = mu[0]
                return (theta_f0, theta_f1)
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v)) * dx(1)
                a1 = u.dx(0) * v.dx(0) * dx(2)
                a2 = u.dx(1) * v.dx(1) * dx(2)
                a3 = self.vel * u.dx(0) * v * dx(1) + self.vel * u.dx(0) * v * dx(2)
                return (a0, a1, a2, a3)
            elif term == "f":
                f0 = ff * v * dx(1)
                f1 = ff * v * dx(2)
                return (f0, f1)
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain()
    @ShapeParametrization(*shape_parametrization_expression)
    class GraetzPullBack(Graetz):
        def __init__(self, V, **kwargs):
            Graetz.__init__(self, "GraetzPullBack")
            self.V = V
            self._solution = Function(V)
            self.vel = ExpressionOnDeformedDomainGenerator(
                self, "x[1] * (1 - x[1])", mu=(1.0, 1.0), element=V.ufl_element())

        def compute_theta(self, term):
            mu = self.mu
            if term == "a":
                theta_a0 = mu[1]
                theta_a1 = 1.0
                return (theta_a0, theta_a1)
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0, )
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v)) * dx
                a1 = self.vel * u.dx(0) * v * dx
                return (a0, a1)
            elif term == "f":
                f0 = ff * v * dx
                return (f0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Check forms
    problem_on_reference_domain = GraetzOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = GraetzPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)

        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)

        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)


# Test forms pull back to reference domain for tutorial 07
@enable_pull_back_to_reference_domain_logging
@check_affine_and_non_affine_shape_parametrizations((
    "nonlinear_theta_on_reference_domain, nonlinear_theta_on_deformed_domain, nonlinear_operator, initial_guess", [
        (
            lambda mu: (mu[2] * mu[3], ),
            lambda mu: (1.0, ),
            lambda mu, u, v, dx: (u * v * dx, ),
            "1 + 2 * x[0] + 3 * x[1]"
        ),
        (
            lambda mu: (mu[2] * mu[3], ),
            lambda mu: (1.0, ),
            lambda mu, u, v, dx: (mu[0] * u * v * dx, ),
            "1 + 2 * x[0] + 3 * x[1]"
        ),
        (
            lambda mu: (mu[2] * mu[3], ),
            lambda mu: (1.0, ),
            lambda mu, u, v, dx: (u**2 * v * dx, ),
            "1 + 2 * x[0] + 3 * x[1]"
        ),
        (
            lambda mu: (mu[2] * mu[3], ),
            lambda mu: (1.0, ),
            lambda mu, u, v, dx: (mu[0] * u**(1 + mu[1]) * v * dx, ),
            "1 + 2 * x[0] + 3 * x[1]"
        ),
        (
            lambda mu: (mu[3] / mu[2], ),
            lambda mu: (1.0, ),
            lambda mu, u, v, dx: (u.dx(0)**2 * v * dx, ),
            "x[0] + pow(x[0], 2) + 3 * x[0] * x[1]"
        ),
        (
            lambda mu: (mu[2]**(- mu[1]) * mu[3], ),
            lambda mu: (1.0, ),
            lambda mu, u, v, dx: (mu[0] * u.dx(0)**(1 + mu[1]) * v * dx, ),
            "x[0] + pow(x[0], 2) + 3 * x[0] * x[1]"
        ),
        (
            lambda mu: (mu[2]**(- mu[0]) * mu[3], mu[2] * mu[3]**(- mu[1])),
            lambda mu: (1.0, 1.0),
            lambda mu, u, v, dx: (u.dx(0)**(1 + mu[0]) * v * dx, u.dx(1)**(1 + mu[1]) * v * dx, ),
            "x[0] + x[1] + pow(x[0], 2) + 3 * x[0] * x[1]"
        ),
        (
            lambda mu: (mu[2] * mu[3], ),
            lambda mu: (1.0, ),
            lambda mu, u, v, dx: (mu[0] * ((exp(mu[1] * u) - 1) / mu[1]) * v * dx, ),
            "0.1 * (2 + sin(2 * pi * x[0] + 6 * pi * x[1]))"
        )
    ]
), is_affine=False)
def test_pull_back_to_reference_domain_nonlinear_elliptic(
        shape_parametrization_preprocessing, AdditionalProblemDecorator,
        ExceptionType, exception_message,
        nonlinear_theta_on_reference_domain, nonlinear_theta_on_deformed_domain, nonlinear_operator, initial_guess):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "square.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "square_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "square_facet_region.xml"))

    # Reset all subdomains to have id 1, as the original tutorial does not account for shape parametrization
    subdomains.set_all(1)

    # Define shape parametrization
    shape_parametrization_expression = [
        ("mu[2] * x[0]", "mu[3] * x[1]")  # subdomain 1
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)

    # Define function space, test/trial functions, measures, auxiliary expressions
    V = FunctionSpace(mesh, "Lagrange", 1)
    du = TrialFunction(V)
    u = project(Expression(initial_guess, element=V.ufl_element()), V)
    v = TestFunction(V)
    dx = Measure("dx")(subdomain_data=subdomains)
    ff = Constant(1)

    # Define base problem
    class NonlinearElliptic(ParametrizedProblem, metaclass=ABCMeta):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1.0, 1.0, 1.0, 1.0)
            self.mu_range = [(0.01, 10.0), (0.01, 10.0), (0.5, 2.0), (0.5, 2.0)]
            self.terms = ["a", "c", "dc", "f"]
            self.operator = dict()
            self.Q = dict()

        def name(self):
            return "___".join([
                self.folder_prefix, shape_parametrization_preprocessing.__name__,
                AdditionalProblemDecorator.__name__])

        def init(self):
            self._init_operators()

        def _init_operators(self):
            pass

        @abstractmethod
        def compute_theta(self, term):
            pass

        @abstractmethod
        def assemble_operator(self, term):
            pass

    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class NonlinearEllipticOnReferenceDomain(NonlinearElliptic):
        def __init__(self, V, **kwargs):
            NonlinearElliptic.__init__(self, "NonlinearEllipticOnReferenceDomain")
            self.V = V
            self._solution = u

        @compute_theta_for_derivative({"dc": "c"})
        def compute_theta(self, term):
            mu = self.mu
            if term == "a":
                theta_a0 = mu[3] / mu[2]
                theta_a1 = mu[2] / mu[3]
                return (theta_a0, theta_a1)
            elif term == "c":
                return nonlinear_theta_on_reference_domain(mu)
            elif term == "f":
                theta_f0 = mu[2] * mu[3]
                return (theta_f0, )
            else:
                raise ValueError("Invalid term for compute_theta().")

        @assemble_operator_for_derivative({"dc": "c"})
        def assemble_operator(self, term):
            if term == "a":
                a0 = du.dx(0) * v.dx(0) * dx
                a1 = du.dx(1) * v.dx(1) * dx
                return (a0, a1)
            elif term == "c":
                return nonlinear_operator(self.mu, u, v, dx)
            elif term == "f":
                f0 = ff * v * dx
                return (f0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain()
    @ShapeParametrization(*shape_parametrization_expression)
    @StoreMapFromSolutionToProblem
    class NonlinearEllipticPullBack(NonlinearElliptic):
        def __init__(self, V, **kwargs):
            NonlinearElliptic.__init__(self, "NonlinearEllipticPullBack")
            self.V = V
            self._solution = u

        @compute_theta_for_derivative({"dc": "c"})
        def compute_theta(self, term):
            if term == "a":
                theta_a0 = 1.0
                return (theta_a0, )
            elif term == "c":
                return nonlinear_theta_on_deformed_domain(self.mu)
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0, )
            else:
                raise ValueError("Invalid term for compute_theta().")

        @assemble_operator_for_derivative({"dc": "c"})
        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(du), grad(v)) * dx
                return (a0, )
            elif term == "c":
                return nonlinear_operator(self.mu, u, v, dx)
            elif term == "f":
                f0 = ff * v * dx
                return (f0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Check forms
    problem_on_reference_domain = NonlinearEllipticOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = NonlinearEllipticPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)

        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)

        c_on_reference_domain = theta_times_operator(problem_on_reference_domain, "c")
        c_pull_back = theta_times_operator(problem_pull_back, "c")
        assert forms_are_close(c_on_reference_domain, c_pull_back)

        dc_on_reference_domain = theta_times_operator(problem_on_reference_domain, "dc")
        dc_pull_back = theta_times_operator(problem_pull_back, "dc")
        assert forms_are_close(dc_on_reference_domain, dc_pull_back)

        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)


# Test forms pull back to reference domain for tutorial 09
@enable_pull_back_to_reference_domain_logging
@check_affine_and_non_affine_shape_parametrizations((
    "CellDiameter, cell_diameter_pull_back", [
        (lambda mesh: Constant(0.), lambda mu: 0),
        (lambda mesh: Constant(1.), lambda mu: 1),
        (CellDiameter, lambda mu: sqrt(mu))
    ]
))
def test_pull_back_to_reference_domain_advection_dominated(
        shape_parametrization_preprocessing, AdditionalProblemDecorator,
        ExceptionType, exception_message, CellDiameter, cell_diameter_pull_back):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "graetz.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "graetz_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "graetz_facet_region.xml"))

    # Define shape parametrization
    shape_parametrization_expression = [
        ("x[0]", "x[1]"),  # subdomain 1
        ("mu[0] * (x[0] - 1) + 1", "x[1]")  # subdomain 2
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)

    # Define function space, test/trial functions, measures, auxiliary expressions
    V = FunctionSpace(mesh, "Lagrange", 2)
    u = TrialFunction(V)
    v = TestFunction(V)
    dx = Measure("dx")(subdomain_data=subdomains)
    vel = Expression("x[1] * (1 - x[1])", element=V.ufl_element())
    ff = Constant(1.)
    h = CellDiameter(V.mesh())

    # Define base problem
    class AdvectionDominated(ParametrizedProblem, metaclass=ABCMeta):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1., 1.)
            self.mu_range = [(0.5, 4.0), (1e-6, 1e-1)]
            self.terms = ["a", "f"]
            self.operator = dict()
            self.Q = dict()

        def name(self):
            return "___".join([
                self.folder_prefix, shape_parametrization_preprocessing.__name__,
                AdditionalProblemDecorator.__name__, str(cell_diameter_pull_back(4.))])

        def init(self):
            self._init_operators()

        def _init_operators(self):
            pass

        @abstractmethod
        def compute_theta(self, term):
            pass

        @abstractmethod
        def assemble_operator(self, term):
            pass

    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class AdvectionDominatedOnReferenceDomain(AdvectionDominated):
        def __init__(self, V, **kwargs):
            AdvectionDominated.__init__(self, "AdvectionDominatedOnReferenceDomain")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            mu = self.mu
            if term == "a":
                theta_a0 = mu[1]
                theta_a1 = mu[1] / mu[0]
                theta_a2 = mu[1] * mu[0]
                theta_a3 = 1.0
                theta_a4 = mu[1]
                theta_a5 = mu[1] / mu[0]**2 * cell_diameter_pull_back(mu[0])
                theta_a6 = mu[1] * cell_diameter_pull_back(mu[0])
                theta_a7 = 1.0
                theta_a8 = 1.0 / mu[0] * cell_diameter_pull_back(mu[0])
                return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7, theta_a8)
            elif term == "f":
                theta_f0 = 1.0
                theta_f1 = mu[0]
                theta_f2 = 1.0
                theta_f3 = cell_diameter_pull_back(mu[0])
                return (theta_f0, theta_f1, theta_f2, theta_f3)
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v)) * dx(1)
                a1 = u.dx(0) * v.dx(0) * dx(2)
                a2 = u.dx(1) * v.dx(1) * dx(2)
                a3 = vel * u.dx(0) * v * dx(1) + vel * u.dx(0) * v * dx(2)
                a4 = - h * inner(div(grad(u)), vel * v.dx(0)) * dx(1)
                a5 = - h * u.dx(0).dx(0) * vel * v.dx(0) * dx(2)
                a6 = - h * u.dx(1).dx(1) * vel * v.dx(0) * dx(2)
                a7 = h * vel * u.dx(0) * vel * v.dx(0) * dx(1)
                a8 = h * vel * u.dx(0) * vel * v.dx(0) * dx(2)
                return (a0, a1, a2, a3, a4, a5, a6, a7, a8)
            elif term == "f":
                f0 = ff * v * dx(1)
                f1 = ff * v * dx(2)
                f2 = h * ff * vel * v.dx(0) * dx(1)
                f3 = h * ff * vel * v.dx(0) * dx(2)
                return (f0, f1, f2, f3)
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain()
    @ShapeParametrization(*shape_parametrization_expression)
    class AdvectionDominatedPullBack(AdvectionDominated):
        def __init__(self, V, **kwargs):
            AdvectionDominated.__init__(self, "AdvectionDominatedPullBack")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            mu = self.mu
            if term == "a":
                theta_a0 = mu[1]
                theta_a1 = 1.0
                return (theta_a0, theta_a1)
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0, )
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v)) * dx - inner(div(grad(u)), h * vel * v.dx(0)) * dx
                a1 = vel * u.dx(0) * v * dx + inner(vel * u.dx(0), h * vel * v.dx(0)) * dx
                return (a0, a1)
            elif term == "f":
                f0 = ff * v * dx + ff * h * vel * v.dx(0) * dx
                return (f0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Check forms
    problem_on_reference_domain = AdvectionDominatedOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = AdvectionDominatedPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)

        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)

        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)


# Test forms pull back to reference domain for tutorial 12
@enable_pull_back_to_reference_domain_logging
@check_affine_and_non_affine_shape_parametrizations()
def test_pull_back_to_reference_domain_stokes(
        shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "t_bypass.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "t_bypass_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "t_bypass_facet_region.xml"))

    # Define shape parametrization
    shape_parametrization_expression = [
        ("mu[4] * x[0] + mu[1] - mu[4]",
         "mu[4] * tan(mu[5]) * x[0] + mu[0] * x[1] + mu[2] - mu[4] * tan(mu[5]) - mu[0]"),  # subdomain 1
        ("mu[4] * x[0] + mu[1] - mu[4]",
         "mu[4] * tan(mu[5]) * x[0] + mu[0] * x[1] + mu[2] - mu[4] * tan(mu[5]) - mu[0]"),  # subdomain 2
        ("mu[1] * x[0]", "mu[3] * x[1] + mu[2] + mu[0] - 2 * mu[3]"),  # subdomain 3
        ("mu[1] * x[0]", "mu[3] * x[1] + mu[2] + mu[0] - 2 * mu[3]"),  # subdomain 4
        ("mu[1] * x[0]", "mu[0] * x[1] + mu[2] - mu[0]"),  # subdomain 5
        ("mu[1] * x[0]", "mu[0] * x[1] + mu[2] - mu[0]"),  # subdomain 6
        ("mu[1] * x[0]", "mu[2] * x[1]"),  # subdomain 7
        ("mu[1] * x[0]", "mu[2] * x[1]"),  # subdomain 8
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)

    # Define function space, test/trial functions, measures, auxiliary expressions
    element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_u, element_p)
    V = FunctionSpace(mesh, element, components=[["u", "s"], "p"])
    up = TrialFunction(V)
    (u, p) = split(up)
    vq = TestFunction(V)
    (v, q) = split(vq)
    dx = Measure("dx")(subdomain_data=subdomains)

    ff = Constant((1.0, -10.0))
    gg = Constant(2.0)
    nu = 1.0

    # Define base problem
    class Stokes(ParametrizedProblem, metaclass=ABCMeta):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1., 1., 1., 1., 1., 0.)
            self.mu_range = [(0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.0, pi / 6.0)]
            self.terms = ["a", "b", "bt", "f", "g"]
            self.operator = dict()
            self.Q = dict()

        def name(self):
            return "___".join([
                self.folder_prefix, shape_parametrization_preprocessing.__name__,
                AdditionalProblemDecorator.__name__])

        def init(self):
            self._init_operators()

        def _init_operators(self):
            pass

        @abstractmethod
        def compute_theta(self, term):
            pass

        @abstractmethod
        def assemble_operator(self, term):
            pass

    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class StokesOnReferenceDomain(Stokes):
        def __init__(self, V, **kwargs):
            Stokes.__init__(self, "StokesOnReferenceDomain")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            mu = self.mu
            if term == "a":
                theta_a0 = nu * (mu[0] / mu[4])
                theta_a1 = nu * (- tan(mu[5]))
                theta_a2 = nu * (mu[4] * (tan(mu[5])**2 + 1) / mu[0])
                theta_a3 = nu * (mu[3] / mu[1])
                theta_a4 = nu * (mu[1] / mu[3])
                theta_a5 = nu * (mu[0] / mu[1])
                theta_a6 = nu * (mu[1] / mu[0])
                theta_a7 = nu * (mu[2] / mu[1])
                theta_a8 = nu * (mu[1] / mu[2])
                return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7, theta_a8)
            elif term in ("b", "bt"):
                theta_b0 = mu[0]
                theta_b1 = - tan(mu[5]) * mu[4]
                theta_b2 = mu[4]
                theta_b3 = mu[3]
                theta_b4 = mu[1]
                theta_b5 = mu[0]
                theta_b6 = mu[1]
                theta_b7 = mu[2]
                theta_b8 = mu[1]
                return (theta_b0, theta_b1, theta_b2, theta_b3, theta_b4, theta_b5, theta_b6, theta_b7, theta_b8)
            elif term == "f":
                theta_f0 = mu[0] * mu[4]
                theta_f1 = mu[1] * mu[3]
                theta_f2 = mu[0] * mu[1]
                theta_f3 = mu[1] * mu[2]
                return (theta_f0, theta_f1, theta_f2, theta_f3)
            elif term == "g":
                theta_g0 = mu[0] * mu[4]
                theta_g1 = mu[1] * mu[3]
                theta_g2 = mu[0] * mu[1]
                theta_g3 = mu[1] * mu[2]
                return (theta_g0, theta_g1, theta_g2, theta_g3)
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = (u[0].dx(0) * v[0].dx(0) + u[1].dx(0) * v[1].dx(0)) * (dx(1) + dx(2))
                a1 = (u[0].dx(0) * v[0].dx(1) + u[0].dx(1) * v[0].dx(0)
                      + u[1].dx(0) * v[1].dx(1) + u[1].dx(1) * v[1].dx(0)) * (dx(1) + dx(2))
                a2 = (u[0].dx(1) * v[0].dx(1) + u[1].dx(1) * v[1].dx(1)) * (dx(1) + dx(2))
                a3 = (u[0].dx(0) * v[0].dx(0) + u[1].dx(0) * v[1].dx(0)) * (dx(3) + dx(4))
                a4 = (u[0].dx(1) * v[0].dx(1) + u[1].dx(1) * v[1].dx(1)) * (dx(3) + dx(4))
                a5 = (u[0].dx(0) * v[0].dx(0) + u[1].dx(0) * v[1].dx(0)) * (dx(5) + dx(6))
                a6 = (u[0].dx(1) * v[0].dx(1) + u[1].dx(1) * v[1].dx(1)) * (dx(5) + dx(6))
                a7 = (u[0].dx(0) * v[0].dx(0) + u[1].dx(0) * v[1].dx(0)) * (dx(7) + dx(8))
                a8 = (u[0].dx(1) * v[0].dx(1) + u[1].dx(1) * v[1].dx(1)) * (dx(7) + dx(8))
                return (a0, a1, a2, a3, a4, a5, a6, a7, a8)
            elif term == "b":
                b0 = - q * u[0].dx(0) * (dx(1) + dx(2))
                b1 = - q * u[0].dx(1) * (dx(1) + dx(2))
                b2 = - q * u[1].dx(1) * (dx(1) + dx(2))
                b3 = - q * u[0].dx(0) * (dx(3) + dx(4))
                b4 = - q * u[1].dx(1) * (dx(3) + dx(4))
                b5 = - q * u[0].dx(0) * (dx(5) + dx(6))
                b6 = - q * u[1].dx(1) * (dx(5) + dx(6))
                b7 = - q * u[0].dx(0) * (dx(7) + dx(8))
                b8 = - q * u[1].dx(1) * (dx(7) + dx(8))
                return (b0, b1, b2, b3, b4, b5, b6, b7, b8)
            elif term == "bt":
                bt0 = - p * v[0].dx(0) * (dx(1) + dx(2))
                bt1 = - p * v[0].dx(1) * (dx(1) + dx(2))
                bt2 = - p * v[1].dx(1) * (dx(1) + dx(2))
                bt3 = - p * v[0].dx(0) * (dx(3) + dx(4))
                bt4 = - p * v[1].dx(1) * (dx(3) + dx(4))
                bt5 = - p * v[0].dx(0) * (dx(5) + dx(6))
                bt6 = - p * v[1].dx(1) * (dx(5) + dx(6))
                bt7 = - p * v[0].dx(0) * (dx(7) + dx(8))
                bt8 = - p * v[1].dx(1) * (dx(7) + dx(8))
                return (bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8)
            elif term == "f":
                f0 = inner(ff, v) * (dx(1) + dx(2))
                f1 = inner(ff, v) * (dx(3) + dx(4))
                f2 = inner(ff, v) * (dx(5) + dx(6))
                f3 = inner(ff, v) * (dx(7) + dx(8))
                return (f0, f1, f2, f3)
            elif term == "g":
                g0 = gg * q * (dx(1) + dx(2))
                g1 = gg * q * (dx(3) + dx(4))
                g2 = gg * q * (dx(5) + dx(6))
                g3 = gg * q * (dx(7) + dx(8))
                return (g0, g1, g2, g3)
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain()
    @ShapeParametrization(*shape_parametrization_expression)
    class StokesPullBack(Stokes):
        def __init__(self, V, **kwargs):
            Stokes.__init__(self, "StokesPullBack")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            if term == "a":
                theta_a0 = nu * 1.0
                return (theta_a0, )
            elif term in ("b", "bt"):
                theta_b0 = 1.0
                return (theta_b0, )
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0, )
            elif term == "g":
                theta_g0 = 1.0
                return (theta_g0, )
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v)) * dx
                return (a0, )
            elif term == "b":
                b0 = - q * div(u) * dx
                return (b0, )
            elif term == "bt":
                bt0 = - p * div(v) * dx
                return (bt0, )
            elif term == "f":
                f0 = inner(ff, v) * dx
                return (f0, )
            elif term == "g":
                g0 = q * gg * dx
                return (g0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Check forms
    problem_on_reference_domain = StokesOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = StokesPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)

        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)

        b_on_reference_domain = theta_times_operator(problem_on_reference_domain, "b")
        b_pull_back = theta_times_operator(problem_pull_back, "b")
        assert forms_are_close(b_on_reference_domain, b_pull_back)

        bt_on_reference_domain = theta_times_operator(problem_on_reference_domain, "bt")
        bt_pull_back = theta_times_operator(problem_pull_back, "bt")
        assert forms_are_close(bt_on_reference_domain, bt_pull_back)

        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)

        g_on_reference_domain = theta_times_operator(problem_on_reference_domain, "g")
        g_pull_back = theta_times_operator(problem_pull_back, "g")
        assert forms_are_close(g_on_reference_domain, g_pull_back)


# Test forms pull back to reference domain for stabilization of Stokes problem
@enable_pull_back_to_reference_domain_logging
@check_affine_and_non_affine_shape_parametrizations((
    "CellDiameter, cell_diameter_pull_back", [
        (lambda mesh: Constant(0.), lambda mu: 0),
        (lambda mesh: Constant(1.), lambda mu: 1),
        (CellDiameter, lambda mu: sqrt(mu))
    ]
))
def test_pull_back_to_reference_domain_stokes_stabilization(
        shape_parametrization_preprocessing, AdditionalProblemDecorator,
        ExceptionType, exception_message, CellDiameter, cell_diameter_pull_back):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "cavity.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "cavity_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "cavity_facet_region.xml"))

    # Define shape parametrization
    shape_parametrization_expression = [
        ("mu[0] * x[0]", "x[1]"),  # subdomain 1
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)

    # Define function space, test/trial functions, measures, auxiliary expressions
    element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_u, element_p)
    V = FunctionSpace(mesh, element, components=[["u", "s"], "p"])
    up = TrialFunction(V)
    (u, p) = split(up)
    vq = TestFunction(V)
    (v, q) = split(vq)
    dx = Measure("dx")(subdomain_data=subdomains)

    ff = Constant((2., 3.))
    gg = Constant(4.)
    h = CellDiameter(mesh)
    alpha_p = Constant(1.)

    # Define base problem
    class StokesStabilization(ParametrizedProblem, metaclass=ABCMeta):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1.,)
            self.mu_range = [(0.5, 3)]
            self.terms = ["a", "b", "bt", "stab", "f", "g"]
            self.operator = dict()
            self.Q = dict()

        def name(self):
            return "___".join([
                self.folder_prefix, shape_parametrization_preprocessing.__name__,
                AdditionalProblemDecorator.__name__, str(cell_diameter_pull_back(4.))])

        def init(self):
            self._init_operators()

        def _init_operators(self):
            pass

        @abstractmethod
        def compute_theta(self, term):
            pass

        @abstractmethod
        def assemble_operator(self, term):
            pass

    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class StokesStabilizationOnReferenceDomain(StokesStabilization):
        def __init__(self, V, **kwargs):
            StokesStabilization.__init__(self, "StokesStabilizationOnReferenceDomain")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            mu = self.mu
            if term == "a":
                theta_a0 = 1. / mu[0]
                theta_a1 = mu[0]
                return (theta_a0, theta_a1)
            elif term in ("b", "bt"):
                theta_b0 = 1.
                theta_b1 = mu[0]
                return (theta_b0, theta_b1)
            elif term == "stab":
                theta_s0 = 1. / mu[0] * cell_diameter_pull_back(mu[0])**2
                theta_s1 = mu[0] * cell_diameter_pull_back(mu[0])**2
                return (theta_s0, theta_s1)
            elif term == "f":
                theta_f0 = mu[0]
                return (theta_f0, )
            elif term == "g":
                theta_g0 = mu[0]
                return (theta_g0, )
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = (u[0].dx(0) * v[0].dx(0) + u[1].dx(0) * v[1].dx(0)) * dx
                a1 = (u[0].dx(1) * v[0].dx(1) + u[1].dx(1) * v[1].dx(1)) * dx
                return (a0, a1)
            elif term == "b":
                b0 = - q * u[0].dx(0) * dx
                b1 = - q * u[1].dx(1) * dx
                return (b0, b1)
            elif term == "bt":
                bt0 = - p * v[0].dx(0) * dx
                bt1 = - p * v[1].dx(1) * dx
                return (bt0, bt1)
            elif term == "stab":
                s0 = p.dx(0) * alpha_p * h**2 * q.dx(0) * dx
                s1 = p.dx(1) * alpha_p * h**2 * q.dx(1) * dx
                return (s0, s1)
            elif term == "f":
                f0 = inner(ff, v) * dx
                return (f0, )
            elif term == "g":
                g0 = gg * q * dx
                return (g0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain()
    @ShapeParametrization(*shape_parametrization_expression)
    class StokesStabilizationPullBack(StokesStabilization):
        def __init__(self, V, **kwargs):
            StokesStabilization.__init__(self, "StokesStabilizationPullBack")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            if term == "a":
                theta_a0 = 1.0
                return (theta_a0, )
            elif term in ("b", "bt"):
                theta_b0 = 1.0
                return (theta_b0, )
            elif term == "stab":
                theta_s0 = 1.0
                return (theta_s0, )
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0, )
            elif term == "g":
                theta_g0 = 1.0
                return (theta_g0, )
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v)) * dx
                return (a0, )
            elif term == "b":
                b0 = - q * div(u) * dx
                return (b0, )
            elif term == "bt":
                bt0 = - p * div(v) * dx
                return (bt0, )
            elif term == "stab":
                s0 = inner(grad(p), alpha_p * h**2 * grad(q)) * dx
                return (s0, )
            elif term == "f":
                f0 = inner(ff, v) * dx
                return (f0, )
            elif term == "g":
                g0 = q * gg * dx
                return (g0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Check forms
    problem_on_reference_domain = StokesStabilizationOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = StokesStabilizationPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)

        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)

        b_on_reference_domain = theta_times_operator(problem_on_reference_domain, "b")
        b_pull_back = theta_times_operator(problem_pull_back, "b")
        assert forms_are_close(b_on_reference_domain, b_pull_back)

        bt_on_reference_domain = theta_times_operator(problem_on_reference_domain, "bt")
        bt_pull_back = theta_times_operator(problem_pull_back, "bt")
        assert forms_are_close(bt_on_reference_domain, bt_pull_back)

        stab_on_reference_domain = theta_times_operator(problem_on_reference_domain, "stab")
        stab_pull_back = theta_times_operator(problem_pull_back, "stab")
        assert forms_are_close(stab_on_reference_domain, stab_pull_back)

        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)

        g_on_reference_domain = theta_times_operator(problem_on_reference_domain, "g")
        g_pull_back = theta_times_operator(problem_pull_back, "g")
        assert forms_are_close(g_on_reference_domain, g_pull_back)


# Test forms pull back to reference domain for tutorial 13
@enable_pull_back_to_reference_domain_logging
@check_affine_and_non_affine_shape_parametrizations()
def test_pull_back_to_reference_domain_elliptic_optimal_control_1(
        shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "elliptic_optimal_control_1.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "elliptic_optimal_control_1_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "elliptic_optimal_control_1_facet_region.xml"))

    # Define shape parametrization
    shape_parametrization_expression = [
        ("x[0]", "x[1]"),  # subdomain 1
        ("mu[0] * (x[0] - 1) + 1", "x[1]"),  # subdomain 2
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)

    # Define function space, test/trial functions, measures, auxiliary expressions
    scalar_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(scalar_element, scalar_element, scalar_element)
    V = FunctionSpace(mesh, element, components=["y", "u", "p"])
    yup = TrialFunction(V)
    (y, u, p) = split(yup)
    zvq = TestFunction(V)
    (z, v, q) = split(zvq)
    dx = Measure("dx")(subdomain_data=subdomains)
    alpha = Constant(0.01)
    y_d = Constant(1.0)
    ff = Constant(2.0)

    # Define base problem
    class EllipticOptimalControl(ParametrizedProblem, metaclass=ABCMeta):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1., 1.)
            self.mu_range = [(1.0, 3.5), (0.5, 2.5)]
            self.terms = ["a", "a*", "c", "c*", "m", "n", "f", "g", "h"]
            self.operator = dict()
            self.Q = dict()

        def name(self):
            return "___".join([
                self.folder_prefix, shape_parametrization_preprocessing.__name__,
                AdditionalProblemDecorator.__name__])

        def init(self):
            self._init_operators()

        def _init_operators(self):
            pass

        @abstractmethod
        def compute_theta(self, term):
            pass

        @abstractmethod
        def assemble_operator(self, term):
            pass

    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class EllipticOptimalControlOnReferenceDomain(EllipticOptimalControl):
        def __init__(self, V, **kwargs):
            EllipticOptimalControl.__init__(self, "EllipticOptimalControlOnReferenceDomain")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            mu = self.mu
            if term in ("a", "a*"):
                theta_a0 = 1.0
                theta_a1 = 1.0 / mu[0]
                theta_a2 = mu[0]
                return (theta_a0, theta_a1, theta_a2)
            elif term in ("c", "c*"):
                theta_c0 = 1.0
                theta_c1 = mu[0]
                return (theta_c0, theta_c1)
            elif term == "m":
                theta_m0 = 1.0
                theta_m1 = mu[0]
                return (theta_m0, theta_m1)
            elif term == "n":
                theta_n0 = alpha
                theta_n1 = alpha * mu[0]
                return (theta_n0, theta_n1)
            elif term == "f":
                theta_f0 = 1.0
                theta_f1 = mu[0]
                return (theta_f0, theta_f1)
            elif term == "g":
                theta_g0 = 1.0
                theta_g1 = mu[0] * mu[1]
                return (theta_g0, theta_g1)
            elif term == "h":
                theta_h0 = 1.0 + mu[0] * mu[1]**2
                return (theta_h0,)
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(y), grad(q)) * dx(1)
                a1 = y.dx(0) * q.dx(0) * dx(2)
                a2 = y.dx(1) * q.dx(1) * dx(2)
                return (a0, a1, a2)
            elif term == "a*":
                as0 = inner(grad(z), grad(p)) * dx(1)
                as1 = z.dx(0) * p.dx(0) * dx(2)
                as2 = z.dx(1) * p.dx(1) * dx(2)
                return (as0, as1, as2)
            elif term == "c":
                c0 = u * q * dx(1)
                c1 = u * q * dx(2)
                return (c0, c1)
            elif term == "c*":
                cs0 = v * p * dx(1)
                cs1 = v * p * dx(2)
                return (cs0, cs1)
            elif term == "m":
                m0 = y * z * dx(1)
                m1 = y * z * dx(2)
                return (m0, m1)
            elif term == "n":
                n0 = u * v * dx(1)
                n1 = u * v * dx(2)
                return (n0, n1)
            elif term == "f":
                f0 = ff * q * dx(1)
                f1 = ff * q * dx(2)
                return (f0, f1)
            elif term == "g":
                g0 = y_d * z * dx(1)
                g1 = y_d * z * dx(2)
                return (g0, g1)
            elif term == "h":
                h0 = 1.0
                return (h0,)
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain()
    @ShapeParametrization(*shape_parametrization_expression)
    class EllipticOptimalControlPullBack(EllipticOptimalControl):
        def __init__(self, V, **kwargs):
            EllipticOptimalControl.__init__(self, "EllipticOptimalControlPullBack")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            mu = self.mu
            if term in ("a", "a*"):
                theta_a0 = 1.0
                return (theta_a0,)
            elif term in ("c", "c*"):
                theta_c0 = 1.0
                return (theta_c0,)
            elif term == "m":
                theta_m0 = 1.0
                return (theta_m0,)
            elif term == "n":
                theta_n0 = alpha
                return (theta_n0,)
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0,)
            elif term == "g":
                theta_g0 = 1.0
                theta_g1 = mu[1]
                return (theta_g0, theta_g1)
            elif term == "h":
                theta_h0 = 1.0
                theta_h1 = mu[1]**2
                return (theta_h0, theta_h1)
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(y), grad(q)) * dx
                return (a0,)
            elif term == "a*":
                as0 = inner(grad(z), grad(p)) * dx
                return (as0,)
            elif term == "c":
                c0 = u * q * dx
                return (c0,)
            elif term == "c*":
                cs0 = v * p * dx
                return (cs0,)
            elif term == "m":
                m0 = y * z * dx
                return (m0,)
            elif term == "n":
                n0 = u * v * dx
                return (n0,)
            elif term == "f":
                f0 = ff * q * dx
                return (f0,)
            elif term == "g":
                g0 = y_d * z * dx(1)
                g1 = y_d * z * dx(2)
                return (g0, g1)
            elif term == "h":
                h0 = y_d * y_d * dx(1, domain=mesh)
                h1 = y_d * y_d * dx(2, domain=mesh)
                return (h0, h1)
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Check forms
    problem_on_reference_domain = EllipticOptimalControlOnReferenceDomain(
        V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = EllipticOptimalControlPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)

        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)

        as_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a*")
        as_pull_back = theta_times_operator(problem_pull_back, "a*")
        assert forms_are_close(as_on_reference_domain, as_pull_back)

        c_on_reference_domain = theta_times_operator(problem_on_reference_domain, "c")
        c_pull_back = theta_times_operator(problem_pull_back, "c")
        assert forms_are_close(c_on_reference_domain, c_pull_back)

        cs_on_reference_domain = theta_times_operator(problem_on_reference_domain, "c*")
        cs_pull_back = theta_times_operator(problem_pull_back, "c*")
        assert forms_are_close(cs_on_reference_domain, cs_pull_back)

        m_on_reference_domain = theta_times_operator(problem_on_reference_domain, "m")
        m_pull_back = theta_times_operator(problem_pull_back, "m")
        assert forms_are_close(m_on_reference_domain, m_pull_back)

        n_on_reference_domain = theta_times_operator(problem_on_reference_domain, "n")
        n_pull_back = theta_times_operator(problem_pull_back, "n")
        assert forms_are_close(n_on_reference_domain, n_pull_back)

        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)

        g_on_reference_domain = theta_times_operator(problem_on_reference_domain, "g")
        g_pull_back = theta_times_operator(problem_pull_back, "g")
        assert forms_are_close(g_on_reference_domain, g_pull_back)

        h_on_reference_domain = theta_times_operator(problem_on_reference_domain, "h")
        h_pull_back = theta_times_operator(problem_pull_back, "h")
        assert forms_are_close(h_on_reference_domain, h_pull_back)


# Test forms pull back to reference domain for tutorial 14
@enable_pull_back_to_reference_domain_logging
@check_affine_and_non_affine_shape_parametrizations()
def test_pull_back_to_reference_domain_stokes_optimal_control_1(
        shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "stokes_optimal_control_1.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "stokes_optimal_control_1_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "stokes_optimal_control_1_facet_region.xml"))

    # Define shape parametrization
    shape_parametrization_expression = [
        ("x[0]", "mu[0] * x[1]")  # subdomain 1
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)

    # Define function space, test/trial functions, measures, auxiliary expressions
    velocity_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    pressure_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(velocity_element, pressure_element, velocity_element, velocity_element, pressure_element)
    V = FunctionSpace(mesh, element, components=[["v", "s"], "p", "u", ["w", "r"], "q"])
    trial = TrialFunction(V)
    (v, p, u, w, q) = split(trial)
    test = TestFunction(V)
    (psi, pi, tau, phi, xi) = split(test)
    dx = Measure("dx")(subdomain_data=subdomains)
    alpha = 0.008
    nu = 0.1
    vx_d = Expression("x[1]", degree=1)
    ll = Constant(1.0)

    # Define base problem
    class StokesOptimalControl(ParametrizedProblem, metaclass=ABCMeta):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1.0, 1.0)
            self.mu_range = [(0.5, 2.0), (0.5, 1.5)]
            self.terms = ["a", "a*", "b", "b*", "bt", "bt*", "c", "c*", "m", "n", "f", "g", "h", "l"]
            self.operator = dict()
            self.Q = dict()

        def name(self):
            return "___".join([
                self.folder_prefix, shape_parametrization_preprocessing.__name__,
                AdditionalProblemDecorator.__name__])

        def init(self):
            self._init_operators()

        def _init_operators(self):
            pass

        @abstractmethod
        def compute_theta(self, term):
            pass

        @abstractmethod
        def assemble_operator(self, term):
            pass

    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class StokesOptimalControlOnReferenceDomain(StokesOptimalControl):
        def __init__(self, V, **kwargs):
            StokesOptimalControl.__init__(self, "StokesOptimalControlOnReferenceDomain")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            mu = self.mu
            if term in ("a", "a*"):
                theta_a0 = nu * mu[0]
                theta_a1 = nu / mu[0]
                return (theta_a0, theta_a1)
            elif term in ("b", "b*", "bt", "bt*"):
                theta_b0 = mu[0]
                theta_b1 = 1.0
                return (theta_b0, theta_b1)
            elif term in ("c", "c*"):
                theta_c0 = mu[0]
                return (theta_c0,)
            elif term == "m":
                theta_m0 = mu[0]
                return (theta_m0,)
            elif term == "n":
                theta_n0 = alpha * mu[0]
                return (theta_n0,)
            elif term == "f":
                theta_f0 = - mu[0] * mu[1]
                return (theta_f0,)
            elif term == "g":
                theta_g0 = mu[0]**2
                return (theta_g0,)
            elif term == "l":
                theta_l0 = mu[0]
                return (theta_l0,)
            elif term == "h":
                theta_h0 = mu[0]**3 / 3.
                return (theta_h0,)
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = v[0].dx(0) * phi[0].dx(0) * dx + v[1].dx(0) * phi[1].dx(0) * dx
                a1 = v[0].dx(1) * phi[0].dx(1) * dx + v[1].dx(1) * phi[1].dx(1) * dx
                return (a0, a1)
            elif term == "a*":
                as0 = psi[0].dx(0) * w[0].dx(0) * dx + psi[1].dx(0) * w[1].dx(0) * dx
                as1 = psi[0].dx(1) * w[0].dx(1) * dx + psi[1].dx(1) * w[1].dx(1) * dx
                return (as0, as1)
            elif term == "b":
                b0 = - xi * v[0].dx(0) * dx
                b1 = - xi * v[1].dx(1) * dx
                return (b0, b1)
            elif term == "bt":
                bt0 = - p * phi[0].dx(0) * dx
                bt1 = - p * phi[1].dx(1) * dx
                return (bt0, bt1)
            elif term == "b*":
                bs0 = - pi * w[0].dx(0) * dx
                bs1 = - pi * w[1].dx(1) * dx
                return (bs0, bs1)
            elif term == "bt*":
                bts0 = - q * psi[0].dx(0) * dx
                bts1 = - q * psi[1].dx(1) * dx
                return (bts0, bts1)
            elif term == "c":
                c0 = inner(u, phi) * dx
                return (c0,)
            elif term == "c*":
                cs0 = inner(tau, w) * dx
                return (cs0,)
            elif term == "m":
                m0 = v[0] * psi[0] * dx
                return (m0,)
            elif term == "n":
                n0 = inner(u, tau) * dx
                return (n0,)
            elif term == "f":
                f0 = phi[1] * dx
                return (f0,)
            elif term == "g":
                g0 = vx_d * psi[0] * dx
                return (g0,)
            elif term == "l":
                l0 = ll * xi * dx
                return (l0,)
            elif term == "h":
                h0 = 1.0
                return (h0,)
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain()
    @ShapeParametrization(*shape_parametrization_expression)
    class StokesOptimalControlPullBack(StokesOptimalControl):
        def __init__(self, V, **kwargs):
            StokesOptimalControl.__init__(self, "StokesOptimalControlPullBack")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            mu = self.mu
            if term in ("a", "a*"):
                theta_a0 = nu * 1.0
                return (theta_a0,)
            elif term in ("b", "b*", "bt", "bt*"):
                theta_b0 = 1.0
                return (theta_b0,)
            elif term in ("c", "c*"):
                theta_c0 = 1.0
                return (theta_c0,)
            elif term == "m":
                theta_m0 = 1.0
                return (theta_m0,)
            elif term == "n":
                theta_n0 = alpha * 1.0
                return (theta_n0,)
            elif term == "f":
                theta_f0 = - mu[1]
                return (theta_f0,)
            elif term == "g":
                theta_g0 = 1.0
                return (theta_g0,)
            elif term == "l":
                theta_l0 = 1.0
                return (theta_l0,)
            elif term == "h":
                theta_h0 = 1.0
                return (theta_h0,)
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(v), grad(phi)) * dx
                return (a0,)
            elif term == "a*":
                ad0 = inner(grad(w), grad(psi)) * dx
                return (ad0,)
            elif term == "b":
                b0 = -xi * div(v) * dx
                return (b0,)
            elif term == "b*":
                btd0 = -pi * div(w) * dx
                return (btd0,)
            elif term == "bt":
                bt0 = -p * div(phi) * dx
                return (bt0,)
            elif term == "bt*":
                bd0 = -q * div(psi) * dx
                return (bd0,)
            elif term == "c":
                c0 = inner(u, phi) * dx
                return (c0,)
            elif term == "c*":
                cd0 = inner(tau, w) * dx
                return (cd0,)
            elif term == "m":
                m0 = v[0] * psi[0] * dx
                return (m0,)
            elif term == "n":
                n0 = inner(u, tau) * dx
                return (n0,)
            elif term == "f":
                f0 = phi[1] * dx
                return (f0,)
            elif term == "g":
                g0 = vx_d * psi[0] * dx
                return (g0,)
            elif term == "l":
                l0 = ll * xi * dx
                return (l0,)
            elif term == "h":
                h0 = vx_d * vx_d * dx(domain=mesh)
                return (h0,)
            else:
                raise ValueError("Invalid term for assemble_operator().")
    # Check forms
    problem_on_reference_domain = StokesOptimalControlOnReferenceDomain(
        V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = StokesOptimalControlPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)

        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)

        as_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a*")
        as_pull_back = theta_times_operator(problem_pull_back, "a*")
        assert forms_are_close(as_on_reference_domain, as_pull_back)

        b_on_reference_domain = theta_times_operator(problem_on_reference_domain, "b")
        b_pull_back = theta_times_operator(problem_pull_back, "b")
        assert forms_are_close(b_on_reference_domain, b_pull_back)

        bt_on_reference_domain = theta_times_operator(problem_on_reference_domain, "bt")
        bt_pull_back = theta_times_operator(problem_pull_back, "bt")
        assert forms_are_close(bt_on_reference_domain, bt_pull_back)

        bs_on_reference_domain = theta_times_operator(problem_on_reference_domain, "b*")
        bs_pull_back = theta_times_operator(problem_pull_back, "b*")
        assert forms_are_close(bs_on_reference_domain, bs_pull_back)

        bts_on_reference_domain = theta_times_operator(problem_on_reference_domain, "bt*")
        bts_pull_back = theta_times_operator(problem_pull_back, "bt*")
        assert forms_are_close(bts_on_reference_domain, bts_pull_back)

        c_on_reference_domain = theta_times_operator(problem_on_reference_domain, "c")
        c_pull_back = theta_times_operator(problem_pull_back, "c")
        assert forms_are_close(c_on_reference_domain, c_pull_back)

        cs_on_reference_domain = theta_times_operator(problem_on_reference_domain, "c*")
        cs_pull_back = theta_times_operator(problem_pull_back, "c*")
        assert forms_are_close(cs_on_reference_domain, cs_pull_back)

        m_on_reference_domain = theta_times_operator(problem_on_reference_domain, "m")
        m_pull_back = theta_times_operator(problem_pull_back, "m")
        assert forms_are_close(m_on_reference_domain, m_pull_back)

        n_on_reference_domain = theta_times_operator(problem_on_reference_domain, "n")
        n_pull_back = theta_times_operator(problem_pull_back, "n")
        assert forms_are_close(n_on_reference_domain, n_pull_back)

        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)

        g_on_reference_domain = theta_times_operator(problem_on_reference_domain, "g")
        g_pull_back = theta_times_operator(problem_pull_back, "g")
        assert forms_are_close(g_on_reference_domain, g_pull_back)

        l_on_reference_domain = theta_times_operator(problem_on_reference_domain, "l")
        l_pull_back = theta_times_operator(problem_pull_back, "l")
        assert forms_are_close(l_on_reference_domain, l_pull_back)

        h_on_reference_domain = theta_times_operator(problem_on_reference_domain, "h")
        h_pull_back = theta_times_operator(problem_pull_back, "h")
        assert forms_are_close(h_on_reference_domain, h_pull_back)


# Test forms pull back to reference domain for tutorial 16
@enable_pull_back_to_reference_domain_logging
@check_affine_and_non_affine_shape_parametrizations()
def test_pull_back_to_reference_domain_stokes_coupled(
        shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "t_bypass.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "t_bypass_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "t_bypass_facet_region.xml"))

    # Define shape parametrization
    shape_parametrization_expression = [
        ("mu[4] * x[0] + mu[1] - mu[4]",
         "mu[4] * tan(mu[5]) * x[0] + mu[0] * x[1] + mu[2] - mu[4] * tan(mu[5]) - mu[0]"),  # subdomain 1
        ("mu[4] * x[0] + mu[1] - mu[4]",
         "mu[4] * tan(mu[5]) * x[0] + mu[0] * x[1] + mu[2] - mu[4] * tan(mu[5]) - mu[0]"),  # subdomain 2
        ("mu[1] * x[0]", "mu[3] * x[1] + mu[2] + mu[0] - 2 * mu[3]"),  # subdomain 3
        ("mu[1] * x[0]", "mu[3] * x[1] + mu[2] + mu[0] - 2 * mu[3]"),  # subdomain 4
        ("mu[1] * x[0]", "mu[0] * x[1] + mu[2] - mu[0]"),  # subdomain 5
        ("mu[1] * x[0]", "mu[0] * x[1] + mu[2] - mu[0]"),  # subdomain 6
        ("mu[1] * x[0]", "mu[2] * x[1]"),  # subdomain 7
        ("mu[1] * x[0]", "mu[2] * x[1]"),  # subdomain 8
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)

    # Define function space, test/trial functions, measures, auxiliary expressions
    element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    U = FunctionSpace(mesh, element_u)
    element_c = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    C = FunctionSpace(mesh, element_c)
    c = TrialFunction(C)
    d = TestFunction(C)
    dx = Measure("dx")(subdomain_data=subdomains)
    vel = project(Expression(("- (16.0 / 25.0) * pow(x[1], 2) + (8.0 / 5.0) * x[1]", "-3.0 / 10.0"), degree=2), U)
    ff = Constant(1.0)

    # Define base problem
    class AdvectionDiffusion(ParametrizedProblem, metaclass=ABCMeta):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
            self.mu_range = [(0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0., pi / 6.)]
            self.terms = ["a", "f"]
            self.operator = dict()
            self.Q = dict()

        def name(self):
            return "___".join([
                self.folder_prefix, shape_parametrization_preprocessing.__name__,
                AdditionalProblemDecorator.__name__])

        def init(self):
            self._init_operators()

        def _init_operators(self):
            pass

        @abstractmethod
        def compute_theta(self, term):
            pass

        @abstractmethod
        def assemble_operator(self, term):
            pass

    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class AdvectionDiffusionOnReferenceDomain(AdvectionDiffusion):
        def __init__(self, V, **kwargs):
            AdvectionDiffusion.__init__(self, "AdvectionDiffusionOnReferenceDomain")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            mu = self.mu
            if term == "a":
                # inner(grad(c), grad(d)) * dx
                theta_a0 = mu[0] / mu[4]
                theta_a1 = mu[4] / (mu[0] * cos(mu[5])**2)
                theta_a2 = - tan(mu[5])
                theta_a3 = mu[0] / mu[4]
                theta_a4 = mu[4] / (mu[0] * cos(mu[5])**2)
                theta_a5 = - tan(mu[5])
                theta_a6 = mu[3] / mu[1]
                theta_a7 = mu[1] / mu[3]
                theta_a8 = mu[3] / mu[1]
                theta_a9 = mu[1] / mu[3]
                theta_a10 = mu[0] / mu[1]
                theta_a11 = mu[1] / mu[0]
                theta_a12 = mu[0] / mu[1]
                theta_a13 = mu[1] / mu[0]
                theta_a14 = mu[2] / mu[1]
                theta_a15 = mu[1] / mu[2]
                theta_a16 = mu[2] / mu[1]
                theta_a17 = mu[1] / mu[2]
                # inner(vel, grad(c)) * d * dx
                theta_a18 = mu[0]
                theta_a19 = mu[3]
                theta_a20 = mu[4] * tan(mu[5])
                theta_a21 = mu[3]
                theta_a22 = mu[1]
                theta_a23 = mu[0]
                theta_a24 = mu[1]
                theta_a25 = mu[2]
                theta_a26 = mu[1]
                return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7, theta_a8,
                        theta_a9, theta_a10, theta_a11, theta_a12, theta_a13, theta_a14, theta_a15, theta_a16,
                        theta_a17, theta_a18, theta_a19, theta_a20, theta_a21, theta_a22, theta_a23, theta_a24,
                        theta_a25, theta_a26)
            elif term == "f":
                theta_f0 = mu[0] * mu[4]
                theta_f1 = mu[1] * mu[3]
                theta_f2 = mu[0] * mu[1]
                theta_f3 = mu[1] * mu[2]
                return (theta_f0, theta_f1, theta_f2, theta_f3)
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                # inner(grad(c), grad(d)) * dx
                a0 = c.dx(0) * d.dx(0) * dx(1)
                a1 = c.dx(1) * d.dx(1) * dx(1)
                a2 = c.dx(0) * d.dx(1) * dx(1) + c.dx(1) * d.dx(0) * dx(1)
                a3 = c.dx(0) * d.dx(0) * dx(2)
                a4 = c.dx(1) * d.dx(1) * dx(2)
                a5 = c.dx(0) * d.dx(1) * dx(2) + c.dx(1) * d.dx(0) * dx(2)
                a6 = c.dx(0) * d.dx(0) * dx(3)
                a7 = c.dx(1) * d.dx(1) * dx(3)
                a8 = c.dx(0) * d.dx(0) * dx(4)
                a9 = c.dx(1) * d.dx(1) * dx(4)
                a10 = c.dx(0) * d.dx(0) * dx(5)
                a11 = c.dx(1) * d.dx(1) * dx(5)
                a12 = c.dx(0) * d.dx(0) * dx(6)
                a13 = c.dx(1) * d.dx(1) * dx(6)
                a14 = c.dx(0) * d.dx(0) * dx(7)
                a15 = c.dx(1) * d.dx(1) * dx(7)
                a16 = c.dx(0) * d.dx(0) * dx(8)
                a17 = c.dx(1) * d.dx(1) * dx(8)
                # inner(vel, grad(c)) * d * dx
                a18 = vel[0] * c.dx(0) * d * dx(1) + vel[0] * c.dx(0) * d * dx(2)
                a19 = vel[1] * c.dx(1) * d * dx(1) + vel[1] * c.dx(1) * d * dx(2)
                a20 = vel[1] * c.dx(0) * d * dx(1) + vel[1] * c.dx(0) * d * dx(2)
                a21 = vel[0] * c.dx(0) * d * dx(3) + vel[0] * c.dx(0) * d * dx(4)
                a22 = vel[1] * c.dx(1) * d * dx(3) + vel[1] * c.dx(1) * d * dx(4)
                a23 = vel[0] * c.dx(0) * d * dx(5) + vel[0] * c.dx(0) * d * dx(6)
                a24 = vel[1] * c.dx(1) * d * dx(5) + vel[1] * c.dx(1) * d * dx(6)
                a25 = vel[0] * c.dx(0) * d * dx(7) + vel[0] * c.dx(0) * d * dx(8)
                a26 = vel[1] * c.dx(1) * d * dx(7) + vel[1] * c.dx(1) * d * dx(8)
                return (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                        a20, a21, a22, a23, a24, a25, a26)
            elif term == "f":
                f0 = ff * d * dx(1) + ff * d * dx(2)
                f1 = ff * d * dx(3) + ff * d * dx(4)
                f2 = ff * d * dx(5) + ff * d * dx(6)
                f3 = ff * d * dx(7) + ff * d * dx(8)
                return (f0, f1, f2, f3)
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain()
    @ShapeParametrization(*shape_parametrization_expression)
    class AdvectionDiffusionPullBack(AdvectionDiffusion):
        def __init__(self, V, **kwargs):
            AdvectionDiffusion.__init__(self, "AdvectionDiffusionPullBack")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            if term == "a":
                theta_a0 = 1.0
                return (theta_a0,)
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0,)
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(c), grad(d)) * dx + inner(vel, grad(c)) * d * dx
                return (a0,)
            elif term == "f":
                f0 = ff * d * dx
                return (f0,)
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Check forms
    problem_on_reference_domain = AdvectionDiffusionOnReferenceDomain(C, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = AdvectionDiffusionPullBack(C, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)

        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)

        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)


# Test forms pull back to reference domain for tutorial 17
@enable_pull_back_to_reference_domain_logging
@check_affine_and_non_affine_shape_parametrizations()
def test_pull_back_to_reference_domain_navier_stokes(
        shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "backward_facing_step.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "backward_facing_step_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "backward_facing_step_facet_region.xml"))

    # Define shape parametrization
    shape_parametrization_expression = [
        ("x[0]", "x[1]"),  # subdomain 1
        ("x[0]", "mu[1] / 2. * x[1] + (2. - mu[1])")  # subdomain 2
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)

    # Define function space, test/trial functions, measures, auxiliary expressions
    element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element_up = MixedElement(element_u, element_p)
    V = FunctionSpace(mesh, element_up, components=[["u", "s"], "p"])
    dup = TrialFunction(V)
    (du, dp) = split(dup)
    up = Function(V)
    assign(up.sub(0), project(Expression(("x[0]", "x[1]"), element=V.sub(0).ufl_element()), V.sub(0).collapse()))
    (u, _) = split(up)
    vq = TestFunction(V)
    (v, q) = split(vq)
    dx = Measure("dx")(subdomain_data=subdomains)

    ff = Constant((1.0, - 10.0))
    gg = Constant(2.0)
    nu = Constant(1.0)

    # Define base problem
    class NavierStokes(ParametrizedProblem, metaclass=ABCMeta):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1.0, 2.0)
            self.mu_range = [(1.0, 80.0), (1.5, 2.5)]
            self.terms = ["a", "b", "bt", "c", "dc", "f", "g"]
            self.operator = dict()
            self.Q = dict()

        def name(self):
            return "___".join([
                self.folder_prefix, shape_parametrization_preprocessing.__name__,
                AdditionalProblemDecorator.__name__])

        def init(self):
            self._init_operators()

        def _init_operators(self):
            pass

        @abstractmethod
        def compute_theta(self, term):
            pass

        @abstractmethod
        def assemble_operator(self, term):
            pass

    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class NavierStokesOnReferenceDomain(NavierStokes):
        def __init__(self, V, **kwargs):
            NavierStokes.__init__(self, "NavierStokesOnReferenceDomain")
            self.V = V
            self._solution = up

        @compute_theta_for_derivative({"dc": "c"})
        def compute_theta(self, term):
            mu = self.mu
            if term == "a":
                theta_a0 = nu * 1.0
                theta_a1 = nu * (mu[1] / 2.0)
                theta_a2 = nu * (2.0 / mu[1])
                return (theta_a0, theta_a1, theta_a2)
            elif term in ("b", "bt"):
                theta_b0 = 1.0
                theta_b1 = mu[1] / 2.0
                theta_b2 = 1.0
                return (theta_b0, theta_b1, theta_b2)
            elif term == "c":
                theta_c0 = 1.0
                theta_c1 = mu[1] / 2.0
                theta_c2 = 1.0
                return (theta_c0, theta_c1, theta_c2)
            elif term == "f":
                theta_f0 = 1.0
                theta_f1 = mu[1] / 2.0
                return (theta_f0, theta_f1)
            elif term == "g":
                theta_g0 = 1.0
                theta_g1 = mu[1] / 2.0
                return (theta_g0, theta_g1)
            else:
                raise ValueError("Invalid term for compute_theta().")

        @assemble_operator_for_derivative({"dc": "c"})
        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(du), grad(v)) * dx(1)
                a1 = du[0].dx(0) * v[0].dx(0) * dx(2) + du[1].dx(0) * v[1].dx(0) * dx(2)
                a2 = du[0].dx(1) * v[0].dx(1) * dx(2) + du[1].dx(1) * v[1].dx(1) * dx(2)
                return (a0, a1, a2)
            elif term == "b":
                b0 = -q * div(du) * dx(1)
                b1 = -q * du[0].dx(0) * dx(2)
                b2 = -q * du[1].dx(1) * dx(2)
                return (b0, b1, b2)
            elif term == "bt":
                bt0 = -dp * div(v) * dx(1)
                bt1 = -dp * v[0].dx(0) * dx(2)
                bt2 = -dp * v[1].dx(1) * dx(2)
                return (bt0, bt1, bt2)
            elif term == "c":
                c0 = inner(grad(u) * u, v) * dx(1)
                c1 = u[0] * u[0].dx(0) * v[0] * dx(2) + u[0] * u[1].dx(0) * v[1] * dx(2)
                c2 = u[1] * u[0].dx(1) * v[0] * dx(2) + u[1] * u[1].dx(1) * v[1] * dx(2)
                return (c0, c1, c2)
            elif term == "f":
                f0 = inner(ff, v) * dx(1)
                f1 = inner(ff, v) * dx(2)
                return (f0, f1)
            elif term == "g":
                g0 = gg * q * dx(1)
                g1 = gg * q * dx(2)
                return (g0, g1)
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain()
    @ShapeParametrization(*shape_parametrization_expression)
    class NavierStokesPullBack(NavierStokes):
        def __init__(self, V, **kwargs):
            NavierStokes.__init__(self, "NavierStokesPullBack")
            self.V = V
            self._solution = up

        @compute_theta_for_derivative({"dc": "c"})
        def compute_theta(self, term):
            if term == "a":
                theta_a0 = nu * 1.0
                return (theta_a0, )
            elif term in ("b", "bt"):
                theta_b0 = 1.0
                return (theta_b0, )
            elif term == "c":
                theta_c0 = 1.0
                return (theta_c0, )
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0, )
            elif term == "g":
                theta_g0 = 1.0
                return (theta_g0, )
            else:
                raise ValueError("Invalid term for compute_theta().")

        @assemble_operator_for_derivative({"dc": "c"})
        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(du), grad(v)) * dx
                return (a0, )
            elif term == "b":
                b0 = - q * div(du) * dx
                return (b0, )
            elif term == "bt":
                bt0 = - dp * div(v) * dx
                return (bt0, )
            elif term == "c":
                c0 = inner(grad(u) * u, v) * dx
                return (c0, )
            elif term == "f":
                f0 = inner(ff, v) * dx
                return (f0, )
            elif term == "g":
                g0 = q * gg * dx
                return (g0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Check forms
    problem_on_reference_domain = NavierStokesOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = NavierStokesPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)

        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)

        b_on_reference_domain = theta_times_operator(problem_on_reference_domain, "b")
        b_pull_back = theta_times_operator(problem_pull_back, "b")
        assert forms_are_close(b_on_reference_domain, b_pull_back)

        bt_on_reference_domain = theta_times_operator(problem_on_reference_domain, "bt")
        bt_pull_back = theta_times_operator(problem_pull_back, "bt")
        assert forms_are_close(bt_on_reference_domain, bt_pull_back)

        c_on_reference_domain = theta_times_operator(problem_on_reference_domain, "c")
        c_pull_back = theta_times_operator(problem_pull_back, "c")
        assert forms_are_close(c_on_reference_domain, c_pull_back)

        dc_on_reference_domain = theta_times_operator(problem_on_reference_domain, "dc")
        dc_pull_back = theta_times_operator(problem_pull_back, "dc")
        assert forms_are_close(dc_on_reference_domain, dc_pull_back)

        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)

        g_on_reference_domain = theta_times_operator(problem_on_reference_domain, "g")
        g_pull_back = theta_times_operator(problem_pull_back, "g")
        assert forms_are_close(g_on_reference_domain, g_pull_back)


# Test forms pull back to reference domain for tutorial 18
@enable_pull_back_to_reference_domain_logging
@check_affine_and_non_affine_shape_parametrizations()
def test_pull_back_to_reference_domain_stokes_unsteady(
        shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "cavity.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "cavity_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "cavity_facet_region.xml"))

    # Define shape parametrization
    shape_parametrization_expression = [
        ("mu[0] * x[0]", "x[1]"),  # subdomain 1
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)

    # Define function space, test/trial functions, measures, auxiliary expressions
    element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_u, element_p)
    V = FunctionSpace(mesh, element, components=[["u", "s"], "p"])
    up = TrialFunction(V)
    (u, p) = split(up)
    vq = TestFunction(V)
    (v, q) = split(vq)
    dx = Measure("dx")(subdomain_data=subdomains)

    ff = Constant((1.0, 2.0))
    gg = Constant(1.0)

    # Define base problem
    class StokesUnsteady(ParametrizedProblem, metaclass=ABCMeta):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1., )
            self.mu_range = [(0.5, 2.5)]
            self.terms = ["a", "b", "bt", "m", "f", "g"]
            self.operator = dict()
            self.Q = dict()

        def name(self):
            return "___".join([
                self.folder_prefix, shape_parametrization_preprocessing.__name__,
                AdditionalProblemDecorator.__name__])

        def init(self):
            self._init_operators()

        def _init_operators(self):
            pass

        @abstractmethod
        def compute_theta(self, term):
            pass

        @abstractmethod
        def assemble_operator(self, term):
            pass

    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class StokesUnsteadyOnReferenceDomain(StokesUnsteady):
        def __init__(self, V, **kwargs):
            StokesUnsteady.__init__(self, "StokesUnsteadyOnReferenceDomain")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            mu = self.mu
            if term == "a":
                theta_a0 = 1. / mu[0]
                theta_a1 = mu[0]
                return (theta_a0, theta_a1)
            elif term in ("b", "bt"):
                theta_b0 = 1.
                theta_b1 = mu[0]
                return (theta_b0, theta_b1)
            elif term == "f":
                theta_f0 = mu[0]
                return (theta_f0, )
            elif term == "g":
                theta_g0 = mu[0]
                return (theta_g0, )
            elif term == "m":
                theta_m0 = mu[0]
                return (theta_m0, )
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = (u[0].dx(0) * v[0].dx(0) + u[1].dx(0) * v[1].dx(0)) * dx
                a1 = (u[0].dx(1) * v[0].dx(1) + u[1].dx(1) * v[1].dx(1)) * dx
                return (a0, a1)
            elif term == "b":
                b0 = - q * u[0].dx(0) * dx
                b1 = - q * u[1].dx(1) * dx
                return (b0, b1)
            elif term == "bt":
                bt0 = - p * v[0].dx(0) * dx
                bt1 = - p * v[1].dx(1) * dx
                return (bt0, bt1)
            elif term == "f":
                f0 = inner(ff, v) * dx
                return (f0, )
            elif term == "g":
                g0 = gg * q * dx
                return (g0, )
            elif term == "m":
                m0 = inner(u, v) * dx
                return (m0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain()
    @ShapeParametrization(*shape_parametrization_expression)
    class StokesUnsteadyPullBack(StokesUnsteady):
        def __init__(self, V, **kwargs):
            StokesUnsteady.__init__(self, "StokesUnsteadyPullBack")
            self.V = V
            self._solution = Function(V)

        def compute_theta(self, term):
            if term == "a":
                theta_a0 = 1.0
                return (theta_a0, )
            elif term in ("b", "bt"):
                theta_b0 = 1.0
                return (theta_b0, )
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0, )
            elif term == "g":
                theta_g0 = 1.0
                return (theta_g0, )
            elif term == "m":
                theta_m0 = 1.0
                return (theta_m0, )
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v)) * dx
                return (a0, )
            elif term == "b":
                b0 = - q * div(u) * dx
                return (b0, )
            elif term == "bt":
                bt0 = - p * div(v) * dx
                return (bt0, )
            elif term == "f":
                f0 = inner(ff, v) * dx
                return (f0, )
            elif term == "g":
                g0 = q * gg * dx
                return (g0, )
            elif term == "m":
                m0 = inner(u, v) * dx
                return (m0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Check forms
    problem_on_reference_domain = StokesUnsteadyOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = StokesUnsteadyPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)

        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)

        b_on_reference_domain = theta_times_operator(problem_on_reference_domain, "b")
        b_pull_back = theta_times_operator(problem_pull_back, "b")
        assert forms_are_close(b_on_reference_domain, b_pull_back)

        bt_on_reference_domain = theta_times_operator(problem_on_reference_domain, "bt")
        bt_pull_back = theta_times_operator(problem_pull_back, "bt")
        assert forms_are_close(bt_on_reference_domain, bt_pull_back)

        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)

        g_on_reference_domain = theta_times_operator(problem_on_reference_domain, "g")
        g_pull_back = theta_times_operator(problem_pull_back, "g")
        assert forms_are_close(g_on_reference_domain, g_pull_back)

        m_on_reference_domain = theta_times_operator(problem_on_reference_domain, "m")
        m_pull_back = theta_times_operator(problem_pull_back, "m")
        assert forms_are_close(m_on_reference_domain, m_pull_back)
