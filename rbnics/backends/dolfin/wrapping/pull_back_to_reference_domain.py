# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from collections import defaultdict, namedtuple, OrderedDict
import itertools
import numbers
import re
from logging import DEBUG, getLogger
from numpy import allclose, isclose, ones as numpy_ones
from mpi4py.MPI import Op
from sympy import (Basic as SympyBase, ccode, collect, Float, ImmutableMatrix, Integer, Matrix as SympyMatrix,
                   Number, preorder_traversal, simplify, symbols, sympify)
from ufl import as_tensor, FiniteElement, Form, Measure, sqrt, TensorElement, VectorElement
from ufl.algorithms import apply_transformer, expand_derivatives, Transformer
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.expand_indices import expand_indices
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import (CellDiameter, CellVolume, Circumradius, FacetArea, FacetJacobianDeterminant, FacetNormal,
                         Grad, Jacobian, JacobianDeterminant, JacobianInverse)
from ufl.core.multiindex import FixedIndex, Index, indices, MultiIndex
from ufl.corealg.multifunction import memoized_handler, MultiFunction
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.traversal import pre_traversal, traverse_unique_terminals
from ufl.indexed import Indexed
from dolfin import assemble, cells, compile_cpp_code, CompiledExpression, Constant, Expression, facets
from dolfin.cpp.la import GenericMatrix, GenericVector
from dolfin.function.expression import BaseExpression
from rbnics.backends.dolfin.wrapping.assemble_operator_for_stability_factor import (
    assemble_operator_for_stability_factor)
from rbnics.backends.dolfin.wrapping.compute_theta_for_stability_factor import compute_theta_for_stability_factor
from rbnics.backends.dolfin.wrapping.expand_sum_product import expand_sum_product
from rbnics.backends.dolfin.wrapping.form_description import form_description
import rbnics.backends.dolfin.wrapping.form_mul  # enable form multiplication and division  # noqa: F401
from rbnics.backends.dolfin.wrapping.parametrized_expression import ParametrizedExpression
from rbnics.backends.dolfin.wrapping.remove_complex_nodes import remove_complex_nodes
from rbnics.eim.utils.decorators import DefineSymbolicParameters
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import (overload, PreserveClassName, ProblemDecoratorFor, ReducedProblemDecoratorFor,
                                     ReductionMethodDecoratorFor)
from rbnics.utils.test import PatchInstanceMethod

logger = getLogger("rbnics/backends/dolfin/wrapping/pull_back_to_reference_domain.py")


# ===== Helper function for sympy/ufl conversion ===== #
@overload
def sympy_to_parametrized_expression(sympy_expression: SympyBase, problem: object):
    """
    Convert a sympy scalar expression to a ParametrizedExpression
    """
    cpp_expression = ccode(sympy_expression).replace(", 0]", "]")
    element = FiniteElement("CG", problem.V.mesh().ufl_cell(), 1)
    return ParametrizedExpression(problem, cpp_expression, mu=problem.mu, element=element)


@overload
def sympy_to_parametrized_expression(sympy_expression: (ImmutableMatrix, SympyMatrix), problem: object):
    """
    Convert a sympy vector/matrix expression to a ParametrizedExpression
    """
    dim = problem.V.mesh().geometry().dim()
    if sympy_expression.shape[1] == 1:
        cpp_expression = list()
        for i in range(dim):
            cpp_expression.append(
                ccode(sympy_expression[i]).replace(", 0]", "]")
            )
        element = VectorElement("CG", problem.V.mesh().ufl_cell(), 1)
    else:
        cpp_expression = list()
        for i in range(dim):
            cpp_expression_i = list()
            for j in range(dim):
                cpp_expression_i.append(
                    ccode(sympy_expression[i, j]).replace(", 0]", "]")
                )
            cpp_expression.append(tuple(cpp_expression_i))
        element = TensorElement("CG", problem.V.mesh().ufl_cell(), 1)
    return ParametrizedExpression(problem, tuple(cpp_expression), mu=problem.mu, element=element)


# ===== Memoization for shape parametrization objects: inspired by ufl/corealg/multifunction.py ===== #
def shape_parametrization_cache(function):
    function._cache = Cache()

    def _memoized_function(shape_parametrization_expression_on_subdomain, problem):
        cache = getattr(function, "_cache")
        try:
            output = cache[shape_parametrization_expression_on_subdomain, problem]
        except KeyError:
            output = function(shape_parametrization_expression_on_subdomain, problem)
            cache[shape_parametrization_expression_on_subdomain, problem] = output
        return output

    return _memoized_function


ShapeParametrizationResult = namedtuple("ShapeParametrizationResult", "sympy ufl")


# ===== Shape parametrization classes related to jacobian, inspired by ufl/geometry.py ===== #
@shape_parametrization_cache
def ShapeParametrizationMap(shape_parametrization_expression_on_subdomain, problem):
    from rbnics.shape_parametrization.utils.symbolic import python_string_to_sympy
    shape_parametrization_map_sympy = python_string_to_sympy(shape_parametrization_expression_on_subdomain, problem)
    if shape_parametrization_map_sympy not in problem._shape_parametrization_expressions_sympy_to_ufl:
        shape_parametrization_map = sympy_to_parametrized_expression(shape_parametrization_map_sympy, problem)
        problem._shape_parametrization_expressions_sympy_to_ufl[
            shape_parametrization_map_sympy] = shape_parametrization_map
        assert shape_parametrization_map not in problem._shape_parametrization_expressions_ufl_to_sympy
        problem._shape_parametrization_expressions_ufl_to_sympy[
            shape_parametrization_map] = shape_parametrization_map_sympy
    else:
        shape_parametrization_map = problem._shape_parametrization_expressions_sympy_to_ufl[
            shape_parametrization_map_sympy]
    return ShapeParametrizationResult(sympy=shape_parametrization_map_sympy, ufl=shape_parametrization_map)


@shape_parametrization_cache
def ShapeParametrizationJacobian(shape_parametrization_expression_on_subdomain, problem):
    from rbnics.shape_parametrization.utils.symbolic import (
        compute_shape_parametrization_gradient, python_string_to_sympy)
    shape_parametrization_jacobian_sympy = python_string_to_sympy(compute_shape_parametrization_gradient(
        shape_parametrization_expression_on_subdomain), problem)
    if shape_parametrization_jacobian_sympy not in problem._shape_parametrization_expressions_sympy_to_ufl:
        shape_parametrization_jacobian = sympy_to_parametrized_expression(
            shape_parametrization_jacobian_sympy, problem)
        problem._shape_parametrization_expressions_sympy_to_ufl[
            shape_parametrization_jacobian_sympy] = shape_parametrization_jacobian
        assert shape_parametrization_jacobian not in problem._shape_parametrization_expressions_ufl_to_sympy
        problem._shape_parametrization_expressions_ufl_to_sympy[
            shape_parametrization_jacobian] = shape_parametrization_jacobian_sympy
    else:
        shape_parametrization_jacobian = problem._shape_parametrization_expressions_sympy_to_ufl[
            shape_parametrization_jacobian_sympy]
    return ShapeParametrizationResult(sympy=shape_parametrization_jacobian_sympy, ufl=shape_parametrization_jacobian)


@shape_parametrization_cache
def ShapeParametrizationJacobianInverse(shape_parametrization_expression_on_subdomain, problem):
    shape_parametrization_jacobian_inverse_sympy = ShapeParametrizationJacobian(
        shape_parametrization_expression_on_subdomain, problem).sympy.inv()
    if shape_parametrization_jacobian_inverse_sympy not in problem._shape_parametrization_expressions_sympy_to_ufl:
        shape_parametrization_jacobian_inverse = sympy_to_parametrized_expression(
            shape_parametrization_jacobian_inverse_sympy, problem)
        problem._shape_parametrization_expressions_sympy_to_ufl[
            shape_parametrization_jacobian_inverse_sympy] = shape_parametrization_jacobian_inverse
        assert shape_parametrization_jacobian_inverse not in problem._shape_parametrization_expressions_ufl_to_sympy
        problem._shape_parametrization_expressions_ufl_to_sympy[
            shape_parametrization_jacobian_inverse] = shape_parametrization_jacobian_inverse_sympy
    else:
        shape_parametrization_jacobian_inverse = problem._shape_parametrization_expressions_sympy_to_ufl[
            shape_parametrization_jacobian_inverse_sympy]
    return ShapeParametrizationResult(sympy=shape_parametrization_jacobian_inverse_sympy,
                                      ufl=shape_parametrization_jacobian_inverse)


@shape_parametrization_cache
def ShapeParametrizationJacobianInverseTranspose(shape_parametrization_expression_on_subdomain, problem):
    shape_parametrization_jacobian_inverse_transpose_sympy = ShapeParametrizationJacobianInverse(
        shape_parametrization_expression_on_subdomain, problem).sympy.transpose()
    if (shape_parametrization_jacobian_inverse_transpose_sympy not in
            problem._shape_parametrization_expressions_sympy_to_ufl):
        shape_parametrization_jacobian_inverse_transpose = sympy_to_parametrized_expression(
            shape_parametrization_jacobian_inverse_transpose_sympy, problem)
        problem._shape_parametrization_expressions_sympy_to_ufl[
            shape_parametrization_jacobian_inverse_transpose_sympy] = shape_parametrization_jacobian_inverse_transpose
        assert (shape_parametrization_jacobian_inverse_transpose
                not in problem._shape_parametrization_expressions_ufl_to_sympy)
        problem._shape_parametrization_expressions_ufl_to_sympy[
            shape_parametrization_jacobian_inverse_transpose] = shape_parametrization_jacobian_inverse_transpose_sympy
    else:
        shape_parametrization_jacobian_inverse_transpose = problem._shape_parametrization_expressions_sympy_to_ufl[
            shape_parametrization_jacobian_inverse_transpose_sympy]
    return ShapeParametrizationResult(sympy=shape_parametrization_jacobian_inverse_transpose_sympy,
                                      ufl=shape_parametrization_jacobian_inverse_transpose)


@shape_parametrization_cache
def ShapeParametrizationJacobianDeterminant(shape_parametrization_expression_on_subdomain, problem):
    shape_parametrization_jacobian_determinant_sympy = ShapeParametrizationJacobian(
        shape_parametrization_expression_on_subdomain, problem).sympy.det()
    if shape_parametrization_jacobian_determinant_sympy not in problem._shape_parametrization_expressions_sympy_to_ufl:
        shape_parametrization_jacobian_determinant = sympy_to_parametrized_expression(
            shape_parametrization_jacobian_determinant_sympy, problem)
        problem._shape_parametrization_expressions_sympy_to_ufl[
            shape_parametrization_jacobian_determinant_sympy] = shape_parametrization_jacobian_determinant
        assert shape_parametrization_jacobian_determinant not in problem._shape_parametrization_expressions_ufl_to_sympy
        problem._shape_parametrization_expressions_ufl_to_sympy[
            shape_parametrization_jacobian_determinant] = shape_parametrization_jacobian_determinant_sympy
    else:
        shape_parametrization_jacobian_determinant = problem._shape_parametrization_expressions_sympy_to_ufl[
            shape_parametrization_jacobian_determinant_sympy]
    return ShapeParametrizationResult(sympy=shape_parametrization_jacobian_determinant_sympy,
                                      ufl=shape_parametrization_jacobian_determinant)


@shape_parametrization_cache
def ShapeParametrizationFacetJacobianDeterminant(shape_parametrization_expression_on_subdomain, problem):
    nanson = (ShapeParametrizationJacobianDeterminant(
              shape_parametrization_expression_on_subdomain, problem).ufl
              * ShapeParametrizationJacobianInverseTranspose(
              shape_parametrization_expression_on_subdomain, problem).ufl
              * FacetNormal(problem.V.mesh().ufl_domain()))
    i = Index()
    return sqrt(nanson[i] * nanson[i])


@shape_parametrization_cache
def ShapeParametrizationCircumradius(shape_parametrization_expression_on_subdomain, problem):
    return ShapeParametrizationJacobianDeterminant(
        shape_parametrization_expression_on_subdomain, problem).ufl**(
            1. / problem.V.mesh().ufl_domain().topological_dimension())


@shape_parametrization_cache
def ShapeParametrizationCellDiameter(shape_parametrization_expression_on_subdomain, problem):
    return ShapeParametrizationJacobianDeterminant(
        shape_parametrization_expression_on_subdomain, problem).ufl**(
            1. / problem.V.mesh().ufl_domain().topological_dimension())


# ===== Pull back form measures: inspired by ufl/algorithms/apply_integral_scaling.py ===== #
def pull_back_measures(shape_parametrization_expression_on_subdomain, problem, integral, subdomain_id):
    # This function is inspired by compute_integrand_scaling_factor in the aforementioned file
    assert integral.ufl_domain() == problem.V.mesh().ufl_domain()
    integral_type = integral.integral_type()
    tdim = integral.ufl_domain().topological_dimension()

    if integral_type == "cell":
        scale = ShapeParametrizationJacobianDeterminant(shape_parametrization_expression_on_subdomain, problem).ufl
    elif integral_type.startswith("exterior_facet") or integral_type.startswith("interior_facet"):
        if tdim > 1:
            # Scaling integral by facet jacobian determinant
            scale = ShapeParametrizationFacetJacobianDeterminant(
                shape_parametrization_expression_on_subdomain, problem)
        else:
            # No need to scale "integral" over a vertex
            scale = 1
    else:
        raise ValueError("Unknown integral type {}, don't know how to scale.".format(integral_type))

    # Prepare measure for the new form (from firedrake/mg/ufl_utils.py)
    measure = Measure(
        integral.integral_type(),
        domain=integral.ufl_domain(),
        subdomain_id=subdomain_id,
        subdomain_data=integral.subdomain_data(),
        metadata=integral.metadata())
    return (scale, measure)


# ===== Pull back form gradients: inspired by ufl/algorithms/change_to_reference.py ===== #
class PullBackGradients(MultiFunction):
    # This class is inspired by OLDChangeToReferenceGrad in the aforementioned file
    def __init__(self, shape_parametrization_expression_on_subdomain, problem):
        MultiFunction.__init__(self)
        self.shape_parametrization_expression_on_subdomain = shape_parametrization_expression_on_subdomain
        self.problem = problem
        # Auxiliary quantities to compute derivatives
        from rbnics.shape_parametrization.utils.symbolic import sympy_symbolic_coordinates
        from rbnics.shape_parametrization.utils.symbolic.python_string_to_sympy import MatrixListSymbol
        self.x_symb = sympy_symbolic_coordinates(problem.V.mesh().geometry().dim(), MatrixListSymbol)

    expr = MultiFunction.reuse_if_untouched

    def div(self, o, f):
        assert self.problem.V.mesh().ufl_domain() == f.ufl_domain()
        # Create shape parametrization Jacobian inverse object
        Jinv = ShapeParametrizationJacobianInverse(
            self.shape_parametrization_expression_on_subdomain, self.problem).ufl

        # Indices to get to the scalar component of f
        first = indices(len(f.ufl_shape) - 1)
        last = Index()
        j = Index()

        # Wrap back in tensor shape
        grad_f = GradWithSympy(f, self.x_symb, self.problem)
        replaced_o = as_tensor(Jinv[j, last] * grad_f[first + (last, j)], first)
        return replaced_o

    def grad(self, o, f):
        assert self.problem.V.mesh().ufl_domain() == f.ufl_domain()
        # Create shape parametrization Jacobian inverse object
        Jinv = ShapeParametrizationJacobianInverse(
            self.shape_parametrization_expression_on_subdomain, self.problem).ufl

        # Indices to get to the scalar component of f
        f_indices = indices(len(f.ufl_shape))

        # Indices for grad definition
        j, k = indices(2)

        # Wrap back in tensor shape, derivative axes at the end
        grad_f = GradWithSympy(f, self.x_symb, self.problem)
        return as_tensor(Jinv[j, k] * grad_f[f_indices + (j,)], f_indices + (k,))

    def reference_div(self, o):
        raise ValueError("Not expecting reference div.")

    def reference_grad(self, o):
        raise ValueError("Not expecting reference grad.")


def pull_back_gradients(shape_parametrization_expression_on_subdomain, problem, integrand):
    # This function is inspired by change_to_reference_grad in the aforementioned file
    return map_expr_dag(PullBackGradients(shape_parametrization_expression_on_subdomain, problem), integrand)


def GradWithSympy(expression, x_symb, problem):
    grad_expression = apply_derivatives(Grad(expression))
    return apply_transformer(grad_expression, GradWithSympyTransformer(x_symb, problem))


class GradWithSympyTransformer(Transformer):
    def __init__(self, x_symb, problem):
        Transformer.__init__(self)
        self.x_symb = x_symb
        self.problem = problem

    expr = Transformer.reuse_if_untouched

    def grad(self, o, f):
        if f in self.problem._shape_parametrization_expressions_ufl_to_sympy:
            f_sympy = self.problem._shape_parametrization_expressions_ufl_to_sympy[f]
            grad_f = list()
            for i in range(self.problem.V.mesh().geometry().dim()):
                grad_f_i_sympy = f_sympy.diff(self.x_symb[i])
                if grad_f_i_sympy not in self.problem._shape_parametrization_expressions_sympy_to_ufl:
                    grad_f_i = self.problem._shape_parametrization_expressions_sympy_to_ufl.get(
                        grad_f_i_sympy, sympy_to_parametrized_expression(grad_f_i_sympy, self.problem))
                    self.problem._shape_parametrization_expressions_sympy_to_ufl[grad_f_i_sympy] = grad_f_i
                    assert grad_f_i not in self.problem._shape_parametrization_expressions_ufl_to_sympy
                    self.problem._shape_parametrization_expressions_ufl_to_sympy[grad_f_i] = grad_f_i_sympy
                else:
                    grad_f_i = self.problem._shape_parametrization_expressions_sympy_to_ufl[grad_f_i_sympy]
                grad_f.append(grad_f_i)
            return as_tensor(grad_f)
        else:
            return o


# ===== Pull back geometric quantities: inspired by ufl/algorithms/apply_geometry_lowering.py ===== #
class PullBackGeometricQuantities(MultiFunction):
    # This class is inspired by GeometryLoweringApplier in the aforementioned file
    def __init__(self, shape_parametrization_expression_on_subdomain, problem):
        MultiFunction.__init__(self)
        self.shape_parametrization_expression_on_subdomain = shape_parametrization_expression_on_subdomain
        self.problem = problem

    expr = MultiFunction.reuse_if_untouched

    def _not_implemented(self, o):
        raise NotImplementedError("Pull back of this geometric quantity has not been implemented")

    @memoized_handler
    def jacobian(self, o):
        assert self.problem.V.mesh().ufl_domain() == o.ufl_domain()
        return ShapeParametrizationJacobian(
            self.shape_parametrization_expression_on_subdomain, self.problem) * Jacobian(o.ufl_domain()).ufl

    @memoized_handler
    def jacobian_inverse(self, o):
        assert self.problem.V.mesh().ufl_domain() == o.ufl_domain()
        return JacobianInverse(o.ufl_domain()) * ShapeParametrizationJacobianInverse(
            self.shape_parametrization_expression_on_subdomain, self.problem).ufl

    @memoized_handler
    def jacobian_determinant(self, o):
        assert self.problem.V.mesh().ufl_domain() == o.ufl_domain()
        return (ShapeParametrizationJacobianDeterminant(
            self.shape_parametrization_expression_on_subdomain, self.problem).ufl
            * JacobianDeterminant(o.ufl_domain()))

    facet_jacobian = _not_implemented
    facet_jacobian_inverse = _not_implemented

    @memoized_handler
    def facet_jacobian_determinant(self, o):
        assert self.problem.V.mesh().ufl_domain() == o.ufl_domain()
        return (ShapeParametrizationFacetJacobianDeterminant(
            self.shape_parametrization_expression_on_subdomain, self.problem)
            * FacetJacobianDeterminant(o.ufl_domain()))

    @memoized_handler
    def spatial_coordinate(self, o):
        assert self.problem.V.mesh().ufl_domain() == o.ufl_domain()
        return ShapeParametrizationMap(self.shape_parametrization_expression_on_subdomain, self.problem).ufl

    @memoized_handler
    def cell_volume(self, o):
        return self.jacobian_determinant(o) * CellVolume(o.ufl_domain())

    @memoized_handler
    def facet_area(self, o):
        return self.facet_jacobian_determinant(o) * FacetArea(o.ufl_domain())

    @memoized_handler
    def circumradius(self, o):
        # This transformation is not exact. The exact transformation would not preserve affinity
        # if the shape parametrization map was affine.
        assert self.problem.V.mesh().ufl_domain() == o.ufl_domain()
        return ShapeParametrizationCircumradius(
            self.shape_parametrization_expression_on_subdomain, self.problem) * Circumradius(o.ufl_domain())

    @memoized_handler
    def cell_diameter(self, o):
        # This transformation is not exact. The exact transformation would not preserve affinity
        # if the shape parametrization map was affine.
        assert self.problem.V.mesh().ufl_domain() == o.ufl_domain()
        return ShapeParametrizationCellDiameter(
            self.shape_parametrization_expression_on_subdomain, self.problem) * CellDiameter(o.ufl_domain())

    min_cell_edge_length = _not_implemented
    max_cell_edge_length = _not_implemented
    min_facet_edge_length = _not_implemented
    max_facet_edge_length = _not_implemented

    cell_normal = _not_implemented

    @memoized_handler
    def facet_normal(self, o):
        assert self.problem.V.mesh().ufl_domain() == o.ufl_domain()
        nanson = (ShapeParametrizationJacobianDeterminant(
            self.shape_parametrization_expression_on_subdomain, self.problem).ufl
            * ShapeParametrizationJacobianInverseTranspose(
                self.shape_parametrization_expression_on_subdomain, self.problem).ufl * FacetNormal(o.ufl_domain()))
        i = Index()
        return nanson / sqrt(nanson[i] * nanson[i])


def pull_back_geometric_quantities(shape_parametrization_expression_on_subdomain, problem, integrand):
    # This function is inspired by apply_geometry_lowering in the aforementioned file
    return map_expr_dag(
        PullBackGeometricQuantities(shape_parametrization_expression_on_subdomain, problem), integrand)


# ===== Pull back expressions to reference domain: inspired by ufl/algorithms/apply_function_pullbacks.py ===== #
def pull_back_expression_code(pull_back_expression_name, expression_constructor):
    return """
        #include <Eigen/Core>
        #include <pybind11/pybind11.h>
        #include <pybind11/eigen.h>
        #include <dolfin/function/Expression.h>

        namespace py = pybind11;

        class PULL_BACK_EXPRESSION_NAME : public dolfin::Expression
        {
        public:
            PULL_BACK_EXPRESSION_NAME(
                    std::shared_ptr<dolfin::Expression> f,
                    std::shared_ptr<dolfin::Expression> shape_parametrization_expression_on_subdomain) :
                EXPRESSION_CONSTRUCTOR,
                f(f),
                shape_parametrization_expression_on_subdomain(shape_parametrization_expression_on_subdomain) {}

            void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x,
                      const ufc::cell& c) const
            {
                Eigen::VectorXd x_o(x.size());
                shape_parametrization_expression_on_subdomain->eval(x_o, x, c);
                f->eval(values, x_o, c);
            }
        private:
            std::shared_ptr<dolfin::Expression> f;
            std::shared_ptr<dolfin::Expression> shape_parametrization_expression_on_subdomain;
        };

        PYBIND11_MODULE(SIGNATURE, m)
        {
            py::class_<PULL_BACK_EXPRESSION_NAME, std::shared_ptr<PULL_BACK_EXPRESSION_NAME>,
                       dolfin::Expression>(m, "PULL_BACK_EXPRESSION_NAME", py::dynamic_attr())
              .def(py::init<std::shared_ptr<dolfin::Expression>, std::shared_ptr<dolfin::Expression>>());
        }
    """.replace(
        "PULL_BACK_EXPRESSION_NAME", pull_back_expression_name).replace(
            "EXPRESSION_CONSTRUCTOR", expression_constructor)


def PullBackExpression(shape_parametrization_expression_on_subdomain, f, problem):
    shape_parametrization_expression_on_subdomain = ShapeParametrizationMap(
        shape_parametrization_expression_on_subdomain, problem).ufl
    assert len(f.ufl_shape) in (0, 1)
    if len(f.ufl_shape) == 0:
        pull_back_expression_name = "PullBackExpressionScalar"
        expression_constructor = "Expression()"
    elif len(f.ufl_shape) == 1:
        pull_back_expression_name = "PullBackExpressionVector" + str(f.ufl_shape[0])
        expression_constructor = "Expression(" + str(f.ufl_shape[0]) + ")"
    else:
        raise ValueError("Invalid shape")
    PullBackExpression = getattr(compile_cpp_code(pull_back_expression_code(
        pull_back_expression_name, expression_constructor)), pull_back_expression_name)
    pulled_back_f_cpp = PullBackExpression(f._cpp_object, shape_parametrization_expression_on_subdomain._cpp_object)
    pulled_back_f_cpp.f_no_upcast = f
    pulled_back_f_cpp.shape_parametrization_expression_on_subdomain_no_upcast = (
        shape_parametrization_expression_on_subdomain)
    pulled_back_f_cpp._parameters = f._parameters
    pulled_back_f = CompiledExpression(pulled_back_f_cpp, element=f.ufl_element())
    return pulled_back_f


def is_pull_back_expression(expression):
    return hasattr(expression, "shape_parametrization_expression_on_subdomain_no_upcast")


def is_pull_back_expression_parametrized(expression):
    parameters = expression.f_no_upcast._parameters
    if "mu_0" in parameters:
        return True
    # mu[*] is provided by default to shape parametrization expressions, check if it is really used
    shape_parametrization_expression_on_subdomain = expression.shape_parametrization_expression_on_subdomain_no_upcast
    shape_parametrization_expression_on_subdomain = shape_parametrization_expression_on_subdomain._cppcode
    for component_expression in shape_parametrization_expression_on_subdomain:
        assert isinstance(component_expression, str)
        if len(is_pull_back_expression_parametrized.regex.findall(component_expression)) > 0:
            return True
    # Otherwise, the expression is not parametrized
    return False


is_pull_back_expression_parametrized.regex = re.compile(r"\bmu\[[0-9]+\]")


def is_pull_back_expression_time_dependent(expression):
    parameters = expression.f_no_upcast._parameters
    return "t" in parameters


def PushForwardToDeformedDomain(problem, expression):
    assert isinstance(expression, Expression), "Other expression types are not handled yet"
    expression._is_push_forward = True
    return expression


def is_push_forward_expression(expression):
    if hasattr(expression, "_is_push_forward"):
        assert expression._is_push_forward is True
        return True
    else:
        return False


def is_space_dependent_coefficient(expression, multiindex=None):
    assert isinstance(expression, (CompiledExpression, Expression)), "Other expression types are not handled yet"
    if isinstance(expression, Expression):
        expression_cppcode = expression._cppcode
        if multiindex is not None:
            for index in multiindex.indices():
                assert isinstance(index, FixedIndex)
                expression_cppcode = expression_cppcode[int(index)]
        if isinstance(expression_cppcode, tuple):
            expression_cppcode = " ".join(expression_cppcode)
        return len(is_space_dependent_coefficient._regex.findall(expression_cppcode)) > 0
    elif isinstance(expression, CompiledExpression):
        assert is_pull_back_expression(expression), "Only the case of pulled back expressions is currently handled"
        return True


is_space_dependent_coefficient._regex = re.compile(r"\bx\[[0-9]+\]")


class PullBackExpressions(MultiFunction):
    def __init__(self, shape_parametrization_expression_on_subdomain, problem):
        MultiFunction.__init__(self)
        self.shape_parametrization_expression_on_subdomain = shape_parametrization_expression_on_subdomain
        self.problem = problem

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, o):
        if isinstance(o, BaseExpression):
            assert isinstance(o, Expression), "Other expression types are not handled yet"
            if is_push_forward_expression(o) or not is_space_dependent_coefficient(o):
                return o
            else:
                return PullBackExpression(self.shape_parametrization_expression_on_subdomain, o, self.problem)
        else:
            return o


def pull_back_expressions(shape_parametrization_expression_on_subdomain, problem, integrand):
    return map_expr_dag(PullBackExpressions(shape_parametrization_expression_on_subdomain, problem), integrand)


# ===== Pull back form: inspired by ufl/algorithms/change_to_reference.py ===== #
def pull_back_form(shape_parametrization_expression, problem, form):
    # This function is inspired by change_integrand_geometry_representation in the aforementioned file
    pulled_back_form = 0
    for (shape_parametrization_expression_id, shape_parametrization_expression_on_subdomain) in enumerate(
            shape_parametrization_expression):
        subdomain_id = shape_parametrization_expression_id + 1
        for integral in form.integrals():
            integral_subdomain_or_facet_id = integral.subdomain_id()
            assert isinstance(integral_subdomain_or_facet_id, int) or integral_subdomain_or_facet_id == "everywhere"
            integral_type = integral.integral_type()
            assert (integral_type == "cell" or integral_type.startswith("exterior_facet")
                    or integral_type.startswith("interior_facet"))
            if integral_type == "cell":
                if integral_subdomain_or_facet_id == "everywhere":
                    measure_subdomain_id = subdomain_id
                elif integral_subdomain_or_facet_id != subdomain_id:
                    continue
                else:
                    measure_subdomain_id = integral_subdomain_or_facet_id
            elif integral_type.startswith("exterior_facet") or integral_type.startswith("interior_facet"):
                if integral_subdomain_or_facet_id == "everywhere":
                    measure_subdomain_id = problem._subdomain_id_to_facet_ids[subdomain_id]
                elif subdomain_id not in problem._facet_id_to_subdomain_ids[integral_subdomain_or_facet_id]:
                    continue
                else:
                    measure_subdomain_id = integral_subdomain_or_facet_id
            else:
                raise ValueError("Unhandled integral type.")
            # Carry out pull back, if loop was not continue-d
            integrand = integral.integrand()
            integrand = pull_back_expressions(shape_parametrization_expression_on_subdomain, problem, integrand)
            integrand = pull_back_geometric_quantities(
                shape_parametrization_expression_on_subdomain, problem, integrand)
            integrand = pull_back_gradients(shape_parametrization_expression_on_subdomain, problem, integrand)
            (scale, measure) = pull_back_measures(
                shape_parametrization_expression_on_subdomain, problem, integral, measure_subdomain_id)
            pulled_back_form += (integrand * scale) * measure
    return pulled_back_form


# ===== Auxiliary function to collect dict values in parallel ===== #
def _dict_collect(dict1, dict2, datatype):
    dict12 = defaultdict(set)
    for dict_ in (dict1, dict2):
        for (key, value) in dict_.items():
            dict12[key].update(value)
    return dict(dict12)


_dict_collect_op = Op.Create(_dict_collect, commute=True)


# ===== Pull back forms decorator ===== #
def PullBackFormsToReferenceDomainDecoratedProblem(**decorator_kwargs):
    @ProblemDecoratorFor(PullBackFormsToReferenceDomain)
    def PullBackFormsToReferenceDomainDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):
        from rbnics.eim.problems import DEIM, EIM, ExactParametrizedFunctions
        from rbnics.scm.problems import ExactStabilityFactor, SCM
        from rbnics.shape_parametrization.problems import AffineShapeParametrization, ShapeParametrization
        assert (
            all([Algorithm not in ParametrizedDifferentialProblem_DerivedClass.ProblemDecorators
                 for Algorithm in (DEIM, EIM, ExactParametrizedFunctions, ExactStabilityFactor, SCM)])), (
            "DEIM, EIM, ExactParametrizedFunctions, ExactStabilityFactor and SCM should be applied"
            + " above PullBackFormsToReferenceDomain")
        assert (
            any([Algorithm in ParametrizedDifferentialProblem_DerivedClass.ProblemDecorators
                 for Algorithm in (AffineShapeParametrization, ShapeParametrization)])), (
            "PullBackFormsToReferenceDomain should be applied above AffineShapeParametrization"
            + " or ShapeParametrization")

        from rbnics.backends.dolfin import SeparatedParametrizedForm
        from rbnics.shape_parametrization.utils.symbolic import sympy_eval

        @DefineSymbolicParameters
        @PreserveClassName
        class PullBackFormsToReferenceDomainDecoratedProblem_Class(ParametrizedDifferentialProblem_DerivedClass):

            # Default initialization of members
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
                # Storage for pull back
                self._stability_factor_terms_blacklist = [
                    "stability_factor_left_hand_matrix", "stability_factor_right_hand_matrix"]
                self._stability_factor_decorated_assemble_operator = None
                self._stability_factor_decorated_compute_theta = None
                self._pull_back_is_affine = dict()
                self._pulled_back_operators = dict()
                self._pulled_back_theta_factors = dict()
                (self._facet_id_to_subdomain_ids,
                 self._subdomain_id_to_facet_ids) = self._map_facet_id_to_subdomain_id(**kwargs)
                self._facet_id_to_normal_direction_if_straight = self._map_facet_id_to_normal_direction_if_straight(
                    **kwargs)
                self._shape_parametrization_expressions_sympy_to_ufl = dict()
                self._shape_parametrization_expressions_ufl_to_sympy = dict()
                # Customize DEIM, EIM and ExactParametrizedFunctions decorators so that forms are pulled back
                # to the reference domain before applying DEIM, EIM or exact initialization.
                if hasattr(self, "_init_DEIM_approximations"):
                    _original_init_DEIM_approximations = self._init_DEIM_approximations

                    def _custom_init_DEIM_approximations(self_):
                        self_._init_pull_back()
                        _original_init_DEIM_approximations()

                    PatchInstanceMethod(self, "_init_DEIM_approximations", _custom_init_DEIM_approximations).patch()
                if hasattr(self, "_init_EIM_approximations"):
                    _original_init_EIM_approximations = self._init_EIM_approximations

                    def _custom_init_EIM_approximations(self_):
                        self_._init_pull_back()
                        _original_init_EIM_approximations()

                    PatchInstanceMethod(self, "_init_EIM_approximations", _custom_init_EIM_approximations).patch()
                if hasattr(self, "_init_operators_exact"):
                    _original_init_operators_exact = self._init_operators_exact

                    def _custom_init_operators_exact(self_):
                        self_._init_pull_back()
                        _original_init_operators_exact()

                    PatchInstanceMethod(self, "_init_operators_exact", _custom_init_operators_exact).patch()

            def _init_pull_back(self):
                # Temporarily replace float parameters with symbols, so that we can detect if operators
                # are parametrized
                self.attach_symbolic_parameters()
                # Initialize pull back forms
                terms_except_stability_factor_blacklist = [
                    term for term in self.terms if term not in self._stability_factor_terms_blacklist]
                initialized_terms = list()
                for term in terms_except_stability_factor_blacklist:
                    assert (term in self._pulled_back_operators) is (term in self._pulled_back_theta_factors)
                    if term not in self._pulled_back_operators:  # initialize only once
                        try:
                            forms = ParametrizedDifferentialProblem_DerivedClass.assemble_operator(self, term)
                        except ValueError:  # raised by assemble_operator if output computation is optional
                            pass
                        else:
                            # skip trivial case associated to scalar outputs
                            if not all(isinstance(form, numbers.Number) for form in forms):
                                # Pull back forms
                                # TODO support forms in the complex field
                                forms = [remove_complex_nodes(form) for form in forms]
                                pulled_back_forms = [pull_back_form(self.shape_parametrization_expression, self, form)
                                                     for form in forms]
                                # Preprocess pulled back forms via SeparatedParametrizedForm for affinity check
                                separated_pulled_back_forms = dict()
                                for (q, pulled_back_form) in enumerate(pulled_back_forms):
                                    pulled_back_form_description = form_description(pulled_back_form)
                                    for (_, v) in pulled_back_form_description.items():
                                        if v.startswith("solution of") or v.startswith("solution_dot of"):
                                            break
                                    else:
                                        # a nonlinear form is obviously non-affine, so skip the (possibly expensive)
                                        # affinity check initialized below
                                        separated_pulled_back_form = SeparatedParametrizedForm(
                                            expand(pulled_back_form), strict=True)
                                        separated_pulled_back_form.separate()
                                        separated_pulled_back_forms[q] = separated_pulled_back_form
                                # Check if the dependence is affine on the parameters.
                                # If so, move parameter dependent coefficients to compute_theta
                                pull_back_is_affine = list()
                                postprocessed_pulled_back_forms = list()
                                postprocessed_pulled_back_theta_factors = list()
                                for (q, pulled_back_form) in enumerate(pulled_back_forms):
                                    if (q in separated_pulled_back_forms
                                            and self._is_affine_parameter_dependent(separated_pulled_back_forms[q])):
                                        postprocessed_pulled_back_forms.append(
                                            self._get_affine_parameter_dependent_forms(
                                                separated_pulled_back_forms[q]))
                                        postprocessed_pulled_back_theta_factors.append(
                                            self._get_affine_parameter_dependent_theta_factors(
                                                separated_pulled_back_forms[q]))
                                        assert len(postprocessed_pulled_back_forms) == q + 1
                                        assert len(postprocessed_pulled_back_theta_factors) == q + 1
                                        (postprocessed_pulled_back_forms[q],
                                         postprocessed_pulled_back_theta_factors[q]) = (
                                            collect_common_forms_theta_factors(
                                                postprocessed_pulled_back_forms[q],
                                                postprocessed_pulled_back_theta_factors[q]))
                                        pull_back_is_affine.append(
                                            (True, ) * len(postprocessed_pulled_back_forms[q]))
                                    else:
                                        assert (
                                            any([Algorithm in self.ProblemDecorators
                                                 for Algorithm in (DEIM, EIM, ExactParametrizedFunctions)])), (
                                            "Non affine parametric dependence detected. Please use one"
                                            + " among DEIM, EIM and ExactParametrizedFunctions")
                                        postprocessed_pulled_back_forms.append((pulled_back_form, ))
                                        postprocessed_pulled_back_theta_factors.append((1, ))
                                        pull_back_is_affine.append((False, ))
                                # Store resulting pulled back forms and theta factors
                                initialized_terms.append(term)
                                self._pull_back_is_affine[term] = pull_back_is_affine
                                self._pulled_back_operators[term] = postprocessed_pulled_back_forms
                                self._pulled_back_theta_factors[term] = postprocessed_pulled_back_theta_factors
                # Restore float parameters
                self.detach_symbolic_parameters()
                # Re-apply stability factors decorators (if required, i.e. if @ExactStabilityFactor or @SCM
                # was declared after @PullBackFormsToReferenceDomain) to make sure that pulled back operators
                # (rather than original ones) are employed when defining lhs/rhs of stability factor eigenproblems
                if len(self.terms) > len(terms_except_stability_factor_blacklist):
                    self._stability_factor_decorated_assemble_operator = assemble_operator_for_stability_factor(
                        PullBackFormsToReferenceDomainDecoratedProblem_Class.assemble_operator)
                    self._stability_factor_decorated_compute_theta = compute_theta_for_stability_factor(
                        PullBackFormsToReferenceDomainDecoratedProblem_Class.compute_theta)
                # If debug is enabled, deform the mesh for a few representative values of the parameters to check
                # the the form assembled on the parametrized domain results in the same tensor as the pulled back one
                if logger.isEnabledFor(DEBUG) and len(initialized_terms) > 0:
                    # Init mesh motion object
                    self.mesh_motion.init(self)
                    # Backup current mu
                    mu_bak = self.mu
                    # Loop over new terms
                    for term in initialized_terms:
                        # Check pull back over all corners of parametric domain
                        for mu in itertools.product(*self.mu_range):
                            self.set_mu(mu)
                            # Assemble from pulled back forms
                            thetas_pull_back = self.compute_theta(term)
                            forms_pull_back = self.assemble_operator(term)
                            tensor_pull_back = tensor_assemble(
                                sum([Constant(theta) * discard_inexact_terms(operator)
                                     for (theta, operator) in zip(thetas_pull_back, forms_pull_back)]))
                            # Assemble from forms on parametrized domain
                            self.mesh_motion.move_mesh()
                            thetas_parametrized_domain = ParametrizedDifferentialProblem_DerivedClass.compute_theta(
                                self, term)
                            forms_parametrized_domain = ParametrizedDifferentialProblem_DerivedClass.assemble_operator(
                                self, term)
                            tensor_parametrized_domain = tensor_assemble(
                                sum([Constant(theta) * discard_inexact_terms(operator)
                                    for (theta, operator) in zip(
                                        thetas_parametrized_domain, forms_parametrized_domain)]))
                            self.mesh_motion.reset_reference()
                            # Log thetas and forms
                            logger.log(DEBUG, "=== DEBUGGING PULL BACK FOR TERM " + term
                                       + " AND mu = " + str(mu) + " ===")
                            logger.log(DEBUG, "Thetas on parametrized domain")
                            for (q, theta) in enumerate(thetas_parametrized_domain):
                                logger.log(DEBUG, "\ttheta_" + str(q) + " = " + str(theta))
                            logger.log(DEBUG, "Theta factors for pull back")
                            q = 0
                            for (parametrized_q, pulled_back_theta_factors) in enumerate(
                                    self._pulled_back_theta_factors[term]):
                                for pulled_back_theta_factor in pulled_back_theta_factors:
                                    logger.log(DEBUG, "\ttheta_factor_" + str(q) + " = "
                                               + str(pulled_back_theta_factor) + " (evals to "
                                               + str(sympy_eval(str(pulled_back_theta_factor), {"mu": mu}))
                                               + ") associated to theta_" + str(parametrized_q))
                                    q += 1
                            logger.log(DEBUG, "Pulled back thetas")
                            for (q, theta) in enumerate(thetas_pull_back):
                                logger.log(DEBUG, "\tpulled_back_theta_" + str(q) + " = " + str(theta))
                            logger.log(DEBUG, "Operators on parametrized domain")
                            for (q, form) in enumerate(forms_parametrized_domain):
                                expanded_form = expand_derivatives(form)
                                description = form_description(expanded_form)
                                logger.log(DEBUG, "\toperator_" + str(q) + " = " + str(expanded_form) + ", where "
                                           + ", ".join(str(k) + ": " + str(v) for (k, v) in description.items()))
                            logger.log(DEBUG, "Affinity of pulled back operators")
                            q = 0
                            for pull_back_is_affine in self._pull_back_is_affine[term]:
                                for pull_back_is_affine_q in pull_back_is_affine:
                                    logger.log(DEBUG, "\tis_affine(pulled_back_operator_" + str(q) + ") = "
                                               + str(pull_back_is_affine_q))
                                    q += 1
                            logger.log(DEBUG, "Pulled back operators")
                            for (q, form) in enumerate(forms_pull_back):
                                expanded_form = expand_derivatives(form)
                                description = form_description(expanded_form)
                                logger.log(DEBUG, "\tpulled_back_operator_" + str(q) + " = "
                                           + str(expanded_form) + ", where "
                                           + ", ".join(str(k) + ": " + str(v) for (k, v) in description.items()))
                            # Assert
                            assert tensors_are_close(tensor_pull_back, tensor_parametrized_domain)
                    # Restore mu
                    self.set_mu(mu_bak)

            def _init_operators(self):
                self._init_pull_back()
                ParametrizedDifferentialProblem_DerivedClass._init_operators(self)

            def assemble_operator(self, term):
                if term in self._pulled_back_operators:
                    return tuple([pulled_back_operator
                                  for pulled_back_operators in self._pulled_back_operators[term]
                                  for pulled_back_operator in pulled_back_operators])
                elif term in self._stability_factor_terms_blacklist:
                    return self._stability_factor_decorated_assemble_operator(self, term)
                else:
                    return ParametrizedDifferentialProblem_DerivedClass.assemble_operator(self, term)

            def compute_theta(self, term):
                if term in self._pulled_back_theta_factors:
                    thetas = ParametrizedDifferentialProblem_DerivedClass.compute_theta(self, term)
                    return tuple([sympy_eval(str(pulled_back_theta_factor), {"mu": self.mu}) * thetas[q]
                                  for (q, pulled_back_theta_factors) in enumerate(
                                      self._pulled_back_theta_factors[term])
                                  for pulled_back_theta_factor in pulled_back_theta_factors])
                elif term in self._stability_factor_terms_blacklist:
                    return self._stability_factor_decorated_compute_theta(self, term)
                else:
                    return ParametrizedDifferentialProblem_DerivedClass.compute_theta(self, term)

            def _map_facet_id_to_subdomain_id(self, **kwargs):
                mesh = self.V.mesh()
                mpi_comm = mesh.mpi_comm()
                assert "subdomains" in kwargs
                subdomains = kwargs["subdomains"]
                assert "boundaries" in kwargs
                boundaries = kwargs["boundaries"]
                # Loop over local part of the mesh
                facet_id_to_subdomain_ids = defaultdict(set)
                subdomain_id_to_facet_ids = defaultdict(set)
                for f in facets(mesh):
                    if boundaries[f] > 0:  # skip unmarked facets
                        for c in cells(f):
                            facet_id_to_subdomain_ids[boundaries[f]].add(subdomains[c])
                            subdomain_id_to_facet_ids[subdomains[c]].add(boundaries[f])
                facet_id_to_subdomain_ids = dict(facet_id_to_subdomain_ids)
                subdomain_id_to_facet_ids = dict(subdomain_id_to_facet_ids)
                # Collect in parallel
                facet_id_to_subdomain_ids = mpi_comm.allreduce(facet_id_to_subdomain_ids, op=_dict_collect_op)
                subdomain_id_to_facet_ids = mpi_comm.allreduce(subdomain_id_to_facet_ids, op=_dict_collect_op)
                # Return
                return (facet_id_to_subdomain_ids, subdomain_id_to_facet_ids)

            def _map_facet_id_to_normal_direction_if_straight(self, **kwargs):
                # Auxiliary pybind11 wrapper
                cpp_code = """
                    #include <pybind11/pybind11.h>
                    #include <dolfin/mesh/MeshEntity.h>

                    std::size_t local_facet_index(std::shared_ptr<dolfin::MeshEntity> cell,
                                                  std::shared_ptr<dolfin::MeshEntity> facet)
                    {
                        return cell->index(*facet);
                    }

                    PYBIND11_MODULE(SIGNATURE, m)
                    {
                        m.def("local_facet_index", &local_facet_index);
                    }
                """
                local_facet_index = compile_cpp_code(cpp_code).local_facet_index
                # Process input arguments
                mesh = self.V.mesh()
                dim = mesh.topology().dim()
                mpi_comm = mesh.mpi_comm()
                assert "subdomains" in kwargs
                subdomains = kwargs["subdomains"]
                assert "boundaries" in kwargs
                boundaries = kwargs["boundaries"]
                # Loop over local part of the mesh
                facet_id_to_normal_directions = defaultdict(set)
                for f in facets(mesh):
                    if boundaries[f] > 0:  # skip unmarked facets
                        if f.exterior():
                            facet_id_to_normal_directions[boundaries[f]].add(
                                tuple([f.normal()[d] for d in range(dim)]))
                        else:
                            cell_id_to_subdomain_id = dict()
                            for (c_id, c) in enumerate(cells(f)):
                                cell_id_to_subdomain_id[c_id] = subdomains[c]
                            assert len(cell_id_to_subdomain_id) == 2
                            assert cell_id_to_subdomain_id[0] != cell_id_to_subdomain_id[1]
                            cell_id_to_restricted_sign = dict()
                            if cell_id_to_subdomain_id[0] > cell_id_to_subdomain_id[1]:
                                cell_id_to_restricted_sign[0] = "+"
                                cell_id_to_restricted_sign[1] = "-"
                            else:
                                cell_id_to_restricted_sign[0] = "-"
                                cell_id_to_restricted_sign[1] = "+"
                            for (c_id, c) in enumerate(cells(f)):
                                facet_id_to_normal_directions[
                                    (boundaries[f], cell_id_to_restricted_sign[c_id])].add(
                                        tuple([c.normal(local_facet_index(c, f))[d] for d in range(dim)]))
                facet_id_to_normal_directions = dict(facet_id_to_normal_directions)
                # Collect in parallel
                facet_id_to_normal_directions = mpi_comm.allreduce(facet_id_to_normal_directions, op=_dict_collect_op)
                # Remove curved facets
                facet_id_to_normal_direction_if_straight = dict()
                for (facet_id, normal_directions) in facet_id_to_normal_directions.items():
                    normal_direction = normal_directions.pop()
                    for other_normal_direction in normal_directions:
                        if not allclose(other_normal_direction, normal_direction):
                            facet_id_to_normal_direction_if_straight[facet_id] = None
                            break
                    else:
                        facet_id_to_normal_direction_if_straight[facet_id] = normal_direction
                # Return
                return facet_id_to_normal_direction_if_straight

            def _is_affine_parameter_dependent(self, separated_pulled_back_form):
                # The pulled back form is not affine if any of its coefficients depend on x
                for addend in separated_pulled_back_form.coefficients:
                    for factor in addend:
                        assert factor.ufl_shape == ()
                        for node in pre_traversal(factor):
                            if isinstance(node, Indexed):
                                operand_0 = node.ufl_operands[0]
                                if isinstance(operand_0, BaseExpression):
                                    operand_1 = node.ufl_operands[1]
                                    if is_space_dependent_coefficient(operand_0, operand_1):
                                        return False
                            elif (isinstance(node, BaseExpression)  # expressions with multiple components
                                  and node.ufl_shape == ()):        # are visited by Indexed
                                if is_space_dependent_coefficient(node):
                                    return False
                # The pulled back form is not affine if it contains a boundary integral on a non-straight boundary,
                # because the normal direction would depend on x
                for form_with_placeholder in separated_pulled_back_form._form_with_placeholders:
                    assert len(form_with_placeholder.integrals()) == 1
                    integral = form_with_placeholder.integrals()[0]
                    integral_type = integral.integral_type()
                    integral_subdomain_id = integral.subdomain_id()
                    if integral_type == "cell":
                        pass
                    elif integral_type.startswith("exterior_facet"):
                        if self._facet_id_to_normal_direction_if_straight[integral_subdomain_id] is None:
                            return False
                    elif integral_type.startswith("interior_facet"):
                        if (self._facet_id_to_normal_direction_if_straight[(integral_subdomain_id, "+")] is None
                                or self._facet_id_to_normal_direction_if_straight[(integral_subdomain_id, "-")]
                                is None):
                            return False
                    else:
                        raise ValueError(
                            "Unknown integral type {}, don't know how to check for affinity.".format(integral_type))
                # Otherwise, the pulled back form is affine
                return True

            def _get_affine_parameter_dependent_forms(self, separated_pulled_back_form):
                affine_parameter_dependent_forms = list()
                # Append forms which were not originally affinely dependent
                for (index, addend) in enumerate(separated_pulled_back_form.coefficients):
                    affine_parameter_dependent_forms.append(
                        separated_pulled_back_form.replace_placeholders(index, [1] * len(addend)))
                # Append forms which were already affinely dependent
                for unchanged_form in separated_pulled_back_form.unchanged_forms:
                    affine_parameter_dependent_forms.append(unchanged_form)
                # Return
                return tuple(affine_parameter_dependent_forms)

            def _get_affine_parameter_dependent_theta_factors(self, separated_pulled_back_form):
                # Prepare theta factors
                affine_parameter_dependent_theta_factors = list()
                # Append factors corresponding to forms which were not originally affinely dependent
                assert len(separated_pulled_back_form.coefficients) == len(
                    separated_pulled_back_form._placeholders)
                assert len(separated_pulled_back_form.coefficients) == len(
                    separated_pulled_back_form._form_with_placeholders)
                for (addend_coefficient, addend_placeholder, addend_form_with_placeholder) in zip(
                        separated_pulled_back_form.coefficients, separated_pulled_back_form._placeholders,
                        separated_pulled_back_form._form_with_placeholders):
                    affine_parameter_dependent_theta_factors.append(
                        self._compute_affine_parameter_dependent_theta_factor(
                            addend_coefficient, addend_placeholder, addend_form_with_placeholder))
                # Append factors corresponding to forms which were already affinely dependent
                for _ in separated_pulled_back_form.unchanged_forms:
                    affine_parameter_dependent_theta_factors.append(1.)
                # Return
                return tuple(affine_parameter_dependent_theta_factors)

            def _compute_affine_parameter_dependent_theta_factor(
                    self, coefficient, placeholder, form_with_placeholder):
                assert len(form_with_placeholder.integrals()) == 1
                integral = form_with_placeholder.integrals()[0]
                integral_type = integral.integral_type()
                integral_subdomain_id = integral.subdomain_id()
                integrand = integral.integrand()
                # Call UFL replacer
                replacer = ComputeAffineParameterDependentThetaFactorReplacer(coefficient, placeholder)
                theta_factor = apply_transformer(integrand, replacer)
                # Convert to sympy
                locals = dict()
                # ... add parameters
                for (p, mu_p) in enumerate(self.mu):
                    assert isinstance(mu_p, Expression)  # because UFL symbolic parameters were attached
                    locals[str(mu_p)] = symbols("mu[" + str(p) + "]")
                    locals["mu[" + str(p) + "]"] = locals[str(mu_p)]
                # ... add shape parametrization jacobians to locals
                for (shape_parametrization_expression_sympy, shape_parametrization_expression_ufl) in (
                        self._shape_parametrization_expressions_sympy_to_ufl.items()):
                    locals[str(shape_parametrization_expression_ufl)] = shape_parametrization_expression_sympy
                # ... add fake unity constants to locals
                for constant in replacer.constants:
                    assert len(constant.ufl_shape) in (0, 1, 2)
                    if len(constant.ufl_shape) == 0:
                        locals[str(constant)] = float(constant)
                    elif len(constant.ufl_shape) == 1:
                        vals = constant.values()
                        for i in range(constant.ufl_shape[0]):
                            locals[str(constant) + "[" + str(i) + "]"] = vals[i]
                    elif len(constant.ufl_shape) == 2:
                        vals = constant.values()
                        vals = vals.reshape(constant.ufl_shape)
                        for i in range(constant.ufl_shape[0]):
                            for j in range(constant.ufl_shape[1]):
                                locals[str(constant) + "[" + str(i) + ", " + str(j) + "]"] = vals[i, j]
                # ... add normal direction (facet integration only)
                if integral_type == "cell":
                    pass
                elif integral_type.startswith("exterior_facet"):
                    assert self._facet_id_to_normal_direction_if_straight[integral_subdomain_id] is not None
                    locals["n"] = self._facet_id_to_normal_direction_if_straight[integral_subdomain_id]
                elif integral_type.startswith("interior_facet"):
                    assert self._facet_id_to_normal_direction_if_straight[(integral_subdomain_id, "+")] is not None
                    assert self._facet_id_to_normal_direction_if_straight[(integral_subdomain_id, "-")] is not None
                    raise NotImplementedError(
                        "compute_affine_parameter_dependent_theta_factor has not been implemented yet"
                        + " for interior_facet")
                else:
                    raise ValueError(
                        "Unknown integral type {}, don't know how to check for affinity.".format(integral_type))
                # ... carry out conversion
                theta_factor_sympy = theta_factor
                theta_factor_sympy = simplify(sympify(str(theta_factor_sympy), locals=locals))
                theta_factor_sympy = simplify(convert_float_to_int_if_possible(theta_factor_sympy))
                return theta_factor_sympy

        # return value (a class) for the decorator
        return PullBackFormsToReferenceDomainDecoratedProblem_Class

    # return the decorator itself
    return PullBackFormsToReferenceDomainDecoratedProblem_Decorator


PullBackFormsToReferenceDomain = PullBackFormsToReferenceDomainDecoratedProblem


@ReductionMethodDecoratorFor(PullBackFormsToReferenceDomain)
def PullBackFormsToReferenceDomainDecoratedReductionMethod(DifferentialProblemReductionMethod_DerivedClass):
    return DifferentialProblemReductionMethod_DerivedClass


@ReducedProblemDecoratorFor(PullBackFormsToReferenceDomain)
def PullBackFormsToReferenceDomainDecoratedReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    return ParametrizedReducedDifferentialProblem_DerivedClass


# ===== Pull back forms decorator (auxiliary functions and classes) ===== #
def expand(form):
    form = expand_derivatives(form)
    form = expand_sum_product(form)
    form = expand_indices(form)
    return form


class ComputeAffineParameterDependentThetaFactorReplacer(Transformer):
    def __init__(self, coefficient, placeholder):
        Transformer.__init__(self)
        assert len(placeholder) == len(coefficient)
        self.placeholder_to_coefficient = dict(zip(placeholder, coefficient))
        self.contains_placeholder = dict()
        self.constants = list()
        # Append to constants all the constants in coefficient, that may result from SeparatedParametrizedForm
        for c in coefficient:
            for node in traverse_unique_terminals(c):
                if isinstance(node, Constant):
                    self.constants.append(node)

    def operator(self, e, *ops):
        replaced_ops = list()
        contains_placeholder = False
        for o in ops:
            assert o in self.contains_placeholder
            if self.contains_placeholder[o]:
                replaced_ops.append(o)
                contains_placeholder = True
            elif isinstance(o, MultiIndex):
                replaced_ops.append(o)
            else:
                replaced_o = Constant(numpy_ones(o.ufl_shape))
                self.constants.append(replaced_o)
                replaced_ops.append(replaced_o)
        replaced_e = e._ufl_expr_reconstruct_(*replaced_ops)
        self.contains_placeholder[replaced_e] = contains_placeholder
        return replaced_e

    def power(self, e, *ops):
        assert len(ops) == 2
        replaced_ops = list()
        assert ops[0] in self.contains_placeholder
        assert not isinstance(ops[0], MultiIndex)
        if self.contains_placeholder[ops[0]]:
            replaced_ops.append(ops[0])
            contains_placeholder = True
        else:
            replaced_ops0 = Constant(numpy_ones(ops[0].ufl_shape))
            self.constants.append(replaced_ops0)
            replaced_ops.append(replaced_ops0)
            contains_placeholder = False
        assert ops[1] in self.contains_placeholder
        assert not self.contains_placeholder[ops[1]]
        replaced_ops.append(e.ufl_operands[1])  # e.ufl_operands contains the former value of ops[1] before replacement
        replaced_e = e._ufl_expr_reconstruct_(*replaced_ops)
        self.contains_placeholder[replaced_e] = contains_placeholder
        return replaced_e

    def terminal(self, o):
        if o in self.placeholder_to_coefficient:
            assert o.ufl_shape == ()
            replaced_o = self.placeholder_to_coefficient[o]
            assert replaced_o.ufl_shape == ()
            self.contains_placeholder[replaced_o] = True
            return replaced_o
        elif isinstance(o, MultiIndex):
            self.contains_placeholder[o] = False
            return o
        else:
            replaced_o = Constant(numpy_ones(o.ufl_shape))
            self.contains_placeholder[replaced_o] = False
            self.constants.append(replaced_o)
            return replaced_o


def convert_float_to_int_if_possible(theta_factor):
    for node in preorder_traversal(theta_factor):
        if isinstance(node, Float) and node == int(node):
            theta_factor = theta_factor.subs(node, Integer(int(node)))
    return theta_factor


def collect_common_forms_theta_factors(postprocessed_pulled_back_forms, postprocessed_pulled_back_theta_factors):
    from rbnics.shape_parametrization.utils.symbolic import sympy_eval
    # Remove all zero theta factors
    postprocessed_pulled_back_forms_non_zero = list()
    postprocessed_pulled_back_theta_factors_non_zero = list()
    assert len(postprocessed_pulled_back_forms) == len(postprocessed_pulled_back_theta_factors)
    for (postprocessed_pulled_back_form, postprocessed_pulled_back_theta_factor) in zip(
            postprocessed_pulled_back_forms, postprocessed_pulled_back_theta_factors):
        if postprocessed_pulled_back_theta_factor != 0:
            postprocessed_pulled_back_forms_non_zero.append(postprocessed_pulled_back_form)
            postprocessed_pulled_back_theta_factors_non_zero.append(postprocessed_pulled_back_theta_factor)
    # Convert forms to sympy symbols
    postprocessed_pulled_back_forms_ufl_to_sympy = dict()
    postprocessed_pulled_back_forms_sympy_id_to_ufl = dict()
    for postprocessed_pulled_back_form in postprocessed_pulled_back_forms_non_zero:
        if postprocessed_pulled_back_form not in postprocessed_pulled_back_forms_ufl_to_sympy:
            sympy_id = "sympyform" + str(len(postprocessed_pulled_back_forms_sympy_id_to_ufl))
            postprocessed_pulled_back_form_sympy = symbols(sympy_id)
            postprocessed_pulled_back_forms_ufl_to_sympy[
                postprocessed_pulled_back_form] = postprocessed_pulled_back_form_sympy
            postprocessed_pulled_back_forms_sympy_id_to_ufl[sympy_id] = postprocessed_pulled_back_form
    # Convert theta factors to sympy symbols
    postprocessed_pulled_back_theta_factors_sympy_independents = list()
    postprocessed_pulled_back_theta_factors_ufl_to_sympy = dict()
    postprocessed_pulled_back_theta_factors_sympy_id_to_ufl = dict()
    for postprocessed_pulled_back_theta_factor in postprocessed_pulled_back_theta_factors_non_zero:
        if postprocessed_pulled_back_theta_factor not in postprocessed_pulled_back_theta_factors_ufl_to_sympy:
            for (previous_theta_factor_ufl, previous_theta_factor_sympy) in (
                    postprocessed_pulled_back_theta_factors_ufl_to_sympy.items()):
                ratio = simplify(postprocessed_pulled_back_theta_factor / previous_theta_factor_ufl)
                if isinstance(ratio, Number):
                    postprocessed_pulled_back_theta_factors_ufl_to_sympy[
                        postprocessed_pulled_back_theta_factor] = ratio * previous_theta_factor_sympy
                    break
            else:
                sympy_id = "sympytheta" + str(len(postprocessed_pulled_back_theta_factors_sympy_id_to_ufl))
                postprocessed_pulled_back_theta_factor_sympy = symbols(sympy_id)
                postprocessed_pulled_back_theta_factors_sympy_independents.append(
                    postprocessed_pulled_back_theta_factor_sympy)
                postprocessed_pulled_back_theta_factors_ufl_to_sympy[
                    postprocessed_pulled_back_theta_factor] = postprocessed_pulled_back_theta_factor_sympy
                postprocessed_pulled_back_theta_factors_sympy_id_to_ufl[
                    sympy_id] = postprocessed_pulled_back_theta_factor
    # Carry out symbolic sum(product())
    postprocessed_pulled_back_sum_product = 0
    for (postprocessed_pulled_back_form, postprocessed_pulled_back_theta_factor) in zip(
            postprocessed_pulled_back_forms_non_zero, postprocessed_pulled_back_theta_factors_non_zero):
        postprocessed_pulled_back_sum_product += (
            postprocessed_pulled_back_theta_factors_ufl_to_sympy[postprocessed_pulled_back_theta_factor]
            * postprocessed_pulled_back_forms_ufl_to_sympy[postprocessed_pulled_back_form])
    # Collect first with respect to theta factors
    collected_with_respect_to_theta = collect(
        postprocessed_pulled_back_sum_product, postprocessed_pulled_back_theta_factors_sympy_independents,
        evaluate=False, exact=True)
    collected_with_respect_to_theta_ordered = OrderedDict()
    for postprocessed_pulled_back_theta_factor in postprocessed_pulled_back_theta_factors_sympy_independents:
        assert postprocessed_pulled_back_theta_factor in collected_with_respect_to_theta
        collected_with_respect_to_theta_ordered[
            postprocessed_pulled_back_theta_factor] = collected_with_respect_to_theta[
                postprocessed_pulled_back_theta_factor]
    collected_sum_product = 0
    for (collected_theta_factor, collected_form) in collected_with_respect_to_theta_ordered.items():
        collected_sum_product += collected_theta_factor * collected_form
    # Collect then with respect to form factors
    collected_with_respect_to_form = collect(
        collected_sum_product, collected_with_respect_to_theta_ordered.values(), evaluate=False, exact=True)
    collected_with_respect_to_form_ordered = OrderedDict()
    for collected_form in collected_with_respect_to_theta_ordered.values():
        if collected_form in collected_with_respect_to_form:
            collected_with_respect_to_form_ordered[collected_form] = collected_with_respect_to_form[collected_form]
        else:  # it may happen that factors get multiplied by a number during collection
            ratios = tuple(simplify(collected_form / collected_with_respect_to_form_key)
                           for collected_with_respect_to_form_key in collected_with_respect_to_form.keys())
            is_numeric_ratio = tuple(isinstance(r, Number) for r in ratios)
            assert sum(is_numeric_ratio) == 1
            ratio = ratios[is_numeric_ratio.index(True)]
            collected_form = collected_form / ratio
            assert collected_form in collected_with_respect_to_form
            collected_with_respect_to_form_ordered[collected_form] = collected_with_respect_to_form[collected_form]
    # Convert back to ufl
    collected_forms = list()
    collected_theta_factors = list()
    for (collected_form, collected_theta_factor) in collected_with_respect_to_form_ordered.items():
        collected_forms.append(
            sympy_eval(str(collected_form), postprocessed_pulled_back_forms_sympy_id_to_ufl))
        collected_theta_factors.append(
            sympy_eval(str(collected_theta_factor), postprocessed_pulled_back_theta_factors_sympy_id_to_ufl))
    # Return
    return (tuple(collected_forms), tuple(collected_theta_factors))


def forms_are_close(form_1, form_2):
    return tensors_are_close(tensor_assemble(form_1), tensor_assemble(form_2))


@overload
def tensors_are_close(tensor_1: GenericMatrix, tensor_2: GenericMatrix):
    return isclose(tensor_1.norm("frobenius"), tensor_2.norm("frobenius"))


@overload
def tensors_are_close(tensor_1: GenericVector, tensor_2: GenericVector):
    return isclose(tensor_1.norm("l2"), tensor_2.norm("l2"))


# use float and not generic Number because sympy is tweaking the float object resulting in
# isinstance(5., Number) being false
@overload
def tensors_are_close(tensor_1: float, tensor_2: float):
    return isclose(tensor_1, tensor_2)


def tensor_assemble(form):
    assert isinstance(form, (Constant, Form, Number))
    if isinstance(form, Constant):
        assert form.ufl_shape == ()
        return float(form)
    elif isinstance(form, Number):
        return form
    elif isinstance(form, Form):
        return assemble(form)


class DiscardInexactTermsReplacer(MultiFunction):
    expr = MultiFunction.reuse_if_untouched

    @memoized_handler
    def circumradius(self, o):
        return Constant(0.)

    @memoized_handler
    def cell_diameter(self, o):
        return Constant(0.)


def discard_inexact_terms(form):
    return map_integrand_dags(DiscardInexactTermsReplacer(), form)
