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

from collections import defaultdict
import re
import types
import math
from numpy import allclose, ones as numpy_ones, zeros as numpy_zeros
from mpi4py.MPI import Op
from sympy import ccode, collect, expand_mul, Float, Integer, MatrixSymbol, Number, preorder_traversal, simplify, symbols, sympify, zeros as sympy_zeros
from ufl import as_tensor, det, Form, inv, Measure, sqrt, TensorElement, tr, transpose, VectorElement
from ufl.algorithms import apply_transformer, Transformer
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.expand_indices import expand_indices, purge_list_tensors
from ufl.classes import FacetJacobian, FacetJacobianDeterminant, FacetNormal, Grad, Sum
from ufl.compound_expressions import determinant_expr, inverse_expr
from ufl.core.multiindex import FixedIndex, Index, indices, MultiIndex
from ufl.corealg.multifunction import memoized_handler, MultiFunction
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.traversal import pre_traversal, traverse_unique_terminals
from ufl.indexed import Indexed
from dolfin import cells, Constant, Expression, facets
import rbnics.backends.dolfin.wrapping.form_mul # enable form multiplication and division
from rbnics.backends.dolfin.wrapping.parametrized_expression import ParametrizedExpression
from rbnics.utils.decorators import PreserveClassName, ProblemDecoratorFor

# ===== Memoization for shape parametrization objects: inspired by ufl/corealg/multifunction.py ===== #
def shape_parametrization_cache(function):
    function._cache = dict()
    def _memoized_function(shape_parametrization_expression_on_subdomain, problem, domain):
        cache = getattr(function, "_cache")
        output = cache.get((shape_parametrization_expression_on_subdomain, problem, domain))
        if output is None:
            output = function(shape_parametrization_expression_on_subdomain, problem, domain)
            cache[shape_parametrization_expression_on_subdomain, problem, domain] = output
        return output
    return _memoized_function

# ===== Shape parametrization classes related to jacobian, inspired by ufl/geometry.py ===== #
@shape_parametrization_cache
def ShapeParametrizationMap(shape_parametrization_expression_on_subdomain, problem, domain):
    from rbnics.shape_parametrization.utils.symbolic import strings_to_number_of_parameters, sympy_symbolic_coordinates
    mu = (0, )*strings_to_number_of_parameters(shape_parametrization_expression_on_subdomain)
    x_symb = sympy_symbolic_coordinates(problem.V.mesh().geometry().dim(), MatrixSymbol)
    mu_symb = MatrixSymbol("mu", len(mu), 1)
    shape_parametrization_expression_on_subdomain_cpp = list()
    for shape_parametrization_component_on_subdomain in shape_parametrization_expression_on_subdomain:
        shape_parametrization_component_on_subdomain_cpp = sympify(shape_parametrization_component_on_subdomain, locals={"x": x_symb, "mu": mu_symb})
        shape_parametrization_expression_on_subdomain_cpp.append(
            ccode(shape_parametrization_component_on_subdomain_cpp).replace(", 0]", "]"),
        )
    element = VectorElement("CG", domain.ufl_cell(), 1)
    shape_parametrization_map = ParametrizedExpression(problem, tuple(shape_parametrization_expression_on_subdomain_cpp), mu=mu, element=element)
    shape_parametrization_map.set_mu(problem.mu)
    return shape_parametrization_map

@shape_parametrization_cache
def ShapeParametrizationJacobian(shape_parametrization_expression_on_subdomain, problem, domain):
    from rbnics.shape_parametrization.utils.symbolic import compute_shape_parametrization_gradient, strings_to_number_of_parameters
    shape_parametrization_gradient_on_subdomain = compute_shape_parametrization_gradient(shape_parametrization_expression_on_subdomain)
    mu = (0, )*strings_to_number_of_parameters(shape_parametrization_expression_on_subdomain)
    element = TensorElement("CG", domain.ufl_cell(), 1)
    shape_parametrization_jacobian = ParametrizedExpression(problem, shape_parametrization_gradient_on_subdomain, mu=mu, element=element) # no need to convert expression to cpp, this is done already by compute_shape_parametrization_gradient()
    shape_parametrization_jacobian.set_mu(problem.mu)
    return shape_parametrization_jacobian
    
@shape_parametrization_cache
def ShapeParametrizationJacobianSympy(shape_parametrization_expression_on_subdomain, problem, domain):
    from rbnics.shape_parametrization.utils.symbolic import strings_to_number_of_parameters, sympy_symbolic_coordinates
    dim = problem.V.mesh().geometry().dim()
    x_symb = sympy_symbolic_coordinates(dim, MatrixSymbol)
    mu_symb = MatrixSymbol("mu", strings_to_number_of_parameters(shape_parametrization_expression_on_subdomain), 1)
    shape_parametrization_jacobian = ShapeParametrizationJacobian(shape_parametrization_expression_on_subdomain, problem, domain).cppcode
    shape_parametrization_jacobian_sympy = sympy_zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            shape_parametrization_jacobian_sympy[i, j] = sympify(shape_parametrization_jacobian[i][j], locals={"x": x_symb, "mu": mu_symb})
    return shape_parametrization_jacobian_sympy

@shape_parametrization_cache
def ShapeParametrizationJacobianInverse(shape_parametrization_expression_on_subdomain, problem, domain):
    return inv(ShapeParametrizationJacobian(shape_parametrization_expression_on_subdomain, problem, domain))
    
@shape_parametrization_cache
def ShapeParametrizationJacobianInverseTranspose(shape_parametrization_expression_on_subdomain, problem, domain):
    return transpose(ShapeParametrizationJacobianInverse(shape_parametrization_expression_on_subdomain, problem, domain))
    
@shape_parametrization_cache
def ShapeParametrizationJacobianDeterminant(shape_parametrization_expression_on_subdomain, problem, domain):
    return det(ShapeParametrizationJacobian(shape_parametrization_expression_on_subdomain, problem, domain))
    
@shape_parametrization_cache
def ShapeParametrizationFacetJacobian(shape_parametrization_expression_on_subdomain, problem, domain):
    return ShapeParametrizationJacobian(shape_parametrization_expression_on_subdomain, problem, domain)*FacetJacobian(domain)

@shape_parametrization_cache
def ShapeParametrizationFacetJacobianInverse(shape_parametrization_expression_on_subdomain, problem, domain):
    return inverse_expr(ShapeParametrizationFacetJacobian(shape_parametrization_expression_on_subdomain, problem, domain))
    
@shape_parametrization_cache
def ShapeParametrizationFacetJacobianDeterminant(shape_parametrization_expression_on_subdomain, problem, domain):
    nanson = ShapeParametrizationJacobianDeterminant(shape_parametrization_expression_on_subdomain, problem, domain)*ShapeParametrizationJacobianInverseTranspose(shape_parametrization_expression_on_subdomain, problem, domain)*FacetNormal(domain)
    i = Index()
    return sqrt(nanson[i]*nanson[i])
    
# ===== Pull back form measures: inspired by ufl/algorithms/apply_integral_scaling.py ===== #
def pull_back_measures(shape_parametrization_expression_on_subdomain, problem, integral, subdomain_id): # inspired by compute_integrand_scaling_factor
    integral_type = integral.integral_type()
    tdim = integral.ufl_domain().topological_dimension()
    
    if integral_type == "cell":
        scale = ShapeParametrizationJacobianDeterminant(shape_parametrization_expression_on_subdomain, problem, integral.ufl_domain())
    elif integral_type.startswith("exterior_facet") or integral_type.startswith("interior_facet"):
        if tdim > 1:
            # Scaling integral by facet jacobian determinant
            scale = ShapeParametrizationFacetJacobianDeterminant(shape_parametrization_expression_on_subdomain, problem, integral.ufl_domain())
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
        metadata=integral.metadata()
    )
    return (scale, measure)
        
# ===== Pull back form gradients: inspired by ufl/algorithms/change_to_reference.py ===== #
class PullBackGradients(MultiFunction): # inspired by OLDChangeToReferenceGrad
    def __init__(self, shape_parametrization_expression_on_subdomain, problem):
        MultiFunction.__init__(self)
        self.shape_parametrization_expression_on_subdomain = shape_parametrization_expression_on_subdomain
        self.problem = problem

    expr = MultiFunction.reuse_if_untouched
    
    def div(self, o, f):
        # Create shape parametrization Jacobian inverse object
        Jinv = ShapeParametrizationJacobianInverse(self.shape_parametrization_expression_on_subdomain, self.problem, f.ufl_domain())
        
        # Indices to get to the scalar component of f
        first = indices(len(f.ufl_shape) - 1)
        last = Index()
        j = Index()
        
        # Wrap back in tensor shape
        return as_tensor(Jinv[j, last]*Grad(f)[first + (last, j)], first)

    def grad(self, _, f):
        # Create shape parametrization Jacobian inverse object
        Jinv = ShapeParametrizationJacobianInverse(self.shape_parametrization_expression_on_subdomain, self.problem, f.ufl_domain())
        
        # Indices to get to the scalar component of f
        f_indices = indices(len(f.ufl_shape))
        j, k = indices(2)
        
        # Wrap back in tensor shape, derivative axes at the end
        return as_tensor(Jinv[j, k]*Grad(f)[f_indices + (j,)], f_indices + (k,))
        
    def reference_div(self, o):
        raise ValueError("Not expecting reference div.")

    def reference_grad(self, o):
        raise ValueError("Not expecting reference grad.")

    def coefficient_derivative(self, o):
        raise ValueError("Coefficient derivatives should be expanded before applying this.")
        
def pull_back_gradients(shape_parametrization_expression_on_subdomain, problem, integrand): # inspired by change_to_reference_grad
    return map_expr_dag(PullBackGradients(shape_parametrization_expression_on_subdomain, problem), integrand)
    
# ===== Pull back geometric quantities: inspired by ufl/algorithms/apply_geometry_lowering.py ===== #
class PullBackGeometricQuantities(MultiFunction): # inspired by GeometryLoweringApplier
    def __init__(self, shape_parametrization_expression_on_subdomain, problem):
        MultiFunction.__init__(self)
        self.shape_parametrization_expression_on_subdomain = shape_parametrization_expression_on_subdomain
        self.problem = problem

    expr = MultiFunction.reuse_if_untouched
    
    def _not_implemented(self, o):
        raise NotImplementedError("Pull back of this geometric quantity has not been implemented")
    
    @memoized_handler
    def jacobian(self, o):
        return ShapeParametrizationJacobian(self.shape_parametrization_expression_on_subdomain, self.problem, o.ufl_domain())
        
    @memoized_handler
    def jacobian_inverse(self, o):
        return ShapeParametrizationJacobianInverse(self.shape_parametrization_expression_on_subdomain, self.problem, o.ufl_domain())
        
    @memoized_handler
    def jacobian_determinant(self, o):
        return ShapeParametrizationJacobianDeterminant(self.shape_parametrization_expression_on_subdomain, self.problem, o.ufl_domain())
        
    @memoized_handler
    def facet_jacobian(self, o):
        return ShapeParametrizationFacetJacobian(self.shape_parametrization_expression_on_subdomain, self.problem, o.ufl_domain())
        
    @memoized_handler
    def facet_jacobian_inverse(self, o):
        return ShapeParametrizationFacetJacobianInverse(self.shape_parametrization_expression_on_subdomain, self.problem, o.ufl_domain())
        
    @memoized_handler
    def facet_jacobian_determinant(self, o):
        return ShapeParametrizationFacetJacobianDeterminant(self.shape_parametrization_expression_on_subdomain, self.problem, o.ufl_domain())
        
    @memoized_handler
    def spatial_coordinate(self, o):
        return ShapeParametrizationMap(self.shape_parametrization_expression_on_subdomain, self.problem, o.ufl_domain())
        
    @memoized_handler
    def cell_volume(self, o):
        return self.jacobian_determinant(o)*CellVolume(o.ufl_domain())
        
    @memoized_handler
    def facet_area(self, o):
        return self.facet_jacobian_determinant(o)*FacetArea(o.ufl_domain())
        
    circumradius = _not_implemented
    min_cell_edge_length = _not_implemented
    max_cell_edge_length = _not_implemented
    min_facet_edge_length = _not_implemented
    max_facet_edge_length = _not_implemented
    
    cell_normal = _not_implemented
    facet_normal = _not_implemented
        
def pull_back_geometric_quantities(shape_parametrization_expression_on_subdomain, problem, integrand): # inspired by apply_geometry_lowering
    return map_expr_dag(PullBackGeometricQuantities(shape_parametrization_expression_on_subdomain, problem), integrand)

# ===== Pull back expressions to reference domain: inspired by ufl/algorithms/apply_function_pullbacks.py ===== #
pull_back_expression_code = """
    class PullBackExpression : public Expression
    {
    public:
        PullBackExpression() : Expression() {}
        
        std::shared_ptr<Expression> f;
        std::shared_ptr<Expression> shape_parametrization_expression_on_subdomain;

        void eval(Array<double>& values, const Array<double>& x, const ufc::cell& c) const
        {
            Array<double> x_o(x.size());
            shape_parametrization_expression_on_subdomain->eval(x_o, x, c);
            f->eval(values, x_o, c);
        }
    };
"""
def PullBackExpression(shape_parametrization_expression_on_subdomain, f, problem, domain):
    pulled_back_f = Expression(pull_back_expression_code, element=f.ufl_element())
    pulled_back_f.f = f
    pulled_back_f.shape_parametrization_expression_on_subdomain = ShapeParametrizationMap(shape_parametrization_expression_on_subdomain, problem, domain)
    return pulled_back_f

class PullBackExpressions(MultiFunction):
    def __init__(self, shape_parametrization_expression_on_subdomain, problem):
        MultiFunction.__init__(self)
        self.shape_parametrization_expression_on_subdomain = shape_parametrization_expression_on_subdomain
        self.problem = problem
    
    expr = MultiFunction.reuse_if_untouched
    
    def terminal(self, o):
        if isinstance(o, Expression):
            return PullBackExpression(self.shape_parametrization_expression_on_subdomain, o, self.problem, self.problem.V.mesh().ufl_domain())
        else:
            return o

def pull_back_expressions(shape_parametrization_expression_on_subdomain, problem, integrand):
    return map_expr_dag(PullBackExpressions(shape_parametrization_expression_on_subdomain, problem), integrand)

# ===== Pull back form: inspired by ufl/algorithms/change_to_reference.py ===== #
def pull_back_form(shape_parametrization_expression, problem, form): # inspired by change_integrand_geometry_representation
    pulled_back_form = 0
    for (shape_parametrization_expression_id, shape_parametrization_expression_on_subdomain) in enumerate(shape_parametrization_expression):
        subdomain_id = shape_parametrization_expression_id + 1
        for integral in form.integrals():
            integral_subdomain_or_facet_id = integral.subdomain_id()
            assert isinstance(integral_subdomain_or_facet_id, int) or integral_subdomain_or_facet_id == "everywhere"
            integral_type = integral.integral_type()
            assert integral_type == "cell" or integral_type.startswith("exterior_facet") or integral_type.startswith("interior_facet")
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
            integrand = pull_back_geometric_quantities(shape_parametrization_expression_on_subdomain, problem, integrand)
            integrand = pull_back_gradients(shape_parametrization_expression_on_subdomain, problem, integrand)
            (scale, measure) = pull_back_measures(shape_parametrization_expression_on_subdomain, problem, integral, measure_subdomain_id)
            pulled_back_form += (integrand*scale)*measure
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
def PullBackFormsToReferenceDomainDecoratedProblem(*terms_to_pull_back, **decorator_kwargs):
    @ProblemDecoratorFor(PullBackFormsToReferenceDomain, terms_to_pull_back=terms_to_pull_back)
    def PullBackFormsToReferenceDomainDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):
        from rbnics.eim.problems import DEIM, EIM, ExactParametrizedFunctions
        from rbnics.shape_parametrization.problems import AffineShapeParametrization, ShapeParametrization
        assert all([Algorithm not in ParametrizedDifferentialProblem_DerivedClass.ProblemDecorators for Algorithm in (DEIM, EIM, ExactParametrizedFunctions)]), "DEIM, EIM and ExactParametrizedFunctions should be applied after PullBackFormsToReferenceDomain"
        assert any([Algorithm in ParametrizedDifferentialProblem_DerivedClass.ProblemDecorators for Algorithm in (AffineShapeParametrization, ShapeParametrization)]), "PullBackFormsToReferenceDomain should be applied after AffineShapeParametrization or ShapeParametrization"
        
        from rbnics.backends.dolfin import SeparatedParametrizedForm
        from rbnics.shape_parametrization.utils.symbolic import sympy_symbolic_coordinates
        
        @PreserveClassName
        class PullBackFormsToReferenceDomainDecoratedProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
            
            # Default initialization of members
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
                # Storage for pull back
                self._terms_to_pull_back = terms_to_pull_back
                self._pulled_back_operators = dict()
                self._pulled_back_theta_factors = dict()
                (self._facet_id_to_subdomain_ids, self._subdomain_id_to_facet_ids) = self._map_facet_id_to_subdomain_id(**kwargs)
                self._facet_id_to_normal_direction_if_straight = self._map_facet_id_to_normal_direction_if_straight(**kwargs)
                self._is_affine_parameter_dependent_regex = re.compile(r"\bx\[[0-9]+\]")
                self._shape_parametrization_jacobians = [ShapeParametrizationJacobian(shape_parametrization_expression_on_subdomain, self, self.V.mesh().ufl_domain()) for shape_parametrization_expression_on_subdomain in self.shape_parametrization_expression]
                self._shape_parametrization_jacobians_sympy = [ShapeParametrizationJacobianSympy(shape_parametrization_expression_on_subdomain, self, self.V.mesh().ufl_domain()) for shape_parametrization_expression_on_subdomain in self.shape_parametrization_expression]
                
            def init(self):
                for term in self._terms_to_pull_back:
                    assert (term in self._pulled_back_operators) is (term in self._pulled_back_theta_factors)
                    if term not in self._pulled_back_operators: # initialize only once
                        # Pull back forms
                        forms = ParametrizedDifferentialProblem_DerivedClass.assemble_operator(self, term)
                        pulled_back_forms = [pull_back_form(self.shape_parametrization_expression, self, form) for form in forms]
                        # Preprocess pulled back forms via SeparatedParametrizedForm
                        separated_pulled_back_forms = [SeparatedParametrizedForm(expand(pulled_back_form)) for pulled_back_form in pulled_back_forms]
                        for separated_pulled_back_form in separated_pulled_back_forms:
                            separated_pulled_back_form.separate()
                        # Check if the dependence is affine on the parameters. If so, move parameter dependent coefficients to compute_theta
                        postprocessed_pulled_back_forms = list()
                        postprocessed_pulled_back_theta_factors = list()
                        for (q, separated_pulled_back_form) in enumerate(separated_pulled_back_forms):
                            if self._is_affine_parameter_dependent(separated_pulled_back_form):
                                postprocessed_pulled_back_forms.append(self._get_affine_parameter_dependent_forms(separated_pulled_back_form))
                                postprocessed_pulled_back_theta_factors.append(self._get_affine_parameter_dependent_theta_factors(separated_pulled_back_form))
                                assert len(postprocessed_pulled_back_forms) == q + 1
                                assert len(postprocessed_pulled_back_theta_factors) == q + 1
                                (postprocessed_pulled_back_forms[q], postprocessed_pulled_back_theta_factors[q]) = collect_common_forms_theta_factors(postprocessed_pulled_back_forms[q], postprocessed_pulled_back_theta_factors[q])
                            else:
                                assert any([Algorithm in self.ProblemDecorators for Algorithm in (DEIM, EIM, ExactParametrizedFunctions)]), "Non affine parametric dependence detected. Please use one among DEIM, EIM and ExactParametrizedFunctions"
                                postprocessed_pulled_back_forms.append((pulled_back_forms[q], ))
                                postprocessed_pulled_back_theta_factors.append((1, ))
                        # Store resulting pulled back forms and theta factors
                        self._pulled_back_operators[term] = postprocessed_pulled_back_forms
                        self._pulled_back_theta_factors[term] = postprocessed_pulled_back_theta_factors
                        # If debug is enabled, deform the mesh for a few representative values of the parameters to check the the form assembled
                        # on the parametrized domain results in the same tensor as the pulled back one
                        if "debug" in decorator_kwargs and decorator_kwargs["debug"] is True:
                            pass # TODO
                # Call parent
                ParametrizedDifferentialProblem_DerivedClass.init(self)
                
            def assemble_operator(self, term):
                if term in self._terms_to_pull_back:
                    return tuple([pulled_back_operator for pulled_back_operators in self._pulled_back_operators[term] for pulled_back_operator in pulled_back_operators])
                else:
                    return ParametrizedDifferentialProblem_DerivedClass.assemble_operator(self, term)
                    
            def compute_theta(self, term):
                thetas = ParametrizedDifferentialProblem_DerivedClass.compute_theta(self, term)
                if term in self._terms_to_pull_back:
                    return tuple([safe_eval(str(pulled_back_theta_factor), {"mu": self.mu})*thetas[q] for (q, pulled_back_theta_factors) in enumerate(self._pulled_back_theta_factors[term]) for pulled_back_theta_factor in pulled_back_theta_factors])
                else:
                    return thetas
                    
            def _map_facet_id_to_subdomain_id(self, **kwargs):
                mesh = self.V.mesh()
                mpi_comm = mesh.mpi_comm().tompi4py()
                assert "subdomains" in kwargs
                subdomains = kwargs["subdomains"]
                assert "boundaries" in kwargs
                boundaries = kwargs["boundaries"]
                # Loop over local part of the mesh
                facet_id_to_subdomain_ids = defaultdict(set)
                subdomain_id_to_facet_ids = defaultdict(set)
                for f in facets(mesh):
                    if boundaries[f] > 0: # skip unmarked facets
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
                mesh = self.V.mesh()
                dim = mesh.topology().dim()
                mpi_comm = mesh.mpi_comm().tompi4py()
                assert "subdomains" in kwargs
                subdomains = kwargs["subdomains"]
                assert "boundaries" in kwargs
                boundaries = kwargs["boundaries"]
                # Loop over local part of the mesh
                facet_id_to_normal_directions = defaultdict(set)
                for f in facets(mesh):
                    if boundaries[f] > 0: # skip unmarked facets
                        if f.exterior():
                            facet_id_to_normal_directions[boundaries[f]].add(tuple([f.normal(d) for d in range(dim)]))
                        else:
                            cell_id_to_subdomain_id = dict()
                            for (c_id, c) in enumerate(cells(f)):
                                cell_id_to_subdomain_id[c_id] = subdomains[c]
                            assert len(cell_id_to_subdomain_id) is 2
                            assert cell_id_to_subdomain_id[0] != cell_id_to_subdomain_id[1]
                            cell_id_to_restricted_sign = dict()
                            if cell_id_to_subdomain_id[0] > cell_id_to_subdomain_id[1]:
                                cell_id_to_restricted_sign[0] = "+"
                                cell_id_to_restricted_sign[1] = "-"
                            else:
                                cell_id_to_restricted_sign[0] = "-"
                                cell_id_to_restricted_sign[1] = "+"
                            for (c_id, c) in enumerate(cells(f)):
                                facet_id_to_normal_directions[(boundaries[f], cell_id_to_restricted_sign[c_id])].add(tuple([c.normal(c.index(f), d) for d in range(dim)]))
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
                                if isinstance(operand_0, Expression):
                                    node_cppcode = operand_0.cppcode
                                    for index in node.ufl_operands[1].indices():
                                        assert isinstance(index, FixedIndex)
                                        node_cppcode = node_cppcode[int(index)]
                                    if len(self._is_affine_parameter_dependent_regex.findall(node_cppcode)) > 0:
                                        return False
                # The pulled back form is not affine if it contains a boundary integral on a non-straight boundary,
                # because the normal direction would depend on x
                for form_with_placeholder in separated_pulled_back_form._form_with_placeholders:
                    assert len(form_with_placeholder.integrals()) is 1
                    integral = form_with_placeholder.integrals()[0]
                    integral_type = integral.integral_type()
                    integral_subdomain_id = integral.subdomain_id()
                    if integral_type == "cell":
                        pass
                    elif integral_type.startswith("exterior_facet"):
                        if self._facet_id_to_normal_direction_if_straight[integral_subdomain_id] is None:
                            return False
                    elif integral_type.startswith("interior_facet"):
                        if (
                            self._facet_id_to_normal_direction_if_straight[(integral_subdomain_id, "+")] is None
                                or
                            self._facet_id_to_normal_direction_if_straight[(integral_subdomain_id, "-")] is None
                        ):
                            return False
                    else:
                        raise ValueError("Unknown integral type {}, don't know how to check for affinity.".format(integral_type))
                # Otherwise, the pulled back form is affine
                return True
                
            def _get_affine_parameter_dependent_forms(self, separated_pulled_back_form):
                affine_parameter_dependent_forms = list()
                # Append forms which were not originally affinely dependent
                for (index, addend) in enumerate(separated_pulled_back_form.coefficients):
                    affine_parameter_dependent_forms.append(
                        separated_pulled_back_form.replace_placeholders(index, [1]*len(addend))
                    )
                # Append forms which were already affinely dependent
                for unchanged_form in separated_pulled_back_form.unchanged_forms:
                    affine_parameter_dependent_forms.append(unchanged_form)
                # Return
                return tuple(affine_parameter_dependent_forms)
                
            def _get_affine_parameter_dependent_theta_factors(self, separated_pulled_back_form):
                # Prepare theta factors
                affine_parameter_dependent_theta_factors = list()
                # Append factors corresponding to forms which were not originally affinely dependent
                assert len(separated_pulled_back_form.coefficients) == len(separated_pulled_back_form._placeholders) == len(separated_pulled_back_form._form_with_placeholders)
                for (addend_coefficient, addend_placeholder, addend_form_with_placeholder) in zip(separated_pulled_back_form.coefficients, separated_pulled_back_form._placeholders, separated_pulled_back_form._form_with_placeholders):
                    affine_parameter_dependent_theta_factors.append(
                        self._compute_affine_parameter_dependent_theta_factor(addend_coefficient, addend_placeholder, addend_form_with_placeholder)
                    )
                # Append factors corresponding to forms which were already affinely dependent
                for _ in separated_pulled_back_form.unchanged_forms:
                    affine_parameter_dependent_theta_factors.append(1.)
                # Return
                return tuple(affine_parameter_dependent_theta_factors)
                
            def _compute_affine_parameter_dependent_theta_factor(self, coefficient, placeholder, form_with_placeholder):
                assert len(form_with_placeholder.integrals()) is 1
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
                for p in range(len(self.mu)):
                    locals["mu_" + str(p)] = symbols("mu[" + str(p) + "]")
                # ... add shape parametrization jacobians to locals
                for (shape_parametrization_jacobian, shape_parametrization_jacobian_sympy) in zip(self._shape_parametrization_jacobians, self._shape_parametrization_jacobians_sympy):
                    locals[str(shape_parametrization_jacobian)] = shape_parametrization_jacobian_sympy
                # ... add fake unity constants to locals
                mesh_point = self.V.mesh().coordinates()[0]
                for constant in replacer.constants:
                    assert len(constant.ufl_shape) in (0, 1, 2)
                    if len(constant.ufl_shape) is 0:
                        locals[str(constant)] = float(constant)
                    elif len(constant.ufl_shape) is 1:
                        vals = numpy_zeros(constant.ufl_shape)
                        constant.eval(vals, mesh_point)
                        for i in range(constant.ufl_shape[0]):
                            locals[str(constant) + "[" + str(i) + "]"] = vals[i]
                    elif len(constant.ufl_shape) is 2:
                        vals = numpy_zeros(constant.ufl_shape).reshape((-1,))
                        constant.eval(vals, mesh_point)
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
                    raise NotImplementedError("compute_affine_parameter_dependent_theta_factor has not been implemented yet for interior_facet")
                else:
                    raise ValueError("Unknown integral type {}, don't know how to check for affinity.".format(integral_type))
                # ... carry out conversion
                theta_factor_sympy = theta_factor
                for i in range(2): # first pass will replace shape parametrization and normals, second one mu_* to self.mu[*]
                    theta_factor_sympy = simplify(sympify(str(theta_factor_sympy), locals=locals))
                theta_factor_sympy = simplify(convert_float_to_int_if_possible(theta_factor_sympy))
                return theta_factor_sympy
                
        # return value (a class) for the decorator
        return PullBackFormsToReferenceDomainDecoratedProblem_Class
        
    # return the decorator itself
    return PullBackFormsToReferenceDomainDecoratedProblem_Decorator
                
PullBackFormsToReferenceDomain = PullBackFormsToReferenceDomainDecoratedProblem
    
def expand(form):
    # Call UFL expander
    expanded_form = expand_indices(apply_derivatives(apply_algebra_lowering(form)))
    # Call sympy replacer
    expanded_form = apply_transformer(expanded_form, SympyExpander())
    # Split sums
    expanded_split_form_integrals = list()
    for integral in expanded_form.integrals():
        split_sum_of_integrals(integral, expanded_split_form_integrals)
    expanded_split_form = Form(expanded_split_form_integrals)
    # Return
    return expanded_split_form
    
class SympyExpander(Transformer):
    def __init__(self):
        Transformer.__init__(self)
        self.ufl_to_sympy = dict()
        self.sympy_to_ufl = dict()
        self.sympy_id_to_ufl = dict()
        
    def operator(self, e, *ops):
        self._store_sympy_symbol(e)
        return e
        
    def sum(self, e, arg1, arg2):
        def op(arg1, arg2):
            return arg1 + arg2
        return self._apply_sympy_simplify(e, arg1, arg2, op)

    def product(self, e, arg1, arg2):
        def op(arg1, arg2):
            return arg1*arg2
        return self._apply_sympy_simplify(e, arg1, arg2, op)
    
    def terminal(self, o):
        self._store_sympy_symbol(o)
        return o
        
    def _apply_sympy_simplify(self, e, arg1, arg2, op):
        assert arg1 in self.ufl_to_sympy
        sympy_arg1 = self.ufl_to_sympy[arg1]
        assert arg2 in self.ufl_to_sympy
        sympy_arg2 = self.ufl_to_sympy[arg2]
        sympy_expanded_e = expand_mul(op(sympy_arg1, sympy_arg2))
        ufl_expanded_e = safe_eval(str(sympy_expanded_e), self.sympy_id_to_ufl)
        self.ufl_to_sympy[ufl_expanded_e] = sympy_expanded_e
        self.sympy_to_ufl[sympy_expanded_e] = ufl_expanded_e
        return ufl_expanded_e
    
    def _store_sympy_symbol(self, o):
        if isinstance(o, MultiIndex):
            pass
        else:
            assert len(o.ufl_shape) in (0, 1, 2)
            if len(o.ufl_shape) is 0:
                self._store_sympy_scalar_symbol(o)
            elif len(o.ufl_shape) is 1:
                for i in range(o.ufl_shape[0]):
                    self._store_sympy_scalar_symbol(o[i])
            elif len(o.ufl_shape) is 2:
                for i in range(o.ufl_shape[0]):
                    for j in range(o.ufl_shape[1]):
                        self._store_sympy_scalar_symbol(o[i, j])
                
    def _store_sympy_scalar_symbol(self, o):
        assert o.ufl_shape == ()
        if o not in self.ufl_to_sympy:
            sympy_id = "sympy" + str(len(self.ufl_to_sympy))
            sympy_o = symbols(sympy_id)
            self.ufl_to_sympy[o] = sympy_o
            self.sympy_to_ufl[sympy_o] = o
            self.sympy_id_to_ufl[sympy_id] = o
            
def split_sum_of_integrals(integral, expanded_split_form_integrals):
    integrand = integral.integrand()
    if isinstance(integrand, Sum):
        for operand in integrand.ufl_operands:
            split_sum_of_integrals(integral.reconstruct(integrand=operand), expanded_split_form_integrals)
    else:
        expanded_split_form_integrals.append(integral)
    
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
    # Remove all zero theta factors
    postprocessed_pulled_back_forms_non_zero = list()
    postprocessed_pulled_back_theta_factors_non_zero = list()
    assert len(postprocessed_pulled_back_forms) == len(postprocessed_pulled_back_theta_factors)
    for (postprocessed_pulled_back_form, postprocessed_pulled_back_theta_factor) in zip(postprocessed_pulled_back_forms, postprocessed_pulled_back_theta_factors):
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
            postprocessed_pulled_back_forms_ufl_to_sympy[postprocessed_pulled_back_form] = postprocessed_pulled_back_form_sympy
            postprocessed_pulled_back_forms_sympy_id_to_ufl[sympy_id] = postprocessed_pulled_back_form
    # Convert theta factors to sympy symbols
    postprocessed_pulled_back_theta_factors_sympy_independents = list()
    postprocessed_pulled_back_theta_factors_ufl_to_sympy = dict()
    postprocessed_pulled_back_theta_factors_sympy_id_to_ufl = dict()
    for postprocessed_pulled_back_theta_factor in postprocessed_pulled_back_theta_factors_non_zero:
        if postprocessed_pulled_back_theta_factor not in postprocessed_pulled_back_theta_factors_ufl_to_sympy:
            for (previous_theta_factor_ufl, previous_theta_factor_sympy) in postprocessed_pulled_back_theta_factors_ufl_to_sympy.items():
                ratio = simplify(postprocessed_pulled_back_theta_factor/previous_theta_factor_ufl)
                if isinstance(ratio, Number):
                    postprocessed_pulled_back_theta_factors_ufl_to_sympy[postprocessed_pulled_back_theta_factor] = ratio*previous_theta_factor_sympy
                    break
            else:
                sympy_id = "sympytheta" + str(len(postprocessed_pulled_back_theta_factors_sympy_id_to_ufl))
                postprocessed_pulled_back_theta_factor_sympy = symbols(sympy_id)
                postprocessed_pulled_back_theta_factors_sympy_independents.append(postprocessed_pulled_back_theta_factor_sympy)
                postprocessed_pulled_back_theta_factors_ufl_to_sympy[postprocessed_pulled_back_theta_factor] = postprocessed_pulled_back_theta_factor_sympy
                postprocessed_pulled_back_theta_factors_sympy_id_to_ufl[sympy_id] = postprocessed_pulled_back_theta_factor
    # Carry out symbolic sum(product())
    postprocessed_pulled_back_sum_product = 0
    for (postprocessed_pulled_back_form, postprocessed_pulled_back_theta_factor) in zip(postprocessed_pulled_back_forms_non_zero, postprocessed_pulled_back_theta_factors_non_zero):
        postprocessed_pulled_back_sum_product += (
            postprocessed_pulled_back_theta_factors_ufl_to_sympy[postprocessed_pulled_back_theta_factor]
                *
            postprocessed_pulled_back_forms_ufl_to_sympy[postprocessed_pulled_back_form]
        )
    # Collect first with respect to theta factors
    collected_with_respect_to_theta = collect(postprocessed_pulled_back_sum_product, postprocessed_pulled_back_theta_factors_sympy_independents, evaluate=False, exact=True)
    collected_sum_product = 0
    for (collected_theta_factor, collected_form) in collected_with_respect_to_theta.items():
        collected_sum_product += collected_theta_factor*collected_form
    # Collect then with respect to form factors
    collected_with_respect_to_form = collect(collected_sum_product, collected_with_respect_to_theta.values(), evaluate=False, exact=True)
    # Convert back to ufl
    collected_forms = list()
    collected_theta_factors = list()
    for (collected_form, collected_theta_factor) in collected_with_respect_to_form.items():
        collected_forms.append(safe_eval(str(collected_form), postprocessed_pulled_back_forms_sympy_id_to_ufl))
        collected_theta_factors.append(safe_eval(str(collected_theta_factor), postprocessed_pulled_back_theta_factors_sympy_id_to_ufl))
    # Return
    return (tuple(collected_forms), tuple(collected_theta_factors))
    
def safe_eval(string, locals):
    for name, function in math.__dict__.items():
        if callable(function):
            locals[name] = function
    return eval(string, {"__builtins__": None}, locals)
