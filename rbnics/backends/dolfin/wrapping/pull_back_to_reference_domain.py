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
from mpi4py.MPI import Op
from ufl import as_tensor, det, inv, Measure, TensorElement, tr, transpose, VectorElement
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.expand_indices import expand_indices, purge_list_tensors
from ufl.classes import FacetJacobian, FacetJacobianDeterminant, Grad
from ufl.compound_expressions import determinant_expr
from ufl.core.multiindex import Index, indices
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dag
from dolfin import cells, Expression, facets
from rbnics.backends.dolfin.wrapping.parametrized_expression import ParametrizedExpression

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
    from rbnics.shape_parametrization.utils.symbolic import strings_to_number_of_parameters
    mu = (0, )*strings_to_number_of_parameters(shape_parametrization_expression_on_subdomain)
    element = VectorElement("CG", domain.ufl_cell(), 1)
    shape_parametrization_map = ParametrizedExpression(problem, shape_parametrization_expression_on_subdomain, mu=mu, element=element)
    shape_parametrization_map.set_mu(problem.mu)
    return shape_parametrization_map

@shape_parametrization_cache
def ShapeParametrizationJacobian(shape_parametrization_expression_on_subdomain, problem, domain):
    from rbnics.shape_parametrization.utils.symbolic import compute_shape_parametrization_gradient, strings_to_number_of_parameters
    shape_parametrization_gradient_on_subdomain = compute_shape_parametrization_gradient(shape_parametrization_expression_on_subdomain)
    mu = (0, )*strings_to_number_of_parameters(shape_parametrization_expression_on_subdomain)
    element = TensorElement("CG", domain.ufl_cell(), 1)
    shape_parametrization_jacobian = ParametrizedExpression(problem, shape_parametrization_gradient_on_subdomain, mu=mu, element=element)
    shape_parametrization_jacobian.set_mu(problem.mu)
    return shape_parametrization_jacobian

@shape_parametrization_cache
def ShapeParametrizationJacobianInverse(shape_parametrization_expression_on_subdomain, problem, domain):
    return inv(ShapeParametrizationJacobian(shape_parametrization_expression_on_subdomain, problem, domain))
    
@shape_parametrization_cache
def ShapeParametrizationJacobianDeterminant(shape_parametrization_expression_on_subdomain, problem, domain):
    return det(ShapeParametrizationJacobian(shape_parametrization_expression_on_subdomain, problem, domain))
    
@shape_parametrization_cache
def ShapeParametrizationFacetJacobian(shape_parametrization_expression_on_subdomain, problem, domain):
    return ShapeParametrizationJacobian(shape_parametrization_expression_on_subdomain, problem, domain)*FacetJacobian(domain)
    
@shape_parametrization_cache
def ShapeParametrizationFacetJacobianDeterminant(shape_parametrization_expression_on_subdomain, problem, domain):
    return determinant_expr(ShapeParametrizationFacetJacobian(shape_parametrization_expression_on_subdomain, problem, domain))
    
# ===== Pull back form measures: inspired by ufl/algorithms/apply_integral_scaling.py ===== #
def pull_back_measures(shape_parametrization_expression_on_subdomain, problem, integral, subdomain_id): # inspired by compute_integrand_scaling_factor
    integral_type = integral.integral_type()
    tdim = integral.ufl_domain().topological_dimension()
    
    if integral_type == "cell":
        scale = ShapeParametrizationJacobianDeterminant(shape_parametrization_expression_on_subdomain, problem, integral.ufl_domain())
    elif integral_type.startswith("exterior_facet") or integral_type.startswith("interior_facet"):
        if tdim > 1:
            # Scaling integral by facet jacobian determinant
            scale = ShapeParametrizationFacetJacobianDeterminant(shape_parametrization_expression_on_subdomain, problem, integral.ufl_domain())/FacetJacobianDeterminant(integral.ufl_domain())
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

    def terminal(self, o):
        return o
        
    def div(self, o, f):
        assert f._ufl_is_terminal_
        
        # Create shape parametrization Jacobian inverse object
        Jinv = ShapeParametrizationJacobianInverse(self.shape_parametrization_expression_on_subdomain, self.problem, f.ufl_domain())
        
        # Indices to get to the scalar component of f
        first = indices(len(f.ufl_shape) - 1)
        last = Index()
        j = Index()
        
        # Wrap back in tensor shape
        return as_tensor(Jinv[j, last]*Grad(f)[first + (last, j)], first)

    def grad(self, _, f):
        assert f._ufl_is_terminal_
        
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
    pass # TODO

def pull_back_geometric_quantities(shape_parametrization_expression_on_subdomain, problem, integrand): # inspired by apply_geometry_lowering
    return integrand # TODO

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
                    measure_subdomain_id = problem.subdomain_id_to_facet_ids[subdomain_id]
                elif subdomain_id not in problem.facet_id_to_subdomain_ids[integral_subdomain_or_facet_id]:
                    continue
                else:
                    measure_subdomain_id = integral_subdomain_or_facet_id
            # Carry out pull back, if loop was not continue-d
            integrand = integral.integrand()
            integrand = pull_back_expressions(shape_parametrization_expression_on_subdomain, problem, integrand)
            integrand = pull_back_geometric_quantities(shape_parametrization_expression_on_subdomain, problem, integrand)
            integrand = pull_back_gradients(shape_parametrization_expression_on_subdomain, problem, integrand)
            (scale, measure) = pull_back_measures(shape_parametrization_expression_on_subdomain, problem, integral, measure_subdomain_id)
            #integrand_times_scale = expand_indices(apply_algebra_lowering(integrand*scale)) # TODO only for affine case
            integrand_times_scale = integrand*scale
            pulled_back_form += integrand_times_scale*measure
    return pulled_back_form
    
# ===== Auxiliary function to store a dict from facet id to subdomain ids ===== #
def fill_facet_id_to_subdomain_id(problem):
    if not hasattr(problem, "facet_id_to_subdomain_ids"):
        assert not hasattr(problem, "subdomain_id_to_facet_ids")
        mesh = problem.V.mesh()
        mpi_comm = mesh.mpi_comm().tompi4py()
        assert "subdomains" in problem.problem_kwargs
        subdomains = problem.problem_kwargs["subdomains"]
        assert "boundaries" in problem.problem_kwargs
        boundaries = problem.problem_kwargs["boundaries"]
        # Loop over local part of the mesh
        facet_id_to_subdomain_ids = defaultdict(set)
        subdomain_id_to_facet_ids = defaultdict(set)
        for f in facets(mesh):
            for c in cells(f):
                facet_id_to_subdomain_ids[boundaries[f]].add(subdomains[c])
                subdomain_id_to_facet_ids[subdomains[c]].add(boundaries[f])
        facet_id_to_subdomain_ids = dict(facet_id_to_subdomain_ids)
        subdomain_id_to_facet_ids = dict(subdomain_id_to_facet_ids)
        # Collect in parallel
        mpi_comm.allreduce(facet_id_to_subdomain_ids, op=_dict_collect_op)
        mpi_comm.allreduce(subdomain_id_to_facet_ids, op=_dict_collect_op)
        # Store
        problem.facet_id_to_subdomain_ids = facet_id_to_subdomain_ids
        problem.subdomain_id_to_facet_ids = subdomain_id_to_facet_ids
    
def _dict_collect(dict1, dict2, datatype):
    dict12 = defaultdict(set)
    for dict_ in (dict1, dict2):
        for (key, value) in d.items():
            dict12[key].update(value)
    return dict(dict12)

_dict_collect_op = Op.Create(_dict_collect, commute=True)
    
# ===== Pull back forms decorator ===== #
def pull_back_forms_to_reference_domain(*args, **kwargs):
    def pull_back_forms_to_reference_domain_decorator(assemble_operator):
        def decoreated_assemble_operator(self, term):
            forms = assemble_operator(self, term)
            if term in args:
                fill_facet_id_to_subdomain_id(self) # carried out only once
                pulled_back_forms = [pull_back_form(self.shape_parametrization_expression, self, form) for form in forms]
                # Check if the dependence is affine on the parameters. If so, move parameter dependent coefficients to compute_theta
                # TODO
                # If debug is enabled, deform the mesh for a few representative values of the parameters to check the the form assembled
                # on the parametrized domain results in the same tensor as the pulled back one
                if "debug" in kwargs and kwargs["debug"] is True:
                    pass # TODO
                return tuple(pulled_back_forms)
            else:
                return forms
            
        return decoreated_assemble_operator
    return pull_back_forms_to_reference_domain_decorator
    
# ===== Pull back Dirichlet BC decorator ===== #
def pull_back_dirichlet_bcs_to_reference_domain(*args, **kwargs):
    def pull_back_dirichlet_bcs_to_reference_domain_decorator(assemble_operator):
        def decoreated_assemble_operator(self, term):
            dirichlet_bcs = assemble_operator(self, term)
            if term in args:
                fill_facet_id_to_subdomain_id(self) # carried out only once
                pulled_back_dirichlet_bcs = None # TODO
                # Check if the dependence is affine on the parameters. If so, move parameter dependent coefficients to compute_theta
                # TODO
                # If debug is enabled, deform the mesh for a few representative values of the parameters to check the the form assembled
                # on the parametrized domain results in the same tensor as the pulled back one
                if "debug" in kwargs and kwargs["debug"] is True:
                    pass # TODO
                return tuple(pulled_back_dirichlet_bcs)
            else:
                return forms
            
        return decoreated_assemble_operator
    return pull_back_dirichlet_bcs_to_reference_domain_decorator
