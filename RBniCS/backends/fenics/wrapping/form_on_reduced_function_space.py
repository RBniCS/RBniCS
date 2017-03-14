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
## @file
#  @brief
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from ufl import Measure, replace
from ufl.algorithms.traversal import iter_expressions
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.geometry import GeometricQuantity
from dolfin import Argument, assign, Function
from RBniCS.backends.fenics.wrapping.function_from_subfunction_if_any import function_from_subfunction_if_any
from RBniCS.utils.decorators import get_problem_from_solution, get_reduced_problem_from_problem, is_problem_solution
from RBniCS.eim.utils.decorators import get_EIM_approximation_from_parametrized_expression

def form_on_reduced_function_space(form_wrapper, at):
    form = form_wrapper._form
    EIM_approximation = get_EIM_approximation_from_parametrized_expression(form_wrapper)
    reduced_V = at.get_reduced_function_spaces()
    reduced_subdomain_data = at.get_reduced_subdomain_data()
    
    if (form, reduced_V) not in form_on_reduced_function_space__form_cache:
        replacements = dict()
        reduced_problem_to_reduced_mesh_solution = dict()
        reduced_problem_to_reduced_Z = dict()
        
        # Look for terminals on truth mesh
        for integral in form.integrals():
            for expression in iter_expressions(integral):
                for node in traverse_unique_terminals(expression):
                    node = function_from_subfunction_if_any(node)
                    if node in replacements:
                        continue
                    # ... test and trial functions
                    elif isinstance(node, Argument):
                        replacements[node] = Argument(reduced_V[node.number()], node.number(), node.part())
                    # ... problem solutions related to nonlinear terms
                    elif isinstance(node, Function) and is_problem_solution(node):
                        truth_problem = get_problem_from_solution(node)
                        reduced_problem = get_reduced_problem_from_problem(truth_problem)
                        # Get the function space corresponding to node on the reduced mesh
                        auxiliary_reduced_V = at.get_auxiliary_reduced_function_space(truth_problem)
                        # Define a replacement
                        replacements[node] = Function(auxiliary_reduced_V)
                        reduced_problem_to_reduced_mesh_solution[reduced_problem] = replacements[node]
                        # Get reduced problem basis functions on reduced mesh
                        reduced_problem_to_reduced_Z[reduced_problem] = at.get_auxiliary_basis_functions_matrix(truth_problem, reduced_problem)
                    # ... geometric quantities
                    elif isinstance(node, GeometricQuantity):
                        if len(reduced_V) == 2:
                            assert reduced_V[0].mesh().ufl_domain() == reduced_V[1].mesh().ufl_domain()
                        replacements[node] = type(node)(reduced_V[0].mesh())
        # ... and replace them
        replaced_form = replace(form, replacements)
        
        # Look for measures and replace them
        replaced_form_with_replaced_measures = 0
        for integral in replaced_form.integrals():
            # Prepare measure for the new form (from firedrake/mg/ufl_utils.py)
            if len(reduced_V) == 2:
                assert reduced_V[0].mesh().ufl_domain() == reduced_V[1].mesh().ufl_domain()
            measure_reduced_domain = reduced_V[0].mesh().ufl_domain()
            measure_subdomain_data = integral.subdomain_data()
            if measure_subdomain_data is not None:
                measure_reduced_subdomain_data = reduced_subdomain_data[measure_subdomain_data]
            else:
                measure_reduced_subdomain_data = None
            measure = Measure(
                integral.integral_type(),
                domain=measure_reduced_domain,
                subdomain_id=integral.subdomain_id(),
                subdomain_data=measure_reduced_subdomain_data,
                metadata=integral.metadata()
            )
            replaced_form_with_replaced_measures += integral.integrand()*measure
        
        # Cache the resulting dicts
        form_on_reduced_function_space__form_cache[(form, reduced_V)] = replaced_form_with_replaced_measures
        form_on_reduced_function_space__reduced_problem_to_reduced_mesh_solution_cache[(form, reduced_V)] = reduced_problem_to_reduced_mesh_solution
        form_on_reduced_function_space__reduced_problem_to_reduced_Z_cache[(form, reduced_V)] = reduced_problem_to_reduced_Z
        
    # Extract from cache
    replaced_form_with_replaced_measures = form_on_reduced_function_space__form_cache[(form, reduced_V)]
    reduced_problem_to_reduced_mesh_solution = form_on_reduced_function_space__reduced_problem_to_reduced_mesh_solution_cache[(form, reduced_V)]
    reduced_problem_to_reduced_Z = form_on_reduced_function_space__reduced_problem_to_reduced_Z_cache[(form, reduced_V)]
    
    # Solve reduced problem associated to nonlinear terms
    for (reduced_problem, reduced_mesh_solution) in reduced_problem_to_reduced_mesh_solution.iteritems():
        reduced_problem.set_mu(EIM_approximation.mu)
        if not hasattr(reduced_problem, "_is_solving"):
            reduced_solution = reduced_problem.solve()
        else:
            reduced_solution = reduced_problem._solution
        reduced_Z = reduced_problem_to_reduced_Z[reduced_problem]
        assign(reduced_mesh_solution, reduced_Z[:reduced_solution.N]*reduced_solution)
    
    return replaced_form_with_replaced_measures
    
form_on_reduced_function_space__form_cache = dict()
form_on_reduced_function_space__reduced_problem_to_reduced_mesh_solution_cache = dict()
form_on_reduced_function_space__reduced_problem_to_reduced_Z_cache = dict()
