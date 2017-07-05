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

from ufl.geometry import GeometricQuantity
import rbnics.backends.fenics
from rbnics.utils.decorators import exact_problem, get_problem_from_solution, get_reduced_problem_from_problem, is_training_finished
from rbnics.utils.mpi import log, PROGRESS
from rbnics.eim.utils.decorators import get_EIM_approximation_from_parametrized_expression

def expression_on_reduced_mesh(expression_wrapper, at, backend=None):
    if backend is None:
        backend = rbnics.backends.fenics
    
    expression = expression_wrapper._expression
    expression_name = expression_wrapper._name
    EIM_approximation = get_EIM_approximation_from_parametrized_expression(expression_wrapper)
    reduced_mesh = at.get_reduced_mesh()
    reduced_V = at.get_reduced_function_space()
    
    if (expression_name, reduced_mesh) not in expression_on_reduced_mesh__expression_cache:
        visited = set()
        replacements = dict()
        truth_problem_to_components = dict()
        truth_problem_to_reduced_mesh_solution = dict()
        truth_problem_to_reduced_mesh_interpolator = dict()
        reduced_problem_to_components = dict()
        reduced_problem_to_reduced_mesh_solution = dict()
        reduced_problem_to_reduced_Z = dict()
        
        # Look for terminals on truth mesh
        for node in backend.wrapping.expression_iterator(expression):
            if node in visited:
                continue
            # ... problem solutions related to nonlinear terms
            elif backend.wrapping.is_problem_solution_or_problem_solution_component_type(node):
                if backend.wrapping.is_problem_solution_or_problem_solution_component(node):
                    (preprocessed_node, component, truth_solution) = backend.wrapping.solution_identify_component(node)
                    truth_problem = get_problem_from_solution(truth_solution)
                    # Get the function space corresponding to preprocessed_node on the reduced mesh
                    auxiliary_reduced_V = at.get_auxiliary_reduced_function_space(truth_problem, component)
                    # Define a replacement
                    replacements[preprocessed_node] = backend.Function(auxiliary_reduced_V)
                    if is_training_finished(truth_problem):
                        reduced_problem = get_reduced_problem_from_problem(truth_problem)
                        # Store the component
                        if reduced_problem not in reduced_problem_to_components:
                            reduced_problem_to_components[reduced_problem] = list()
                        reduced_problem_to_components[reduced_problem].append(component)
                        # Store the replacement
                        if reduced_problem not in reduced_problem_to_reduced_mesh_solution:
                            reduced_problem_to_reduced_mesh_solution[reduced_problem] = list()
                        reduced_problem_to_reduced_mesh_solution[reduced_problem].append(replacements[preprocessed_node])
                        # Get reduced problem basis functions on reduced mesh
                        if reduced_problem not in reduced_problem_to_reduced_Z:
                            reduced_problem_to_reduced_Z[reduced_problem] = list()
                        reduced_problem_to_reduced_Z[reduced_problem].append(at.get_auxiliary_basis_functions_matrix(truth_problem, reduced_problem, component))
                    else:
                        if not hasattr(truth_problem, "_is_solving"):
                            exact_truth_problem = exact_problem(truth_problem)
                            exact_truth_problem.init()
                            # Store the component
                            if exact_truth_problem not in truth_problem_to_components:
                                truth_problem_to_components[exact_truth_problem] = list()
                            truth_problem_to_components[exact_truth_problem].append(component)
                            # Store the replacement
                            if exact_truth_problem not in truth_problem_to_reduced_mesh_solution:
                                truth_problem_to_reduced_mesh_solution[exact_truth_problem] = list()
                            truth_problem_to_reduced_mesh_solution[exact_truth_problem].append(replacements[preprocessed_node])
                            # Get interpolator on reduced mesh
                            if exact_truth_problem not in truth_problem_to_reduced_mesh_interpolator:
                                truth_problem_to_reduced_mesh_interpolator[exact_truth_problem] = list()
                            truth_problem_to_reduced_mesh_interpolator[exact_truth_problem].append(at.get_auxiliary_function_interpolator(exact_truth_problem, component))
                        else:
                            # Store the component
                            if truth_problem not in truth_problem_to_components:
                                truth_problem_to_components[truth_problem] = list()
                            truth_problem_to_components[truth_problem].append(component)
                            # Store the replacement
                            if exact_truth_problem not in truth_problem_to_reduced_mesh_solution:
                                truth_problem_to_reduced_mesh_solution[truth_problem] = list()
                            truth_problem_to_reduced_mesh_solution[truth_problem].append(replacements[preprocessed_node])
                            # Get interpolator on reduced mesh
                            if truth_problem not in truth_problem_to_reduced_mesh_interpolator:
                                truth_problem_to_reduced_mesh_interpolator[truth_problem] = list()
                            truth_problem_to_reduced_mesh_interpolator[truth_problem].append(at.get_auxiliary_function_interpolator(truth_problem, component))
                else:
                    (auxiliary_problem, component) = backend.wrapping.get_auxiliary_problem_for_non_parametrized_function(node)
                    preprocessed_node = node
                    # Get the function space corresponding to preprocessed_node on the reduced mesh
                    auxiliary_reduced_V = at.get_auxiliary_reduced_function_space(auxiliary_problem, component)
                    # Get interpolator on reduced mesh
                    auxiliary_truth_problem_to_reduced_mesh_interpolator = at.get_auxiliary_function_interpolator(auxiliary_problem, component)
                    # Define and store the replacement
                    replacements[preprocessed_node] = auxiliary_truth_problem_to_reduced_mesh_interpolator(preprocessed_node)
                # Make sure to skip any parent solution related to this one
                visited.add(node)
                visited.add(preprocessed_node)
                for parent_node in backend.wrapping.solution_iterator(preprocessed_node):
                    visited.add(parent_node)
            # ... geometric quantities
            elif isinstance(node, GeometricQuantity):
                replacements[node] = type(node)(reduced_mesh)
                visited.add(node)
        # ... and replace them
        replaced_expression = backend.wrapping.expression_replace(expression, replacements)
        
        # Cache the resulting dicts
        expression_on_reduced_mesh__expression_cache[(expression_name, reduced_mesh)] = replaced_expression
        expression_on_reduced_mesh__truth_problem_to_components_cache[(expression_name, reduced_mesh)] = truth_problem_to_components
        expression_on_reduced_mesh__truth_problem_to_reduced_mesh_solution_cache[(expression_name, reduced_mesh)] = truth_problem_to_reduced_mesh_solution
        expression_on_reduced_mesh__truth_problem_to_reduced_mesh_interpolator_cache[(expression_name, reduced_mesh)] = truth_problem_to_reduced_mesh_interpolator
        expression_on_reduced_mesh__reduced_problem_to_components_cache[(expression_name, reduced_mesh)] = reduced_problem_to_components
        expression_on_reduced_mesh__reduced_problem_to_reduced_mesh_solution_cache[(expression_name, reduced_mesh)] = reduced_problem_to_reduced_mesh_solution
        expression_on_reduced_mesh__reduced_problem_to_reduced_Z_cache[(expression_name, reduced_mesh)] = reduced_problem_to_reduced_Z
        
    # Extract from cache
    replaced_expression = expression_on_reduced_mesh__expression_cache[(expression_name, reduced_mesh)]
    truth_problem_to_components = expression_on_reduced_mesh__truth_problem_to_components_cache[(expression_name, reduced_mesh)]
    truth_problem_to_reduced_mesh_solution = expression_on_reduced_mesh__truth_problem_to_reduced_mesh_solution_cache[(expression_name, reduced_mesh)]
    truth_problem_to_reduced_mesh_interpolator = expression_on_reduced_mesh__truth_problem_to_reduced_mesh_interpolator_cache[(expression_name, reduced_mesh)]
    reduced_problem_to_components = expression_on_reduced_mesh__reduced_problem_to_components_cache[(expression_name, reduced_mesh)]
    reduced_problem_to_reduced_mesh_solution = expression_on_reduced_mesh__reduced_problem_to_reduced_mesh_solution_cache[(expression_name, reduced_mesh)]
    reduced_problem_to_reduced_Z = expression_on_reduced_mesh__reduced_problem_to_reduced_Z_cache[(expression_name, reduced_mesh)]
    
    # Solve truth problems (which have not been reduced yet) associated to nonlinear terms
    for truth_problem in truth_problem_to_reduced_mesh_solution:
        truth_problem.set_mu(EIM_approximation.mu)
        if not hasattr(truth_problem, "_is_solving"):
            log(PROGRESS, "In expression_on_reduced_mesh, requiring truth problem solve for problem " + str(truth_problem))
            truth_problem.solve()
        else:
            log(PROGRESS, "In expression_on_reduced_mesh, loading current truth problem solution for problem " + str(truth_problem))
        for (reduced_mesh_solution, reduced_mesh_interpolator) in zip(truth_problem_to_reduced_mesh_solution[truth_problem], truth_problem_to_reduced_mesh_interpolator[truth_problem]):
            solution_to = reduced_mesh_solution
            solution_from = reduced_mesh_interpolator(truth_problem._solution)
            backend.assign(solution_to, solution_from)
    
    # Solve reduced problems associated to nonlinear terms
    for reduced_problem in reduced_problem_to_reduced_mesh_solution:
        reduced_problem.set_mu(EIM_approximation.mu)
        if not hasattr(reduced_problem, "_is_solving"):
            log(PROGRESS, "In expression_on_reduced_mesh, requiring reduced problem solve for problem " + str(reduced_problem))
            reduced_problem.solve()
        else:
            log(PROGRESS, "In expression_on_reduced_mesh, loading current reduced problem solution for problem " + str(reduced_problem))
        for (reduced_mesh_solution, reduced_Z) in zip(reduced_problem_to_reduced_mesh_solution[reduced_problem], reduced_problem_to_reduced_Z[reduced_problem]):
            solution_to = reduced_mesh_solution
            solution_from = reduced_Z[:reduced_problem._solution.N]*reduced_problem._solution
            backend.assign(solution_to, solution_from)
    
    # Interpolate and return
    reduced_space = at.get_reduced_function_space()
    backend.wrapping.assert_lagrange_1(reduced_space)
    interpolated_replaced_expression = backend.Function(reduced_space)
    backend.wrapping.ufl_lagrange_interpolation(interpolated_replaced_expression, replaced_expression)
    return interpolated_replaced_expression
    
expression_on_reduced_mesh__expression_cache = dict()
expression_on_reduced_mesh__truth_problem_to_components_cache = dict()
expression_on_reduced_mesh__truth_problem_to_reduced_mesh_solution_cache = dict()
expression_on_reduced_mesh__truth_problem_to_reduced_mesh_interpolator_cache = dict()
expression_on_reduced_mesh__reduced_problem_to_components_cache = dict()
expression_on_reduced_mesh__reduced_problem_to_reduced_mesh_solution_cache = dict()
expression_on_reduced_mesh__reduced_problem_to_reduced_Z_cache = dict()

