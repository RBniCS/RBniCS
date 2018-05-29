# Copyright (C) 2015-2018 by the RBniCS authors
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
from rbnics.eim.utils.decorators import get_problem_from_parametrized_expression, get_problem_from_solution, get_reduced_problem_from_problem, is_training_finished, is_training_started
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import exact_problem
from rbnics.utils.io import OnlineSizeDict
from rbnics.utils.mpi import log, PROGRESS

def basic_expression_on_reduced_mesh(backend, wrapping, online_backend, online_wrapping):
    def _basic_expression_on_reduced_mesh(expression_wrapper, at):
        expression = expression_wrapper._expression
        expression_name = expression_wrapper.name()
        reduced_space = at.get_reduced_function_space()
        mu = get_problem_from_parametrized_expression(expression_wrapper).mu
        reduced_mesh = at.get_reduced_mesh()
        
        if (expression_name, reduced_mesh) not in expression_on_reduced_mesh__expression_cache:
            visited = set()
            replacements = dict()
            truth_problems = list()
            truth_problem_to_components = dict()
            truth_problem_to_exact_truth_problem = dict()
            truth_problem_to_reduced_mesh_solution = dict()
            truth_problem_to_reduced_mesh_interpolator = dict()
            reduced_problem_to_components = dict()
            reduced_problem_to_reduced_mesh_solution = dict()
            reduced_problem_to_reduced_basis_functions = dict()
            
            # Look for terminals on truth mesh
            for node in wrapping.expression_iterator(expression):
                if node in visited:
                    continue
                # ... problem solutions related to nonlinear terms
                elif wrapping.is_problem_solution_or_problem_solution_component_type(node):
                    if wrapping.is_problem_solution_or_problem_solution_component(node):
                        (preprocessed_node, component, truth_solution) = wrapping.solution_identify_component(node)
                        truth_problem = get_problem_from_solution(truth_solution)
                        truth_problems.append(truth_problem)
                        # Store the component
                        if truth_problem not in truth_problem_to_components:
                            truth_problem_to_components[truth_problem] = list()
                        truth_problem_to_components[truth_problem].append(component)
                        # Get the function space corresponding to preprocessed_node on the reduced mesh
                        auxiliary_reduced_V = at.get_auxiliary_reduced_function_space(truth_problem, component)
                        # Define and store the replacement
                        if truth_problem not in truth_problem_to_reduced_mesh_solution:
                            truth_problem_to_reduced_mesh_solution[truth_problem] = list()
                        replacements[preprocessed_node] = backend.Function(auxiliary_reduced_V)
                        truth_problem_to_reduced_mesh_solution[truth_problem].append(replacements[preprocessed_node])
                        # Get interpolator on reduced mesh
                        if truth_problem not in truth_problem_to_reduced_mesh_interpolator:
                            truth_problem_to_reduced_mesh_interpolator[truth_problem] = list()
                        truth_problem_to_reduced_mesh_interpolator[truth_problem].append(at.get_auxiliary_function_interpolator(truth_problem, component))
                    else:
                        (auxiliary_problem, component) = wrapping.get_auxiliary_problem_for_non_parametrized_function(node)
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
                    for parent_node in wrapping.solution_iterator(preprocessed_node):
                        visited.add(parent_node)
                # ... geometric quantities
                elif isinstance(node, GeometricQuantity):
                    replacements[node] = type(node)(reduced_mesh)
                    visited.add(node)
            # ... and replace them
            replaced_expression = wrapping.expression_replace(expression, replacements)
            
            # Cache the resulting dicts
            expression_on_reduced_mesh__expression_cache[(expression_name, reduced_mesh)] = replaced_expression
            expression_on_reduced_mesh__truth_problems_cache[(expression_name, reduced_mesh)] = truth_problems
            expression_on_reduced_mesh__truth_problem_to_components_cache[(expression_name, reduced_mesh)] = truth_problem_to_components
            expression_on_reduced_mesh__truth_problem_to_exact_truth_problem_cache[(expression_name, reduced_mesh)] = truth_problem_to_exact_truth_problem
            expression_on_reduced_mesh__truth_problem_to_reduced_mesh_solution_cache[(expression_name, reduced_mesh)] = truth_problem_to_reduced_mesh_solution
            expression_on_reduced_mesh__truth_problem_to_reduced_mesh_interpolator_cache[(expression_name, reduced_mesh)] = truth_problem_to_reduced_mesh_interpolator
            expression_on_reduced_mesh__reduced_problem_to_components_cache[(expression_name, reduced_mesh)] = reduced_problem_to_components
            expression_on_reduced_mesh__reduced_problem_to_reduced_mesh_solution_cache[(expression_name, reduced_mesh)] = reduced_problem_to_reduced_mesh_solution
            expression_on_reduced_mesh__reduced_problem_to_reduced_basis_functions_cache[(expression_name, reduced_mesh)] = reduced_problem_to_reduced_basis_functions
            
        # Extract from cache
        replaced_expression = expression_on_reduced_mesh__expression_cache[(expression_name, reduced_mesh)]
        truth_problems = expression_on_reduced_mesh__truth_problems_cache[(expression_name, reduced_mesh)]
        truth_problem_to_components = expression_on_reduced_mesh__truth_problem_to_components_cache[(expression_name, reduced_mesh)]
        truth_problem_to_exact_truth_problem = expression_on_reduced_mesh__truth_problem_to_exact_truth_problem_cache[(expression_name, reduced_mesh)]
        truth_problem_to_reduced_mesh_solution = expression_on_reduced_mesh__truth_problem_to_reduced_mesh_solution_cache[(expression_name, reduced_mesh)]
        truth_problem_to_reduced_mesh_interpolator = expression_on_reduced_mesh__truth_problem_to_reduced_mesh_interpolator_cache[(expression_name, reduced_mesh)]
        reduced_problem_to_components = expression_on_reduced_mesh__reduced_problem_to_components_cache[(expression_name, reduced_mesh)]
        reduced_problem_to_reduced_mesh_solution = expression_on_reduced_mesh__reduced_problem_to_reduced_mesh_solution_cache[(expression_name, reduced_mesh)]
        reduced_problem_to_reduced_basis_functions = expression_on_reduced_mesh__reduced_problem_to_reduced_basis_functions_cache[(expression_name, reduced_mesh)]
        
        # Get list of truth and reduced problems that need to be solved, possibly updating cache
        required_truth_problems = list()
        required_reduced_problems = list()
        for truth_problem in truth_problems:
            truth_problem_is_solving = hasattr(truth_problem, "_is_solving")
            if is_training_started(truth_problem):
                reduced_problem = get_reduced_problem_from_problem(truth_problem)
                reduced_problem_is_solving = hasattr(reduced_problem, "_is_solving")
            else:
                reduced_problem = None
                reduced_problem_is_solving = False
            if not truth_problem_is_solving:
                if is_training_finished(truth_problem):
                    # Store the component
                    if reduced_problem not in reduced_problem_to_components:
                        reduced_problem_to_components[reduced_problem] = truth_problem_to_components[truth_problem]
                    # Store the replacement
                    if reduced_problem not in reduced_problem_to_reduced_mesh_solution:
                        reduced_problem_to_reduced_mesh_solution[reduced_problem] = truth_problem_to_reduced_mesh_solution[truth_problem]
                    # Get reduced problem basis functions on reduced mesh
                    if reduced_problem not in reduced_problem_to_reduced_basis_functions:
                        reduced_problem_to_reduced_basis_functions[reduced_problem] = list()
                        for component in reduced_problem_to_components[reduced_problem]:
                            reduced_problem_to_reduced_basis_functions[reduced_problem].append(at.get_auxiliary_basis_functions_matrix(truth_problem, reduced_problem, component))
                    # Append to list of required reduced problems
                    required_reduced_problems.append((reduced_problem, reduced_problem_is_solving))
                else:
                    if (
                        hasattr(truth_problem, "_apply_exact_evaluation_at_stages")
                            and
                        not hasattr(truth_problem, "_apply_EIM_at_stages")
                            and
                        not hasattr(truth_problem, "_apply_DEIM_at_stages")
                    ):
                        # Init truth problem (if required), as it may not have been initialized
                        truth_problem.init()
                        # Append to list of required truth problems which are not currently solving
                        required_truth_problems.append((truth_problem, False, reduced_problem_is_solving))
                    else:
                        # Store the corresponding exact truth problem
                        if truth_problem not in truth_problem_to_exact_truth_problem:
                            exact_truth_problem = exact_problem(truth_problem)
                            truth_problem_to_exact_truth_problem[truth_problem] = exact_truth_problem
                            # Init exact truth problem (if required), as it may not have been initialized
                            exact_truth_problem.init()
                        else:
                            exact_truth_problem = truth_problem_to_exact_truth_problem[truth_problem]
                        # Store the component
                        if exact_truth_problem not in truth_problem_to_components:
                            truth_problem_to_components[exact_truth_problem] = truth_problem_to_components[truth_problem]
                        # Store the replacement
                        if exact_truth_problem not in truth_problem_to_reduced_mesh_solution:
                            truth_problem_to_reduced_mesh_solution[exact_truth_problem] = truth_problem_to_reduced_mesh_solution[truth_problem]
                        # Get interpolator on reduced mesh
                        if exact_truth_problem not in truth_problem_to_reduced_mesh_interpolator:
                            truth_problem_to_reduced_mesh_interpolator[exact_truth_problem] = list()
                            for component in truth_problem_to_components[exact_truth_problem]:
                                truth_problem_to_reduced_mesh_interpolator[exact_truth_problem].append(at.get_auxiliary_function_interpolator(exact_truth_problem, component))
                        # Append to list of required truth problems which are not currently solving
                        required_truth_problems.append((exact_truth_problem, False, reduced_problem_is_solving))
            else:
                assert not reduced_problem_is_solving
                # Append to list of required truth problems which are currently solving
                required_truth_problems.append((truth_problem, True, False))
        
        # Solve truth problems (which have not been reduced yet) associated to nonlinear terms
        for (truth_problem, truth_problem_is_solving, reduced_problem_is_solving) in required_truth_problems:
            if not reduced_problem_is_solving:
                # Solve (if necessary) ...
                truth_problem.set_mu(mu)
                if not truth_problem_is_solving:
                    log(PROGRESS, "In expression_on_reduced_mesh, requiring truth problem solve for problem " + truth_problem.name())
                    truth_problem.solve()
                else:
                    log(PROGRESS, "In expression_on_reduced_mesh, loading current truth problem solution for problem " + truth_problem.name())
            else:
                reduced_problem = get_reduced_problem_from_problem(truth_problem)
                log(PROGRESS, "In expression_on_reduced_mesh, replacing current truth problem solution with reduced solution for problem " + reduced_problem.truth_problem.name())
            # ... and assign to reduced_mesh_solution
            for (reduced_mesh_solution, reduced_mesh_interpolator) in zip(truth_problem_to_reduced_mesh_solution[truth_problem], truth_problem_to_reduced_mesh_interpolator[truth_problem]):
                solution_to = reduced_mesh_solution
                if not reduced_problem_is_solving:
                    solution_from = reduced_mesh_interpolator(truth_problem._solution)
                else:
                    solution_from = reduced_mesh_interpolator(reduced_problem.basis_functions[:reduced_problem._solution.N]*reduced_problem._solution)
                backend.assign(solution_to, solution_from)
        
        # Solve reduced problems associated to nonlinear terms
        for (reduced_problem, is_solving) in required_reduced_problems:
            # Solve (if necessary) ...
            reduced_problem.set_mu(mu)
            if not is_solving:
                log(PROGRESS, "In expression_on_reduced_mesh, requiring reduced problem solve for problem " + reduced_problem.truth_problem.name())
                reduced_problem.solve()
            else:
                log(PROGRESS, "In expression_on_reduced_mesh, loading current reduced problem solution for problem " + reduced_problem.truth_problem.name())
            # ... and assign to reduced_mesh_solution
            for (reduced_mesh_solution, reduced_basis_functions) in zip(reduced_problem_to_reduced_mesh_solution[reduced_problem], reduced_problem_to_reduced_basis_functions[reduced_problem]):
                solution_to = reduced_mesh_solution
                solution_from_N = OnlineSizeDict()
                for c, v in reduced_problem._solution.N.items():
                    if c in reduced_basis_functions._components_name:
                        solution_from_N[c] = v
                solution_from = online_backend.OnlineFunction(solution_from_N)
                online_backend.online_assign(solution_from, reduced_problem._solution)
                solution_from = reduced_basis_functions[:solution_from_N]*solution_from
                backend.assign(solution_to, solution_from)
        
        # Evaluate and return
        reduced_function = backend.Function(reduced_space)
        wrapping.evaluate_expression(expression, reduced_function, replaced_expression)
        return reduced_function
        
    expression_on_reduced_mesh__expression_cache = Cache()
    expression_on_reduced_mesh__truth_problems_cache = Cache()
    expression_on_reduced_mesh__truth_problem_to_components_cache = Cache()
    expression_on_reduced_mesh__truth_problem_to_exact_truth_problem_cache = Cache()
    expression_on_reduced_mesh__truth_problem_to_reduced_mesh_solution_cache = Cache()
    expression_on_reduced_mesh__truth_problem_to_reduced_mesh_interpolator_cache = Cache()
    expression_on_reduced_mesh__reduced_problem_to_components_cache = Cache()
    expression_on_reduced_mesh__reduced_problem_to_reduced_mesh_solution_cache = Cache()
    expression_on_reduced_mesh__reduced_problem_to_reduced_basis_functions_cache = Cache()
    
    return _basic_expression_on_reduced_mesh

# No explicit instantiation for backend = rbnics.backends.dolfin to avoid
# circular dependencies. The concrete instatiation will be carried out in
# rbnics.backends.function.evaluate
