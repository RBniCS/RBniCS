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

from rbnics.backends.dolfin.wrapping.function_extend_or_restrict import _sub_from_tuple
from rbnics.eim.utils.decorators import get_problem_from_parametrized_expression
from rbnics.utils.decorators import exact_problem, get_problem_from_solution, get_reduced_problem_from_problem, is_training_finished, is_training_started
from rbnics.utils.mpi import log, PROGRESS

def basic_expression_on_truth_mesh(backend, wrapping):
    def _basic_expression_on_truth_mesh(expression_wrapper, function=None):
        expression = expression_wrapper._expression
        expression_name = expression_wrapper._name
        space = expression_wrapper._space
        mu = get_problem_from_parametrized_expression(expression_wrapper).mu
        
        if expression_name not in expression_on_truth_mesh__reduced_problem_to_truth_solution_cache:
            visited = set()
            truth_problems = list()
            truth_problem_to_components = dict()
            truth_problem_to_exact_truth_problem = dict()
            truth_problem_to_truth_solution = dict()
            reduced_problem_to_components = dict()
            reduced_problem_to_truth_solution = dict()
            
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
                        # Store the solution
                        truth_problem_to_truth_solution[truth_problem] = truth_solution
                        # Store the component
                        if truth_problem not in truth_problem_to_components:
                            truth_problem_to_components[truth_problem] = list()
                        truth_problem_to_components[truth_problem].append(component)
                    else:
                        preprocessed_node = node
                    # Make sure to skip any parent solution related to this one
                    visited.add(node)
                    visited.add(preprocessed_node)
                    for parent_node in wrapping.solution_iterator(preprocessed_node):
                        visited.add(parent_node)
                        
            # Cache the resulting dicts
            expression_on_truth_mesh__truth_problems_cache[expression_name] = truth_problems
            expression_on_truth_mesh__truth_problem_to_components_cache[expression_name] = truth_problem_to_components
            expression_on_truth_mesh__truth_problem_to_exact_truth_problem_cache[expression_name] = truth_problem_to_exact_truth_problem
            expression_on_truth_mesh__truth_problem_to_truth_solution_cache[expression_name] = truth_problem_to_truth_solution
            expression_on_truth_mesh__reduced_problem_to_components_cache[expression_name] = reduced_problem_to_components
            expression_on_truth_mesh__reduced_problem_to_truth_solution_cache[expression_name] = reduced_problem_to_truth_solution
            
        # Extract from cache
        truth_problems = expression_on_truth_mesh__truth_problems_cache[expression_name]
        truth_problem_to_components = expression_on_truth_mesh__truth_problem_to_components_cache[expression_name]
        truth_problem_to_exact_truth_problem = expression_on_truth_mesh__truth_problem_to_exact_truth_problem_cache[expression_name]
        truth_problem_to_truth_solution = expression_on_truth_mesh__truth_problem_to_truth_solution_cache[expression_name]
        reduced_problem_to_components = expression_on_truth_mesh__reduced_problem_to_components_cache[expression_name]
        reduced_problem_to_truth_solution = expression_on_truth_mesh__reduced_problem_to_truth_solution_cache[expression_name]
        
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
                    # Store the solution
                    if reduced_problem not in reduced_problem_to_truth_solution:
                        reduced_problem_to_truth_solution[reduced_problem] = truth_problem_to_truth_solution[truth_problem]
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
                        # Store the solution
                        if exact_truth_problem not in truth_problem_to_truth_solution:
                            truth_problem_to_truth_solution[exact_truth_problem] = truth_problem_to_truth_solution[truth_problem]
                        # Append to list of required truth problems which are not currently solving
                        required_truth_problems.append((exact_truth_problem, False, reduced_problem_is_solving))
            else:
                assert not reduced_problem_is_solving
                # Append to list of required truth problems which are currently solving
                required_truth_problems.append((truth_problem, True, False))
        
        # Solve truth problems (which have not been reduced yet) associated to nonlinear terms
        truth_problem_to_truth_solution_copy = dict()
        for (truth_problem, truth_problem_is_solving, reduced_problem_is_solving) in required_truth_problems:
            if not reduced_problem_is_solving:
                # Solve (if necessary) ...
                truth_problem.set_mu(mu)
                if not truth_problem_is_solving:
                    log(PROGRESS, "In expression_on_truth_mesh, requiring truth problem solve for problem " + truth_problem.name())
                    truth_problem.solve()
                else:
                    log(PROGRESS, "In expression_on_truth_mesh, loading current truth problem solution for problem " + truth_problem.name())
            else:
                reduced_problem = get_reduced_problem_from_problem(truth_problem)
                log(PROGRESS, "In expression_on_truth_mesh, replacing current truth problem solution with reduced solution for problem " + reduced_problem.truth_problem.name())
            # ... and assign to truth_solution
            truth_solution = truth_problem_to_truth_solution[truth_problem]
            truth_problem_to_truth_solution_copy[truth_problem] = backend.copy(truth_solution)
            for component in truth_problem_to_components[truth_problem]:
                solution_to = _sub_from_tuple(truth_solution, component)
                if not reduced_problem_is_solving:
                    solution_from = _sub_from_tuple(truth_problem._solution, component)
                else:
                    solution_from = _sub_from_tuple(reduced_problem.basis_functions[:reduced_problem._solution.N]*reduced_problem._solution, component)
                backend.assign(solution_to, solution_from)
            
        # Solve reduced problems associated to nonlinear terms
        reduced_problem_to_truth_solution_copy = dict()
        for (reduced_problem, is_solving) in required_reduced_problems:
            # Solve (if necessary) ...
            reduced_problem.set_mu(mu)
            if not is_solving:
                log(PROGRESS, "In expression_on_truth_mesh, requiring reduced problem solve for problem " + reduced_problem.truth_problem.name())
                reduced_problem.solve()
            else:
                log(PROGRESS, "In expression_on_truth_mesh, loading current reduced problem solution for problem " + reduced_problem.truth_problem.name())
            # ... and assign to truth_solution
            truth_solution = reduced_problem_to_truth_solution[reduced_problem]
            reduced_problem_to_truth_solution_copy[reduced_problem] = backend.copy(truth_solution)
            for component in reduced_problem_to_components[reduced_problem]:
                solution_to = _sub_from_tuple(truth_solution, component)
                solution_from = _sub_from_tuple(reduced_problem.basis_functions[:reduced_problem._solution.N]*reduced_problem._solution, component)
                backend.assign(solution_to, solution_from)
        
        # Evaluate
        if function is None:
            function = backend.Function(space)
        wrapping.evaluate_expression(expression, function)
        
        # Undo any side effect of truth problem solves
        for (truth_problem, _, _) in required_truth_problems:
            truth_solution = truth_problem_to_truth_solution[truth_problem]
            truth_solution_copy = truth_problem_to_truth_solution_copy[truth_problem]
            for component in truth_problem_to_components[truth_problem]:
                solution_to = _sub_from_tuple(truth_solution, component)
                solution_from = _sub_from_tuple(truth_solution_copy, component)
                backend.assign(solution_to, solution_from)
        
        # Undo any side effect of reduced problem solves
        for (reduced_problem, _) in required_reduced_problems:
            truth_solution = reduced_problem_to_truth_solution[reduced_problem]
            truth_solution_copy = reduced_problem_to_truth_solution_copy[reduced_problem]
            for component in reduced_problem_to_components[reduced_problem]:
                solution_to = _sub_from_tuple(truth_solution, component)
                solution_from = _sub_from_tuple(truth_solution_copy, component)
                backend.assign(solution_to, solution_from)
        
        # Return
        return function
    
    expression_on_truth_mesh__truth_problems_cache = dict()
    expression_on_truth_mesh__truth_problem_to_components_cache = dict()
    expression_on_truth_mesh__truth_problem_to_exact_truth_problem_cache = dict()
    expression_on_truth_mesh__truth_problem_to_truth_solution_cache = dict()
    expression_on_truth_mesh__reduced_problem_to_components_cache = dict()
    expression_on_truth_mesh__reduced_problem_to_truth_solution_cache = dict()
    
    return _basic_expression_on_truth_mesh

# No explicit instantiation for backend = rbnics.backends.dolfin to avoid
# circular dependencies. The concrete instatiation will be carried out in
# rbnics.backends.dolfin.evaluate
