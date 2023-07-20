# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from logging import DEBUG, getLogger
try:
    from ufl_legacy.geometry import GeometricQuantity
except ImportError:
    from ufl.geometry import GeometricQuantity
from rbnics.eim.utils.decorators import get_problem_from_parametrized_expression
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import (exact_problem, get_problem_from_solution, get_problem_from_solution_dot,
                                     get_reduced_problem_from_problem, is_training_finished, is_training_started)
from rbnics.utils.io import OnlineSizeDict

logger = getLogger("rbnics/backends/dolfin/expression_on_reduced_mesh.py")


def basic_expression_on_reduced_mesh(backend, wrapping, online_backend, online_wrapping):

    def _basic_expression_on_reduced_mesh(expression_wrapper, at):
        expression = expression_wrapper._expression
        expression_name = expression_wrapper.name()
        expression_problem = get_problem_from_parametrized_expression(expression_wrapper)
        reduced_space = at.get_reduced_function_space()
        reduced_mesh = at.get_reduced_mesh()
        mu = expression_problem.mu
        if hasattr(expression_problem, "set_time"):
            t = expression_problem.t
        else:
            t = None

        if (expression_name, reduced_mesh) not in expression_cache:
            visited = set()
            replacements = dict()
            truth_problems = list()
            truth_problem_to_components = {  # outer dict index over time derivative
                0: dict(),
                1: dict()}
            truth_problem_to_exact_truth_problem = dict()
            truth_problem_to_reduced_mesh_solution = dict()
            truth_problem_to_reduced_mesh_solution_dot = dict()
            truth_problem_to_reduced_mesh_interpolator = {  # outer dict index over time derivative
                0: dict(),
                1: dict()}
            reduced_problem_to_components = {  # outer dict index over time derivative
                0: dict(),
                1: dict()}
            reduced_problem_to_reduced_mesh_solution = dict()
            reduced_problem_to_reduced_mesh_solution_dot = dict()
            reduced_problem_to_reduced_basis_functions = {  # outer dict index over time derivative
                0: dict(),
                1: dict()}

            # Look for terminals on truth mesh
            logger.log(DEBUG, "Traversing terminals of expression " + expression_name)
            for node in wrapping.expression_iterator(expression):
                if node in visited:
                    continue
                # ... problem solutions related to nonlinear terms
                elif wrapping.is_problem_solution_type(node):
                    node_is_problem_solution = wrapping.is_problem_solution(node)
                    node_is_problem_solution_dot = wrapping.is_problem_solution_dot(node)
                    if node_is_problem_solution or node_is_problem_solution_dot:
                        if node_is_problem_solution:
                            (preprocessed_node, component, truth_solution) = wrapping.solution_identify_component(node)
                            truth_problem = get_problem_from_solution(truth_solution)
                            logger.log(DEBUG, "\tFound problem solution of truth problem " + truth_problem.name()
                                       + " (exact problem decorator: " + str(hasattr(truth_problem, "__is_exact__"))
                                       + ", component: " + str(component) + ")")
                            # Time derivative key for components and interpolator dicts
                            time_derivative = 0
                        elif node_is_problem_solution_dot:
                            (preprocessed_node, component,
                             truth_solution_dot) = wrapping.solution_dot_identify_component(node)
                            truth_problem = get_problem_from_solution_dot(truth_solution_dot)
                            logger.log(DEBUG, "\tFound problem solution dot of truth problem " + truth_problem.name()
                                       + " (exact problem decorator: " + str(hasattr(truth_problem, "__is_exact__"))
                                       + ", component: " + str(component) + ")")
                            # Time derivative key for components and interpolator dicts
                            time_derivative = 1
                        # Store truth problem
                        if truth_problem not in truth_problems:
                            truth_problems.append(truth_problem)
                        # Store the component
                        if truth_problem not in truth_problem_to_components[time_derivative]:
                            truth_problem_to_components[time_derivative][truth_problem] = list()
                        if component not in truth_problem_to_components[time_derivative][truth_problem]:
                            truth_problem_to_components[time_derivative][truth_problem].append(component)
                            # Get the function space corresponding to preprocessed_node on the reduced mesh
                            auxiliary_reduced_V = at.get_auxiliary_reduced_function_space(truth_problem, component)
                            # Define and store the replacement
                            assert preprocessed_node not in replacements
                            replacements[preprocessed_node] = backend.Function(auxiliary_reduced_V)
                            if time_derivative == 0:
                                if truth_problem not in truth_problem_to_reduced_mesh_solution:
                                    truth_problem_to_reduced_mesh_solution[truth_problem] = list()
                                truth_problem_to_reduced_mesh_solution[truth_problem].append(
                                    replacements[preprocessed_node])
                            elif time_derivative == 1:
                                if truth_problem not in truth_problem_to_reduced_mesh_solution_dot:
                                    truth_problem_to_reduced_mesh_solution_dot[truth_problem] = list()
                                truth_problem_to_reduced_mesh_solution_dot[truth_problem].append(
                                    replacements[preprocessed_node])
                            # Get interpolator on reduced mesh
                            if truth_problem not in truth_problem_to_reduced_mesh_interpolator[time_derivative]:
                                truth_problem_to_reduced_mesh_interpolator[time_derivative][truth_problem] = list()
                            truth_problem_to_reduced_mesh_interpolator[time_derivative][truth_problem].append(
                                at.get_auxiliary_function_interpolator(truth_problem, component))
                    else:
                        (preprocessed_node, component,
                         auxiliary_problem) = wrapping.get_auxiliary_problem_for_non_parametrized_function(node)
                        logger.log(DEBUG, "\tFound non parametrized function " + str(preprocessed_node)
                                   + " associated to auxiliary problem " + str(auxiliary_problem.name())
                                   + ", component: " + str(component))
                        if preprocessed_node not in replacements:
                            # Get interpolator on reduced mesh
                            auxiliary_truth_problem_to_reduced_mesh_interpolator = (
                                at.get_auxiliary_function_interpolator(auxiliary_problem, component))
                            # Define and store the replacement
                            replacements[preprocessed_node] = auxiliary_truth_problem_to_reduced_mesh_interpolator(
                                preprocessed_node)
                    # Make sure to skip any parent solution related to this one
                    visited.add(node)
                    visited.add(preprocessed_node)
                    for parent_node in wrapping.solution_iterator(preprocessed_node):
                        visited.add(parent_node)
                # ... geometric quantities
                elif isinstance(node, GeometricQuantity):
                    logger.log(DEBUG, "\tFound geometric quantity " + str(node))
                    replacements[node] = type(node)(reduced_mesh)
                    visited.add(node)
                else:
                    visited.add(node)
            # ... and replace them
            replaced_expression = wrapping.expression_replace(expression, replacements)

            # Cache the resulting dicts
            expression_cache[(expression_name, reduced_mesh)] = replaced_expression
            truth_problems_cache[(expression_name, reduced_mesh)] = truth_problems
            truth_problem_to_components_cache[(expression_name, reduced_mesh)] = truth_problem_to_components
            truth_problem_to_exact_truth_problem_cache[
                (expression_name, reduced_mesh)] = truth_problem_to_exact_truth_problem
            truth_problem_to_reduced_mesh_solution_cache[
                (expression_name, reduced_mesh)] = truth_problem_to_reduced_mesh_solution
            truth_problem_to_reduced_mesh_solution_dot_cache[
                (expression_name, reduced_mesh)] = truth_problem_to_reduced_mesh_solution_dot
            truth_problem_to_reduced_mesh_interpolator_cache[
                (expression_name, reduced_mesh)] = truth_problem_to_reduced_mesh_interpolator
            reduced_problem_to_components_cache[
                (expression_name, reduced_mesh)] = reduced_problem_to_components
            reduced_problem_to_reduced_mesh_solution_cache[
                (expression_name, reduced_mesh)] = reduced_problem_to_reduced_mesh_solution
            reduced_problem_to_reduced_mesh_solution_dot_cache[
                (expression_name, reduced_mesh)] = reduced_problem_to_reduced_mesh_solution_dot
            reduced_problem_to_reduced_basis_functions_cache[
                (expression_name, reduced_mesh)] = reduced_problem_to_reduced_basis_functions

        # Extract from cache
        replaced_expression = expression_cache[(expression_name, reduced_mesh)]
        truth_problems = truth_problems_cache[(expression_name, reduced_mesh)]
        truth_problem_to_components = truth_problem_to_components_cache[(expression_name, reduced_mesh)]
        truth_problem_to_exact_truth_problem = truth_problem_to_exact_truth_problem_cache[
            (expression_name, reduced_mesh)]
        truth_problem_to_reduced_mesh_solution = truth_problem_to_reduced_mesh_solution_cache[
            (expression_name, reduced_mesh)]
        truth_problem_to_reduced_mesh_solution_dot = truth_problem_to_reduced_mesh_solution_dot_cache[
            (expression_name, reduced_mesh)]
        truth_problem_to_reduced_mesh_interpolator = truth_problem_to_reduced_mesh_interpolator_cache[
            (expression_name, reduced_mesh)]
        reduced_problem_to_components = reduced_problem_to_components_cache[(expression_name, reduced_mesh)]
        reduced_problem_to_reduced_mesh_solution = reduced_problem_to_reduced_mesh_solution_cache[
            (expression_name, reduced_mesh)]
        reduced_problem_to_reduced_mesh_solution_dot = reduced_problem_to_reduced_mesh_solution_dot_cache[
            (expression_name, reduced_mesh)]
        reduced_problem_to_reduced_basis_functions = reduced_problem_to_reduced_basis_functions_cache[
            (expression_name, reduced_mesh)]

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
                    logger.log(DEBUG, "Truth problem " + truth_problem.name()
                               + " (exact problem decorator: " + str(hasattr(truth_problem, "__is_exact__"))
                               + ") is not currently solving, and its offline stage has finished:"
                               + " truth problem will be replaced by reduced problem")
                    # Store the replacement for solution
                    if (reduced_problem not in reduced_problem_to_reduced_mesh_solution
                            and truth_problem in truth_problem_to_reduced_mesh_solution):
                        reduced_problem_to_reduced_mesh_solution[
                            reduced_problem] = truth_problem_to_reduced_mesh_solution[truth_problem]
                        # Store the component
                        assert reduced_problem not in reduced_problem_to_components[0]
                        assert truth_problem in truth_problem_to_components[0]
                        reduced_problem_to_components[
                            0][reduced_problem] = truth_problem_to_components[0][truth_problem]
                        # Get reduced problem basis functions on reduced mesh
                        assert reduced_problem not in reduced_problem_to_reduced_basis_functions[0]
                        reduced_problem_to_reduced_basis_functions[0][reduced_problem] = [
                            at.get_auxiliary_basis_functions_matrix(truth_problem, component)
                            for component in reduced_problem_to_components[0][reduced_problem]]
                    # Store the replacement for solution_dot
                    if (reduced_problem not in reduced_problem_to_reduced_mesh_solution_dot
                            and truth_problem in truth_problem_to_reduced_mesh_solution_dot):
                        reduced_problem_to_reduced_mesh_solution_dot[
                            reduced_problem] = truth_problem_to_reduced_mesh_solution_dot[truth_problem]
                        # Store the component
                        assert reduced_problem not in reduced_problem_to_components[1]
                        assert truth_problem in truth_problem_to_components[1]
                        reduced_problem_to_components[
                            1][reduced_problem] = truth_problem_to_components[1][truth_problem]
                        # Get reduced problem basis functions on reduced mesh
                        assert reduced_problem not in reduced_problem_to_reduced_basis_functions[1]
                        reduced_problem_to_reduced_basis_functions[1][reduced_problem] = [
                            at.get_auxiliary_basis_functions_matrix(truth_problem, component)
                            for component in reduced_problem_to_components[1][reduced_problem]]
                    # Append to list of required reduced problems
                    required_reduced_problems.append((reduced_problem, reduced_problem_is_solving))
                else:
                    if (hasattr(truth_problem, "_apply_exact_evaluation_at_stages")
                            and not hasattr(truth_problem, "_apply_EIM_at_stages")
                            and not hasattr(truth_problem, "_apply_DEIM_at_stages")):
                        logger.log(DEBUG, "Truth problem " + truth_problem.name()
                                   + " (exact problem decorator: " + str(hasattr(truth_problem, "__is_exact__"))
                                   + ") is not currently solving, its offline stage has not finished,"
                                   + " and only @ExactParametrizedFunctions has been used:"
                                   + " truth solve of this truth problem instance will be called")
                        # Init truth problem (if required), as it may not have been initialized
                        truth_problem.init()
                        # Append to list of required truth problems which are not currently solving
                        required_truth_problems.append((truth_problem, False, reduced_problem_is_solving))
                    else:
                        logger.log(DEBUG, "Truth problem " + truth_problem.name()
                                   + " (exact problem decorator: " + str(hasattr(truth_problem, "__is_exact__"))
                                   + ") is not currently solving, its offline stage has not finished,"
                                   + " and either @ExactParametrizedFunctions has not been used"
                                   + " or it has been used in combination with @DEIM or @EIM:"
                                   + " truth solve on an auxiliary instance (with exact problem decorator)"
                                   + " will be called, to prevent early initialization of DEIM/EIM data structures")
                        # Store the corresponding exact truth problem
                        if truth_problem not in truth_problem_to_exact_truth_problem:
                            exact_truth_problem = exact_problem(truth_problem)
                            truth_problem_to_exact_truth_problem[truth_problem] = exact_truth_problem
                            # Init exact truth problem (if required), as it may not have been initialized
                            exact_truth_problem.init()
                        else:
                            exact_truth_problem = truth_problem_to_exact_truth_problem[truth_problem]
                        # Store the replacement for solution
                        if (exact_truth_problem not in truth_problem_to_reduced_mesh_solution
                                and truth_problem in truth_problem_to_reduced_mesh_solution):
                            truth_problem_to_reduced_mesh_solution[
                                exact_truth_problem] = truth_problem_to_reduced_mesh_solution[truth_problem]
                            # Store the component
                            assert exact_truth_problem not in truth_problem_to_components[0]
                            assert truth_problem in truth_problem_to_components[0]
                            truth_problem_to_components[
                                0][exact_truth_problem] = truth_problem_to_components[0][truth_problem]
                            # Get interpolator on reduced mesh
                            assert exact_truth_problem not in truth_problem_to_reduced_mesh_interpolator[0]
                            assert truth_problem in truth_problem_to_reduced_mesh_interpolator[0]
                            truth_problem_to_reduced_mesh_interpolator[
                                0][exact_truth_problem] = truth_problem_to_reduced_mesh_interpolator[0][truth_problem]
                        # Store the replacement for solution_dot
                        if (exact_truth_problem not in truth_problem_to_reduced_mesh_solution_dot
                                and truth_problem in truth_problem_to_reduced_mesh_solution_dot):
                            truth_problem_to_reduced_mesh_solution_dot[
                                exact_truth_problem] = truth_problem_to_reduced_mesh_solution_dot[truth_problem]
                            # Store the component
                            assert exact_truth_problem not in truth_problem_to_components[1]
                            assert truth_problem in truth_problem_to_components[1]
                            truth_problem_to_components[
                                1][exact_truth_problem] = truth_problem_to_components[1][truth_problem]
                            # Get interpolator on reduced mesh
                            assert exact_truth_problem not in truth_problem_to_reduced_mesh_interpolator[1]
                            assert truth_problem in truth_problem_to_reduced_mesh_interpolator[1]
                            truth_problem_to_reduced_mesh_interpolator[
                                1][exact_truth_problem] = truth_problem_to_reduced_mesh_interpolator[1][truth_problem]
                        # Append to list of required truth problems which are not currently solving
                        required_truth_problems.append((exact_truth_problem, False, reduced_problem_is_solving))
            else:
                logger.log(DEBUG, "Truth problem " + truth_problem.name()
                           + " (exact problem decorator: " + str(hasattr(truth_problem, "__is_exact__"))
                           + ") is currently solving: current truth solution will be loaded")
                assert not reduced_problem_is_solving
                # Append to list of required truth problems which are currently solving
                required_truth_problems.append((truth_problem, True, False))

        # Solve truth problems (which have not been reduced yet) associated to nonlinear terms
        for (truth_problem, truth_problem_is_solving, reduced_problem_is_solving) in required_truth_problems:
            if not reduced_problem_is_solving:
                # Solve (if necessary)
                truth_problem.set_mu(mu)
                if not truth_problem_is_solving:
                    logger.log(DEBUG, "Requiring truth problem solve for problem " + truth_problem.name()
                               + " (exact problem decorator: " + str(hasattr(truth_problem, "__is_exact__")) + ")")
                    truth_problem.solve()
                else:
                    logger.log(DEBUG, "Loading current truth problem solution for problem " + truth_problem.name()
                               + " (exact problem decorator: " + str(hasattr(truth_problem, "__is_exact__")) + ")")
            else:
                reduced_problem = get_reduced_problem_from_problem(truth_problem)
                logger.log(DEBUG, "Replacing current truth problem solution with reduced solution for problem "
                           + reduced_problem.truth_problem.name())
            # Assign to reduced_mesh_solution
            if truth_problem in truth_problem_to_reduced_mesh_solution:
                for (reduced_mesh_solution, reduced_mesh_interpolator) in zip(
                        truth_problem_to_reduced_mesh_solution[truth_problem],
                        truth_problem_to_reduced_mesh_interpolator[0][truth_problem]):
                    solution_to = reduced_mesh_solution
                    if t is None:
                        if not reduced_problem_is_solving:
                            solution_from = reduced_mesh_interpolator(truth_problem._solution)
                        else:
                            solution_from = reduced_mesh_interpolator(
                                reduced_problem.basis_functions[:reduced_problem._solution.N]
                                * reduced_problem._solution)
                    else:
                        if not reduced_problem_is_solving:
                            if not truth_problem_is_solving:
                                solution_from = reduced_mesh_interpolator(truth_problem._solution_over_time.at(t))
                            else:
                                solution_from = reduced_mesh_interpolator(truth_problem._solution)
                        else:
                            solution_from = reduced_mesh_interpolator(
                                reduced_problem.basis_functions[:reduced_problem._solution.N]
                                * reduced_problem._solution)
                    backend.assign(solution_to, solution_from)
            # Assign to reduced_mesh_solution_dot
            if truth_problem in truth_problem_to_reduced_mesh_solution_dot:
                for (reduced_mesh_solution_dot, reduced_mesh_interpolator) in zip(
                        truth_problem_to_reduced_mesh_solution_dot[truth_problem],
                        truth_problem_to_reduced_mesh_interpolator[1][truth_problem]):
                    solution_dot_to = reduced_mesh_solution_dot
                    assert t is not None
                    if not reduced_problem_is_solving:
                        if not truth_problem_is_solving:
                            solution_dot_from = reduced_mesh_interpolator(truth_problem._solution_dot_over_time.at(t))
                        else:
                            solution_dot_from = reduced_mesh_interpolator(truth_problem._solution_dot)
                    else:
                        solution_dot_from = reduced_mesh_interpolator(
                            reduced_problem.basis_functions[:reduced_problem._solution_dot.N]
                            * reduced_problem._solution_dot)
                    backend.assign(solution_dot_to, solution_dot_from)

        # Solve reduced problems associated to nonlinear terms
        for (reduced_problem, is_solving) in required_reduced_problems:
            # Solve (if necessary)
            reduced_problem.set_mu(mu)
            if not is_solving:
                logger.log(DEBUG, "In expression_on_reduced_mesh, requiring reduced problem solve for problem "
                           + reduced_problem.truth_problem.name())
                reduced_problem.solve()
            else:
                logger.log(DEBUG, "In expression_on_reduced_mesh, loading current reduced problem solution"
                           + " for problem " + reduced_problem.truth_problem.name())
            # Assign to reduced_mesh_solution
            if reduced_problem in reduced_problem_to_reduced_mesh_solution:
                for (reduced_mesh_solution, reduced_basis_functions) in zip(
                        reduced_problem_to_reduced_mesh_solution[reduced_problem],
                        reduced_problem_to_reduced_basis_functions[0][reduced_problem]):
                    solution_to = reduced_mesh_solution
                    solution_from_N = OnlineSizeDict()
                    for c, v in reduced_problem._solution.N.items():
                        if c in reduced_basis_functions._components_name:
                            solution_from_N[c] = v
                    solution_from = online_backend.OnlineFunction(solution_from_N)
                    if t is None or is_solving:
                        online_backend.online_assign(solution_from, reduced_problem._solution)
                    else:
                        online_backend.online_assign(solution_from, reduced_problem._solution_over_time.at(t))
                    solution_from = reduced_basis_functions[:solution_from_N] * solution_from
                    backend.assign(solution_to, solution_from)
            # Assign to reduced_mesh_solution_dot
            if reduced_problem in reduced_problem_to_reduced_mesh_solution_dot:
                for (reduced_mesh_solution_dot, reduced_basis_functions) in zip(
                        reduced_problem_to_reduced_mesh_solution_dot[reduced_problem],
                        reduced_problem_to_reduced_basis_functions[1][reduced_problem]):
                    solution_dot_to = reduced_mesh_solution_dot
                    solution_dot_from_N = OnlineSizeDict()
                    for c, v in reduced_problem._solution_dot.N.items():
                        if c in reduced_basis_functions._components_name:
                            solution_dot_from_N[c] = v
                    solution_dot_from = online_backend.OnlineFunction(solution_dot_from_N)
                    assert t is not None
                    if is_solving:
                        online_backend.online_assign(solution_dot_from, reduced_problem._solution_dot)
                    else:
                        online_backend.online_assign(solution_dot_from, reduced_problem._solution_dot_over_time.at(t))
                    solution_dot_from = reduced_basis_functions[:solution_dot_from_N] * solution_dot_from
                    backend.assign(solution_dot_to, solution_dot_from)

        # Evaluate and return
        reduced_function = backend.Function(reduced_space)
        wrapping.evaluate_expression(expression, reduced_function, replaced_expression)
        return reduced_function

    expression_cache = Cache()
    truth_problems_cache = Cache()
    truth_problem_to_components_cache = Cache()
    truth_problem_to_exact_truth_problem_cache = Cache()
    truth_problem_to_reduced_mesh_solution_cache = Cache()
    truth_problem_to_reduced_mesh_solution_dot_cache = Cache()
    truth_problem_to_reduced_mesh_interpolator_cache = Cache()
    reduced_problem_to_components_cache = Cache()
    reduced_problem_to_reduced_mesh_solution_cache = Cache()
    reduced_problem_to_reduced_mesh_solution_dot_cache = Cache()
    reduced_problem_to_reduced_basis_functions_cache = Cache()

    return _basic_expression_on_reduced_mesh

# No explicit instantiation for backend = rbnics.backends.dolfin to avoid
# circular dependencies. The concrete instatiation will be carried out in
# rbnics.backends.function.evaluate
