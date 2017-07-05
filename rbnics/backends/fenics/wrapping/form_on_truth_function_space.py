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

import rbnics.backends.fenics
from rbnics.utils.decorators import exact_problem, get_problem_from_solution, get_reduced_problem_from_problem, is_training_finished
from rbnics.backends.fenics.wrapping.function_extend_or_restrict import _sub_from_tuple
from rbnics.utils.mpi import log, PROGRESS
from rbnics.eim.utils.decorators import get_EIM_approximation_from_parametrized_expression

def form_on_truth_function_space(form_wrapper, backend=None):
    if backend is None:
        backend = rbnics.backends.fenics
    
    form = form_wrapper._form
    form_name = form_wrapper._name
    EIM_approximation = get_EIM_approximation_from_parametrized_expression(form_wrapper)
    
    if form_name not in form_on_truth_function_space__reduced_problem_to_truth_solution_cache:
        visited = set()
        truth_problem_to_components = dict() # from truth problem to components
        truth_problem_to_truth_solution = dict() # from truth problem to solution
        reduced_problem_to_components = dict() # from reduced problem to components
        reduced_problem_to_truth_solution = dict() # from reduced problem to solution
        
        # Look for terminals on truth mesh
        for node in backend.wrapping.form_iterator(form):
            if node in visited:
                continue
            # ... problem solutions related to nonlinear terms
            elif backend.wrapping.is_problem_solution_or_problem_solution_component_type(node):
                if backend.wrapping.is_problem_solution_or_problem_solution_component(node):
                    (preprocessed_node, component, truth_solution) = backend.wrapping.solution_identify_component(node)
                    truth_problem = get_problem_from_solution(truth_solution)
                    if is_training_finished(truth_problem):
                        reduced_problem = get_reduced_problem_from_problem(truth_problem)
                        if reduced_problem not in reduced_problem_to_components:
                            reduced_problem_to_components[reduced_problem] = list()
                        reduced_problem_to_components[reduced_problem].append(component)
                        reduced_problem_to_truth_solution[reduced_problem] = truth_solution
                    else:
                        if not hasattr(truth_problem, "_is_solving"):
                            exact_truth_problem = exact_problem(truth_problem)
                            exact_truth_problem.init()
                            if exact_truth_problem not in truth_problem_to_components:
                                truth_problem_to_components[exact_truth_problem] = list()
                            truth_problem_to_components[exact_truth_problem].append(component)
                            truth_problem_to_truth_solution[exact_truth_problem] = truth_solution
                        else:
                            if truth_problem not in truth_problem_to_components:
                                truth_problem_to_components[truth_problem] = list()
                            truth_problem_to_components[truth_problem].append(component)
                            truth_problem_to_truth_solution[truth_problem] = truth_solution
                else:
                    preprocessed_node = node
                # Make sure to skip any parent solution related to this one
                visited.add(node)
                visited.add(preprocessed_node)
                for parent_node in backend.wrapping.solution_iterator(preprocessed_node):
                    visited.add(parent_node)
        
        # Cache the resulting dicts
        form_on_truth_function_space__truth_problem_to_components_cache[form_name] = truth_problem_to_components
        form_on_truth_function_space__truth_problem_to_truth_solution_cache[form_name] = truth_problem_to_truth_solution
        form_on_truth_function_space__reduced_problem_to_components_cache[form_name] = reduced_problem_to_components
        form_on_truth_function_space__reduced_problem_to_truth_solution_cache[form_name] = reduced_problem_to_truth_solution
        
    # Extract from cache
    truth_problem_to_components = form_on_truth_function_space__truth_problem_to_components_cache[form_name]
    truth_problem_to_truth_solution = form_on_truth_function_space__truth_problem_to_truth_solution_cache[form_name]
    reduced_problem_to_components = form_on_truth_function_space__reduced_problem_to_components_cache[form_name]
    reduced_problem_to_truth_solution = form_on_truth_function_space__reduced_problem_to_truth_solution_cache[form_name]
    
    # Solve truth problems (which have not been reduced yet) associated to nonlinear terms
    for (truth_problem, truth_solution) in truth_problem_to_truth_solution.iteritems():
        truth_problem.set_mu(EIM_approximation.mu)
        if not hasattr(truth_problem, "_is_solving"):
            log(PROGRESS, "In form_on_truth_function_space, requiring truth problem solve for problem " + str(truth_problem))
            truth_problem.solve()
        else:
            log(PROGRESS, "In form_on_truth_function_space, loading current truth problem solution for problem " + str(truth_problem))
        for component in truth_problem_to_components[truth_problem]:
            solution_to = _sub_from_tuple(truth_solution, component)
            solution_from = _sub_from_tuple(truth_problem._solution, component)
            backend.assign(solution_to, solution_from)
    
    # Solve reduced problems associated to nonlinear terms
    for (reduced_problem, truth_solution) in reduced_problem_to_truth_solution.iteritems():
        reduced_problem.set_mu(EIM_approximation.mu)
        assert not hasattr(reduced_problem, "_is_solving")
        log(PROGRESS, "In form_on_truth_function_space, requiring reduced problem solve for problem " + str(reduced_problem))
        reduced_problem.solve()
        for component in reduced_problem_to_components[reduced_problem]:
            solution_to = _sub_from_tuple(truth_solution, component)
            solution_from = _sub_from_tuple(reduced_problem.Z[:reduced_problem._solution.N]*reduced_problem._solution, component)
            backend.assign(solution_to, solution_from)
    
    # Assemble and return
    tensor = backend.wrapping.assemble(form)
    tensor.generator = form_wrapper # for I/O
    return tensor

form_on_truth_function_space__truth_problem_to_components_cache = dict()    
form_on_truth_function_space__truth_problem_to_truth_solution_cache = dict()
form_on_truth_function_space__reduced_problem_to_components_cache = dict()
form_on_truth_function_space__reduced_problem_to_truth_solution_cache = dict()
