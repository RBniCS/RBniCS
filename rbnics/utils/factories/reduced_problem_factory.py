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
## @file reduced_problem_factory.py
#  @brief Factory to generate a reduced problem corresponding to a given reduction method and truth problem
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

#~~~~~~~~~~~~~~~~~~~~~~~~~     PARAMETRIZED PROBLEM BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ReducedProblemFactory
#

from rbnics.utils.decorators import CustomizeReducedProblemFor, ReducedProblemFor, ReducedProblemDecoratorFor
from rbnics.utils.factories.factory_helper import FactoryGenerateTypes
from rbnics.utils.mpi import log, DEBUG

# Factory to generate a reduced problem corresponding to a given reduction method and truth problem
def ReducedProblemFactory(truth_problem, reduction_method, **kwargs):
    
    log(DEBUG,
        "In ReducedProblemFactory with\n" +
        "\ttruth problem = " + str(type(truth_problem)) + "\n" +
        "\treduction_method = " + str(type(reduction_method)) + "\n" +
        "\tkwargs = " + str(kwargs)
    )
    if hasattr(type(truth_problem), "ProblemDecorators"):
        log(DEBUG,
            "\ttruth problem decorators = " + 
                "\n\t\t".join([str(Decorator) for Decorator in type(truth_problem).ProblemDecorators])
            + "\n"
        )

    TypesList = list()
    
    # Generate ReducedProblem types based on Problem type
    def ReducedProblem_condition_on_dict_key(Problem):
        return isinstance(truth_problem, Problem)
    def ReducedProblem_condition_for_valid_candidate(tuple_):
        return isinstance(reduction_method, tuple_[1]) # 1-th entry stores the reduction method type
    def ReducedProblem_condition_for_candidate_replacement(candidate_replaces_if):
        return (
            candidate_replaces_if is None # replace in any case
                or
            candidate_replaces_if(truth_problem, **kwargs)
        )
    log(DEBUG, "Generate ReducedProblem types based on Problem type")
    TypesList.extend(
        FactoryGenerateTypes(ReducedProblemFor._all_reduced_problems, ReducedProblem_condition_on_dict_key, ReducedProblem_condition_for_valid_candidate, ReducedProblem_condition_for_candidate_replacement)
    )
    
    # Look if any customizer has been defined
    for (Problem, customizer) in CustomizeReducedProblemFor._all_reduced_problems_customizers.iteritems():
        if isinstance(truth_problem, Problem):
            TypesList.append(customizer)
    
    # Append ReducedProblemDecorator types based on Algorithm type
    if hasattr(type(truth_problem), "ProblemDecorators"):
        def ReducedProblemDecorator_condition_on_dict_key(Algorithm):
            return Algorithm in type(truth_problem).ProblemDecorators
        def ReducedProblemDecorator_condition_for_valid_candidate(tuple_):
            return True # always a valid candidate
        def ReducedProblemDecorator_condition_for_candidate_replacement(candidate_replaces_if):
            return (
                candidate_replaces_if is None # replace in any case
                    or
                candidate_replaces_if(truth_problem, **kwargs)
            )
        log(DEBUG, "Append ReducedProblemDecorator types based on Algorithm type")
        TypesList.extend(
            FactoryGenerateTypes(ReducedProblemDecoratorFor._all_reduced_problems_decorators, ReducedProblemDecorator_condition_on_dict_key, ReducedProblemDecorator_condition_for_valid_candidate, ReducedProblemDecorator_condition_for_candidate_replacement)
        )
    
    # Log
    log(DEBUG, "The reduced problem is a composition of the following types:")
    for t in range(len(TypesList) - 1, -1, -1):
        log(DEBUG, str(TypesList[t]))
    log(DEBUG, "\n")
    
    # Compose all types
    assert len(TypesList) > 0
    ComposedType = TypesList[0]
    for t in range(1, len(TypesList)):
        ComposedType = TypesList[t](ComposedType)
        
    # Finally, return an instance of the generated class
    return ComposedType(truth_problem, **kwargs)
    
