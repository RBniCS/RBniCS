# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file reduced_method_factory.py
#  @brief Factory to generate a reduction method corresponding to a category (e.g. RB or POD) and a given truth problem
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

#~~~~~~~~~~~~~~~~~~~~~~~~~     REDUCED METHOD FACTORY CLASSES     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ReducedBasis
#

from RBniCS.utils.decorators import ReductionMethodFor, ReductionMethodDecoratorFor
from RBniCS.utils.factories.factory_helper import FactoryGenerateTypes
from RBniCS.utils.io import log, DEBUG

# Factory to generate a reduction method corresponding to a category (e.g. RB or POD) and a given truth problem
def ReductionMethodFactory(truth_problem, category):

    log(DEBUG,
        "In ReductionMethodFactory with\n" +
        "\ttruth problem = " + str(type(truth_problem)) + "\n" +
        "\tcategory = " + str(category)
    )
    if hasattr(type(truth_problem), "ProblemDecorators"):
        log(DEBUG,
            "\ttruth problem decorators = " + 
                "\n\t\t".join([str(Decorator) for Decorator in type(truth_problem).ProblemDecorators])
            + "\n"
        )
    
    TypesList = list()
    
    # Generate ReductionMethod type based on Problem type
    def ReductionMethod_condition_on_dict_key(Problem):
        return isinstance(truth_problem, Problem)
    def ReductionMethod_condition_for_valid_candidate(tuple_):
        return category == tuple_[1] # 1-th entry stores the reduction method category
    def ReductionMethod_condition_for_candidate_replacement(candidate_replaces_if):
        return (
            candidate_replaces_if is None # replace in any case
                or
            candidate_replaces_if(truth_problem)
        )
    log(DEBUG, "Generate ReductionMethod type based on Problem type")
    TypesList.extend(
        FactoryGenerateTypes(ReductionMethodFor._all_reduction_methods, ReductionMethod_condition_on_dict_key, ReductionMethod_condition_for_valid_candidate, ReductionMethod_condition_for_candidate_replacement)
    )
    
    # Append ReductionMethodDecorator types based on Algorithm type
    if hasattr(type(truth_problem), "ProblemDecorators"):
        def ReductionMethodDecorator_condition_on_dict_key(Algorithm):
            return Algorithm in type(truth_problem).ProblemDecorators
        def ReductionMethodDecorator_condition_for_valid_candidate(tuple_):
            return True # always a valid candidate
        def ReductionMethodDecorator_condition_for_candidate_replacement(candidate_replaces_if):
            return (
                candidate_replaces_if is None # replace in any case
                    or
                candidate_replaces_if(truth_problem)
            )
        log(DEBUG, "Append ReductionMethodDecorator types based on Algorithm type")
        TypesList.extend(
            FactoryGenerateTypes(ReductionMethodDecoratorFor._all_reduction_method_decorators, ReductionMethodDecorator_condition_on_dict_key, ReductionMethodDecorator_condition_for_valid_candidate, ReductionMethodDecorator_condition_for_candidate_replacement)
        )
    
    # Log
    log(DEBUG, "The reduction method is a composition of the following types:")
    for t in range(len(TypesList) - 1, -1, -1):
        log(DEBUG, str(TypesList[t]))
    log(DEBUG, "\n")
    
    # Compose all types
    assert len(TypesList) > 0
    ComposedType = TypesList[0]
    for t in range(1, len(TypesList)):
        ComposedType = TypesList[t](ComposedType)
        
    return ComposedType(truth_problem)
    
def ReducedBasis(truth_problem):
    return ReductionMethodFactory(truth_problem, "ReducedBasis")

def PODGalerkin(truth_problem):
    return ReductionMethodFactory(truth_problem, "PODGalerkin")
    
