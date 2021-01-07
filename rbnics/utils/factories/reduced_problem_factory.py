# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from logging import DEBUG, getLogger
from rbnics.utils.decorators.customize_reduced_problem_for import _cache as customize_reduced_problem_cache
from rbnics.utils.decorators.reduced_problem_for import _cache as reduced_problem_cache
from rbnics.utils.decorators.reduced_problem_decorator_for import _cache as reduced_problem_decorator_cache

logger = getLogger("rbnics/utils/factories/reduced_problem_factory.py")


# Factory to generate a reduced problem corresponding to a given reduction method and truth problem
def ReducedProblemFactory(truth_problem, reduction_method, **kwargs):

    logger.log(DEBUG, "In ReducedProblemFactory with")
    logger.log(DEBUG, "\ttruth problem = " + str(type(truth_problem)))
    logger.log(DEBUG, "\treduction_method = " + str(type(reduction_method)))
    logger.log(DEBUG, "\tkwargs = " + str(kwargs))

    if hasattr(type(truth_problem), "ProblemDecorators"):
        logger.log(DEBUG, "\ttruth problem decorators = ")
        for Decorator in type(truth_problem).ProblemDecorators:
            logger.log(DEBUG, "\t\t" + str(Decorator))
        logger.log(DEBUG, "")

    TypesList = list()

    # Generate ReducedProblem types based on Problem and ReductionMethod type
    logger.log(DEBUG, "Generate ReducedProblem types based on Problem and ReductionMethod type")
    ReducedProblemGenerator = getattr(reduced_problem_cache, "ReducedProblem")
    TypesList.append(ReducedProblemGenerator(truth_problem, reduction_method))

    # Look if any customizer has been defined
    for (Problem, customizer) in customize_reduced_problem_cache.items():
        if isinstance(truth_problem, Problem):
            TypesList.append(customizer)

    # Append ReducedProblemDecorator types based on Algorithm type
    if hasattr(type(truth_problem), "ProblemDecorators"):
        logger.log(DEBUG, "Append ReducedProblemDecorator types based on Algorithm type")
        for Decorator in type(truth_problem).ProblemDecorators:
            ReducedProblemDecoratorGenerator = getattr(reduced_problem_decorator_cache, Decorator.__name__)
            TypesList.append(ReducedProblemDecoratorGenerator(truth_problem, reduction_method, **kwargs))

    # Log
    logger.log(DEBUG, "The reduced problem is a composition of the following types:")
    for t in range(len(TypesList) - 1, -1, -1):
        logger.log(DEBUG, str(TypesList[t]))
    logger.log(DEBUG, "")

    # Compose all types
    assert len(TypesList) > 0
    ComposedType = TypesList[0]
    for t in range(1, len(TypesList)):
        ComposedType = TypesList[t](ComposedType)

    # Finally, return an instance of the generated class
    return ComposedType(truth_problem, **kwargs)
