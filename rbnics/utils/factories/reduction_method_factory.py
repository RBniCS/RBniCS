# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from logging import DEBUG, getLogger
from rbnics.utils.decorators.customize_reduction_method_for import _cache as customize_reduction_method_cache
from rbnics.utils.decorators.reduction_method_for import _cache as reduction_method_cache
from rbnics.utils.decorators.reduction_method_decorator_for import _cache as reduction_method_decorator_cache

logger = getLogger("rbnics/utils/factories/reduction_method_factory.py")


# Factory to generate a reduction method corresponding to a category (e.g. RB or POD) and a given truth problem
def ReductionMethodFactory(truth_problem, category, **kwargs):

    logger.log(DEBUG, "In ReductionMethodFactory with")
    logger.log(DEBUG, "\ttruth problem = " + str(type(truth_problem)))
    logger.log(DEBUG, "\tcategory = " + str(category))
    logger.log(DEBUG, "\tkwargs = " + str(kwargs))

    if hasattr(type(truth_problem), "ProblemDecorators"):
        logger.log(DEBUG, "\ttruth problem decorators = ")
        for Decorator in type(truth_problem).ProblemDecorators:
            logger.log(DEBUG, "\t\t" + str(Decorator))
        logger.log(DEBUG, "")

    TypesList = list()

    # Generate ReductionMethod type based on Problem type
    logger.log(DEBUG, "Generate ReductionMethod type based on Problem type")
    ReductionMethodGenerator = getattr(reduction_method_cache, category)
    TypesList.append(ReductionMethodGenerator(truth_problem))

    # Look if any customizer has been defined
    for (Problem, customizer) in customize_reduction_method_cache.items():
        if isinstance(truth_problem, Problem):
            TypesList.append(customizer)

    # Append ReductionMethodDecorator types based on Algorithm type
    if hasattr(type(truth_problem), "ProblemDecorators"):
        logger.log(DEBUG, "Append ReductionMethodDecorator types based on Algorithm type")
        for Decorator in type(truth_problem).ProblemDecorators:
            ReductionMethodDecoratorGenerator = getattr(reduction_method_decorator_cache, Decorator.__name__)
            TypesList.append(ReductionMethodDecoratorGenerator(truth_problem, **kwargs))

    # Log
    logger.log(DEBUG, "The reduction method is a composition of the following types:")
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


def ReducedBasis(truth_problem, **kwargs):
    return ReductionMethodFactory(truth_problem, "ReducedBasis", **kwargs)


def PODGalerkin(truth_problem, **kwargs):
    return ReductionMethodFactory(truth_problem, "PODGalerkin", **kwargs)
