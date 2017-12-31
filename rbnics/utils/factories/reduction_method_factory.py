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

from rbnics.utils.decorators import MultiLevelReductionMethod
from rbnics.utils.decorators.customize_reduction_method_for import _cache as customize_reduction_method_cache
from rbnics.utils.decorators.reduction_method_for import _cache as reduction_method_cache
from rbnics.utils.decorators.reduction_method_decorator_for import _cache as reduction_method_decorator_cache
from rbnics.utils.mpi import log, DEBUG

# Factory to generate a reduction method corresponding to a category (e.g. RB or POD) and a given truth problem
def ReductionMethodFactory(truth_problem, category, **kwargs):
    
    log(DEBUG,
        "In ReductionMethodFactory with\n" +
        "\ttruth problem = " + str(type(truth_problem)) + "\n" +
        "\tcategory = " + str(category) + "\n" +
        "\tkwargs = " + str(kwargs)
        )
    if hasattr(type(truth_problem), "ProblemDecorators"):
        log(DEBUG,
            "\ttruth problem decorators = " +
            "\n\t\t".join([str(Decorator) for Decorator in type(truth_problem).ProblemDecorators]) +
            "\n"
            )
    
    TypesList = list()
    
    # Generate ReductionMethod type based on Problem type
    log(DEBUG, "Generate ReductionMethod type based on Problem type")
    ReductionMethodGenerator = getattr(reduction_method_cache, category)
    TypesList.append(ReductionMethodGenerator(truth_problem))
    
    # Look if any customizer has been defined
    for (Problem, customizer) in customize_reduction_method_cache.items():
        if isinstance(truth_problem, Problem):
            TypesList.append(customizer)
    
    # Append ReductionMethodDecorator types based on Algorithm type
    if hasattr(type(truth_problem), "ProblemDecorators"):
        log(DEBUG, "Append ReductionMethodDecorator types based on Algorithm type")
        for Decorator in type(truth_problem).ProblemDecorators:
            ReductionMethodDecoratorGenerator = getattr(reduction_method_decorator_cache, Decorator.__name__)
            TypesList.append(ReductionMethodDecoratorGenerator(truth_problem, **kwargs))
                
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
        
    # Decorate with multilevel reduction method
    ComposedType = MultiLevelReductionMethod(ComposedType)
    
    # Finally, return an instance of the generated class
    return ComposedType(truth_problem, **kwargs)
    
def ReducedBasis(truth_problem, **kwargs):
    return ReductionMethodFactory(truth_problem, "ReducedBasis", **kwargs)

def PODGalerkin(truth_problem, **kwargs):
    return ReductionMethodFactory(truth_problem, "PODGalerkin", **kwargs)
