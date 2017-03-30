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
## @file numpy_io.py
#  @brief I/O helper functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.decorators.for_decorators_helper import ForDecoratorsStore, ForDecoratorsLogging
from RBniCS.utils.decorators.reduction_method_for import ReductionMethodFor, ReductionMethodFor_Impl
from RBniCS.utils.mpi import log, DEBUG

def ReducedProblemFor(Problem, ReductionMethod, replaces=None, replaces_if=None):
    impl = ReducedProblemFor_Impl(Problem, ReductionMethod, replaces, replaces_if)
    def ReducedProblemFor_Decorator(ReducedProblem):
        output = impl(ReducedProblem)
        # The current reduced problem can also be used as a truth problem for multilevel reduction
        category = ReductionMethodFor._all_reduction_methods_categories[ReductionMethod]
        multilevel_reduction_method_for_impl = ReductionMethodFor_Impl(ReducedProblem, category)      # there is no need for replacements, since
        multilevel_reduced_problem_for_impl = ReducedProblemFor_Impl(ReducedProblem, ReductionMethod) # reduced problems can only generate again themselves
        multilevel_reduction_method_for_impl(ReductionMethod)
        multilevel_reduced_problem_for_impl(ReducedProblem)
        return output
    return ReducedProblemFor_Decorator
    
def ReducedProblemFor_Impl(Problem, ReductionMethod, replaces=None, replaces_if=None):
    def ReducedProblemFor_ImplDecorator(ReducedProblem):
        # Add to local storage
        log(DEBUG,
            "In ReducedProblemFor with\n" +
            "\tProblem = " + str(Problem) + "\n" +
            "\tReductionMethod = " + str(ReductionMethod) + "\n" +
            "\tReducedProblem = " + str(ReducedProblem) + "\n" +
            "\treplaces = " + str(replaces) + "\n" +
            "\treplaces_if = " + str(replaces_if) + "\n"
        )
        def go_to_next_level(Key, StoredKey):
            # List the keys in order of inheritance: base classes will come first
            # in the list, then their children, and then children of their children.
            return Key is not StoredKey and issubclass(Key, StoredKey)
        ForDecoratorsStore(Problem, ReducedProblemFor._all_reduced_problems, (ReducedProblem, ReductionMethod, replaces, replaces_if), go_to_next_level)
        log(DEBUG, "ReducedProblemFor storage now contains:")
        ForDecoratorsLogging(ReducedProblemFor._all_reduced_problems, "Problem", "ReducedProblem", "ReductionMethod")
        log(DEBUG, "\n")
        # Done with the storage, return the unchanged reduced problem class
        return ReducedProblem
    return ReducedProblemFor_ImplDecorator

ReducedProblemFor._all_reduced_problems = list() # (over inheritance level) of dicts from Problem to list of (ReducedProblem, ReductionMethod, replaces, replaces_if)
