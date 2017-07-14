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

from rbnics.utils.decorators.for_decorators_helper import ForDecoratorsStore, ForDecoratorsLogging
from rbnics.utils.decorators.multi_level_reduction_method import MultiLevelReductionMethod
from rbnics.utils.mpi import log, DEBUG

def ReductionMethodFor(Problem, category, enabled_if=None, replaces=None, replaces_if=None):
    impl = ReductionMethodFor_Impl(Problem, category, enabled_if, replaces, replaces_if)
    def ReductionMethodFor_Decorator(ReductionMethod):
        output = impl(ReductionMethod)
        return output
    return ReductionMethodFor_Decorator
    
def ReductionMethodFor_Impl(Problem, category, enabled_if=None, replaces=None, replaces_if=None):
    def ReductionMethodFor_ImplDecorator(ReductionMethod):
        # Decorate with multilevel reduction method
        ReductionMethod = MultiLevelReductionMethod(ReductionMethod)
        # Add to local storage
        log(DEBUG,
            "In ReductionMethodFor with\n" +
            "\tProblem = " + str(Problem) + "\n" +
            "\tReductionMethod = " + str(ReductionMethod) + "\n" +
            "\tcategory = " + str(category) + "\n" +
            "\tenabled_if = " + str(enabled_if) + "\n" +
            "\treplaces = " + str(replaces) + "\n" +
            "\treplaces_if = " + str(replaces_if)
        )
        def go_to_next_level(Key, StoredKey):
            # List the keys in order of inheritance: base classes will come first
            # in the list, then their children, and then children of their children.
            return Key is not StoredKey and issubclass(Key, StoredKey)
        ForDecoratorsStore(Problem, ReductionMethodFor._all_reduction_methods, (ReductionMethod, category, enabled_if, replaces, replaces_if), go_to_next_level)
        log(DEBUG, "ReductionMethodFor storage now contains:")
        ForDecoratorsLogging(ReductionMethodFor._all_reduction_methods, "Problem", "ReductionMethod", "category")
        log(DEBUG, "")
        # Moreover also add to storage the category to generate recursively reduction methods in ReducedProblemFor
        ReductionMethodFor._all_reduction_methods_categories[ReductionMethod] = category
        # Done with the storage, return
        return ReductionMethod
    return ReductionMethodFor_ImplDecorator

ReductionMethodFor._all_reduction_methods = list() # (over inheritance level) of dicts from Problem to list of (ReductionMethod, category, enabled_if, replaces, replaces_if)
ReductionMethodFor._all_reduction_methods_categories = dict() # from reduction method to category
