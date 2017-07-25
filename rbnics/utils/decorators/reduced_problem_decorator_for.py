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
from rbnics.utils.mpi import log, DEBUG

def ReducedProblemDecoratorFor(Algorithm, enabled_if=None, replaces=None, replaces_if=None):
    def ReducedProblemDecoratorFor_Decorator(ReducedProblemDecorator):
        # Add to local storage
        log(DEBUG,
            "In ReducedProblemDecoratorFor with\n" +
            "\tAlgorithm = " + str(Algorithm) + "\n" +
            "\tReducedProblemDecorator = " + str(ReducedProblemDecorator) + "\n" +
            "\tenabled_if = " + str(enabled_if) + "\n" +
            "\treplaces = " + str(replaces) + "\n" +
            "\treplaces_if = " + str(replaces_if) + "\n"
        )
        def go_to_next_level(Key, StoredKey):
            # Algorithms are functions (decorators) so it is not possible
            # to define inheritance levels. Flatten the storage levels
            # and thus rely only on explicit replaces provided by the user
            return False
        ForDecoratorsStore(Algorithm, ReducedProblemDecoratorFor._all_reduced_problems_decorators, (ReducedProblemDecorator, None, enabled_if, replaces, replaces_if), go_to_next_level)
        log(DEBUG, "ReducedProblemDecoratorFor storage now contains:")
        ForDecoratorsLogging(ReducedProblemDecoratorFor._all_reduced_problems_decorators, "Algorithm", "ReducedProblemDecorator", None)
        log(DEBUG, "")
        # Done with the storage, return the unchanged reduced problem decorator
        return ReducedProblemDecorator
    return ReducedProblemDecoratorFor_Decorator

ReducedProblemDecoratorFor._all_reduced_problems_decorators = list() # (over inheritance level) of dicts from Algorithm to list of (ReducedProblemDecorator, None, enabled_if, replaces, replaces_if)
