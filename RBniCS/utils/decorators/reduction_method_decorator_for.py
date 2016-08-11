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
## @file numpy_io.py
#  @brief I/O helper functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.decorators.for_decorators_helper import ForDecoratorsStore, ForDecoratorsLogging
from RBniCS.utils.mpi import log, DEBUG

def ReductionMethodDecoratorFor(Algorithm, replaces=None, replaces_if=None):
    def ReductionMethodDecoratorFor_Decorator(ReductionMethodDecorator):
        # Add to local storage
        log(DEBUG,
            "In ReductionMethodDecoratorFor with\n" +
            "\tAlgorithm = " + str(Algorithm) + "\n" +
            "\tReductionMethodDecorator = " + str(ReductionMethodDecorator) + "\n" +
            "\treplaces = " + str(replaces) + "\n" +
            "\treplaces_if = " + str(replaces_if)
        )
        def go_to_next_level(Key, StoredKey):
            # Algorithms are functions (decorators) to it is not possible
            # to define inheritance levels. Flatten the storage levels
            # and thus rely only on explicit replaces provided by the user
            return False
        ForDecoratorsStore(Algorithm, ReductionMethodDecoratorFor._all_reduction_method_decorators, (ReductionMethodDecorator, None, replaces, replaces_if), go_to_next_level)
        log(DEBUG, "ReductionMethodDecoratorFor storage now contains:")
        ForDecoratorsLogging(ReductionMethodDecoratorFor._all_reduction_method_decorators, "Algorithm", "ReductionMethodDecorator", None)
        log(DEBUG, "")
        # Done with the storage, return the unchanged reduction method decorator
        return ReductionMethodDecorator
    return ReductionMethodDecoratorFor_Decorator
    
ReductionMethodDecoratorFor._all_reduction_method_decorators = list() # (over inheritance level) of dicts from Algorithm to list of (ReductionMethodDecorator, None, replaces, replaces_if)
