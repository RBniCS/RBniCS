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
## @file reduced_problem_factory.py
#  @brief Factory to generate a reduced problem corresponding to a given reduction method and truth problem
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

#~~~~~~~~~~~~~~~~~~~~~~~~~     PARAMETRIZED PROBLEM BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ReducedProblemFactory
#

import inspect
from RBniCS.utils.io import log, DEBUG

def FactoryGenerateTypes(list_of_dicts, condition_on_dict_key, condition_for_valid_candidate, condition_for_candidate_replacement):
    
    TypesList = list()
    
    for (level, dict_) in enumerate(list_of_dicts):
        log(DEBUG, "\tOn level " + str(level))
        for key in dict_:
            if condition_on_dict_key(key):
                log(DEBUG, "\tProcessing key " + str(key))
                candidates = list()
                candidates_replaces = list()
                candidates_replaces_if = list()
                for tuple_ in dict_[key]:
                    if condition_for_valid_candidate(tuple_):
                        log(DEBUG, "\t\tProcessing candidate " + str(tuple_[0]) + ": valid")
                        candidates.append(tuple_[0]) # 0-th entry stores the type
                        candidates_replaces.append(tuple_[2]) # 2-th entry stores what other type to replace
                        candidates_replaces_if.append(tuple_[3]) # 3-th entry stores when to replace it
                    else:
                        log(DEBUG, "\t\tProcessing candidate " + str(tuple_[0]) + ": invalid")
                assert len(candidates) > 0
                if len(candidates) > 1:
                    log(DEBUG, "\t\tFound several candidates: " + str(candidates))
                    candidates_to_be_removed = list()
                    for c in range(len(candidates)):
                        if candidates_replaces[c] is not None:
                            if condition_for_candidate_replacement(candidates_replaces_if[c]):
                                log(DEBUG, "\t\t\tRemoving candidate " + candidates_replaces[c] + " because of user provided replacement")
                                candidates_to_be_removed.append(c)
                            else:
                                log(DEBUG, "\t\t\tRemoving candidate " + candidates[c] + " because of user provided replacement")
                                candidates_to_be_removed.append(c)
                    for c in candidates_to_be_removed:
                        del candidates[c]
                    if inspect.isclass(candidates[0]): # they will all be classes
                        candidates_to_be_removed = list()
                        for c1 in range(len(candidates)):
                            assert inspect.isclass(candidates[c1])
                            for c2 in range(c1 + 1, len(candidates)):
                                assert inspect.isclass(candidates[c2])
                                if issubclass(candidates[c1], candidates[c2]):
                                    assert candidates[c1] is not candidates[c2]
                                    log(DEBUG, "\t\t\tRemoving candidate " + str(candidates[c2]) + " in favor of its child " + str(candidates[c1]))
                                    candidates_to_be_removed.append(c2)
                                elif issubclass(candidates[c2], candidates[c1]):
                                    assert candidates[c2] is not candidates[c1]
                                    log(DEBUG, "\t\t\tRemoving candidate " + str(candidates[c1]) + " in favor of its child " + str(candidates[c2]))
                                    candidates_to_be_removed.append(c1)
                                else:
                                    log(DEBUG, "\t\t\tCandidates " + str(candidates[c1]) + " and " + str(candidates[c2]) + " do not inherit from the other, keeping both of them")
                        for c in candidates_to_be_removed:
                            del candidates[c]
                    assert len(candidates) == 1
                    log(DEBUG, "\t\t\tOnly remaining candidate is " + str(candidates[0]))
                else:
                    log(DEBUG, "\t\tFound only one candidate: " + str(candidates[0]))
                if inspect.isclass(candidates[0]):
                    log(DEBUG, "\tDiscarding parent classes of " + str(candidates[0]) + " from previous levels, if any")
                    TypesList_to_be_removed = list() # of bools
                    for t in TypesList:
                        assert inspect.isclass(t)
                        if issubclass(candidates[0], t):
                            assert candidates[0] is not t
                            log(DEBUG, "\t\tDiscarding " + str(t) + " in favor of its child " + str(candidates[0]))
                            TypesList_to_be_removed.append(True)
                    for c in TypesList_to_be_removed:
                        del TypesList[c]
                TypesList.append(candidates[0])
            else:
                log(DEBUG, "\tSkipping key " + str(key))
    
    assert len(TypesList) > 0
    return TypesList
    
