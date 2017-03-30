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

import inspect
from rbnics.utils.mpi import log, DEBUG

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
                    assert len(candidates) == len(candidates_replaces_if)
                    assert len(candidates) == len(candidates_replaces)
                    for (candidate, candidate_replaces_if, candidate_replaces) in zip(candidates, candidates_replaces_if, candidates_replaces):
                        if candidate_replaces is not None:
                            if condition_for_candidate_replacement(candidate_replaces_if):
                                log(DEBUG, "\t\t\tKeeping candidate " + str(candidate) + " because of successful user provided replacement with " + str(candidate_replaces))
                                candidates_to_be_removed.append(candidate_replaces)
                            else:
                                log(DEBUG, "\t\t\tRemoving candidate " + str(candidate) + " because of failed user provided replacement with " + str(candidate_replaces))
                                candidates_to_be_removed.append(candidate)
                    for c in candidates_to_be_removed:
                        candidates.remove(c)
                    if inspect.isclass(candidates[0]): # they will all be classes
                        candidates_to_be_removed = list()
                        for (index1, candidate1) in enumerate(candidates):
                            assert inspect.isclass(candidate1)
                            for (index2, candidate2) in enumerate(candidates[index1 + 1:], start=index1 + 1):
                                assert inspect.isclass(candidate2)
                                if issubclass(candidate1, candidate2):
                                    assert candidate1 is not candidate2
                                    log(DEBUG, "\t\t\tRemoving candidate " + str(candidate2) + " in favor of its child " + str(candidate1))
                                    candidates_to_be_removed.append(index2)
                                elif issubclass(candidate2, candidate1):
                                    assert candidate2 is not candidate1
                                    log(DEBUG, "\t\t\tRemoving candidate " + str(candidate1) + " in favor of its child " + str(candidate2))
                                    candidates_to_be_removed.append(index1)
                                else:
                                    log(DEBUG, "\t\t\tCandidates " + str(candidates[c1]) + " and " + str(candidates[c2]) + " do not inherit from each other, keeping both of them")
                        for c in candidates_to_be_removed:
                            del candidates[c]
                    assert len(candidates) == 1
                    log(DEBUG, "\t\t\tOnly remaining candidate is " + str(candidates[0]))
                else:
                    log(DEBUG, "\t\tFound only one candidate: " + str(candidates[0]))
                if inspect.isclass(candidates[0]):
                    log(DEBUG, "\tDiscarding parent classes of " + str(candidates[0]) + " from previous levels, if any")
                    TypesList_to_be_removed = list() # of int
                    for (index, t) in enumerate(TypesList):
                        assert inspect.isclass(t)
                        if issubclass(candidates[0], t):
                            assert candidates[0] is not t
                            log(DEBUG, "\t\tDiscarding " + str(t) + " in favor of its child " + str(candidates[0]))
                            TypesList_to_be_removed.append(index)
                    for c in TypesList_to_be_removed:
                        del TypesList[c]
                TypesList.append(candidates[0])
            else:
                log(DEBUG, "\tSkipping key " + str(key))
    
    assert len(TypesList) > 0
    return TypesList
    
