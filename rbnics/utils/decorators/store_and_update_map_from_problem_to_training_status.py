# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.cache import Cache
from rbnics.utils.decorators.preserve_class_name import PreserveClassName


def StoreMapFromProblemToTrainingStatus(ParametrizedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class StoreMapFromProblemToTrainingStatus_Class(ParametrizedDifferentialProblem_DerivedClass):

        def __init__(self, V, **kwargs):
            # Call the parent initialization
            ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)

            # Populate problem to training status
            init_map_from_problem_to_training_status(self)

    # return value (a class) for the decorator
    return StoreMapFromProblemToTrainingStatus_Class


def UpdateMapFromProblemToTrainingStatus(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class UpdateMapFromProblemToTrainingStatus_Class(DifferentialProblemReductionMethod_DerivedClass):

        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)

        def _init_offline(self):
            # Update reduced problem to training status
            set_map_from_problem_to_training_status_off(self.truth_problem)

            # Call the parent initialization
            return DifferentialProblemReductionMethod_DerivedClass._init_offline(self)

        def _finalize_offline(self):
            # Call the parent finalization
            DifferentialProblemReductionMethod_DerivedClass._finalize_offline(self)

            # Update reduced problem to training status
            set_map_from_problem_to_training_status_on(self.truth_problem)

    # return value (a class) for the decorator
    return UpdateMapFromProblemToTrainingStatus_Class


def init_map_from_problem_to_training_status(problem):
    if problem not in _problem_to_training_status:
        _problem_to_training_status[problem] = None
    else:
        assert _problem_to_training_status[problem] is None


def set_map_from_problem_to_training_status_on(problem):
    assert problem in _problem_to_training_status
    _problem_to_training_status[problem] = True


def set_map_from_problem_to_training_status_off(problem):
    assert problem in _problem_to_training_status
    _problem_to_training_status[problem] = False


def is_training_started(problem):
    assert problem in _problem_to_training_status
    return _problem_to_training_status[problem] is not None


def is_training_finished(problem):
    assert problem in _problem_to_training_status
    return (_problem_to_training_status[problem] is not None
            and _problem_to_training_status[problem])


_problem_to_training_status = Cache()
