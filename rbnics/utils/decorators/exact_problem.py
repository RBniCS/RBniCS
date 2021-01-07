# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.cache import Cache
from rbnics.utils.decorators.preserve_class_name import PreserveClassName
from rbnics.utils.decorators.sync_setters import sync_setters


def exact_problem(decorated_problem, preserve_class_name=True):
    if (decorated_problem, preserve_class_name) not in _all_exact_problems:
        DecoratedProblem = type(decorated_problem)
        if hasattr(DecoratedProblem, "ProblemDecorators"):
            assert hasattr(DecoratedProblem, "UndecoratedProblemClass")

            @PreserveClassName
            class ExactProblem_Class(DecoratedProblem.UndecoratedProblemClass):
                @sync_setters(decorated_problem, "set_mu", "mu")
                @sync_setters(decorated_problem, "set_mu_range", "mu_range")
                def __init__(self, V, **kwargs):
                    # Store reference to original problem
                    self.__decorated_problem__ = decorated_problem
                    # Call Parent constructor
                    DecoratedProblem.UndecoratedProblemClass.__init__(self, V, **kwargs)

            if hasattr(decorated_problem, "set_time"):
                ExactProblem_Class_Base = ExactProblem_Class

                @PreserveClassName
                class ExactProblem_Class(ExactProblem_Class_Base):
                    @sync_setters(decorated_problem, "set_time", "t")
                    @sync_setters(decorated_problem, "set_initial_time", "t0")
                    @sync_setters(decorated_problem, "set_time_step_size", "dt")
                    @sync_setters(decorated_problem, "set_final_time", "T")
                    def __init__(self, V, **kwargs):
                        ExactProblem_Class_Base.__init__(self, V, **kwargs)

            if not preserve_class_name:
                # there would be no point in having class names like ExactExactProblem
                assert not hasattr(ExactProblem_Class, "__is_exact__")
                setattr(ExactProblem_Class, "__name__", "Exact" + ExactProblem_Class.__name__)

                ExactProblem_Class_Base = ExactProblem_Class

                @PreserveClassName
                class ExactProblem_Class(ExactProblem_Class_Base):
                    def name(self):
                        return "Exact" + decorated_problem.name()

            setattr(ExactProblem_Class, "__is_exact__", True)
            setattr(ExactProblem_Class, "__DecoratedProblem__", DecoratedProblem)

            # Re-apply decorators, replacing e.g. EIM with ExactParametrizedFunctions:
            for (Decorator, ExactDecorator, kwargs) in zip(
                    DecoratedProblem.ProblemDecorators, DecoratedProblem.ProblemExactDecorators,
                    DecoratedProblem.ProblemDecoratorsKwargs):
                if ExactDecorator is not None:
                    ExactProblem_Class = ExactDecorator(**kwargs)(ExactProblem_Class)
                else:
                    ExactProblem_Class = Decorator(**kwargs)(ExactProblem_Class)

            # Create a new instance of ExactProblem_Class
            exact_problem = ExactProblem_Class(decorated_problem.V, **decorated_problem.problem_kwargs)

            # Save
            _all_exact_problems[(decorated_problem, preserve_class_name)] = exact_problem
            _all_exact_problems[(exact_problem, True)] = exact_problem
            # shortcut for generation of an exact problem from itself
        else:
            _all_exact_problems[(decorated_problem, preserve_class_name)] = decorated_problem

    # Return
    return _all_exact_problems[(decorated_problem, preserve_class_name)]


_all_exact_problems = Cache()
