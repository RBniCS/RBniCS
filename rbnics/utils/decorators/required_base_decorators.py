# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later


def RequiredBaseDecorators(*BaseDecorators):

    def RequiredBaseDecorators_FunctionDecorator(Decorator):

        def RequiredBaseDecorators_ClassDecorator(Class):
            BaseClass = Class
            AlreadyAppliedBaseDecorators = list()
            if hasattr(Class, "AlreadyAppliedBaseDecorators"):
                AlreadyAppliedBaseDecorators.extend(Class.AlreadyAppliedBaseDecorators)

            for BaseDecorator in BaseDecorators:
                if (BaseDecorator is not None
                        and BaseDecorator.__name__ not in AlreadyAppliedBaseDecorators):
                    BaseClass = BaseDecorator(BaseClass)
                    AlreadyAppliedBaseDecorators.append(BaseDecorator.__name__)

            if Decorator not in AlreadyAppliedBaseDecorators:
                DecoratedClass = Decorator(BaseClass)
                DecoratedClass.AlreadyAppliedBaseDecorators = list()
                DecoratedClass.AlreadyAppliedBaseDecorators.extend(AlreadyAppliedBaseDecorators)
                DecoratedClass.AlreadyAppliedBaseDecorators.append(Decorator.__name__)
                return DecoratedClass
            else:
                return BaseClass

        # Preserve class decorator name and return
        setattr(RequiredBaseDecorators_ClassDecorator, "__name__", Decorator.__name__)
        setattr(RequiredBaseDecorators_ClassDecorator, "__module__", Decorator.__module__)
        return RequiredBaseDecorators_ClassDecorator

    return RequiredBaseDecorators_FunctionDecorator
