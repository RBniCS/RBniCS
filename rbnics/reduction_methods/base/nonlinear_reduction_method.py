# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(None)
def NonlinearReductionMethod(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class NonlinearReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):
        pass

    # return value (a class) for the decorator
    return NonlinearReductionMethod_Class
