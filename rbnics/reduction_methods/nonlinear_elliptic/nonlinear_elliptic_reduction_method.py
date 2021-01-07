# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base import NonlinearReductionMethod


def NonlinearEllipticReductionMethod(EllipticReductionMethod_DerivedClass):

    NonlinearEllipticReductionMethod_Base = NonlinearReductionMethod(EllipticReductionMethod_DerivedClass)

    class NonlinearEllipticReductionMethod_Class(NonlinearEllipticReductionMethod_Base):
        pass

    # return value (a class) for the decorator
    return NonlinearEllipticReductionMethod_Class
