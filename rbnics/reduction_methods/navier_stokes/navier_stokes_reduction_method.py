# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base import NonlinearReductionMethod


def NavierStokesReductionMethod(StokesReductionMethod_DerivedClass):

    NavierStokesReductionMethod_Base = NonlinearReductionMethod(StokesReductionMethod_DerivedClass)

    class NavierStokesReductionMethod_Class(NavierStokesReductionMethod_Base):
        pass

    # return value (a class) for the decorator
    return NavierStokesReductionMethod_Class
