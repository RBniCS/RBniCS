# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.stokes_unsteady.stokes_unsteady_reduction_method import (
    AbstractCFDUnsteadyReductionMethod)
from rbnics.reduction_methods.base import NonlinearTimeDependentReductionMethod


# Base class containing the interface of a projection based ROM
# for saddle point problems.
def NavierStokesUnsteadyReductionMethod(NavierStokesReductionMethod_DerivedClass):

    NavierStokesUnsteadyReductionMethod_Base = AbstractCFDUnsteadyReductionMethod(
        NonlinearTimeDependentReductionMethod(NavierStokesReductionMethod_DerivedClass))

    class NavierStokesUnsteadyReductionMethod_Class(NavierStokesUnsteadyReductionMethod_Base):
        pass

    # return value (a class) for the decorator
    return NavierStokesUnsteadyReductionMethod_Class
