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
## @file elliptic_coercive_reduced_problem.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.problems.stokes.stokes_reduced_problem import StokesReducedProblem
from RBniCS.utils.decorators import Extends, ReducedProblemFor
from RBniCS.problems.stokes.stokes_problem import StokesProblem
from RBniCS.problems.base import PODGalerkinReducedProblem
from RBniCS.reduction_methods.stokes import StokesPODGalerkinReduction

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class StokesReducedOrderModelBase
#

StokesPODGalerkinReducedProblem_Base = PODGalerkinReducedProblem(StokesReducedProblem)

# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
@Extends(StokesPODGalerkinReducedProblem_Base) # needs to be first in order to override for last the methods
@ReducedProblemFor(StokesProblem, StokesPODGalerkinReduction)
class StokesPODGalerkinReducedProblem(StokesPODGalerkinReducedProblem_Base):
    pass
        
