# Copyright (C) 2015-2019 by the RBniCS authors
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

from rbnics.reduction_methods.stokes_optimal_control.stokes_optimal_control_pod_galerkin_reduction import StokesOptimalControlPODGalerkinReduction
# from rbnics.reduction_methods.stokes_optimal_control.stokes_optimal_control_rb_reduction import StokesOptimalControlRBReduction
from rbnics.reduction_methods.stokes_optimal_control.stokes_optimal_control_reduction_method import StokesOptimalControlReductionMethod

__all__ = [
    'StokesOptimalControlPODGalerkinReduction',
    # 'StokesOptimalControlRBReduction',
    'StokesOptimalControlReductionMethod'
]
