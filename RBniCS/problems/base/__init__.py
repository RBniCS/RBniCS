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
## @file __init__.py
#  @brief Init file for auxiliary problems module
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.problems.base.nonlinear_pod_galerkin_reduced_problem import NonlinearPODGalerkinReducedProblem
from RBniCS.problems.base.nonlinear_problem import NonlinearProblem
from RBniCS.problems.base.nonlinear_rb_reduced_problem import NonlinearRBReducedProblem
from RBniCS.problems.base.nonlinear_reduced_problem import NonlinearReducedProblem
from RBniCS.problems.base.nonlinear_time_dependent_pod_galerkin_reduced_problem import NonlinearTimeDependentPODGalerkinReducedProblem
from RBniCS.problems.base.nonlinear_time_dependent_problem import NonlinearTimeDependentProblem
from RBniCS.problems.base.nonlinear_time_dependent_rb_reduced_problem import NonlinearTimeDependentRBReducedProblem
from RBniCS.problems.base.nonlinear_time_dependent_reduced_problem import NonlinearTimeDependentReducedProblem
from RBniCS.problems.base.parametrized_differential_problem import ParametrizedDifferentialProblem
from RBniCS.problems.base.parametrized_problem import ParametrizedProblem
from RBniCS.problems.base.parametrized_reduced_differential_problem import ParametrizedReducedDifferentialProblem
from RBniCS.problems.base.pod_galerkin_reduced_problem import PODGalerkinReducedProblem
from RBniCS.problems.base.rb_reduced_problem import RBReducedProblem
from RBniCS.problems.base.time_dependent_pod_galerkin_reduced_problem import TimeDependentPODGalerkinReducedProblem
from RBniCS.problems.base.time_dependent_problem import TimeDependentProblem
from RBniCS.problems.base.time_dependent_rb_reduced_problem import TimeDependentRBReducedProblem
from RBniCS.problems.base.time_dependent_reduced_problem import TimeDependentReducedProblem


__all__ = [
    'NonlinearPODGalerkinReducedProblem',
    'NonlinearProblem',
    'NonlinearRBReducedProblem',
    'NonlinearReducedProblem',
    'NonlinearTimeDependentPODGalerkinProblem',
    'NonlinearTimeDependentProblem',
    'NonlinearTimeDependentRBReducedProblem',
    'NonlinearTimeDependentReducedProblem',
    'ParametrizedDifferentialProblem',
    'ParametrizedProblem',
    'ParametrizedReducedDifferentialProblem',
    'PODGalerkinReducedProblem',
    'RBReducedProblem',
    'TimeDependentPODGalerkinReducedProblem',
    'TimeDependentProblem',
    'TimeDependentRBReducedProblem',
    'TimeDependentReducedProblem'
]
