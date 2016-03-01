# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file reduced_method_factory.py
#  @brief Factories to associate parametrized problem classes to reduction method classes.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

#~~~~~~~~~~~~~~~~~~~~~~~~~     REDUCED METHOD FACTORY CLASSES     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ReducedBasis
#
# Factories to associate parametrized problem classes to reduction method classes: reduced basis method
def ReducedBasis(truth_problem):
    if isinstance(truth_problem, EllipticCoerciveProblem):
        return EllipticCoerciveRBReduction(truth_problem)
    else:
        raise RuntimeError("Invalid arguments in ReducedBasis factory.")
        
## @class ReducedBasis
#
# Factories to associate parametrized problem classes to reduction method classes: POD-Galerkin method
def PODGalerkin(truth_problem):
    if isinstance(truth_problem, EllipticCoerciveProblem):
        return EllipticCoercivePODGalerkinReduction(truth_problem)
    else:
        raise RuntimeError("Invalid arguments in PODGalerkin factory.")
