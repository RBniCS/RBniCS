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
## @file elliptic_optimal_control_pod_galerkin_reduction.py
#  @brief Implementation of a POD-Galerkin ROM for elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from RBniCS.backends import FunctionsList
from RBniCS.utils.mpi import print
from RBniCS.utils.decorators import Extends, override, ReductionMethodFor
from RBniCS.problems.elliptic_optimal_control.elliptic_optimal_control_problem import EllipticOptimalControlProblem
from RBniCS.reduction_methods.base import PODGalerkinReduction
from RBniCS.reduction_methods.elliptic_optimal_control.elliptic_optimal_control_reduction_method import EllipticOptimalControlReductionMethod

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE POD BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticOptimalControlPODGalerkinReduction
#

EllipticOptimalControlPODGalerkinReduction_Base = PODGalerkinReduction(EllipticOptimalControlReductionMethod)

# Base class containing the interface of a POD-Galerkin ROM
# for elliptic coercive problems
@Extends(EllipticOptimalControlPODGalerkinReduction_Base) # needs to be first in order to override for last the methods
@ReductionMethodFor(EllipticOptimalControlProblem, "PODGalerkin")
class EllipticOptimalControlPODGalerkinReduction(EllipticOptimalControlPODGalerkinReduction_Base):    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the POD-Galerkin ROM object
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call the parent initialization
        EllipticOptimalControlPODGalerkinReduction_Base.__init__(self, truth_problem, **kwargs)
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Compute basis functions performing POD: overridden to handle aggregated spaces
    def compute_basis_functions(self):
        # Carry out POD
        Z = dict()
        N = dict()
        for component in self.truth_problem.components:
            print("# POD for component", component)
            POD = self.POD[component]
            (_, Z[component], N[component]) = POD.apply(self.Nmax, self.tol)
            POD.print_eigenvalues(N[component])
            POD.save_eigenvalues_file(self.folder["post_processing"], "eigs_" + component)
            POD.save_retained_energy_file(self.folder["post_processing"], "retained_energy_" + component)
        
        # Store POD modes related to control as usual
        self.reduced_problem.Z.enrich(Z["u"], component="u")
        self.reduced_problem.N["u"] += N["u"]
        
        # Aggregate POD modes related to state and adjoint
        for component_to in ("y", "p"):
            for component_from in ("y", "p"):
                for i in range(N[component_from]):
                    self.reduced_problem.Z.enrich(Z[component_from][i], component={component_from: component_to})
                self.reduced_problem.N[component_to] += N[component_from]
        
        # Save
        self.reduced_problem.Z.save(self.reduced_problem.folder["basis"], "basis")
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
