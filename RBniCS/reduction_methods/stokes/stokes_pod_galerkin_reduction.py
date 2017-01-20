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
## @file stokes_pod_galerkin_reduction.py
#  @brief Implementation of a POD-Galerkin ROM for elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.backends import ProperOrthogonalDecomposition
from RBniCS.utils.decorators import Extends, override, ReductionMethodFor
from RBniCS.problems.stokes.stokes_problem import StokesProblem
from RBniCS.reduction_methods.base import PODGalerkinReduction
from RBniCS.reduction_methods.stokes.stokes_reduction_method import StokesReductionMethod

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE POD BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class StokesPODGalerkinReduction
#

StokesPODGalerkinReduction_Base = PODGalerkinReduction(StokesReductionMethod)

# Base class containing the interface of a POD-Galerkin ROM
# for elliptic coercive problems
@Extends(StokesPODGalerkinReduction_Base) # needs to be first in order to override for last the methods
@ReductionMethodFor(StokesProblem, "PODGalerkin")
class StokesPODGalerkinReduction(StokesPODGalerkinReduction_Base):    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the POD-Galerkin ROM object
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call the parent initialization
        StokesPODGalerkinReduction_Base.__init__(self, truth_problem, **kwargs)
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Initialize data structures required for the offline phase: overridden version because supremizer POD is different from a standard component
    @override
    def _init_offline(self):
        # We cannot use the standard initialization provided by PODGalerkinReduction because
        # supremizer POD requires a custom initialization. We thus duplicate here part of its code
        
        # Call parent of parent (!) to initialize inner product and reduced problem
        output = StokesReductionMethod._init_offline(self)
        
        # Declare a new POD for each basis component
        self.POD = dict()
        for component in ("u", "p"):
            inner_product = self.truth_problem.inner_product[component][0]
            self.POD[component] = ProperOrthogonalDecomposition(self.truth_problem.V, inner_product)
        for component in ("s", ):
            inner_product = self.truth_problem.inner_product[component][0]
            self.POD[component] = ProperOrthogonalDecomposition(self.truth_problem.V, inner_product, component="s")
            
        # Return
        return output
        
    ## Update the snapshots matrix: overridden version because supremizer POD is different from a standard component
    def update_snapshots_matrix(self, snapshot_and_supremizer):
        assert isinstance(snapshot_and_supremizer, tuple)
        assert len(snapshot_and_supremizer) == 2
        snapshot = snapshot_and_supremizer[0]
        supremizer = snapshot_and_supremizer[1]
        for component in ("u", "p"):
            self.POD[component].store_snapshot(snapshot, component=component)
        for component in ("s", ):
            self.POD[component].store_snapshot(supremizer)
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the testing set
    @override
    def error_analysis(self, N=None, **kwargs):        
        components = ["u", "p"] # but not "s"
        kwargs["components"] = components
                
        StokesPODGalerkinReduction_Base.error_analysis(self, N, **kwargs)
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
