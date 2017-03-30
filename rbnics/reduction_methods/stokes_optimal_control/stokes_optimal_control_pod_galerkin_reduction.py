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
## @file
#  @brief
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from RBniCS.backends import FunctionsList, ProperOrthogonalDecomposition
from RBniCS.utils.mpi import print
from RBniCS.utils.decorators import Extends, override, ReductionMethodFor
from RBniCS.problems.stokes_optimal_control.stokes_optimal_control_problem import StokesOptimalControlProblem
from RBniCS.reduction_methods.base import PODGalerkinReduction
from RBniCS.reduction_methods.stokes_optimal_control.stokes_optimal_control_reduction_method import StokesOptimalControlReductionMethod

StokesOptimalControlPODGalerkinReduction_Base = PODGalerkinReduction(StokesOptimalControlReductionMethod)

@Extends(StokesOptimalControlPODGalerkinReduction_Base) # needs to be first in order to override for last the methods
@ReductionMethodFor(StokesOptimalControlProblem, "PODGalerkin")
class StokesOptimalControlPODGalerkinReduction(StokesOptimalControlPODGalerkinReduction_Base):    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the POD-Galerkin ROM object
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call the parent initialization
        StokesOptimalControlPODGalerkinReduction_Base.__init__(self, truth_problem, **kwargs)
        
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
        output = StokesOptimalControlReductionMethod._init_offline(self)
        
        # Declare a new POD for each basis component
        self.POD = dict()
        for component in ("v", "p", "u", "w", "q"):
            inner_product = self.truth_problem.inner_product[component][0]
            self.POD[component] = ProperOrthogonalDecomposition(self.truth_problem.V, inner_product)
        for component in ("s", "r"):
            inner_product = self.truth_problem.inner_product[component][0]
            self.POD[component] = ProperOrthogonalDecomposition(self.truth_problem.V, inner_product, component=component)
            
        # Return
        return output
    
    ## Update the snapshots matrix: overridden version because supremizer POD is different from a standard component
    def update_snapshots_matrix(self, snapshot_and_supremizers):
        assert isinstance(snapshot_and_supremizers, tuple)
        assert len(snapshot_and_supremizers) == 3
        snapshot = snapshot_and_supremizers[0]
        supremizer = dict()
        supremizer["s"] = snapshot_and_supremizers[1]
        supremizer["r"] = snapshot_and_supremizers[2]
        for component in ("v", "p", "u", "w", "q"):
            self.POD[component].store_snapshot(snapshot, component=component)
        for component in ("s", "r"):
            self.POD[component].store_snapshot(supremizer[component])
    
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
        for pair in (("v", "w"), ("s", "r"), ("p", "q")):
            for component_to in pair:
                for component_from in pair:
                    for i in range(N[component_from]):
                        self.reduced_problem.Z.enrich(Z[component_from][i], component={component_from: component_to})
                    self.reduced_problem.N[component_to] += N[component_from]
        
        # Save
        self.reduced_problem.Z.save(self.reduced_problem.folder["basis"], "basis")
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the testing set
    @override
    def error_analysis(self, N=None, **kwargs):        
        components = ["v", "p", "u", "w", "q"] # but not supremizers
        kwargs["components"] = components
                
        StokesOptimalControlPODGalerkinReduction_Base.error_analysis(self, N, **kwargs)
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    
