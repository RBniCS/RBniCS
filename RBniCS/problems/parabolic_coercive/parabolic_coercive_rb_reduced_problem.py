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
## @file parabolic_coercive_reduced_problem.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.problems.elliptic_coercive import EllipticCoerciveRBReducedProblem
from RBniCS.problems.parabolic_coercive.parabolic_coercive_reduced_problem import ParabolicCoerciveReducedProblem
from RBniCS.utils.decorators import Extends, override, ReducedProblemFor
from RBniCS.problems.parabolic_coercive.parabolic_coercive_problem import ParabolicCoerciveProblem
from RBniCS.reduction_methods.parabolic_coercive import ParabolicCoerciveRBReduction
from RBniCS.backends import Function, FunctionsList, LinearSolver, product, sum, TimeQuadrature, transpose
from RBniCS.backends.online import OnlineAffineExpansionStorage

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ParabolicCoerciveReducedOrderModelBase
#

ParabolicCoerciveRBReducedProblem_Base = ParabolicCoerciveReducedProblem(EllipticCoerciveRBReducedProblem)

# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
@Extends(ParabolicCoerciveRBReducedProblem_Base) # needs to be first in order to override for last the methods
@ReducedProblemFor(ParabolicCoerciveProblem, ParabolicCoerciveRBReduction)
class ParabolicCoerciveRBReducedProblem(ParabolicCoerciveRBReducedProblem_Base):
    
    ## Default initialization of members.
    @override
    def __init__(self, truth_problem):
        # Call to parent
        ParabolicCoerciveRBReducedProblem_Base.__init__(self, truth_problem)
        
    def _init_error_estimation_operators(self, current_stage="online"):
        ParabolicCoerciveRBReducedProblem_Base._init_error_estimation_operators(self, current_stage)
        # Also initialize data structures related to error estimation (mass term)
        assert current_stage in ("online", "offline")
        if current_stage == "online":
            self.riesz_product["mm"] = self.assemble_error_estimation_operators("riesz_product_mm", "online")
            self.riesz_product["ma"] = self.assemble_error_estimation_operators("riesz_product_ma", "online")
            self.riesz_product["mf"] = self.assemble_error_estimation_operators("riesz_product_mf", "online")
        elif current_stage == "offline":
            self.riesz["m"] = OnlineAffineExpansionStorage(self.Q["m"])
            for qm in range(self.Q["m"]):
                self.riesz["m"][qm] = FunctionsList(self.truth_problem.V)
            self.riesz_product["mm"] = OnlineAffineExpansionStorage(self.Q["m"], self.Q["m"])
            self.riesz_product["ma"] = OnlineAffineExpansionStorage(self.Q["m"], self.Q["a"])
            self.riesz_product["mf"] = OnlineAffineExpansionStorage(self.Q["m"], self.Q["f"])
        else:
            raise AssertionError("Invalid stage in _init_error_estimation_operators().")
            
    ## Return the numerator of the error bound for the current solution
    def get_residual_norm_squared(self):
        residual_norm_squared_over_time = list() # of float
        for (k, (solution, solution_dot)) in enumerate(zip(self._solution_over_time, self._solution_dot_over_time)):
            # Set current time
            self.t = k*self.dt
            # Set current solution and solution_dot
            self._solution = solution
            self._solution_dot = solution_dot
            # Compute the numerator of the error bound at the current time
            if k > 0:
                # Compute residual of elliptic part
                elliptic_residual_norm_squared = ParabolicCoerciveRBReducedProblem_Base.get_residual_norm_squared(self)
                # Add terms related to time derivative
                N = self._solution.N
                theta_m = self.compute_theta("m")
                theta_a = self.compute_theta("a")
                theta_f = self.compute_theta("f")
                residual_norm_squared_over_time.append(
                      elliptic_residual_norm_squared
                    + 2.0*(transpose(self._solution_dot)*sum(product(theta_m, self.riesz_product["mf"][:N], theta_f)))
                    + 2.0*(transpose(self._solution_dot)*sum(product(theta_m, self.riesz_product["ma"][:N, :N], theta_a))*self._solution)
                    + transpose(self._solution_dot)*sum(product(theta_m, self.riesz_product["mm"][:N, :N], theta_m))*self._solution_dot
                )
            else:
                # Initial condition is approximated exactly
                residual_norm_squared_over_time.append(0.)
        # Integrate in time
        time_quadrature = TimeQuadrature((0., self.T), self.dt)
        return time_quadrature.integrate(residual_norm_squared_over_time)
        
    ## Build operators for error estimation
    def build_error_estimation_operators(self):
        ParabolicCoerciveRBReducedProblem_Base.build_error_estimation_operators(self)
            
        # Update the Riesz representation of -M*Z with the new basis function(s)
        self.update_riesz_m()
        # Update the (m, f) Riesz representors product with the new basis function
        self.assemble_error_estimation_operators("riesz_product_mf", "offline")
        # Update the (m, a) Riesz representors product with the new basis function
        self.assemble_error_estimation_operators("riesz_product_ma", "offline")
        # Update the (m, m) Riesz representors product with the new basis function
        self.assemble_error_estimation_operators("riesz_product_mm", "offline")
        
    ## Compute the Riesz representation of m
    def update_riesz_m(self):
        # We need to clear out previously computed Riesz representors, because
        # POD-Greedy basis are not hierarchical from one greedy iteration to the next one
        for qm in range(self.Q["m"]):
            self.riesz["m"][qm].clear()
        
        # Re-compute all Riesz representor of m
        assert len(self.truth_problem.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
        inner_product = self.truth_problem.inner_product[0]
        for qm in range(self.Q["m"]):
            for n in range(len(self.riesz["m"][qm]), self.N + self.N_bc):
                if self.truth_problem.dirichlet_bc is not None:
                    theta_bc = (0.,)*len(self.truth_problem.dirichlet_bc)
                    homogeneous_dirichlet_bc = sum(product(theta_bc, self.truth_problem.dirichlet_bc))
                else:
                    homogeneous_dirichlet_bc = None
                solver = LinearSolver(inner_product, self._riesz_solve_storage, -1.*self.truth_problem.operator["m"][qm]*self.Z[n], homogeneous_dirichlet_bc)
                solver.solve()
                self.riesz["m"][qm].enrich(self._riesz_solve_storage)
                
    ## Compute the Riesz representation of a
    def update_riesz_a(self):
        # We need to clear out previously computed Riesz representors, because
        # POD-Greedy basis are not hierarchical from one greedy iteration to the next one
        for qa in range(self.Q["a"]):
            self.riesz["a"][qa].clear()
            
        # Re-compute all Riesz representor of a
        ParabolicCoerciveRBReducedProblem_Base.update_riesz_a(self)
                
    ## Assemble operators for error estimation
    def assemble_error_estimation_operators(self, term, current_stage="online"):
        assert current_stage in ("online", "offline")
        short_term = term.replace("riesz_product_", "")
        if current_stage == "online": # load from file
            if not short_term in self.riesz_product:
                self.riesz_product[short_term] = OnlineAffineExpansionStorage(0, 0) # it will be resized by load
            if term == "riesz_product_mm":
                self.riesz_product["mm"].load(self.folder["error_estimation"], "riesz_product_mm")
            elif term == "riesz_product_ma":
                self.riesz_product["ma"].load(self.folder["error_estimation"], "riesz_product_ma")
            elif term == "riesz_product_mf":
                self.riesz_product["mf"].load(self.folder["error_estimation"], "riesz_product_mf")
            else:
                return ParabolicCoerciveRBReducedProblem_Base.assemble_error_estimation_operators(self, term, current_stage)
            return self.riesz_product[short_term]
        elif current_stage == "offline":
            assert len(self.truth_problem.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
            inner_product = self.truth_problem.inner_product[0]
            if term == "riesz_product_mm":
                for qm in range(self.Q["m"]):
                    assert len(self.riesz["m"][qm]) == self.N + self.N_bc
                    for qmp in range(qm, self.Q["m"]):
                        assert len(self.riesz["m"][qmp]) == self.N + self.N_bc
                        self.riesz_product["mm"][qm, qmp] = transpose(self.riesz["m"][qm])*inner_product*self.riesz["m"][qmp]
                        if qm != qmp:
                            self.riesz_product["mm"][qmp, qm] = self.riesz_product["mm"][qm, qmp]
                self.riesz_product["mm"].save(self.folder["error_estimation"], "riesz_product_mm")
            elif term == "riesz_product_ma":
                for qm in range(self.Q["m"]):
                    assert len(self.riesz["m"][qm]) == self.N + self.N_bc
                    for qa in range(0, self.Q["a"]):
                        assert len(self.riesz["a"][qa]) == self.N + self.N_bc
                        self.riesz_product["ma"][qm, qa] = transpose(self.riesz["m"][qm])*inner_product*self.riesz["a"][qa]
                self.riesz_product["ma"].save(self.folder["error_estimation"], "riesz_product_ma")
            elif term == "riesz_product_mf":
                for qm in range(self.Q["m"]):
                    assert len(self.riesz["m"][qm]) == self.N + self.N_bc
                    for qf in range(0, self.Q["f"]):
                        assert len(self.riesz["f"][qf]) == 1
                        self.riesz_product["mf"][qm, qf] = transpose(self.riesz["m"][qm])*inner_product*self.riesz["f"][qf][0]
                self.riesz_product["mf"].save(self.folder["error_estimation"], "riesz_product_mf")
            else:
                return ParabolicCoerciveRBReducedProblem_Base.assemble_error_estimation_operators(self, term, current_stage)
            return self.riesz_product[short_term]
        else:
            raise AssertionError("Invalid stage in assemble_error_estimation_operators().")
        
