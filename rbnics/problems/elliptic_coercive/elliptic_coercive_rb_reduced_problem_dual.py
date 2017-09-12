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

from rbnics.backends import product, transpose, sum
from rbnics.backends.online import OnlineAffineExpansionStorage
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.base import DualReducedProblem
from rbnics.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from rbnics.problems.elliptic_coercive.elliptic_coercive_problem_dual import EllipticCoerciveProblem_Dual
from rbnics.problems.elliptic_coercive.elliptic_coercive_rb_reduced_problem import EllipticCoerciveRBReducedProblem
from rbnics.reduction_methods.elliptic_coercive import EllipticCoerciveRBReduction

EllipticCoerciveRBReducedProblem_Dual_Base = DualReducedProblem(EllipticCoerciveRBReducedProblem)

# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
@ReducedProblemFor(EllipticCoerciveProblem_Dual, EllipticCoerciveRBReduction)
class EllipticCoerciveRBReducedProblem_Dual(EllipticCoerciveRBReducedProblem_Dual_Base):
    
    ## Default initialization of members.
    def __init__(self, truth_problem, **kwargs):
        # Call to parent
        EllipticCoerciveRBReducedProblem_Dual_Base.__init__(self, truth_problem, **kwargs)
        
        # Residual terms
        self.output_correction_and_estimation = dict() # from string to OnlineAffineExpansionStorage
        
    ## Initialize data structures required for the online phase
    def init(self, current_stage="online"):
        EllipticCoerciveRBReducedProblem_Dual_Base.init(self, current_stage)
        self._init_output_correction_and_estimation_operators(current_stage)
        
    def _init_output_correction_and_estimation_operators(self, current_stage="online"):
        # Also initialize data structures related to output correction and output error estimation
        assert current_stage in ("online", "offline")
        if current_stage == "online":
            self.output_correction_and_estimation["a"] = self.assemble_output_correction_and_estimation_operators("output_correction_and_estimation_a", "online")
            self.output_correction_and_estimation["f"] = self.assemble_output_correction_and_estimation_operators("output_correction_and_estimation_f", "online")
        elif current_stage == "offline":
            self.output_correction_and_estimation["a"] = OnlineAffineExpansionStorage(self.primal_truth_problem.Q["a"])
            self.output_correction_and_estimation["f"] = OnlineAffineExpansionStorage(self.primal_truth_problem.Q["f"])
            # Save empty files to avoid triggering an assert while finalizing dual offline stage
            self.output_correction_and_estimation["a"].save(self.folder["error_estimation"], "output_correction_and_estimation_a")
            self.output_correction_and_estimation["f"].save(self.folder["error_estimation"], "output_correction_and_estimation_f")
        else:
            raise AssertionError("Invalid stage in _init_output_correction_and_estimation_operators().")
        
    # Perform an online evaluation of the output correction
    def _compute_output(self, dual_N):
        primal_solution = self.primal_reduced_problem._solution
        primal_N = primal_solution.N
        dual_solution = self._solution
        assembled_output_correction_and_estimation_operator = dict()
        assembled_output_correction_and_estimation_operator["a"] = sum(product(self.primal_reduced_problem.compute_theta("a"), self.output_correction_and_estimation["a"][:dual_N, :primal_N]))
        assembled_output_correction_and_estimation_operator["f"] = sum(product(self.primal_reduced_problem.compute_theta("f"), self.output_correction_and_estimation["f"][:dual_N]))
        self._output = transpose(dual_solution)*assembled_output_correction_and_estimation_operator["f"] - transpose(dual_solution)*assembled_output_correction_and_estimation_operator["a"]*primal_solution
    
    ## Build operators for output correction and error estimation
    def build_output_correction_and_estimation_operators(self):
        self.assemble_output_correction_and_estimation_operators("output_correction_and_estimation_a", "offline")
        self.assemble_output_correction_and_estimation_operators("output_correction_and_estimation_f", "offline")
    
    ## Assemble operators for output correction and error estimation
    def assemble_output_correction_and_estimation_operators(self, term, current_stage="online"):
        assert current_stage in ("online", "offline")
        short_term = term.replace("output_correction_and_estimation_", "")
        if current_stage == "online": # load from file
            if not term in self.output_correction_and_estimation:
                self.output_correction_and_estimation[short_term] = OnlineAffineExpansionStorage(0, 0) # it will be resized by load
            if term == "output_correction_and_estimation_a":
                self.output_correction_and_estimation["a"].load(self.folder["error_estimation"], "output_correction_and_estimation_a")
            elif term == "output_correction_and_estimation_f":
                self.output_correction_and_estimation["f"].load(self.folder["error_estimation"], "output_correction_and_estimation_f")
            else:
                raise ValueError("Invalid term for assemble_output_correction_and_estimation_operators().")
            return self.output_correction_and_estimation[short_term]
        elif current_stage == "offline":
            if term == "output_correction_and_estimation_a":
                for qa in range(self.primal_truth_problem.Q["a"]):
                    self.output_correction_and_estimation["a"][qa] = transpose(self.Z)*self.primal_truth_problem.operator["a"][qa]*self.primal_reduced_problem.Z
                self.output_correction_and_estimation["a"].save(self.folder["error_estimation"], "output_correction_and_estimation_a")
            elif term == "output_correction_and_estimation_f":
                for qf in range(self.primal_truth_problem.Q["f"]):
                    self.output_correction_and_estimation["f"][qf] = transpose(self.Z)*self.primal_truth_problem.operator["f"][qf]
                self.output_correction_and_estimation["f"].save(self.folder["error_estimation"], "output_correction_and_estimation_f")
            else:
                raise ValueError("Invalid term for assemble_output_correction_and_estimation_operators().")
            return self.output_correction_and_estimation[short_term]
        else:
            raise AssertionError("Invalid stage in assemble_output_correction_and_estimation_operators().")
    
