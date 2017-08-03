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

from ufl import replace
from rbnics.backends import product, sum, transpose
from rbnics.backends.online import OnlineAffineExpansionStorage
from rbnics.problems.navier_stokes.navier_stokes_reduced_problem import NavierStokesReducedProblem
from rbnics.utils.decorators import Extends, override

def NavierStokesTensor3ReducedProblem(NavierStokesReducedProblem_DerivedClass):
    
    NavierStokesTensor3ReducedProblem_Base = NavierStokesReducedProblem(NavierStokesReducedProblem_DerivedClass)
    
    @Extends(NavierStokesTensor3ReducedProblem_Base)
    class NavierStokesTensor3ReducedProblem_Class(NavierStokesTensor3ReducedProblem_Base):
        @override
        def __init__(self, truth_problem, **kwargs):
            # Store value of N passed to solve
            self._N_solve = None
            
        @override
        def _init_operators(self, current_stage="online"):
            assert current_stage in ("online", "offline")
            if current_stage == "online":
                # Call Parent
                NavierStokesTensor3ReducedProblem_Base._init_operators(self, "online")
            elif current_stage == "offline":
                # Call Parent
                NavierStokesTensor3ReducedProblem_Base._init_operators(self, "offline")
                # Make sure to disable Q and operators associated to c and c,
                # since they will be correctly resized in assemble_operator()
                self.Q["c"] = self.Q["dc"] = None
                self.operator["c"] = self.operator["dc"] = OnlineAffineExpansionStorage()
            else:
                raise AssertionError("Invalid stage in _init_operators().")
                
        @override
        def _solve(self, N, **kwargs):
            self._update_N_solve(N)
            NavierStokesTensor3ReducedProblem_Base._solve(self, N, **kwargs)
            
        def _update_N_solve(self, N):
            self._N_solve = N
            
        @override
        def compute_theta(self, term):
            if term in ("c", "dc"):
                N = dict()
                N["u"] = self._N_solve["u"]
                N["s"] = self._N_solve["s"]
                truth_Q = self.truth_problem.Q[term]
                truth_theta = NavierStokesTensor3ReducedProblem_Base.compute_theta(self, term)
                reduced_theta = list()
                reduced_solution = self._solution
                if term == "c":
                    for truth_q in range(truth_Q):
                        for c1 in ("u", "s"):
                            for n1 in range(N[c1]):
                                for c2 in ("u", "s"):
                                    for n2 in range(N[c2]):
                                        reduced_theta.append(truth_theta[truth_q]*reduced_solution[c1][n1]*reduced_solution[c2][n2])
                elif term == "dc":
                    for truth_q in range(truth_Q):
                        for c in ("u", "s"):
                            for n in range(N[c]):
                                reduced_theta.append(truth_theta[truth_q]*reduced_solution[c][n])
                return tuple(reduced_theta)
            else:
                return NavierStokesTensor3ReducedProblem_Base.compute_theta(self, term)
        
        @override
        def assemble_operator(self, term, current_stage="online"):
            assert current_stage in ("online", "offline")
            if current_stage == "online": # load from file
                return NavierStokesTensor3ReducedProblem_Base.assemble_operator(self, term, "online")
            elif current_stage == "offline":
                if term in ("c", "dc"):
                    N = dict()
                    N["u"] = len(Z["u"])
                    N["s"] = len(Z["s"])
                    truth_Q = self.truth_problem.Q[term]
                    reduced_Q = original_Q*(N["u"] + N["s"])
                    self.Q[term] = reduced_Q
                    self.operator[term] = OnlineAffineExpansionStorage(reduced_Q)
                    test = TestFunction(self.truth_problem.V)
                    reduced_q = 0
                    if term == "c":
                        for truth_q in range(truth_Q):
                            for c1 in ("u", "s"):
                                for n1 in range(N[c1]):
                                    for c2 in ("u", "s"):
                                        for n2 in range(N[c2]):
                                            truth_operator_q_c_n = replace(
                                                self.truth_problem.operator[term + "_tensor3"][truth_q],
                                                {
                                                    self.truth_problem.solution_placeholder_1: self.Z[c1][n1],
                                                    self.truth_problem.solution_placeholder_2: self.Z[c2][n2],
                                                    self.truth_problem.solution_placeholder_3: test
                                                }
                                            )
                                            self.operator[term][reduced_q] = transpose(self.Z)*truth_operator_q_c_n
                                            reduced_q += 1
                    elif term == "dc":
                        trial = TrialFunction(self.truth_problem.V)
                        for truth_q in range(truth_Q):
                            for c in ("u", "s"):
                                for n in range(N[c]):
                                    truth_operator_q_c_n = replace(
                                        self.truth_problem.operator[term + "_tensor3"][truth_q],
                                        {
                                            self.truth_problem.solution_placeholder_1: self.Z[c][n],
                                            self.truth_problem.solution_placeholder_2: trial,
                                            self.truth_problem.solution_placeholder_3: test
                                        }
                                    )
                                    self.operator["dc"][reduced_q] = transpose(self.Z)*truth_operator_q_c_n*self.Z
                                    reduced_q += 1
                    if "reduced_operators" in self.folder:
                        self.operator[term].save(self.folder["reduced_operators"], "operator_" + term)
                    return self.operator[term]
                else:
                    return NavierStokesTensor3ReducedProblem_Base.assemble_operator(self, term, "offline")
            else:
                raise AssertionError("Invalid stage in assemble_operator().")
        
    # return value (a class) for the decorator
    return NavierStokesTensor3ReducedProblem_Class

