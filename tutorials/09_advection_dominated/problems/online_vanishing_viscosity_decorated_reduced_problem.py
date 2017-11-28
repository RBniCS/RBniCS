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

from rbnics.backends import AffineExpansionStorage, LinearSolver, product, sum
from rbnics.backends.online import OnlineAffineExpansionStorage, OnlineFunction, OnlineMatrix
from rbnics.utils.decorators import is_training_finished, PreserveClassName, ReducedProblemDecoratorFor
from .online_vanishing_viscosity import OnlineVanishingViscosity

@ReducedProblemDecoratorFor(OnlineVanishingViscosity)
def OnlineVanishingViscosityDecoratedReducedProblem(EllipticCoerciveReducedProblem_DerivedClass):
    
    @PreserveClassName
    class OnlineVanishingViscosityDecoratedReducedProblem_Class(EllipticCoerciveReducedProblem_DerivedClass):
        
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            EllipticCoerciveReducedProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # Store vanishing viscosity data
            self._viscosity = truth_problem._viscosity
            self._N_threshold_min = truth_problem._N_threshold_min
            self._N_threshold_max = truth_problem._N_threshold_max
            
            # Temporary storage for vanishing viscosity eigenvalues
            self.vanishing_viscosity_eigenvalues = list()
            
        def _init_operators(self, current_stage="online"):
            if current_stage == "online":
                # Initialize additional reduced operators
                self.operator["vanishing_viscosity"] = self.assemble_operator("vanishing_viscosity", "online")
            elif current_stage == "offline":
                # Initialize additional truth operators
                self.truth_problem.operator["k"] = AffineExpansionStorage(self.truth_problem.assemble_operator("k"))
                self.truth_problem.operator["m"] = AffineExpansionStorage(self.truth_problem.assemble_operator("m"))
                # Initialize additional reduced operators
                self.operator["vanishing_viscosity"] = OnlineAffineExpansionStorage(1)
            # Call Parent
            EllipticCoerciveReducedProblem_DerivedClass._init_operators(self, current_stage)
            
        def build_reduced_operators(self):
            # Call Parent
            EllipticCoerciveReducedProblem_DerivedClass.build_reduced_operators(self)
            # Compute reduced vanishing viscosity stabilization operator
            self.operator["vanishing_viscosity"] = self.assemble_operator("vanishing_viscosity", "offline")
            
        def assemble_operator(self, term, current_stage="online"):
            if term == "vanishing_viscosity":
                assert current_stage in ("online", "offline")
                if current_stage == "online": # load from file
                    self.operator["vanishing_viscosity"] = OnlineAffineExpansionStorage(1)
                    self.operator["vanishing_viscosity"].load(self.folder["reduced_operators"], "operator_vanishing_viscosity")
                    return self.operator["vanishing_viscosity"]
                elif current_stage == "offline":
                    if len(self.vanishing_viscosity_eigenvalues) > 0: # basis was rotated
                        print("build reduced vanishing viscosity operator")
                        N = self.N
                        vanishing_viscosity_eigenvalues = self.vanishing_viscosity_eigenvalues
                        vanishing_viscosity_operator = OnlineMatrix(N, N)
                        N_min = int(N*self._N_threshold_min)
                        N_max = int(N*self._N_threshold_max)
                        lambda_N_min = vanishing_viscosity_eigenvalues[N_min]
                        lambda_N_max = vanishing_viscosity_eigenvalues[N_max]
                        for i in range(N):
                            lambda_i = vanishing_viscosity_eigenvalues[i]
                            if i < N_min:
                                viscosity_i = 0.
                            elif i < N_max:
                                viscosity_i = (
                                    self._viscosity *
                                    (lambda_i - lambda_N_min)**2/(lambda_N_max - lambda_N_min)**3 *
                                    (2*lambda_N_max**2 - (lambda_N_min + lambda_N_max)*lambda_i)
                                )
                            else:
                                viscosity_i = self._viscosity*lambda_i
                            vanishing_viscosity_operator[i, i] = viscosity_i*lambda_i
                        # Store and save
                        self.operator["vanishing_viscosity"][0] = vanishing_viscosity_operator
                        self.operator["vanishing_viscosity"].save(self.folder["reduced_operators"], "operator_vanishing_viscosity")
                        return self.operator["vanishing_viscosity"]
                    else: # basis has not been rotated yet
                        return self.operator["vanishing_viscosity"]
                else:
                    raise ValueError("Invalid stage in assemble_operator().")
            else:
                return EllipticCoerciveReducedProblem_DerivedClass.assemble_operator(self, term, current_stage)
                
        def _solve(self, N, **kwargs):
            if is_training_finished(self.truth_problem):
                # Temporarily change value of stabilized attribute in truth problem
                bak_stabilized = self.truth_problem.stabilized
                self.truth_problem.stabilized = kwargs.get("online_stabilization", False)
                # Solve reduced problem
                if kwargs.get("online_vanishing_viscosity", True):
                    assembled_operator = dict()
                    assembled_operator["a"] = (
                        sum(product(self.compute_theta("a"), self.operator["a"][:N, :N])) +
                        self.operator["vanishing_viscosity"][0][:N, :N]
                    )
                    assembled_operator["f"] = sum(product(self.compute_theta("f"), self.operator["f"][:N]))
                    self._solution = OnlineFunction(N)
                    solver = LinearSolver(assembled_operator["a"], self._solution, assembled_operator["f"])
                    solver.solve()
                else:
                    EllipticCoerciveReducedProblem_DerivedClass._solve(self, N, **kwargs)
                # Restore original value of stabilized attribute in truth problem
                self.truth_problem.stabilized = bak_stabilized
            else:
                EllipticCoerciveReducedProblem_DerivedClass._solve(self, N, **kwargs)
            
    # return value (a class) for the decorator
    return OnlineVanishingViscosityDecoratedReducedProblem_Class
