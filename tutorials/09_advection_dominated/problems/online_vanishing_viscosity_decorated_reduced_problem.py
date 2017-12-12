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

from collections import OrderedDict
from rbnics.backends import AffineExpansionStorage, LinearSolver, product, sum
from rbnics.backends.online import OnlineAffineExpansionStorage, OnlineFunction
from rbnics.utils.decorators import PreserveClassName, ReducedProblemDecoratorFor
from backends.online import OnlineMatrix, OnlineNonHierarchicalAffineExpansionStorage, OnlineSolveKwargsGenerator
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
            
            # Default values for keyword arguments in solve
            self._online_solve_default_kwargs = OrderedDict()
            self._online_solve_default_kwargs["online_stabilization"] = False
            self._online_solve_default_kwargs["online_vanishing_viscosity"] = True
            self.OnlineSolveKwargs = OnlineSolveKwargsGenerator(**self._online_solve_default_kwargs)
            
        def _init_operators(self, current_stage="online"):
            # The difference between this method and the parent one is that non-hierarchical affine
            # expansion storage is requested.
            # (note that the non-hierarchical storage will be used also during the offline stage,
            #  where the bases are still hierarchical, and not only during the offline postprocessing)
            assert current_stage in ("online", "offline")
            if current_stage == "online":
                # Initialize all terms using a non-hierarchical affine expansion storage, and then loading from file
                for term in self.terms:
                    self.operator[term] = OnlineNonHierarchicalAffineExpansionStorage(0) # it will be resized by assemble_operator
                    self.assemble_operator(term, "online")
                    self.Q[term] = len(self.operator[term])
                # Initialize additional reduced operator related to vanishing viscosity
                self.operator["vanishing_viscosity"] = OnlineNonHierarchicalAffineExpansionStorage(1)
                self.assemble_operator("vanishing_viscosity", "online")
            elif current_stage == "offline":
                # Initialize additional truth operators
                self.truth_problem.operator["k"] = AffineExpansionStorage(self.truth_problem.assemble_operator("k"))
                self.truth_problem.operator["m"] = AffineExpansionStorage(self.truth_problem.assemble_operator("m"))
                # Initialize all terms using a non-hierarchical affine expansion storage
                for term in self.terms:
                    self.Q[term] = self.truth_problem.Q[term]
                    self.operator[term] = OnlineNonHierarchicalAffineExpansionStorage(self.Q[term])
                # Initialize additional reduced operator related to vanishing viscosity
                self.operator["vanishing_viscosity"] = OnlineNonHierarchicalAffineExpansionStorage(1)
            else:
                raise ValueError("Invalid stage in _init_operators().")
                
        def _init_inner_products(self, current_stage="online"):
            # The difference between this method and the parent one is that non-hierarchical affine
            # expansion storage is requested.
            self.inner_product = OnlineNonHierarchicalAffineExpansionStorage(1)
            self.projection_inner_product = OnlineNonHierarchicalAffineExpansionStorage(1)
            assert current_stage in ("online", "offline")
            if current_stage == "online":
                self.assemble_operator("inner_product", "online")
                self.assemble_operator("projection_inner_product", "online")
                self._combined_inner_product = self._combine_all_inner_products()
                self._combined_projection_inner_product = self._combine_all_projection_inner_products()
            elif current_stage == "offline":
                pass # nothing more to be done
            else:
                raise ValueError("Invalid stage in _init_inner_products().")
            
        def assemble_operator(self, term, current_stage="online"):
            if term == "vanishing_viscosity":
                assert current_stage in ("online", "offline")
                if current_stage == "online": # load from file
                    self.operator["vanishing_viscosity"].load(self.folder["reduced_operators"], "operator_vanishing_viscosity")
                    return self.operator["vanishing_viscosity"]
                elif current_stage == "offline":
                    assert len(self.vanishing_viscosity_eigenvalues) is self.N
                    assert all([isinstance(vanishing_viscosity_eigenvalues_n, list) for vanishing_viscosity_eigenvalues_n in self.vanishing_viscosity_eigenvalues])
                    assert all([len(vanishing_viscosity_eigenvalues_n) is n + 1 for (n, vanishing_viscosity_eigenvalues_n) in enumerate(self.vanishing_viscosity_eigenvalues)])
                    print("build reduced vanishing viscosity operator")
                    for n in range(1, N + 1):
                        vanishing_viscosity_expansion = OnlineAffineExpansionStorage(1)
                        vanishing_viscosity_eigenvalues = self.vanishing_viscosity_eigenvalues[n]
                        vanishing_viscosity_operator = OnlineMatrix(n, n)
                        n_min = int(n*self._N_threshold_min)
                        n_max = int(n*self._N_threshold_max)
                        lambda_n_min = vanishing_viscosity_eigenvalues[n_min]
                        lambda_n_max = vanishing_viscosity_eigenvalues[n_max]
                        for i in range(n):
                            lambda_i = vanishing_viscosity_eigenvalues[i]
                            if i < n_min:
                                viscosity_i = 0.
                            elif i < n_max:
                                viscosity_i = (
                                    self._viscosity *
                                    (lambda_i - lambda_n_min)**2/(lambda_n_max - lambda_n_min)**3 *
                                    (2*lambda_n_max**2 - (lambda_n_min + lambda_n_max)*lambda_i)
                                )
                            else:
                                viscosity_i = self._viscosity*lambda_i
                            vanishing_viscosity_operator[i, i] = viscosity_i*lambda_i
                        vanishing_viscosity_expansion[0] = vanishing_viscosity_operator
                        self.operator["vanishing_viscosity"][:n, :n] = vanishing_viscosity_expansion
                    # Save to file
                    self.operator["vanishing_viscosity"].save(self.folder["reduced_operators"], "operator_vanishing_viscosity")
                    return self.operator["vanishing_viscosity"]
                else:
                    raise ValueError("Invalid stage in assemble_operator().")
            else:
                return EllipticCoerciveReducedProblem_DerivedClass.assemble_operator(self, term, current_stage)
        
        def _online_size_from_kwargs(self, N, **kwargs):
            N, kwargs = EllipticCoerciveReducedProblem_DerivedClass._online_size_from_kwargs(self, N, **kwargs)
            kwargs = self.OnlineSolveKwargs(**kwargs)
            return N, kwargs
        
        def _solve(self, N, **kwargs):
            # Temporarily change value of stabilized attribute in truth problem
            bak_stabilized = self.truth_problem.stabilized
            self.truth_problem.stabilized = kwargs["online_stabilization"]
            # Solve reduced problem
            if kwargs["online_vanishing_viscosity"]:
                assembled_operator = dict()
                assembled_operator["a"] = (
                    sum(product(self.compute_theta("a"), self.operator["a"][:N, :N])) +
                    self.operator["vanishing_viscosity"][:N, :N][0]
                )
                assembled_operator["f"] = sum(product(self.compute_theta("f"), self.operator["f"][:N]))
                self._solution = OnlineFunction(N)
                solver = LinearSolver(assembled_operator["a"], self._solution, assembled_operator["f"])
                solver.solve()
            else:
                EllipticCoerciveReducedProblem_DerivedClass._solve(self, N, **kwargs)
            # Restore original value of stabilized attribute in truth problem
            self.truth_problem.stabilized = bak_stabilized
            
    # return value (a class) for the decorator
    return OnlineVanishingViscosityDecoratedReducedProblem_Class
