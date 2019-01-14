# Copyright (C) 2015-2019 by the RBniCS authors
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

from rbnics.utils.decorators import PreserveClassName, ReductionMethodDecoratorFor
from rbnics.utils.io import TextBox
from backends.online import OnlineSolveKwargsGenerator
from problems import OnlineRectification

@ReductionMethodDecoratorFor(OnlineRectification)
def OnlineRectificationDecoratedReductionMethod(EllipticCoerciveReductionMethod_DerivedClass):
    
    @PreserveClassName
    class OnlineRectificationDecoratedReductionMethod_Class(EllipticCoerciveReductionMethod_DerivedClass):
        def _offline(self):
            # Change default online solve arguments during offline stage to not use rectification
            # (which will be prepared in a postprocessing stage)
            self.reduced_problem._online_solve_default_kwargs["online_rectification"] = False
            self.reduced_problem.OnlineSolveKwargs = OnlineSolveKwargsGenerator(**self.reduced_problem._online_solve_default_kwargs)
            
            # Call standard offline phase
            EllipticCoerciveReductionMethod_DerivedClass._offline(self)
            
            # Start rectification postprocessing
            print(TextBox(self.truth_problem.name() + " " + self.label + " offline rectification postprocessing phase begins", fill="="))
            print("")
            
            # Compute projection of truth and reduced snapshots
            self.reduced_problem.init("offline_rectification_postprocessing")
            self.reduced_problem.build_reduced_operators("offline_rectification_postprocessing")
            
            # Carry out a consistency verification of the rectified solution
            for n in range(1, self.reduced_problem.N + 1):
                print("consistency verification of rectified solutions for n =", n)
                for online_solve_kwargs in self.reduced_problem.online_solve_kwargs_with_rectification:
                    print("\tonline solve options:", dict(online_solve_kwargs))
                    for mu_i in self.reduced_problem.snapshots_mu[:n]:
                        self.reduced_problem.set_mu(mu_i)
                        self.reduced_problem.solve(n, **online_solve_kwargs)
                        error = self.reduced_problem.compute_error(**online_solve_kwargs)
                        print("\t\tmu = " + str(mu_i) + ", absolute error = " + str(error))
                        
            print(TextBox(self.truth_problem.name() + " " + self.label + " offline rectification postprocessing phase ends", fill="="))
            print("")
            
            # Restore default online solve arguments for online stage
            self.reduced_problem._online_solve_default_kwargs["online_rectification"] = True
            self.reduced_problem.OnlineSolveKwargs = OnlineSolveKwargsGenerator(**self.reduced_problem._online_solve_default_kwargs)
            
        def update_basis_matrix(self, snapshot):
            # Store
            self.reduced_problem.snapshots_mu.append(self.truth_problem.mu)
            self.reduced_problem.snapshots.enrich(snapshot)
            # Call Parent
            EllipticCoerciveReductionMethod_DerivedClass.update_basis_matrix(self, snapshot)
            
    # return value (a class) for the decorator
    return OnlineRectificationDecoratedReductionMethod_Class
