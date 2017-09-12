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


from numpy import array_equal
from dolfin import cells
from rbnics.sampling.distributions import Distribution, EquispacedDistribution

def DiscardInadmissibleDeformations(Distribution_DerivedClass):
    assert not issubclass(Distribution_DerivedClass, EquispacedDistribution) # we would have no way to replace inadmissible parameters
    
    class DiscardInadmissibleDeformations_Class(Distribution_DerivedClass):
        def __init__(self, truth_problem):
            self.truth_problem = truth_problem
            self.mesh = self.truth_problem.V.mesh()
            self.cells_orientation = [c.orientation() for c in cells(self.mesh)]
            
        def sample(self, box, n):
            print("Generating parameter space subset discarding inadmissible deformations.")
            # Backup truth problem mu
            mu_bak = tuple(list(self.truth_problem.mu))
            # Generate the parameter space subset
            set_ = list() # of tuples
            for i in range(n):
                deformation_admissible = False
                while not deformation_admissible:
                    set_i = Distribution_DerivedClass.sample(self, box, 1)
                    # Check if the deformation is admissible
                    self.truth_problem.set_mu(set_i[0])
                    deformed_cells_orientation = [c.orientation() for c in cells(self.mesh)]
                    deformation_admissible = array_equal(deformed_cells_orientation, self.cells_orientation)
                    if not deformation_admissible:
                        print("Inadmissible deformation corresponding to mu =", self.truth_problem.mu, "has been discarded.")
                set_.append(self.truth_problem.mu)
            # Restore original truth problem mu
            self.truth_problem.set_mu(mu_bak)
            # Return
            print("Parameter space subset generation completed.")
            return set_
            
    return DiscardInadmissibleDeformations_Class
    
