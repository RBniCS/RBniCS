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
## @file scm.py
#  @brief Implementation of the successive constraints method for the approximation of the coercivity constant
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from numpy import hstack, zeros
from pygem.freeform import FFD
from pygem.radial import RBF
from pygem.params import FFDParameters, RBFParameters
from RBniCS.utils.decorators import Extends, override, ProblemDecoratorFor

def PyGeMDecoratedProblem(pygem_morphing_type, pygem_parameters_filename, pygem_index_and_component_to_mu_index_map, **decorator_kwargs):
    assert pygem_morphing_type in ("FFD", "RBF")
    
    @ProblemDecoratorFor(PyGeM,
        pygem_morphing_type=pygem_morphing_type,
        pygem_parameters_filename=pygem_parameters_filename,
        pygem_index_and_component_to_mu_index_map=pygem_index_and_component_to_mu_index_map
    )
    def PyGeMDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):
        @Extends(ParametrizedDifferentialProblem_DerivedClass, preserve_class_name=True)
        class PyGeMDecoratedProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
        
            @override
            def __init__(self, V, **kwargs):
                # Call the standard initialization
                ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
                # Initialize pygem deformation
                self.mesh = V.mesh()
                self.dim = self.mesh.ufl_cell().geometric_dimension()
                self.reference_mesh_points = self.mesh.coordinates().copy()
                if pygem_morphing_type == "FFD":
                    ParametersType = FFDParameters
                    MorphingType = FFD
                elif pygem_morphing_type == "RBF":
                    ParametersType = RBFParameters
                    MorphingType = RBF
                else:
                    raise RuntimeError("Invalid morphing.")
                self.params = ParametersType()
                self.params.read_parameters(filename=pygem_parameters_filename)
                self.mesh_motion = MorphingType(self.params, self._to_3d(self.reference_mesh_points))
                
            @override
            def set_mu(self, mu):
                ParametrizedDifferentialProblem_DerivedClass.set_mu(self, mu)
                # Update pygem parameters and data structures
                if pygem_morphing_type == "FFD":
                    for (pygem_index_and_component, mu_index) in pygem_index_and_component_to_mu_index_map.iteritems():
                        assert len(pygem_index_and_component) == 2
                        pygem_index = pygem_index_and_component[0]
                        assert len(pygem_index) == 3
                        pygem_component = pygem_index_and_component[1]
                        assert pygem_component in ("x", "y", "z")
                        if pygem_component == "x":
                            self.params.array_mu_x[pygem_index[0]][pygem_index[1]][pygem_index[2]] = self.mu[mu_index]
                        elif pygem_component == "y":
                            self.params.array_mu_y[pygem_index[0]][pygem_index[1]][pygem_index[2]] = self.mu[mu_index]
                        elif pygem_component == "z":
                            self.params.array_mu_z[pygem_index[0]][pygem_index[1]][pygem_index[2]] = self.mu[mu_index]
                        else:
                            raise AssertionError("Invalid component.")
                elif pygem_morphing_type == "RBF":
                    control_points_displacements = zeros((self.params.n_control_points, 3))
                    for (pygem_index_and_component, mu_index) in pygem_index_and_component_to_mu_index_map.iteritems():
                        assert len(pygem_index_and_component) == 2
                        pygem_index = pygem_index_and_component[0]
                        assert isinstance(pygem_index, int)
                        pygem_component = pygem_index_and_component[1]
                        assert pygem_component in ("x", "y", "z")
                        control_points_displacements[pygem_index] = self.mu[mu_index]
                    self.params.deformed_control_points = control_points_displacements + self.params.original_control_points
                    self.mesh_motion.weights = self._get_weights(self.params.original_control_points, self.params.deformed_control_points)
                else:
                    raise RuntimeError("Invalid morphing.")
                # Deform the mesh
                self.mesh_motion.perform()
                self.mesh.coordinates()[:] = self._from_3d(self.mesh_motion.modified_mesh_points)
                 
            ## Initialize data structures required for the offline phase
            @override
            def init(self):
                ParametrizedDifferentialProblem_DerivedClass.init(self)
                # Check consistency between self.mu and parameters related to deformation.
                # Cannot do that in the initialization becase self.mu is not available yet.
                assert len(pygem_index_to_mu_index_map) <= len(self.mu)
                assert min(pygem_index_to_mu_index_map.values()) >= 0
                assert max(pygem_index_to_mu_index_map.values()) < len(self.mu)
                
            def _to_3d(self, coordinates):
                for _ in range(self.dim, 3):
                    coordinates = np.hstack((coordinates, np.zeros((coordinates.shape[0], 1), dtype=coordinates.dtype)))
                return coordinates
                
            def _from_3d(self, coordinates):
                return coordinates[:, :self.dim]
        
        # return value (a class) for the decorator
        return PyGeMDecoratedProblem_Class
    
    # return the decorator itself
    return PyGeMDecoratedProblem_Decorator
    
# For the sake of the user, since this is the only class that he/she needs to use, rename it to an easier name
PyGeM = PyGeMDecoratedProblem
