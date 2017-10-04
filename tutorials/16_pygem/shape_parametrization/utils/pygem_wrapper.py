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

from numpy import hstack, zeros
from pygem.freeform import FFD
from pygem.radial import RBF
from pygem.params import FFDParameters, RBFParameters

class PyGeMWrapper(object):
    def __init__(self, other_pygem_wrapper=None, **kwargs):
        assert (
            (other_pygem_wrapper is not None and len(kwargs) == 0)
                or
            (other_pygem_wrapper is None and len(kwargs) > 0)
        )
        if other_pygem_wrapper is None:
            self.pygem_morphing_type = kwargs["pygem_morphing_type"]
            self.pygem_parameters_filename = kwargs["pygem_parameters_filename"]
            self.pygem_index_and_component_to_mu_index_map = kwargs["pygem_index_and_component_to_mu_index_map"]
        else:
            self.pygem_morphing_type = other_pygem_wrapper.pygem_morphing_type
            self.pygem_parameters_filename = other_pygem_wrapper.pygem_parameters_filename
            self.pygem_index_and_component_to_mu_index_map = other_pygem_wrapper.pygem_index_and_component_to_mu_index_map
            
        if self.pygem_morphing_type == "FFD":
            ParametersType = FFDParameters
        elif self.pygem_morphing_type == "RBF":
            ParametersType = RBFParameters
        else:
            raise RuntimeError("Invalid morphing.")
            
        self.params = ParametersType()
        self.params.read_parameters(filename=self.pygem_parameters_filename)
        
        # Mesh storage
        self.mesh = None
        self.dim = None
        self.reference_mesh_points = None
        self.mesh_motion = None
        
    def init(self, mesh):
        if self.pygem_morphing_type == "FFD":
            MorphingType = FFD
        elif self.pygem_morphing_type == "RBF":
            MorphingType = RBF
        else:
            raise RuntimeError("Invalid morphing.")
            
        self.mesh = mesh
        self.dim = self.mesh.ufl_cell().geometric_dimension()
        self.reference_mesh_points = self.mesh.coordinates().copy()
        self.mesh_motion = MorphingType(self.params, self._to_3d(self.reference_mesh_points))
        
    def update(self, mu):
        if self.pygem_morphing_type == "FFD":
            for (pygem_index_and_component, mu_index) in self.pygem_index_and_component_to_mu_index_map.items():
                assert len(pygem_index_and_component) == 2
                pygem_index = pygem_index_and_component[0]
                assert len(pygem_index) == 3
                pygem_component = pygem_index_and_component[1]
                assert pygem_component in ("x", "y", "z")
                if pygem_component == "x":
                    self.params.array_mu_x[pygem_index[0]][pygem_index[1]][pygem_index[2]] = mu[mu_index]
                elif pygem_component == "y":
                    self.params.array_mu_y[pygem_index[0]][pygem_index[1]][pygem_index[2]] = mu[mu_index]
                elif pygem_component == "z":
                    self.params.array_mu_z[pygem_index[0]][pygem_index[1]][pygem_index[2]] = mu[mu_index]
                else:
                    raise AssertionError("Invalid component.")
        elif self.pygem_morphing_type == "RBF":
            control_points_displacements = zeros((self.params.n_control_points, 3))
            for (pygem_index_and_component, mu_index) in self.pygem_index_and_component_to_mu_index_map.items():
                assert len(pygem_index_and_component) == 2
                pygem_index = pygem_index_and_component[0]
                assert isinstance(pygem_index, int)
                pygem_component = pygem_index_and_component[1]
                assert pygem_component in ("x", "y", "z")
                control_points_displacements[pygem_index] = mu[mu_index]
            self.params.deformed_control_points = control_points_displacements + self.params.original_control_points
            self.mesh_motion.weights = self._get_weights(self.params.original_control_points, self.params.deformed_control_points)
        else:
            raise RuntimeError("Invalid morphing.")
        
    def move_mesh(self):
        self.mesh_motion.perform()
        self.mesh.coordinates()[:] = self._from_3d(self.mesh_motion.modified_mesh_points)
        
    def reset_reference(self):
        self.mesh.coordinates()[:] = self.reference_mesh_points
    
    def _to_3d(self, coordinates):
        for _ in range(self.dim, 3):
            coordinates = hstack((coordinates, zeros((coordinates.shape[0], 1), dtype=coordinates.dtype)))
        return coordinates
        
    def _from_3d(self, coordinates):
        return coordinates[:, :self.dim]
        
