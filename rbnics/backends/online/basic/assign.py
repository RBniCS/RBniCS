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

from rbnics.utils.decorators import list_of, overload
from rbnics.utils.io import OnlineSizeDict

def assign(backend):
    class _Assign(object):
        @overload(backend.Function.Type(), backend.Function.Type())
        def __call__(self, object_to, object_from):
            if object_from is not object_to:
                assert isinstance(object_to.vector().N, (dict, int))
                assert isinstance(object_from.vector().N, (dict, int))
                if isinstance(object_from.vector().N, dict) and isinstance(object_to.vector().N, dict):
                    from_N_keys = set(object_from.vector().N.keys())
                    to_N_keys = set(object_to.vector().N.keys())
                    components_in_both = from_N_keys & to_N_keys
                    for c in components_in_both:
                        assert object_to.vector().N[c] == object_from.vector().N[c]
                    components_only_in_from = from_N_keys - to_N_keys
                    components_only_in_to = to_N_keys - from_N_keys
                    from_N_dict = OnlineSizeDict()
                    to_N_dict = OnlineSizeDict()
                    for c in components_in_both:
                        from_N_dict[c] = object_from.vector().N[c]
                        to_N_dict[c] = object_to.vector().N[c]
                    for c in components_only_in_from:
                        from_N_dict[c] = 0
                    for c in components_only_in_to:
                        to_N_dict[c] = 0
                    object_to.vector()[:to_N_dict] = object_from.vector()[:from_N_dict]
                    self._preserve_vector_attributes(object_to.vector(), object_from.vector(), len(components_only_in_from) > 0, len(components_only_in_to) > 0)
                elif isinstance(object_from.vector().N, int) and isinstance(object_to.vector().N, dict):
                    assert len(object_to.vector().N) == 1
                    raise ValueError("Refusing to assign a dict dimension N to an int dimension N")
                elif isinstance(object_from.vector().N, dict) and isinstance(object_to.vector().N, int):
                    assert len(object_from.vector().N) == 1
                    for (c, N_c) in object_from.vector().N.items():
                        break
                    assert N_c == object_to.vector().N
                    N = OnlineSizeDict()
                    N[c] = N_c
                    object_to.vector().N = N
                    object_to.vector()[:] = object_from.vector()
                    self._preserve_vector_attributes(object_to.vector(), object_from.vector())
                else: # isinstance(object_from.vector().N, int) and isinstance(object_to.vector().N, int):
                    assert object_to.vector().N == object_from.vector().N
                    object_to.vector()[:] = object_from.vector()
                    self._preserve_vector_attributes(object_to.vector(), object_from.vector())
                
        @overload(list_of(backend.Function.Type()), list_of(backend.Function.Type()))
        def __call__(self, object_to, object_from):
            if object_from is not object_to:
                del object_to[:]
                object_to.extend(object_from)
        
        @overload(backend.Matrix.Type(), backend.Matrix.Type())
        def __call__(self, object_to, object_from):
            if object_from is not object_to:
                assert object_to.N == object_from.N
                assert object_to.M == object_from.M
                object_to[:, :] = object_from
                self._preserve_matrix_attributes(object_to, object_from)
        
        @overload(backend.Vector.Type(), backend.Vector.Type())
        def __call__(self, object_to, object_from):
            if object_from is not object_to:
                assert object_to.N == object_from.N
                object_to[:] = object_from
                self._preserve_vector_attributes(object_to, object_from)
                
        def _preserve_vector_attributes(self, object_to, object_from, subset=False, superset=False):
            # Preserve auxiliary attributes related to basis functions matrix
            assert (object_to._component_name_to_basis_component_index is None) == (object_to._component_name_to_basis_component_length is None)
            if object_to._component_name_to_basis_component_index is None:
                assert not subset
                object_to._component_name_to_basis_component_index = object_from._component_name_to_basis_component_index
                object_to._component_name_to_basis_component_length = object_from._component_name_to_basis_component_length
            else:
                if not subset and not superset:
                    assert object_from._component_name_to_basis_component_index == object_to._component_name_to_basis_component_index
                    assert object_from._component_name_to_basis_component_length == object_to._component_name_to_basis_component_length
                elif subset:
                    assert set(object_to._component_name_to_basis_component_index.keys()) <= set(object_from._component_name_to_basis_component_index.keys())
                    assert object_to._component_name_to_basis_component_length.items() <= object_from._component_name_to_basis_component_length.items()
                else: # is superset
                    assert set(object_to._component_name_to_basis_component_index.keys()) >= set(object_from._component_name_to_basis_component_index.keys())
                    assert object_to._component_name_to_basis_component_length.items() >= object_from._component_name_to_basis_component_length.items()
                
        def _preserve_matrix_attributes(self, object_to, object_from):
            # Preserve auxiliary attributes related to basis functions matrix
            assert (object_to._component_name_to_basis_component_index[0] is None) == (object_to._component_name_to_basis_component_length[0] is None)
            assert (object_to._component_name_to_basis_component_index[0] is None) == (object_to._component_name_to_basis_component_index[1] is None)
            assert (object_to._component_name_to_basis_component_length[0] is None) == (object_to._component_name_to_basis_component_length[1] is None)
            if object_to._component_name_to_basis_component_index[0] is None:
                object_to._component_name_to_basis_component_index = object_from._component_name_to_basis_component_index
                object_to._component_name_to_basis_component_length = object_from._component_name_to_basis_component_length
            else:
                assert object_from._component_name_to_basis_component_index == object_to._component_name_to_basis_component_index
                assert object_from._component_name_to_basis_component_length == object_to._component_name_to_basis_component_length
    return _Assign()
