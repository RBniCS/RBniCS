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

from rbnics.utils.decorators import list_of, overload

def assign(backend):
    class _Assign(object):
        @overload(backend.Function.Type(), backend.Function.Type())
        def __call__(self, object_to, object_from):
            if object_from is not object_to:
                assert isinstance(object_to.vector().N, (dict, int))
                assert isinstance(object_to.vector().N, dict) is isinstance(object_from.vector().N, dict)
                if isinstance(object_from.vector().N, dict):
                    from_N_keys = set(object_from.vector().N.keys())
                    to_N_keys = set(object_to.vector().N.keys())
                    components_in_both = from_N_keys & to_N_keys
                    for c in components_in_both:
                        assert object_to.vector().N[c] == object_from.vector().N[c]
                    components_only_in_from = from_N_keys - to_N_keys
                    components_only_in_to = to_N_keys - from_N_keys
                    assert len(components_only_in_to) is 0
                    from_N_dict = dict()
                    for c in components_in_both:
                        from_N_dict[c] = object_from.vector().N[c]
                    for c in components_only_in_from:
                        from_N_dict[c] = 0
                    object_to.vector()[:] = object_from.vector()[:from_N_dict]
                else:
                    assert object_to.vector().N == object_from.vector().N
                    object_to.vector()[:] = object_from.vector()
            
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
        
        @overload(backend.Vector.Type(), backend.Vector.Type())
        def __call__(self, object_to, object_from):
            if object_from is not object_to:
                assert object_to.N == object_from.N
                object_to[:] = object_from
    return _Assign()
