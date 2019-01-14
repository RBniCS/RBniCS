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

from ufl.algorithms.traversal import iter_expressions
from ufl.corealg.traversal import pre_traversal

def form_iterator(form, iterator_type="nodes"):
    assert iterator_type in ("nodes", "integrals")
    if iterator_type == "nodes":
        for integral in form.integrals():
            for expression in iter_expressions(integral):
                for node in pre_traversal(expression): # pre_traversal algorithms guarantees that subsolutions are processed before solutions
                    yield node
    elif iterator_type == "integrals":
        for integral in form.integrals():
            yield integral
    else:
        raise ValueError("Invalid iterator type")
