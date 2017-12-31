# Copyright (C) 2015-2018 by the RBniCS authors
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

import dolfin
from dolfin import plot as original_plot
from rbnics.backends.online import OnlineFunction

def plot(obj, *args, **kwargs):
    if isinstance(obj, OnlineFunction.Type()):
        assert "reduced_problem" in kwargs, "Please use this method as plot(reduced_solution, reduced_problem=my_reduced_problem) when plotting a reduced solution"
        N = obj.N
        Z = kwargs["reduced_problem"].Z[:N]
        del kwargs["reduced_problem"]
        original_plot(Z*obj, *args, **kwargs)
    else:
        original_plot(obj, *args, **kwargs)
dolfin.plot = plot
