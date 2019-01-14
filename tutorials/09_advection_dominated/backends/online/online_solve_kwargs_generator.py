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

from collections import namedtuple

def OnlineSolveKwargsGenerator(**kwargs):
    OnlineSolveKwargsTuple = namedtuple("OnlineSolveKwargs", kwargs.keys())
    OnlineSolveKwargsTuple.__new__.__defaults__ = tuple(kwargs.values())
    def OnlineSolveKwargs(*args_, **kwargs_):
        return OnlineSolveKwargsTuple(*args_, **kwargs_)._asdict()
    return OnlineSolveKwargs
