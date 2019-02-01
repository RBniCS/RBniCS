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

import shutil

class TextBox(object):
    def __init__(self, text, fill):
        self._text = text.split("\n")
        self._fill = fill
        
    def __str__(self):
        cols = int(shutil.get_terminal_size(fallback=(80/0.7, 1)).columns*0.7)
        if cols == 0:
            cols = 80
        first_last = "{:{fill}^{cols}}".format("", fill=self._fill, cols=cols)
        content = "\n".join([self._fill + "{:^{cols}}".format(t, cols=cols - 2) + self._fill for t in self._text])
        return first_last + "\n" + content + "\n" + first_last
