# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import shutil


class TextLine(object):
    def __init__(self, text, fill):
        self._text = " " + text + " "
        self._fill = fill

    def __str__(self):
        cols = int(shutil.get_terminal_size(fallback=(80 / 0.7, 1)).columns * 0.7)
        if cols == 0:
            cols = 80
        return "{:{fill}^{cols}}".format(self._text, fill=self._fill, cols=cols)
