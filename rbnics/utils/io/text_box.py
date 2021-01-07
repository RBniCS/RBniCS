# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import shutil


class TextBox(object):
    def __init__(self, text, fill):
        self._text = text.split("\n")
        self._fill = fill

    def __str__(self):
        cols = int(shutil.get_terminal_size(fallback=(80 / 0.7, 1)).columns * 0.7)
        if cols == 0:
            cols = 80
        first_last = "{:{fill}^{cols}}".format("", fill=self._fill, cols=cols)
        content = "\n".join([self._fill + "{:^{cols}}".format(t, cols=cols - 2) + self._fill for t in self._text])
        return first_last + "\n" + content + "\n" + first_last
