# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.io import TextIO


class VerticesMappingIO(object):
    # Save a variable to file
    @staticmethod
    def save_file(vertices_mapping, directory, filename):
        if not filename.endswith(".vmp"):
            filename = filename + ".vmp"
        TextIO.save_file(vertices_mapping, directory, filename)

    # Load a variable from file
    @staticmethod
    def load_file(directory, filename):
        if not filename.endswith(".vmp"):
            filename = filename + ".vmp"
        return TextIO.load_file(directory, filename)

    # Check if the file exists
    @staticmethod
    def exists_file(directory, filename):
        if not filename.endswith(".vmp"):
            filename = filename + ".vmp"
        return TextIO.exists_file(directory, filename)
