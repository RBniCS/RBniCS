# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later


def matrix_mul_vector(matrix, vector):
    return matrix * vector


def vectorized_matrix_inner_vectorized_matrix(matrix, other_matrix):
    return (matrix * other_matrix).sum()
