# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import re


def strings_to_number_of_parameters(strings):
    P = -1
    for string in strings:
        for match in mu_regex.findall(string):
            P = max(P, int(match))
    return P + 1


mu_regex = re.compile(r"mu\[([0-9]+)\]")
