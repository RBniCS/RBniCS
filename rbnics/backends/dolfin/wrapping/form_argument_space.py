# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later


def form_argument_space(form, number):
    all_arguments = form.arguments()
    number_arguments = list()
    for argument in all_arguments:
        if argument.number() == number:
            number_arguments.append(argument)
    assert len(number_arguments) == 1
    return number_arguments[0].function_space()
