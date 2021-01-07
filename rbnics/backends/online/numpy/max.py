# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.online.numpy.abs import AbsOutput
from rbnics.utils.decorators import backend_for


# max function to compute the maximum absolute value of entries in EIM. To be used in combination with abs,
# even though abs actually carries out both the max and the abs!
@backend_for("numpy", inputs=(AbsOutput, ))
def max(abs_output):
    return (abs_output.max_abs_return_value, abs_output.max_abs_return_location)
