# Copyright (C) 2015-2022 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import matplotlib
import matplotlib.pyplot as plt
matplotlib_backend = matplotlib.get_backend()


def disable_matplotlib():
    plt.switch_backend("agg")


def enable_matplotlib():
    plt.switch_backend(matplotlib_backend)
    plt.close("all")  # do not trigger matplotlib max_open_warning
