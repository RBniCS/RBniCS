# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Force import of reduction methods package, so that @ReductionMethodDecoratorFor
# decorators are processed
import rbnics.scm.reduction_methods  # noqa: F401
