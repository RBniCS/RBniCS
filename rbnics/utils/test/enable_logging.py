# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import sys
import functools
from logging import basicConfig, NOTSET


def enable_logging(loggers_and_levels):
    basicConfig(stream=sys.stdout)

    def enable_logging_decorator(original_test):
        @functools.wraps(original_test)
        def decorated_test(*args, **kwargs):
            for (logger, level) in loggers_and_levels.items():
                logger.setLevel(level)
            original_test(*args, **kwargs)
            for logger in loggers_and_levels.keys():
                logger.setLevel(NOTSET)
        return decorated_test

    return enable_logging_decorator
