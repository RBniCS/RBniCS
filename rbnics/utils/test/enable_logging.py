# Copyright (C) 2015-2019 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

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
