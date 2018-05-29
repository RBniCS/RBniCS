# Copyright (C) 2015-2018 by the RBniCS authors
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

import os
import sys
from rbnics.utils.config import Config

def test_config(tempdir):
    # Create a default configuration
    config = Config()

    # Write config to stdout
    print("===============")
    config.write(sys.stdout)
    print("===============")

    # Change options
    config.set("backends", "online backend", "online")
    config.set("problems", "cache", {"disk"})

    # Write config to stdout
    print("===============")
    config.write(sys.stdout)
    print("===============")

    # Write config to file
    with open(os.path.join(tempdir, ".rbnicsrc"), "w") as configfile:
        config.write(configfile)
        
    # Check that file has been written
    assert os.path.isfile(os.path.join(tempdir, ".rbnicsrc"))
    
    # Read back in
    config2 = Config()
    config2.read(tempdir)
    
    # Write config2 to stdout
    print("===============")
    config2.write(sys.stdout)
    print("===============")

    # Check that read was successful
    assert config == config2
