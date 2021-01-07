# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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
    config.write(os.path.join(tempdir, ".rbnicsrc"))

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
