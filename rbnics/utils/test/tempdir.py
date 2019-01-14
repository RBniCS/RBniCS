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

try:
    import pytest
except ImportError:
    def tempdir(request):
        return NotImplemented
        
    def save_tempdir(request):
        return NotImplemented
        
    def load_tempdir(request):
        return NotImplemented
else:
    import os
    import shutil
    from collections import defaultdict
    from mpi4py.MPI import COMM_WORLD, SUM
    
    def _create_tempdir(request, mode=None):
        """
        Adapted from DOLFIN's dolfin_utils/test/fixtures.py.
        """
        
        # Get directory name of test_foo.py file
        testfile = request.module.__file__
        testfiledir = os.path.dirname(os.path.abspath(testfile))

        # Construct name test_foo_tempdir from name test_foo.py
        testfilename = os.path.basename(testfile)
        if hasattr(request.config, 'slaveinput'):
            outputname = testfilename.replace(".py", "_tempdir_{}".format(request.config.slaveinput['slaveid']))
        else:
            outputname = testfilename.replace(".py", "_tempdir")

        # Get function name test_something from test_foo.py
        function = request.function.__name__
        if mode == "save":
            function = function.replace("_save", "_io")
        elif mode == "load":
            function = function.replace("_load", "_io")

        # Join all of these to make a unique path for this test function
        basepath = os.path.join(testfiledir, outputname)
        path = os.path.join(basepath, function)

        # Add a sequence number to avoid collisions when tests are otherwise parameterized
        if COMM_WORLD.rank == 0:
            _create_tempdir._sequencenumber[path] += 1
            sequencenumber = _create_tempdir._sequencenumber[path]
            sequencenumber = COMM_WORLD.allreduce(sequencenumber, op=SUM)
        else:
            sequencenumber = COMM_WORLD.allreduce(0, op=SUM)
        path += "__" + str(sequencenumber)

        # Delete and re-create directory on root node
        if COMM_WORLD.rank == 0:
            # First time visiting this basepath, delete the old and create
            # a new if mode is not load
            if basepath not in _create_tempdir._basepaths:
                _create_tempdir._basepaths.add(basepath)
                if mode == "load":
                    assert os.path.exists(basepath)
                else:
                    if os.path.exists(basepath):
                        shutil.rmtree(basepath)
                # Make sure we have the base path test_foo_tempdir for
                # this test_foo.py file
                if not os.path.exists(basepath):
                    os.mkdir(basepath)

            # Delete path from old test run if mode is not load
            if mode == "load":
                assert os.path.exists(path)
            else:
                if os.path.exists(path):
                    shutil.rmtree(path)
            # Make sure we have the path for this test execution:
            # e.g. test_foo_tempdir/test_something__3
            if not os.path.exists(path):
                os.mkdir(path)
        COMM_WORLD.barrier()

        return path
        
    _create_tempdir._sequencenumber = defaultdict(int)
    _create_tempdir._basepaths = set()
        
    @pytest.fixture(scope="function")
    def tempdir(request):
        return _create_tempdir(request)
        
    @pytest.fixture(scope="function")
    def save_tempdir(request):
        return _create_tempdir(request, mode="save")

    @pytest.fixture(scope="function")
    def load_tempdir(request):
        return _create_tempdir(request, mode="load")
