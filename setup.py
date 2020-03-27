# Copyright (C) 2015-2020 by the RBniCS authors
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
from setuptools import find_packages, setup
from setuptools.command.egg_info import egg_info as setuptools_egg_info
from setuptools.command.install import install as setuptools_install
import shutil
import subprocess
import tempfile

def AdditionalBackendsOptions(SetuptoolsClass):
    class AdditionalBackendsOptions_Class(SetuptoolsClass):
        user_options = SetuptoolsClass.user_options + [
            ('additional-backends=', None, 'Desired additional backends'),
            ('additional-backends-directory=', None, 'Additional backends directory')
        ]

        def initialize_options(self):
            SetuptoolsClass.initialize_options(self)
            self.additional_backends = None
            self.additional_backends_directory = None
            self.additional_backends_directory_cloned = False

        def finalize_options(self):
            if self.additional_backends is not None:
                self.additional_backends = set(self.additional_backends.split())
                if len(self.additional_backends) > 0 and self.additional_backends_directory is None:
                    self.additional_backends_directory = tempfile.mkdtemp()
                    for additional_backend in self.additional_backends:
                        additional_backend = additional_backend.replace("online/", "")
                        subprocess.check_call(["git", "clone", "git@gitlab.com:RBniCS-backends/" + additional_backend + ".git"], cwd=self.additional_backends_directory)
                    self.additional_backends_directory_cloned = True
            SetuptoolsClass.finalize_options(self)
    return AdditionalBackendsOptions_Class

@AdditionalBackendsOptions
class install(setuptools_install):
    def run(self):
        egg_info = self.get_finalized_command("egg_info")
        egg_info.additional_backends = self.additional_backends
        egg_info.additional_backends_directory = self.additional_backends_directory
        setuptools_install.do_egg_install(self)
        if self.additional_backends_directory_cloned:
            shutil.rmtree(self.additional_backends_directory)
        if self.additional_backends is not None:
            for symlink in egg_info.additional_backends_symlinks:
                os.unlink(symlink)

@AdditionalBackendsOptions
class egg_info(setuptools_egg_info):
    def run(self):
        if self.additional_backends is not None:
            self.additional_backends_symlinks = list()
            for additional_backend in self.additional_backends:
                src = os.path.join(self.additional_backends_directory, additional_backend, additional_backend).replace("online/", "")
                dst = os.path.join("rbnics", "backends", additional_backend)
                if not os.path.islink(dst):
                    os.symlink(src, dst)
                    self.additional_backends_symlinks.append(dst)
                additional_module = additional_backend.replace("/", ".")
                self.distribution.packages.append("rbnics.backends." + additional_module)
                self.distribution.packages.append("rbnics.backends." + additional_module + ".wrapping")
        setuptools_egg_info.run(self)

setup(name="RBniCS",
      description="Reduced order modelling in FEniCS",
      long_description="Reduced order modelling in FEniCS",
      author="Francesco Ballarin (and contributors)",
      author_email="francesco.ballarin@sissa.it",
      version="0.1.dev1",
      license="GNU Library or Lesser General Public License (LGPL)",
      url="http://mathlab.sissa.it/rbnics",
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules",
      ],
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          "cvxopt>=1.2.0",
          "mpi4py",
          "multipledispatch>=0.5.0",
          "pylru",
          "pytest-runner",
          "sympy>=1.0",
          "toposort"
      ],
      tests_require=[
          "pytest",
          "pytest-benchmark",
          "pytest-dependency",
          "pytest-flake8",
          "pytest-html",
          "pytest-instafail",
          "pytest-xdist"
      ],
      zip_safe=False,
      cmdclass={
          "egg_info": egg_info,
          "install": install
      }
      )
