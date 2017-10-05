# Copyright (C) 2015-2017 by the RBniCS authors
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

from setuptools import find_packages, setup

setup(name="RBniCS",
      description="Reduced order modelling in FEniCS",
      long_description="Reduced order modelling in FEniCS",
      author="Francesco Ballarin, Gianluigi Rozza, Alberto Sartori (and contributors)",
      author_email="francesco.ballarin@sissa.it, gianluigi.rozza@sissa.it, alberto.sartori@sissa.it",
      version="0.0.dev2",
      license="GNU Library or Lesser General Public License (LGPL)",
      url="http://mathlab.sissa.it/rbnics",
      classifiers=[
          "Development Status :: 3 - Alpha"
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
          "cvxopt",
          "mpi4py",
          "multipledispatch",
          "pytest-runner",
          "sympy",
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
      zip_safe=False
      )
