# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from setuptools import find_packages, setup

setup(name="RBniCS",
      description="Reduced order modelling in FEniCS",
      long_description="Reduced order modelling in FEniCS",
      author="Francesco Ballarin (and contributors)",
      author_email="francesco.ballarin@unicatt.it",
      version="0.1.dev1",
      license="GNU Library or Lesser General Public License (LGPL)",
      url="https://www.rbnicsproject.org",
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
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
          "pytest-flake8"
      ],
      zip_safe=False
      )
