# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

FROM quay.io/fenicsproject/dev
MAINTAINER Francesco Ballarin <francesco.ballarin@sissa.it>

USER root
RUN apt-get -qq update && \
    apt-get -qq remove python3-pytest && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    pip3 -q install --upgrade cvxopt multipledispatch pylru pytest pytest-benchmark pytest-dependency pytest-flake8 pytest-gc pytest-html pytest-instafail pytest-xdist sympy toposort && \
    sed -i "s/pytest_report_header/DISABLED_pytest_report_header/g" /usr/local/lib/python3.6/dist-packages/pytest_metadata/plugin.py && \
    cat /dev/null > $FENICS_HOME/WELCOME

USER fenics
COPY --chown=fenics . /tmp/RBniCS

USER root
WORKDIR /tmp/RBniCS
RUN python3 setup.py -q install

USER fenics
WORKDIR $FENICS_HOME
RUN mkdir RBniCS && \
    ln -s $FENICS_PREFIX/lib/python3.6/dist-packages/RBniCS*egg/rbnics RBniCS/source && \
    mv /tmp/RBniCS/tests RBniCS/ && \
    mv /tmp/RBniCS/tutorials RBniCS

USER root
RUN rm -rf /tmp/RBniCS
