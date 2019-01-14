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

FROM quay.io/fenicsproject/dev
MAINTAINER Francesco Ballarin <francesco.ballarin@sissa.it>

USER root
RUN apt-get -qq update && \
    apt-get -qq remove python3-pytest && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    pip3 -q install --upgrade cvxopt multipledispatch pylru pytest pytest-benchmark pytest-dependency pytest-flake8 pytest-html pytest-instafail pytest-xdist sympy toposort && \
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
    ln -s $FENICS_PREFIX/lib/python3.6/dist-packages/rbnics RBniCS/source && \
    mv /tmp/RBniCS/tests RBniCS/ && \
    mv /tmp/RBniCS/tutorials RBniCS
    
USER root
RUN rm -rf /tmp/RBniCS
