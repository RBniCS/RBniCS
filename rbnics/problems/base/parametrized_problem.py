# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.io import Folders


class ParametrizedProblem(object):
    """This is the base class, which is inherited by all other
    classes. It defines the base interface with variables and
    functions that the derived classes have to set and/or
    overwrite.
    Current parameter mu and its range are initialized.

    :param folder_prefix: prefix for the folder name.
    """

    def __init__(self, folder_prefix):

        # Current parameters value
        self.mu = tuple()  # tuple of real numbers
        # Parameter ranges
        self.mu_range = list()  # list of (min, max) pairs, such that len(self.mu) == len(self.mu_range)
        #
        self.folder_prefix = folder_prefix
        self.folder = Folders()

    def set_mu_range(self, mu_range):
        """
        Set the range of the parameters.

        :param mu_range: the range into which the parameter changes.
        :type mu_range: list of (min, max) pairs, such that len(self.mu) == len(self.mu_range)
        """
        self.mu_range = mu_range
        # Initialize mu so that it has the correct length
        self.set_mu(tuple([r[0] for r in self.mu_range]))

    def set_mu(self, mu):
        """
        Set the current value of the parameter

        :param mu: the value of the current parameter.
        :type mu: tuple of real numbers
        """
        assert len(mu) == len(self.mu_range), "mu and mu_range must have the same length"
        self.mu = mu
