# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
from rbnics.backends import assign, copy, evaluate
from rbnics.eim.problems.eim_approximation import EIMApproximation
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import sync_setters


def set_mu_decorator(set_mu):
    def decorated_set_mu(self, mu):
        assert isinstance(mu, (EnlargedMu, tuple))
        if isinstance(mu, tuple):
            set_mu(self, mu)
        elif isinstance(mu, EnlargedMu):
            assert len(mu) == 2
            assert "mu" in mu
            assert isinstance(mu["mu"], tuple)
            set_mu(self, mu["mu"])
            assert "t" in mu
            assert isinstance(mu["t"], Number)
            self.set_time(mu["t"])
        else:
            raise ValueError("Invalid mu")

    return decorated_set_mu


class TimeDependentEIMApproximation(EIMApproximation):

    @sync_setters("truth_problem", "set_mu", "mu", set_mu_decorator)
    @sync_setters("truth_problem", "set_time", "t")
    @sync_setters("truth_problem", "set_initial_time", "t0")
    @sync_setters("truth_problem", "set_time_step_size", "dt")
    @sync_setters("truth_problem", "set_final_time", "T")
    def __init__(self, truth_problem, parametrized_expression, folder_prefix, basis_generation):
        # Call the parent initialization
        EIMApproximation.__init__(self, truth_problem, parametrized_expression, folder_prefix, basis_generation)

        # Store quantities related to the time discretization
        self.t0 = 0.
        self.t = 0.
        self.dt = None
        self.T = None

        # I/O
        def _snapshot_cache_key_generator(*args, **kwargs):
            assert len(args) == 2
            assert args[0] == self.mu
            assert args[1] == self.t
            assert len(kwargs) == 0
            return self._cache_key()

        def _snapshot_cache_import(filename):
            snapshot = copy(self.snapshot)
            self.import_solution(self.folder["cache"], filename, snapshot)
            return snapshot

        def _snapshot_cache_export(filename):
            self.export_solution(self.folder["cache"], filename)

        def _snapshot_cache_filename_generator(*args, **kwargs):
            assert len(args) == 2
            assert args[0] == self.mu
            assert args[1] == self.t
            assert len(kwargs) == 0
            return self._cache_file()

        self._snapshot_cache = Cache(
            "EIM",
            key_generator=_snapshot_cache_key_generator,
            import_=_snapshot_cache_import,
            export=_snapshot_cache_export,
            filename_generator=_snapshot_cache_filename_generator
        )

    # Set initial time
    def set_initial_time(self, t0):
        assert isinstance(t0, Number)
        self.t0 = t0

    # Set current time
    def set_time(self, t):
        assert isinstance(t, Number)
        self.t = t

    # Set time step size
    def set_time_step_size(self, dt):
        assert isinstance(dt, Number)
        self.dt = dt

    # Set final time
    def set_final_time(self, T):
        assert isinstance(T, Number)
        self.T = T

    def evaluate_parametrized_expression(self):
        try:
            assign(self.snapshot, self._snapshot_cache[self.mu, self.t])
        except KeyError:
            self.snapshot = evaluate(self.parametrized_expression)
            self._snapshot_cache[self.mu, self.t] = copy(self.snapshot)

    def _cache_key(self):
        return (self.mu, self.t)


class EnlargedMu(dict):
    def __str__(self):
        assert len(self) == 2
        assert "mu" in self
        assert isinstance(self["mu"], tuple)
        assert "t" in self
        assert isinstance(self["t"], Number)
        output = str(self["mu"]) + " and t = " + str(self["t"])
        return output
