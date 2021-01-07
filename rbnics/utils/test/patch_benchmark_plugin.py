# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import sys
import gc
import time
from math import ceil
import matplotlib.pyplot as plt
from rbnics.utils.io import Timer


def patch_benchmark_plugin(benchmark_plugin):
    from pytest_benchmark.fixture import BenchmarkFixture as OriginalBenchmarkFixture
    from pytest_benchmark.session import BenchmarkSession as OriginalBenchmarkSession
    from pytest_benchmark.timers import compute_timer_precision as original_compute_timer_precision
    from pytest_benchmark.utils import format_time

    class BenchmarkFixture(OriginalBenchmarkFixture):
        def __init__(self, *args, **kwargs):
            OriginalBenchmarkFixture.__init__(self, *args, **kwargs)
            # Customize self._timer
            self._timer = Timer("parallel")

        @classmethod
        def _get_precision(cls, timer):
            try:
                return cls._precisions[timer]
            except KeyError:
                cls._precisions[timer] = compute_timer_precision(timer)
                return cls._precisions[timer]

        def _raw(self, function_to_benchmark, *args, **kwargs):
            """
            This is a customization of the original implementation so that:
              * a setup kwarg is necessary;
              * an optional teardown kwarg is allowed;
              * calibration is done with respect to setup + function + teardown, since in our case also setup
                may be expensive.
            """

            assert len(args) == 0   # arguments will be provided by the setup function

            assert "setup" in kwargs
            setup = kwargs.pop("setup")
            teardown = kwargs.pop("teardown", _do_nothing)

            assert len(kwargs) == 0  # no kwargs allowed, except setup and teardown

            if not self.disabled:

                # Choose how many time we must repeat the test, basing the timing on the setup + function + teardown
                def calibration_setup():
                    return ()

                def calibration_function():
                    args = setup()
                    result = function_to_benchmark(*args)
                    teardown(*args, result)

                def calibration_teardown(result):
                    pass

                setup_function_teardown_runner = self._make_runner(
                    calibration_setup, calibration_function, calibration_teardown, (), {})
                duration, iterations, loops_range = self._calibrate_timer(setup_function_teardown_runner)
                rounds = int(ceil(self._max_time / duration))
                rounds = max(rounds, self._min_rounds)
                rounds = min(rounds, sys.maxsize)
                stats = self._make_stats(iterations)

                # Create a runner on the function to benchmark
                runner = self._make_runner(setup, function_to_benchmark, teardown, args, kwargs)
                self._logger.debug(
                    "  Running %s rounds x %s iterations ..." % (rounds, iterations), yellow=True, bold=True)
                run_start = time.time()
                if self._warmup:
                    warmup_rounds = min(rounds, max(1, int(self._warmup / iterations)))
                    self._logger.debug("  Warmup %s rounds x %s iterations ..." % (warmup_rounds, iterations))
                    for _ in range(warmup_rounds):
                        runner(loops_range)
                for _ in range(rounds):
                    stats.update(runner(loops_range))
                self._logger.debug("  Ran for %ss." % format_time(time.time() - run_start), yellow=True, bold=True)
            else:
                args = setup()
                result = function_to_benchmark(*args)
                teardown(*args, result)

        def _make_runner(self, setup, function_to_benchmark, teardown, args_, kwargs_):
            assert len(args_) == 0    # arguments will be provided
            assert len(kwargs_) == 0  # by the setup function

            def runner(loops_range, timer=self._timer):
                gc_enabled = gc.isenabled()
                if self._disable_gc:
                    gc.disable()
                tracer = sys.gettrace()
                sys.settrace(None)
                try:
                    assert loops_range
                    # Warmup
                    warmup_args = setup()
                    warmup_result = function_to_benchmark(*warmup_args)
                    teardown(*warmup_args, warmup_result)
                    # Call setup function
                    args = list()
                    for _ in loops_range:
                        args.append(setup())
                    # Start benchamrk
                    results = list()
                    timer.start()
                    for i in loops_range:
                        results.append(function_to_benchmark(*args[i]))
                    elapsed = timer.stop()
                    # Call teardown function
                    for i in loops_range:
                        teardown(*args[i], results[i])
                    # Return
                    return elapsed
                finally:
                    sys.settrace(tracer)
                    if gc_enabled:
                        gc.enable()

            return runner

    class BenchmarkSession(OriginalBenchmarkSession):
        def display(self, tr):
            OriginalBenchmarkSession.display(self, tr)
            # Speedup/overhead computation
            datetimes = list()  # over runs
            speedups_tmp = dict()  # from (test name, test type, arguments) to list (over runs) of speedups
            for (idx, run) in enumerate(list(self.storage.load())[:-8:-1]):
                datetimes.append(time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(time.mktime(
                        time.strptime(run[1]["datetime"], "%Y-%m-%dT%H:%M:%S.%f")))))
                # Convert benchmarks to a dict over name
                benchmarks = dict()
                for benchmark in run[1]["benchmarks"]:
                    name_with_parametrized = benchmark["name"]
                    assert name_with_parametrized not in benchmarks
                    benchmarks[name_with_parametrized] = benchmark
                # Get mean time for each benchmark
                for (_, benchmark) in benchmarks.items():
                    name = benchmark["name"].split("[")[0]
                    params_str = benchmark["param"]
                    params_dict = benchmark["params"]
                    params_dict_without_test_type = dict(benchmark["params"])
                    params_dict_without_test_type.pop("test_type")
                    # Skip test_type set to builtin
                    if params_dict["test_type"] == "builtin":
                        continue
                    else:
                        builtin_name_with_parametrized = name + "[" + params_str.replace(
                            params_dict["test_type"], "builtin") + "]"
                        builtin_benchmark = benchmarks[builtin_name_with_parametrized]
                        backend_mean_time = benchmark["stats"]["mean"]
                        builtin_mean_time = builtin_benchmark["stats"]["mean"]
                        key = (name, params_dict["test_type"], tuple(sorted(params_dict_without_test_type.items())))
                        if key not in speedups_tmp:
                            assert idx == 0
                            speedups_tmp[key] = list()
                        speedups_tmp[key].append(builtin_mean_time / backend_mean_time)
            # Convert speedups to overheads if the computed number is less than 1, and
            # change the indexing to collect tests with the same name
            speedups = dict()   # from (test_name, test type) to a dict from arguments to
            overheads = dict()  # a list (over runs) of speedups/overheads
            for (key, values) in speedups_tmp.items():
                external_key = key[0:2]
                internal_key = key[2]
                if all([value < 1. for value in values]):
                    if external_key not in overheads:
                        overheads[external_key] = dict()
                    overheads[external_key][internal_key] = [1. / value for value in values]
                else:
                    if external_key not in speedups:
                        speedups[external_key] = dict()
                    speedups[external_key][internal_key] = values
            # Prepare a plot
            storage_dir = self.config.getoption("overhead_speedup_storage")
            for (dict_, ylabel) in ((speedups, "speedup"), (overheads, "overhead")):
                for (external_key, internal_dict) in dict_.items():
                    fig = plt.figure()
                    plt_args = list()
                    plt_legends = list()
                    for (internal_key, values) in internal_dict.items():
                        plt_args.append(list(range(len(values))))
                        plt_args.append(values)
                        plt_legends.append(", ".join(["=".join(str(kk) for kk in k) for k in internal_key]))
                    plt.plot(*plt_args, marker="o")
                    ax = fig.gca()
                    ax.set_title(external_key[0] + ", backend: " + external_key[1])
                    ax.set_xticks([i for i in range(len(datetimes))])
                    ax.set_xticklabels(datetimes, rotation="vertical")
                    ax.get_xticklabels()[0].set_color("blue")
                    ax.set_ylabel(ylabel)
                    legend = plt.legend(plt_legends, loc="center left", bbox_to_anchor=(1, 0.5))
                    plt_filename = os.path.join(storage_dir, "_".join(external_key) + "_" + str(int(
                        time.mktime(time.strptime(datetimes[0], "%Y-%m-%d %H:%M")))) + ".png")
                    plt.savefig(plt_filename, bbox_extra_artists=(legend,), bbox_inches="tight")

    # Auxiliary do nothing function
    def _do_nothing(*args):
        pass

    # Auxiliary timer precision computation
    def compute_timer_precision(timer):
        if isinstance(timer, Timer):
            precision = None
            for points in range(5):
                dt = 0.
                for _ in range(10):
                    timer.start()
                    dt_ = timer.stop()
                    assert dt_ >= 0.
                    dt += dt_
                if precision is not None:
                    precision = min(precision, dt)
                else:
                    precision = dt
            return precision
        else:
            return original_compute_timer_precision(timer)

    benchmark_plugin.BenchmarkFixture = BenchmarkFixture
    benchmark_plugin.BenchmarkSession = BenchmarkSession
