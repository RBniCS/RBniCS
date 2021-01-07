# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numpy import arange, isclose
from petsc4py import PETSc


def BasicPETScTSIntegrator(backend, wrapping):

    class _BasicPETScTSIntegrator(object):
        def __init__(self, problem, solution, solution_dot):
            self.solution = solution
            self.solution_dot = solution_dot
            # Create PETSc's TS object
            self.ts = PETSc.TS().create(wrapping.get_mpi_comm(solution))
            # ... and associate residual and jacobian
            self.ts.setIFunction(problem.residual_vector_eval, wrapping.to_petsc4py(problem.residual_vector))
            self.ts.setIJacobian(problem.jacobian_matrix_eval, wrapping.to_petsc4py(problem.jacobian_matrix))
            self.monitor = _Monitor(self.ts)
            self.ts.setMonitor(self.monitor)
            # Set sensible default values to parameters
            default_parameters = {
                "exact_final_time": "stepover",
                "integrator_type": "beuler",
                "problem_type": "linear",
                "report": True
            }
            self.set_parameters(default_parameters)

        def set_parameters(self, parameters):
            for (key, value) in parameters.items():
                if key == "exact_final_time":
                    self.ts.setExactFinalTime(getattr(self.ts.ExactFinalTime, value.upper()))
                elif key == "final_time":
                    self.ts.setMaxTime(value)
                elif key == "initial_time":
                    self.ts.setTime(value)
                elif key == "integrator_type":
                    self.ts.setType(getattr(self.ts.Type, value.upper()))
                elif key == "linear_solver":
                    snes = self.ts.getSNES()
                    ksp = snes.getKSP()
                    ksp.setType("preonly")
                    ksp.getPC().setType("lu")
                    if value == "default":
                        value = wrapping.get_default_linear_solver()
                    if hasattr(ksp.getPC(), "setFactorSolverType"):  # PETSc >= 3.9
                        ksp.getPC().setFactorSolverType(value)
                    else:
                        ksp.getPC().setFactorSolverPackage(value)
                elif key == "max_time_steps":
                    self.ts.setMaxSteps(value)
                elif key == "monitor":
                    assert isinstance(value, dict)
                    assert all(key_monitor in ("initial_time", "time_step_size") for key_monitor in value)
                    if "initial_time" in value:
                        self.monitor.monitor_t0 = value["initial_time"]
                    if "time_step_size" in value:
                        self.monitor.monitor_dt = value["time_step_size"]
                elif key == "problem_type":
                    assert value in ("linear", "nonlinear")
                    self.ts.setProblemType(getattr(self.ts.ProblemType, value.upper()))
                    snes = self.ts.getSNES()
                    if value == "linear":
                        snes.setType("ksponly")
                    elif value == "nonlinear":
                        snes.setType("newtonls")
                    else:
                        raise ValueError("Invalid paramater passed as problem type.")
                elif key == "report":
                    if value is True:
                        def print_time(ts):
                            t = ts.getTime()
                            dt = ts.getTimeStep()
                            print("# t = {0:g}".format(t + dt))
                        self.ts.setPreStep(print_time)
                    else:
                        def do_nothing(ts):
                            pass
                        self.ts.setPreStep(do_nothing)
                    self._report = value
                elif key == "snes_solver":
                    snes_tolerances = [1.e-10, 1.e-9, 1.e-16, 50]
                    for (key_snes, value_snes) in value.items():
                        snes = self.ts.getSNES()
                        if key_snes == "absolute_tolerance":
                            snes_tolerances[0] = value_snes
                        elif key_snes == "linear_solver":
                            ksp = snes.getKSP()
                            ksp.setType("preonly")
                            ksp.getPC().setType("lu")
                            if value == "default":
                                value = wrapping.get_default_linear_solver()
                            if hasattr(ksp.getPC(), "setFactorSolverType"):  # PETSc >= 3.9
                                ksp.getPC().setFactorSolverType(value_snes)
                            else:
                                ksp.getPC().setFactorSolverPackage(value_snes)
                        elif key_snes == "line_search":
                            raise ValueError("Line search is not wrapped yet by petsc4py")
                        elif key_snes == "maximum_iterations":
                            snes_tolerances[3] = value_snes
                        elif key_snes == "method":
                            snes.setType(value_snes)
                        elif key_snes == "relative_tolerance":
                            snes_tolerances[1] = value_snes
                        elif key_snes == "report":
                            snes.cancelMonitor()

                            def monitor(snes, it, fgnorm):
                                print("  " + str(it) + " SNES Function norm " + "{:e}".format(fgnorm))

                            snes.setMonitor(monitor)
                        elif key_snes == "solution_tolerance":
                            snes_tolerances[2] = value_snes
                        else:
                            raise ValueError("Invalid paramater passed to PETSc SNES object.")
                    snes.setTolerances(*snes_tolerances)
                elif key == "time_step_size":
                    self.ts.setTimeStep(value)
                else:
                    raise ValueError("Invalid paramater passed to PETSc TS object.")
            # Finally, read in additional options from the command line
            self.ts.setFromOptions()

        def solve(self):
            # Assert consistency of final time and time step size
            t0, dt, T = self.ts.getTime(), self.ts.getTimeStep(), self.ts.getMaxTime()
            final_time_consistency = (T - t0) / dt
            assert isclose(round(final_time_consistency), final_time_consistency), (
                "Final time should be occuring after an integer number of time steps")
            # Init monitor
            self.monitor.init(self.solution, self.solution_dot)
            # Create copy to avoid possible internal storage overwriting by linesearch
            solution_copy = wrapping.function_copy(self.solution)
            petsc_solution_copy = wrapping.to_petsc4py(solution_copy)
            # Solve
            self.ts.solve(petsc_solution_copy)
            text_output = "Total time steps %d (%d rejected, %d SNES fails)" % (
                self.ts.getStepNumber(), self.ts.getStepRejections(), self.ts.getSNESFailures())
            if self.ts.getProblemType() == self.ts.ProblemType.NONLINEAR:
                text_output += ", with total %d nonlinear iterations" % (self.ts.getSNESIterations(), )
            if self._report:
                print(text_output)
            # Evaluate solution and solution at the final time. Note that the value store in solution_copy might
            # not be correct if TS has stepped over the final time.
            self.monitor._evaluate_solution(T, self.solution)
            self.monitor._evaluate_solution_dot(T, self.solution_dot)

    class _Monitor(object):
        def __init__(self, ts):
            self.ts = ts
            self.t0 = None
            self.dt = None
            self.T = None
            self.monitor_solution = None
            self.monitor_solution_prev = None
            self.monitor_solution_dot = None
            self.monitor_eps = None
            self.monitor_t0 = None
            self.monitor_dt = None
            self.monitor_t = None
            self.monitor_T = None
            self.monitor_callback = None

        def init(self, solution, solution_dot):
            if self.monitor_callback is not None:
                self.monitor_solution = wrapping.function_copy(solution)
                self.monitor_solution_prev = wrapping.function_copy(solution)
                self.monitor_solution_dot = wrapping.function_copy(solution_dot)
                self.t0, self.dt, self.T = self.ts.getTime(), self.ts.getTimeStep(), self.ts.getMaxTime()
                if self.monitor_t0 is None:
                    self.monitor_t0 = self.t0
                monitor_t0_consistency = (self.monitor_t0 - self.t0) / self.dt
                assert isclose(round(monitor_t0_consistency), monitor_t0_consistency), (
                    "Monitor initial time should be occuring after an integer number of time steps")
                self.monitor_t = self.monitor_t0
                self.monitor_eps = 0.1 * self.dt
                if self.monitor_dt is None:
                    self.monitor_dt = self.dt
                monitor_dt_consistency = self.monitor_dt / self.dt
                assert isclose(round(monitor_dt_consistency), monitor_dt_consistency), (
                    "Monitor time step size should be a multiple of the time step size")
                assert self.monitor_T is None
                self.monitor_T = self.T
                monitor_T_consistency = (self.monitor_T - self.t0) / self.dt
                assert isclose(round(monitor_T_consistency), monitor_T_consistency), (
                    "Monitor initial time should be occuring after an integer number of time steps")

        def __call__(self, ts, step, time, solution):
            """
               TSMonitorSet - Sets an ADDITIONAL function that is to be used at every
               timestep to display the iteration's  progress.

               Logically Collective on TS

               Input Parameters:
                    +  ts - the TS context obtained from TSCreate()
                    .  monitor - monitoring routine
                    .  mctx - [optional] user-defined context for private data for the
                                 monitor routine (use NULL if no context is desired)
                    -  monitordestroy - [optional] routine that frees monitor context
                              (may be NULL)

               Calling sequence of monitor:
                    $    int monitor(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx)

                    +    ts - the TS context
                    .    steps - iteration number (after the final time step the monitor routine
                                 may be called with a step of -1, this indicates the solution has
                                 been interpolated to this time)
                    .    time - current time
                    .    u - current iterate
                    -    mctx - [optional] monitoring context
            """

            if self.monitor_callback is not None:
                monitor_times = arange(self.monitor_t, min(self.monitor_T, time) + self.monitor_eps,
                                       self.monitor_dt).tolist()
                assert all(monitor_t <= time or isclose(monitor_t, time, self.monitor_eps)
                           for monitor_t in monitor_times)
                monitor_times = [min(monitor_t, time) for monitor_t in monitor_times]
                for self.monitor_t in monitor_times:
                    self._evaluate_solution(self.monitor_t, self.monitor_solution)
                    self._evaluate_solution_dot(self.monitor_t, self.monitor_solution_dot)
                    # Apply monitor
                    self.monitor_callback(self.monitor_t, self.monitor_solution, self.monitor_solution_dot)
                # Prepare for next time step
                if len(monitor_times) > 0:
                    self.monitor_t += self.monitor_dt

        def _evaluate_solution(self, t, result):
            assert t >= self.t0 or isclose(t, self.t0, atol=self.monitor_eps)
            assert t <= self.T or isclose(t, self.T, atol=self.monitor_eps)
            if isclose(t, self.t0, atol=self.monitor_eps):  # t = t0
                pass  # assuming that result already contains the initial solution
            else:
                result_petsc = wrapping.to_petsc4py(result)
                self.ts.interpolate(t, result_petsc)
                result_petsc.assemble()
                result_petsc.ghostUpdate()

        def _evaluate_solution_dot(self, t, result):
            assert t >= self.t0 or isclose(t, self.t0, atol=self.monitor_eps)
            assert t <= self.T or isclose(t, self.T, atol=self.monitor_eps)
            if isclose(t, self.t0, atol=self.monitor_eps):  # t = t0
                pass  # assuming that result already contains the initial solution
            else:
                # There is no equivalent TSInterpolate for solution dot, so we approximate it
                # by a first order time discretization
                current_dt = self.ts.getTimeStep()  # might be different from self.dt with adaptivity
                current_dt *= 0.5  # make sure to be in the latest time interval
                if t - current_dt >= self.ts.getTime() - self.ts.getTimeStep():  # use backward finite difference
                    self._evaluate_solution(t, result)
                    self._evaluate_solution(t - current_dt, self.monitor_solution_prev)
                else:  # use forward finite difference
                    assert isclose(t, self.T, atol=self.monitor_eps), (
                        "This case should only happen when TS steps over final time")
                    assert t + current_dt <= self.ts.getTime()
                    bak_monitor_eps = self.monitor_eps
                    self.monitor_eps = 2 * current_dt  # disable assert inside self._evaluate_solution
                    self._evaluate_solution(t + current_dt, result)
                    self.monitor_eps = bak_monitor_eps
                    self._evaluate_solution(t, self.monitor_solution_prev)
                result_petsc = wrapping.to_petsc4py(result)
                result_petsc -= wrapping.to_petsc4py(self.monitor_solution_prev)
                result_petsc /= current_dt
                result_petsc.assemble()
                result_petsc.ghostUpdate()

    return _BasicPETScTSIntegrator
