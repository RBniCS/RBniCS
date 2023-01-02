# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.io import SpeedupAnalysisTable, OnlineSizeDict, TextBox, TextLine, Timer

def speedup_analysis_coanda(self, N_max, filename=None, **kwargs):

    print(TextBox(self.truth_problem.name() + " " + self.label + " speedup analysis begins", fill="="))
    print("")

    speedup_analysis_table = SpeedupAnalysisTable(self.testing_set)
    speedup_analysis_table.set_Nmax(N_max)
    speedup_analysis_table.add_column(
        "speedup_solve", group_name="speedup_solve", operations=("min", "mean", "max"))
    speedup_analysis_table.add_column(
        "speedup_output", group_name="speedup_output", operations=("min", "mean", "max"))

    truth_timer = Timer("parallel")
    reduced_timer = Timer("serial")

    for (n_int, n_arg) in zip(range(1, N_max+1),range(1, N_max+1)):
        print(TextLine(str(n_int), fill="#"))
        self.reduced_problem.flag = False
        self.truth_problem._solution_cache.clear()
        self.reduced_problem._solution_cache.clear()
        for (mu_index, mu) in enumerate(self.testing_set):
            print(TextLine(str(mu_index), fill="-"))

            self.reduced_problem.set_mu(mu)
            if n_int == 1:
                truth_timer.start()
                self.truth_problem.solve(**kwargs)
                elapsed_truth_solve = truth_timer.stop()

                truth_timer.start()
                self.truth_problem.compute_output()
                elapsed_truth_output = truth_timer.stop()

            reduced_timer.start()
            solution = self.reduced_problem.solve(n_arg, **kwargs)
            elapsed_reduced_solve = reduced_timer.stop()

            reduced_timer.start()
            output = self.reduced_problem.compute_output()
            elapsed_reduced_output = reduced_timer.stop()

            if solution is not NotImplemented:
                speedup_analysis_table[
                    "speedup_solve", n_int, mu_index] = elapsed_truth_solve / elapsed_reduced_solve
            else:
                speedup_analysis_table["speedup_solve", n_int, mu_index] = NotImplemented
            if output is not NotImplemented:
                speedup_analysis_table[
                    "speedup_output", n_int, mu_index] = (
                        elapsed_truth_solve + elapsed_truth_output) / (
                            elapsed_reduced_solve + elapsed_reduced_output)
            else:
                speedup_analysis_table["speedup_output", n_int, mu_index] = NotImplemented

    # Print
    print("")
    print(speedup_analysis_table)

    print("")
    print(TextBox(self.truth_problem.name() + " " + self.label + " speedup analysis ends", fill="="))
    print("")

    # Export speedup analysis table
    speedup_analysis_table.save(
        self.folder["speedup_analysis"], "speedup_analysis" if filename is None else filename)
