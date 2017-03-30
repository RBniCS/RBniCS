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
## @file numpy_io.py
#  @brief I/O helper functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.decorators.store_map_from_problem_to_reduced_problem import _problem_to_reduced_problem_map
from RBniCS.utils.factories import ReducedProblemFactory

def regenerate_reduced_problem_from_exact_reduced_problem(truth_problem, reduction_method, exact_reduced_problem):
    # Initialize the affine expansion in the truth problem
    truth_problem.init()
    # Initialize reduced order data structures in the reduced problem
    assert exact_reduced_problem == _problem_to_reduced_problem_map[truth_problem]
    del _problem_to_reduced_problem_map[truth_problem]
    reduced_problem = ReducedProblemFactory(truth_problem, reduction_method, **reduction_method._init_kwargs)
    reduced_problem._error_computation_override__default_with_respect_to = exact_reduced_problem.truth_problem
    # Copy the basis functions from the exact reduced problem
    reduced_problem.Z = exact_reduced_problem.Z
    reduced_problem.N = exact_reduced_problem.N
    reduced_problem.N_bc = exact_reduced_problem.N_bc
    # Re-assemble, if necessary, operators
    assert "reduced_operators" in reduced_problem.folder
    if reduced_problem.folder["reduced_operators"].create(): # precomputation should be carried out now
        reduced_problem._init_operators("offline")
        reduced_problem.build_reduced_operators()
    else: # load from file
        reduced_problem._init_operators("online")
    # Re-assemble, if necessary, error estimation operators
    if "error_estimation" in reduced_problem.folder:
        if reduced_problem.folder["error_estimation"].create(): # precomputation should be carried out now
           reduced_problem._init_error_estimation_operators("offline")
           reduced_problem.build_error_estimation_operators()
        else: # load from file
           reduced_problem._init_error_estimation_operators("online")
    return reduced_problem
        
