# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file test_main.py
#  @brief Main class for tests
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import sys
import timeit
from numpy import mean, zeros

class Timer(object):
    def __init__(self, test):
        self.timeit_timer = timeit.Timer(lambda: test.run())
        # Inspired by python's cpython/master/Lib/timeit.py,
        # which is called by python -mtimeit script,
        # determine number so that 0.2 <= total time < 2.0
        for i in range(1, 10):
            self.number = 10**i
            r = self.timeit_timer.timeit(self.number)
            if r >= 0.2:
                break
        
class TestBase(object):
    def __init__(self, N, test_id, test_subid=None):
        self.N = N
        self.test_id = test_id
        self.test_subid = test_subid
        self.timer = Timer(self)
    
    def run(self):
        pass
        
    def timeit(self):
        r = self.timer.timeit_timer.repeat(repeat=3, number=self.timer.number)
        return min(r) * 1e6 / self.timer.number # usec
        
    def average(self):
        all_outputs = zeros((3*self.timer.number))
        for i in range(len(all_outputs)):
            all_outputs[i] = self.run()
        return mean(all_outputs)
    
    def number_of_runs(self):
        return self.timer.number
    
def main():
    t = Timer(stmt="ttt += 1", setup="import test_v1_dot_v2; ttt = 0")
    # ... copying from python's cpython/master/Lib/timeit.py ...
    # determine number so that 0.2 <= total time < 2.0
    for i in range(1, 10):
        number = 10**i
        try:
            x = t.timeit(number)
        except:
            t.print_exc()
            return 1
        if x >= 0.2:
            break
    try:
        r = t.repeat(repeat=3, number=number)
    except:
        t.print_exc()
        return 1
    usec = min(r) * 1e6 / number
    print(usec)

if __name__ == "__main__":
    sys.exit(main())
