## ![RBniCS - reduced order modelling in FEniCS](https://gitlab.com/RBniCS/RBniCS/raw/master/doc/rbnics-logo-small.png "RBniCS - reduced order modelling in FEniCS") RBniCS - reduced order modelling in FEniCS ##

### 1. Classes hierarchy
                                     --------------> parametrized_problem.py <----------
                                     |                                   ^             |
                                     |                                   |             |
                                     |                                   |             |
                     ----> elliptic_coercive_base.py <------          eim.py         scm.py
                     |                            ^         |
                     |                            |         |
                     |                            |         |
               --->elliptic_coercive_rb_base.py   |    elliptic_coercive_pod_base.py
               |                |                 |                          |
               |                |                 |                          |
               |                |                 |                          |
               |                |                 |                          |
      elliptic_coercive_rb_     |            parabolic_coercive_base.py      |
      non_compliant_base.py     |               ^                ^           |
                                |               |                |           |
                                |               |                |           |
                         parabolic_coercive_rb_base.py       parabolic_coercive_pod_base.py

### 2. Classes description
RBniCS provides the following classes:
1. **parametrized_problem.py**: base class describing an offline/online decomposition of parametrized problems
2. **elliptic_coercive_base.py**: base class for projection based reduced order models of elliptic coervice problems
3. **elliptic_coercive_rb_base.py**: class for reduced basis method of (compliant) elliptic coervice problems
4. **elliptic_coercive_pod_base.py**: class for POD-Galerkin ROMs of elliptic coervice problems
5. **elliptic_coercive_rb_non_compliant_base.py**: class for reduced basis method of non-compliant elliptic coervice problems
6. **parabolic_coercive_base.py**: base class for projection based reduced order models of parabolic coervice problems
7. **parabolic_coercive_rb_base.py**: class for reduced basis method of parabolic coervice problems
8. **parabolic_coercive_pod_base.py**: class for POD-Galerkin ROMs of parabolic coervice problems
9. **eim.py**: class for empirical interpolation method (interpolation of parametrized functions)
10. **scm.py**: class for successive constraints method (approximation of the coercivity constant)
11. **gram_schmidt.py**: auxiliary class containing the implementation of the Gram Schmidt procedure
12. **proper_orthogonal_decomposition.py**: auxiliary class containing the implementation of the POD
