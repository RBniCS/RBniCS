## ![RBniCS - reduced order modelling in FEniCS](https://gitlab.com/RBniCS/RBniCS/raw/master/doc/rbnics-logo-small.png "RBniCS - reduced order modelling in FEniCS") RBniCS - reduced order modelling in FEniCS ##

### 0. Introduction
**RBniCS** is an implementation in **FEniCS** of several reduced order modelling techniques (and, in particular, certified reduced basis method and Proper Orthogonal Decomposition-Galerkin methods) for coercive problems. It is ideally suited for an introductory course on reduced basis methods and reduced order modelling, thanks to an object-oriented approach and an intuitive and versatile python interface. To this end, it has been employed in several doctoral courses on "Reduced Basis Methods for Computational Mechanics".

**RBniCS** can also be used as a basis for more advanced projects that would like to assess the capability of reduced order models in their existing **FEniCS**-based software, thanks to the availability of several reduced order methods (such as reduced basis and proper orthogonal decomposition) and algorithms (such as successive constraint method, empirical interpolation method) in the library.

Several tutorials are provided. This software is also a companion of the introductory reduced basis handbook: 

> [J. S. Hesthaven, G. Rozza, B. Stamm. **Certified Reduced Basis Methods for Parametrized Partial Differential Equations**. SpringerBriefs in Mathematics. Springer International Publishing, 2015](http://www.springer.com/us/book/9783319224695)

### 1. Prerequisites
**RBniCS** requires
* **FEniCS**, with petsc4py and slepc4py, for computations during the offline stage;
* **numpy** for computations during the online stage;
* **python-glpk**, a python interface to the glpk library for linear programming, necessary for the successive constraints method.

### 2. Installation and usage
Simply clone the **RBniCS** public repository:
```
git clone https://gitlab.com/RBniCS/RBniCS.git /RB/ni/CS/path
```
The core of the **RBniCS** code is in the **RBniCS** subfolder of the repository.

Then Make sure that both **FEniCS** and **RBniCS** are in your PYTHONPATH,
```
source /FE/ni/CS/path/share/fenics/fenics.conf
export PYTHONPATH="/RB/ni/CS/path:$PYTHONPATH"
```
and run a **RBniCS** python script (such as the tutorials) as follows:
```
python RBniCS_example.py
```

### 3. RBniCS virtual machine
If you want to try **RBniCS** out but do not have **FEniCS** already installed, a virtual machine (Ubuntu 14.04 LTS) with RBniCS (and all required dependencies) installed is available. [link 1](http://1drv.ms/1LZq4VI) [link 2](https://drive.google.com/file/d/0B3Jdl3uI0HHPTUNBTzZrcU1QT0k/view?usp=sharing) 

Username and password of the main Ubuntu user are both **rbnics**. **RBniCS** software is located in $HOME/RBniCS.

### 4. Tutorials
Several tutorials are provided the [**tutorials** subfolder](https://gitlab.com/RBniCS/RBniCS/tree/master/tutorials).
* **Tutorial 1**: introduction to the capabilities of **RBniCS**: reduced basis method for scalar problems.
* **Tutorial 2**: introduction to the capabilities of **RBniCS**: POD-Galerkin method for vector problems.
* **Tutorial 3**: geometrical parametrization.
* **Tutorial 4**: non-compliant problems, successive constraint method.
* **Tutorial 5**: empirical interpolation method.

### 5. Authors and contributors
**RBniCS** is currently developed and mantained at [SISSA mathLab](http://mathlab.sissa.it/) by
* [Dr. Francesco Ballarin](mailto:francesco.ballarin@sissa.it)
* [Dr. Alberto Sartori](mailto:alberto.sartori@sissa.it)
* [Prof. Gianluigi Rozza](mailto:gianluigi.rozza@sissa.it)

Contact us by email for further information or questions about **RBniCS**, or open an ''Issue'' on this website. **RBniCS** is at an early development stage, so contributions improving either the code or the documentation are welcome, both as patches or merge requests on this website.

### 6. How to cite
If you use **RBniCS** in your work, please use the following citations to reference **RBniCS**
```
@book{HesthavenRozzaStamm2015,
  author    = {Hesthaven, Jan S. and Rozza, Gianluigi and Stamm, Benjamin},
  title     = {Certified Reduced Basis Methods for Parametrized Partial Differential Equations},
  publisher = {Springer International Publishing},
  year      = 2015,
  series    = {SpringerBriefs in Mathematics},
  isbn      = {978-3-319-22469-5}
}
```
and cite the [RBniCS website](http://mathlab.sissa.it/rbnics).

A forthcoming publication will also provide more details on **RBniCS**.

### 7. License
Like all core **FEniCS** components, **RBniCS** is freely available under the GNU LGPL, version 3.

![Google Analytics](https://ga-beacon.appspot.com/UA-66224794-1/rbnics/readme?pixel)
