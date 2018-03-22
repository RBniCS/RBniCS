## RBniCS - reduced order modelling in FEniCS ##
![RBniCS - reduced order modelling in FEniCS](https://gitlab.com/RBniCS/RBniCS/raw/master/docs/rbnics-logo-small.png "RBniCS - reduced order modelling in FEniCS")

### 0. Introduction
**RBniCS** is an implementation in **FEniCS** of several reduced order modelling techniques (and, in particular, certified reduced basis method and Proper Orthogonal Decomposition-Galerkin methods) for coercive problems. It is ideally suited for an introductory course on reduced basis methods and reduced order modelling, thanks to an object-oriented approach and an intuitive and versatile python interface. To this end, it has been employed in several doctoral courses on "Reduced Basis Methods for Computational Mechanics".

**RBniCS** can also be used as a basis for more advanced projects that would like to assess the capability of reduced order models in their existing **FEniCS**-based software, thanks to the availability of several reduced order methods (such as reduced basis and proper orthogonal decomposition) and algorithms (such as successive constraint method, empirical interpolation method) in the library.

Several tutorials are provided. This software is also a companion of the introductory reduced basis handbook: 

> [J. S. Hesthaven, G. Rozza, B. Stamm. **Certified Reduced Basis Methods for Parametrized Partial Differential Equations**. SpringerBriefs in Mathematics. Springer International Publishing, 2015](http://www.springer.com/us/book/9783319224695)

### 1. Prerequisites
**RBniCS** requires
* **FEniCS** (>= 2017.2.0, python 3), with PETSc, SLEPc, petsc4py and slepc4py for computations during the offline stage;
* **numpy** and **scipy** for computations during the online stage.

Additional requirements are automatically handled during the setup.

### 2. Installation and usage
Simply clone the **RBniCS** public repository:
```
git clone https://gitlab.com/RBniCS/RBniCS.git
```
and install the package by typing
```
python3 setup.py install
```

### 2.1. RBniCS docker image
If you want to try **RBniCS** out but do not have **FEniCS** already installed, you can [pull our docker image from Docker Hub](https://hub.docker.com/r/rbnics/rbnics/). All required dependencies are already installed. **RBniCS** tutorials and tests are located at
```
$FENICS_HOME/local/share/RBniCS
```

### 3. Tutorials
Several tutorials are provided the [**tutorials** subfolder](https://gitlab.com/RBniCS/RBniCS/tree/master/tutorials).
* **Tutorial 01**: introduction to the capabilities of **RBniCS**: reduced basis method for (scalar) elliptic problems.
* **Tutorial 02**: introduction to the capabilities of **RBniCS**: POD-Galerkin method for (vector) elliptic problems.
* **Tutorial 03**: geometrical parametrization.
* **Tutorial 04**: successive constraint method.
* **Tutorial 05**: empirical interpolation methods for non-affine elliptic problems.
* **Tutorial 06**: reduced basis and POD-Galerkin methods for parabolic problems.
* **Tutorial 07**: empirical interpolation methods for nonlinear elliptic problems.
* **Tutorial 08**: empirical interpolation methods for nonlinear parabolic problems.
* **Tutorial 09**: reduced order methods for advection dominated elliptic problems.
* **Tutorial 10**: weighted reduced order methods for uncertainty quantification problems.
* **Tutorial 11**: POD-Galerkin methods for quasi geostrophic equations, as an example on how to customize and extend RBniCS beyond the set of problems provided in the core of the library.
* **Tutorial 12**: reduced basis and POD-Galerkin methods for Stokes problems.
* **Tutorial 13**: reduced basis and POD-Galerkin methods for optimal control problems governed by elliptic equations.
* **Tutorial 14**: reduced basis and POD-Galerkin methods for optimal control problems governed by Stokes equations.
* **Tutorial 15**: POD-Galerkin methods for optimal control problems governed by quasi geostrophic equations.
* **Tutorial 16**: one-way coupling between a fluid dynamics problem based on Stokes and an elliptic equation (e.g., temperature, concentration).
* **Tutorial 17**: POD-Galerkin and empirical interpolation methods for Navier-Stokes problems.
* **Tutorial 18**: POD-Galerkin methods for unsteady Stokes problems.
* **Tutorial 19**: POD-Galerkin methods for unsteady Navier-Stokes problems.

### 4. Authors and contributors
**RBniCS** is currently developed and mantained at [SISSA mathLab](http://mathlab.sissa.it/) by
* [Dr. Francesco Ballarin](http://people.sissa.it/~fballarin/)
* [Dr. Alberto Sartori](https://scholar.google.it/citations?user=rdoHp_EAAAAJ&hl=en)
* [Prof. Gianluigi Rozza](http://people.sissa.it/~grozza/)

in the framework of the [AROMA-CFD ERC CoG project](http://people.sissa.it/~grozza/aroma-cfd/). Please see the [AUTHORS file](https://gitlab.com/RBniCS/RBniCS/raw/master/AUTHORS) for a list of contributors.

Contact us by [email](mailto:francesco.ballarin@sissa.it) for further information or questions about **RBniCS**, or open an issue on [our issue tracker](https://gitlab.com/RBniCS/RBniCS/issues). **RBniCS** is at an early development stage, so contributions improving either the code or the documentation are welcome, both as patches or [merge requests](https://gitlab.com/RBniCS/RBniCS/merge_requests).

### 5. How to cite
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

### 6. License
Like all core **FEniCS** components, **RBniCS** is freely available under the GNU LGPL, version 3.

![Google Analytics](https://ga-beacon.appspot.com/UA-66224794-1/rbnics/readme?pixel)
