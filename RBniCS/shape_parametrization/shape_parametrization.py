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
## @file scm.py
#  @brief Implementation of the successive constraints method for the approximation of the coercivity constant
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import VectorFunctionSpace, cells, LagrangeInterpolator, Function
from RBniCS.io_utils import ParametrizedExpression

def ShapeParametrization(*shape_parametrization_expression):
    def ShapeParametrization_Decorator(ParametrizedProblem_DerivedClass):
        #~~~~~~~~~~~~~~~~~~~~~~~~~     SHAPE PARAMETRIZATION CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
        ## @class ShapeParametrization
        #
        # A decorator class that allows to overload methods related to shape parametrization and mesh motion
        class ShapeParametrization_Class(ParametrizedProblem_DerivedClass):
        
            ###########################     CONSTRUCTORS     ########################### 
            ## @defgroup Constructors Methods related to the construction of the SCM object
            #  @{
            
            ## Default initialization of members
            # The shape parametrization expression is a list of tuples. The i-th list element
            # corresponds to shape parametrization of the i-th subdomain, the j-th tuple element
            # corresponds to the expression of the j-th component of the shape parametrization
            def __init__(self, V, **kwargs):
                # Call the standard initialization
                ParametrizedProblem_DerivedClass.__init__(self, V, **kwargs)
                # Store FEniCS data structure related to the geometrical parametrization
                assert "subdomains" in kwargs
                self.subdomains = kwargs["subdomains"]
                self.mesh = V.mesh()
                self.reference_coordinates = self.mesh.coordinates().copy()
                self.deformation_V = VectorFunctionSpace(self.mesh, "Lagrange", 1)
                self.subdomain_id_to_deformation_dofs = dict() # from int to list
                for cell in cells(self.mesh):
                    subdomain_id = int(self.subdomains[cell])
                    if subdomain_id not in self.subdomain_id_to_deformation_dofs:
                        self.subdomain_id_to_deformation_dofs[subdomain_id] = list()
                    dofs = self.deformation_V.dofmap().cell_dofs(cell.index())
                    for dof in dofs:
                        self.subdomain_id_to_deformation_dofs[subdomain_id].append(dof)
                
                # Store the shape parametrization expression
                self.shape_parametrization_expression = shape_parametrization_expression
                 
            #  @}
            ########################### end - CONSTRUCTORS - end ###########################
            
            ###########################     SETTERS     ########################### 
            ## @defgroup Setters Set properties of the reduced order approximation
            #  @{
        
            # Propagate the values of all setters also to the parametrized expression object
            
            ## OFFLINE/ONLINE: set the current value of the parameter
            def set_mu(self, mu):
                ParametrizedProblem_DerivedClass.set_mu(self, mu)
                try:
                    self._set_mu_for_displacement_expression(mu)
                except AttributeError:
                    # this will happen when setting mu for the first time,
                    # because the init() method has not been called yet
                    pass
                
            def _set_mu_for_displacement_expression(self, mu):
                for i in range(len(self.displacement_expression)):
                    self.displacement_expression[i].set_mu(mu)
                
            #  @}
            ########################### end - SETTERS - end ########################### 
            
            ###########################     OFFLINE STAGE     ########################### 
            ## @defgroup OfflineStage Methods related to the offline stage
            #  @{
            
            ## Initialize data structures required for the offline phase
            def init(self):
                ParametrizedProblem_DerivedClass.init(self)
                # Preprocess the shape parametrization expression to convert it in the displacement expression
                # This cannot be done during __init__ because at construction time the number
                # of parameters is still unknown
                self.displacement_expression = list()
                for i in range(len(self.shape_parametrization_expression)):
                    displacement_expression_i = list()
                    assert len(self.shape_parametrization_expression[i]) == self.mesh.geometry().dim()
                    for j in range(len(self.shape_parametrization_expression[i])):
                        # convert from shape parametrization T to displacement d = T - I
                        displacement_expression_i.append(
                            self.shape_parametrization_expression[i][j] + " - x[" + str(j) + "]",
                        )
                    self.displacement_expression.append(
                        ParametrizedExpression(
                            tuple(displacement_expression_i),
                            mu=self.mu,
                            element=self.deformation_V.ufl_element()
                        )
                    )
                # Now that displacement_expression has been initialized, make sure
                # that the mu is up-to-date (see the try/except in the set_mu method)
                self._set_mu_for_displacement_expression(self.mu)
            
            #  @}
            ########################### end - OFFLINE STAGE - end ########################### 
            
            ###########################     I/O     ########################### 
            ## @defgroup IO Input/output methods
            #  @{
                
            ## Deform the mesh as a function of the geometrical parameters
            def move_mesh(self):
                print("moving mesh")
                displacement = self.compute_displacement()
                ALE.move(self.mesh, displacement)
            
            ## Restore the reference mesh
            def reset_reference(self):
                print("back to the reference mesh")
                self.mesh.coordinates()[:] = self.reference_coordinates
            
            ## Auxiliary method to deform the domain
            def compute_displacement(self):
                interpolator = LagrangeInterpolator()
                displacement = Function(self.deformation_V)
                for i in range(len(self.displacement_expression)):
                    displacement_subdomain_i = Function(self.deformation_V)
                    interpolator.interpolate(self.displacement_expression[i])
                    subdomain_dofs = self.subdomain_id_to_deformation_dofs[i]
                    displacement.vector()[subdomain_dofs] = displacement_subdomains_i.vector()[subdomain_dofs]                    
                return displacement
                
            ## Get the name of the problem, to be used as a prefix for output folders.
            # Overridden to use the parent name
            @classmethod
            def name(cls):
                assert len(cls.__bases__) == 1
                return cls.__bases__[0].name()
                        
            #  @}
            ########################### end - I/O - end ########################### 
        
        # return value (a class) for the decorator
        return ShapeParametrization_Class
    
    # return the decorator itself
    return ShapeParametrization_Decorator
