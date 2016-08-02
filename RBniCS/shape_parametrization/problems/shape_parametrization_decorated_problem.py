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

from __future__ import print_function
from dolfin import VectorFunctionSpace, cells, LagrangeInterpolator, Function, ALE
from RBniCS.utils.ufl import ParametrizedExpression
from RBniCS.utils.mpi import print
from RBniCS.utils.decorators import Extends, override, ProblemDecoratorFor

def ShapeParametrizationDecoratedProblem(*shape_parametrization_expression, **decorator_kwargs):
    @ProblemDecoratorFor(ShapeParametrization,
        shape_parametrization_expression=shape_parametrization_expression
    )
    def ShapeParametrizationDecoratedProblem_Decorator(ParametrizedProblem_DerivedClass):
        #~~~~~~~~~~~~~~~~~~~~~~~~~     SHAPE PARAMETRIZATION CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
        ## @class ShapeParametrizationDecoratedProblem
        #
        # A decorator class that allows to overload methods related to shape parametrization and mesh motion
        @Extends(ParametrizedProblem_DerivedClass, preserve_class_name=True)
        class ShapeParametrizationDecoratedProblem_Class(ParametrizedProblem_DerivedClass):
        
            ###########################     CONSTRUCTORS     ########################### 
            ## @defgroup Constructors Methods related to the construction of the SCM object
            #  @{
            
            ## Default initialization of members
            # The shape parametrization expression is a list of tuples. The i-th list element
            # corresponds to shape parametrization of the i-th subdomain, the j-th tuple element
            # corresponds to the expression of the j-th component of the shape parametrization
            @override
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
                    subdomain_id = int(self.subdomains[cell]) - 1 # tuple start from 0, while subdomains from 1
                    if subdomain_id not in self.subdomain_id_to_deformation_dofs:
                        self.subdomain_id_to_deformation_dofs[subdomain_id] = list()
                    dofs = self.deformation_V.dofmap().cell_dofs(cell.index())
                    for dof in dofs:
                        self.subdomain_id_to_deformation_dofs[subdomain_id].append(dof)
                assert min(self.subdomain_id_to_deformation_dofs.keys()) == 0
                assert len(self.subdomain_id_to_deformation_dofs.keys()) == max(self.subdomain_id_to_deformation_dofs.keys()) + 1
                
                # Store the shape parametrization expression
                if len(shape_parametrization_expression) > 0:
                    assert "shape_parametrization_expression" not in decorator_kwargs
                    self.shape_parametrization_expression = shape_parametrization_expression
                else:
                    assert "shape_parametrization_expression" in decorator_kwargs
                    self.shape_parametrization_expression = decorator_kwargs["shape_parametrization_expression"]
                assert len(self.shape_parametrization_expression) == len(self.subdomain_id_to_deformation_dofs.keys())
                 
            #  @}
            ########################### end - CONSTRUCTORS - end ###########################
            
            ###########################     OFFLINE STAGE     ########################### 
            ## @defgroup OfflineStage Methods related to the offline stage
            #  @{
            
            ## Initialize data structures required for the offline phase
            @override
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
                            self,
                            tuple(displacement_expression_i),
                            mu=self.mu,
                            element=self.deformation_V.ufl_element()
                        )
                    )
            
            #  @}
            ########################### end - OFFLINE STAGE - end ########################### 
            
            ###########################     I/O     ########################### 
            ## @defgroup IO Input/output methods
            #  @{
                
            ## Deform the mesh as a function of the geometrical parameters
            @override
            def move_mesh(self):
                print("moving mesh")
                displacement = self.compute_displacement()
                ALE.move(self.mesh, displacement)
            
            ## Restore the reference mesh
            @override
            def reset_reference(self):
                print("back to the reference mesh")
                self.mesh.coordinates()[:] = self.reference_coordinates
            
            ## Auxiliary method to deform the domain
            def compute_displacement(self):
                interpolator = LagrangeInterpolator()
                displacement = Function(self.deformation_V)
                for i in range(len(self.displacement_expression)):
                    displacement_subdomain_i = Function(self.deformation_V)
                    interpolator.interpolate(displacement_subdomain_i, self.displacement_expression[i])
                    subdomain_dofs = self.subdomain_id_to_deformation_dofs[i]
                    displacement.vector()[subdomain_dofs] = displacement_subdomain_i.vector()[subdomain_dofs]
                return displacement
                
            #  @}
            ########################### end - I/O - end ########################### 
        
        # return value (a class) for the decorator
        return ShapeParametrizationDecoratedProblem_Class
    
    # return the decorator itself
    return ShapeParametrizationDecoratedProblem_Decorator
    
# For the sake of the user, since this is the only class that he/she needs to use, rename it to an easier name
ShapeParametrization = ShapeParametrizationDecoratedProblem
