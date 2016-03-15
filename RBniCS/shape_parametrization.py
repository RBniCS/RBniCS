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

from dolfin import *
import numpy as np

def ShapeParametrization(ParametrizedProblem_DerivedClass):
    #~~~~~~~~~~~~~~~~~~~~~~~~~     SHAPE PARAMETRIZATION CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
    ## @class ShapeParametrization
    #
    # A decorator class that allows to overload methods related to shape parametrization and mesh motion
    class ShapeParametrization(ParametrizedProblem_DerivedClass):
    
        ###########################     CONSTRUCTORS     ########################### 
        ## @defgroup Constructors Methods related to the construction of the SCM object
        #  @{
        
        ## Default initialization of members
        # The shape parametrization expression is a list of tuples. The i-th list element
        # corresponds to shape parametrization of the i-th subdomain, the j-th tuple element
        # corresponds to the expression of the j-th component of the shape parametrization
        def __init__(self, mesh, subd, V, bc_list, shape_parametrization_expression):
            # Call the standard initialization
            ParametrizedProblem_DerivedClass.__init__(self, V, bc_list)
            # Store FEniCS data structure related to the geometrical parametrization
            self.mesh = mesh
            self.subd = subd
            self.xref = mesh.coordinates()[:,0].copy()
            self.yref = mesh.coordinates()[:,1].copy()
            self.deformation_V = VectorFunctionSpace(self.mesh, "Lagrange", 1)
            self.subdomain_id_to_deformation_dofs = ()
            for subdomain_id in np.unique(self.subd.array()):
                self.subdomain_id_to_deformation_dofs += ([],)
            for cell in cells(mesh):
                subdomain_id = int(self.subd.array()[cell.index()] - 1) # tuple start from 0, while subdomains from 1
                dofs = self.deformation_V.dofmap().cell_dofs(cell.index())
                for dof in dofs:
                    self.subdomain_id_to_deformation_dofs[subdomain_id].append(dof)
            
            # Store the shape parametrization expression
            self.shape_parametrization_expression = shape_parametrization_expression
        
        # Preprocess the shape parametrization expression, to convert mu[p] to mu_p
        # as instant requires, and to convert it in the displacement expression
        # It cannot be done during __init__ because at construction time the number
        # of parameters is still unknown
        def __init_displacement_expression(self):
            if self.__init_displacement_expression.__func__.done:
                return
            self.displacement_expression = []
            for i in range(len(self.shape_parametrization_expression)):
                displacement_expression_i = ()
                for j in range(len(self.shape_parametrization_expression[i])):
                    displacement_expression_ij = self.shape_parametrization_expression[i][j]
                    for p in range(len(self.mu)):
                        displacement_expression_ij = \
                            displacement_expression_ij.replace("mu[" + str(p) + "]", "mu_" + str(p))
                    displacement_expression_ij = \
                        displacement_expression_ij + " - x[" + str(j) + "]" # convert from shape parametrization T to displacement d = T - I
                    displacement_expression_i += (displacement_expression_ij,)
                self.displacement_expression.append(displacement_expression_i)
            self.__init_displacement_expression.__func__.done = True
        # Default value for static variable
        __init_displacement_expression.done = False    
        
        #  @}
        ########################### end - CONSTRUCTORS - end ###########################
        
        ###########################     I/O     ########################### 
        ## @defgroup IO Input/output methods
        #  @{
            
        ## Deform the mesh as a function of the geometrical parameters
        def move_mesh(self):
            print "moving mesh"
            displacement = self.compute_displacement()
            ALE.move(self.mesh, displacement)
        
        ## Restore the reference mesh
        def reset_reference(self):
            print "back to the reference mesh"
            new_coor = np.array([self.xref, self.yref]).transpose()
            self.mesh.coordinates()[:] = new_coor
        
        ## Auxiliary method to deform the domain
        def compute_displacement(self):
            self.__init_displacement_expression() # done only once
            mu_dict = {}
            for p in range(len(self.mu)):
                mu_dict[ "mu_" + str(p) ] = self.mu[p]
            displacement_subdomains = ()
            for i in range(len(self.displacement_expression)):
                displacement_expression_i = Expression(self.displacement_expression[i], degree=self.deformation_V.ufl_element().degree(), **mu_dict)
                displacement_subdomains += (interpolate(displacement_expression_i, self.deformation_V),)
            displacement = Function(self.deformation_V)
            for i in range(len(displacement_subdomains)):
                subdomain_dofs = self.subdomain_id_to_deformation_dofs[i]
                displacement.vector()[subdomain_dofs] = displacement_subdomains[i].vector()[subdomain_dofs]
            return displacement
                    
        #  @}
        ########################### end - I/O - end ########################### 
    
    # return value (a class) for the decorator
    return ShapeParametrization  
