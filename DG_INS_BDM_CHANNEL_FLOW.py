"""

This code tests the validity of a Discontinuous Galerkin BDM elements
Incompressible Navier-Stokes solution by solving the following equation:

u_t + div(u x u) - nu*div(grad(u)) + grad(p) = f
div(u) = 0

u = g                                               on Dirichlet Boundaries
(u x u - nu*grad(u) + p*I)*n - max(u*n,0)u = h      on Neumann Boundaries

This file was created by Kyle Booker and James Lowman under the supervision of
Sander Rhebergen and Nasser Abukhdeir at the University of Waterloo in 2019.

This program is designed and operated using the NGSolve Finite Element Package.
www.ngsolve.org

"""

from ngsolve import *
from netgen.geom2d import unit_square
from netgen.geom2d import SplineGeometry
from math import pi
import time
start_time = time.time()

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def CalcL2Error(sol, sol2):
    u_ex = 1.5*4*y*(0.41-y)/(0.41*0.41)
    u_exact = CoefficientFunction((u_ex, 0.0)) # Dispersed Phase Velocity at Inlet

    err_u = sqrt(Integrate((sol.components[0]-u_exact)**2, mesh))

    #err_p = sqrt(Integrate((sol.components[1]-p_exact)**2, mesh))

    err_div = sqrt(Integrate(Trace(Grad(sol.components[0]))**2, mesh))
    return (err_u, 1.0, err_div)



#------------------------------------------------------------------------------#
#   User Settings:
#------------------------------------------------------------------------------#

Verbose_Mode        = 1     #   0/1 == Yes/No -- Outputs solution information to terminal
Polynomial_Order    = 1  #   Int           -- Order of approximation polynomials
Initial_Mesh_Size   = 1/4   #   Float         -- Initial mesh Size
No_Refinments       = 5     #   Int           -- Number of times to refine the mesh
Time_Step           = 0.01 #   Float         -- Size of the time step to take
No_Time_Solutions   = 1000   #   Int           -- Number of transient solutions
nu                  = 0.001   #   Float         -- Kinematic viscosity
t_final             = 2  #   Final time

#------------------------------------------------------------------------------#
#   Mesh Generation:
#------------------------------------------------------------------------------#

geo = SplineGeometry()
geo.AddRectangle( (0, 0), (1.0, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
mesh = Mesh( geo.GenerateMesh(maxh=2))

mesh.Curve(Polynomial_Order)
Draw(mesh)

if Verbose_Mode == 1:
    print ("\n\t Boundary Labels: ", mesh.GetBoundaries(),"\n") # Check boundary labels

#------------------------------------------------------------------------------#
#   Define Function Spaces:
#------------------------------------------------------------------------------#

"""

A BDM finite element space (HDiv) is defined on the mesh for the velocity while
an L2 space is defined for the pressure.

BDM elements have that property that u * n is continuous across
elements (i.e. u_1 * n_1 = u_2 * n_1 on element boundaries).

"""

# Velocity Space - HDiv BDM space
V    =  HDiv(mesh, order = Polynomial_Order, dgjumps = True, dirichlet="inlet|wall|cyl")
# Pressure Space - one polynomial degree less than V
Q    =  L2(mesh, order = Polynomial_Order-1, dgjumps = True)
# Mixed Finite Element space
X    =  FESpace ([V, Q], dgjumps = True) # Mixed finite element space (u,p)
print("DOF: ", X.ndof)
drag = []
lift = []

#------------------------------------------------------------------------------#
#   Define trial and test functions, and solution storage sunctions:
#------------------------------------------------------------------------------#

(u, p), (v, q) = X.TnT()    #   Define Trial functions (u,p) and Test functions (v,q)

# NGSolve utilizes grid functions as mutable scalar/vector/tensor variables
UN   =  GridFunction(X)      #   Grid Function for the solution space
UOld =  GridFunction(X)      #   Grid Function for the solution space at previous time step
Pressure = GridFunction(Q)

# Temporary storage variables for previous time step data
U0  =   CoefficientFunction (UOld.components[0])     #   Previous velocity
P0  =   CoefficientFunction (UOld.components[1])     #   Previous pressure

#------------------------------------------------------------------------------#
#   Special variable definitions:
#------------------------------------------------------------------------------#

# Definition of the outward facing normal for every facet in the domain
n   =   specialcf.normal(mesh.dim) # Normal vector on an interface

# Definition of the individual cell sizes
h   =   specialcf.mesh_size

# A Nitsche penalty parameter is defined in the weak forumation for all facets
alpha   =   10.0*Polynomial_Order**2/h

#------------------------------------------------------------------------------#
#   Helper Functions:
#------------------------------------------------------------------------------#

# NGSolve has no native "Max" function, therefore Max is defined explicitly
def Max(A,B):
    return IfPos(A-B,A,B) # If A-B>0 return A; else return B

# A custom function to calculate the L2 Norm error for a given solution
def CalcL2Error(sol):
    err_u = sqrt(Integrate((Norm(sol.components[0])-Norm(u_exact))**2, mesh))
    p_mean = sqrt(Integrate(sol.components[1]**2, mesh))
    p_exact_mean = sqrt(Integrate(p_exact**2, mesh))
    err_p = sqrt(Integrate((sol.components[1]-p_mean-p_exact+p_exact_mean)**2, mesh))
    #err_p = sqrt(Integrate((sol.components[1]-p_exact)**2, mesh))
    err_div = sqrt(Integrate(Trace(Grad(sol.components[0]))**2, mesh))
    return (err_u, err_p, err_div)


#------------------------------------------------------------------------------#
#   NGSolve-Mutable Variables:
#------------------------------------------------------------------------------#

# A special coefficient function class, Parameter, is required to update time.
#       This is required as the exact solution is dependent on time, and as such
#       requires the Dirichlet boundary conditions to be dependent on time
var_time    =   Parameter(0.0)

#------------------------------------------------------------------------------#
#   Exact Solution:
#------------------------------------------------------------------------------#

u_x     =   CoefficientFunction(1.5*4*y*(0.41-y)/(0.41*0.41))
u_y     =   CoefficientFunction(0.0)
u_exact =   CoefficientFunction((u_x, u_y))

p_exact = 0 #sin(pi*x)*cos(pi*y)

#------------------------------------------------------------------------------#
#   Generation of forcing function, f, that enables exact solution:
#------------------------------------------------------------------------------#

f_st            =   CoefficientFunction((0.0,0.0))

h_stokes        =   CoefficientFunction((0.0,0.0))


force_navier_stokes   = CoefficientFunction((0.0,0.0))

# Neumann boundary condition for INS
h_ins = CoefficientFunction((0.0,0.0))

#------------------------------------------------------------------------------#
#   Setting up the time stepping variables:
#------------------------------------------------------------------------------#

dt      = Time_Step                     #   Time step
t       = 0.0                           #   Initial time

#------------------------------------------------------------------------------#
#   Mutable helper functions:
#------------------------------------------------------------------------------#

avg_u       = 0.5*(u + u.Other())               #   Average of Velocity     {{u}}
jump_u      = u-u.Other()                       #   Jump of Velocity        [[u]]
jump_v      = v-v.Other()                       #   Jump of Basis Functions [[v]]
avggrad_u   = 0.5*(Grad(u) + Grad(u.Other()))   #   Average of Vel Grad     {{Grad(u)}}
avggrad_v   = 0.5*(Grad(v) + Grad(v.Other()))   #   Average of BFs          {{Grad(v)}}

u_time_bl   = u*v/dt                            #   Blinear du/dt           (U^N+1)*v/dt
u_time_l    = U0*v/dt                           #   Linear du/dt            (U^N)*v/dt



Draw (UOld.components[0], mesh, "velocity")
Draw (UOld.components[1], mesh, "pressure")
Draw (Norm(UOld.components[0]), mesh, "|velocity|")

#------------------------------------------------------------------------------#
#   Setup the Incompressible Navier-Stokes problem:
#------------------------------------------------------------------------------#

# Navier-Stokes Bilinear Form
bl_ns   =   BilinearForm(X)

bl_ns  +=   u_time_bl                                               * dx \
        +   nu * InnerProduct(Grad(u), Grad(v))                     * dx \
        +   nu * alpha * InnerProduct(jump_u, jump_v)               * dx(skeleton=True) \
        -   nu * InnerProduct(avggrad_u, OuterProduct(jump_v, n))   * dx(skeleton=True) \
        -   nu * InnerProduct(avggrad_v, OuterProduct(jump_u, n))   * dx(skeleton=True) \
        +   nu * alpha * u * v                                      * ds(skeleton=True, definedon=mesh.Boundaries("inlet|wall|cyl")) \
        -   nu * InnerProduct(Grad(u), OuterProduct(v, n))          * ds(skeleton=True, definedon=mesh.Boundaries("inlet|wall|cyl")) \
        -   nu * InnerProduct(Grad(v), OuterProduct(u, n))          * ds(skeleton=True, definedon=mesh.Boundaries("inlet|wall|cyl")) \
        -   div(v)*p                                                * dx \
        -   div(u)*q                                                * dx \
        -   InnerProduct(OuterProduct(u,U0), Grad(v))               * dx \
        +   jump_v * (U0 * n * avg_u + 0.5 * Norm(U0 * n) * jump_u) * dx(skeleton=True) \
        +   v * (0.5 * (U0 * n) * u + 0.5 * Norm(U0 * n) * u )      * ds(skeleton=True, definedon=mesh.Boundaries("inlet|wall|cyl")) \
        +   v *( Max(U0*n, 0.0) * u)                                * ds(skeleton=True, definedon=mesh.Boundaries("outlet"))

# Navier-Stokes Linear Form
l_ns    =   LinearForm(X)

l_ns   +=   force_navier_stokes * v * dx + u_time_l                 * dx \
        +   nu * alpha * u_exact * v                                * ds(skeleton=True, definedon=mesh.Boundaries("inlet")) \
        -   nu * InnerProduct(Grad(v), OuterProduct(u_exact, n))    * ds(skeleton=True, definedon=mesh.Boundaries("inlet")) \
        -   v * (0.5*U0 * n * u_exact - 0.5*Norm(U0 * n) * u_exact) * ds(skeleton=True, definedon=mesh.Boundaries("inlet")) \
        -   v * h_ins                                               * ds(skeleton=True, definedon=mesh.Boundaries("outlet"))

#------------------------------------------------------------------------------#
#   Setup the Incompressible Navier-Stokes Preconditioner:
#------------------------------------------------------------------------------#

c = Preconditioner(type="direct", bf=bl_ns, flags = {"inverse" : "umfpack" })

#------------------------------------------------------------------------------#
#   Solution function for the Incompressible Navier-Stokes problem:
#------------------------------------------------------------------------------#


f = LinearForm(X)
f.Assemble()

store = []


def SolveBVP_NavierStokes():

    # Interpolate the exact solution onto the boundary facets
    UN.components[0].Set((u_exact*n)*n, definedon=mesh.Boundaries("inlet"))

    # Resolve Stokes with new mesh
    #SolveBVP_Stokes()
    Redraw (blocking=False)

    t = 0
    # Solve transient INS
    step    =   0  # Iteration step counter
    with TaskManager():
        while t < t_final:
            step += 1
            # Increase time by time step
            # Interpolate the exact solution onto the boundary facets
            UN.components[0].Set((u_exact*n)*n, definedon=mesh.Boundaries("inlet"))

            t += float(dt)

            # Set the parameter time to update mutable variables

            if Verbose_Mode == 1:
                print ('\t Time step: %d \t\t Time: %1.1e' %(step,t))

            # Assemble the linear and bi-linear matrices, and preconditioner
            bl_ns.Assemble()
            l_ns.Assemble()
            c.Update()

            # solve system
            BVP(bf=bl_ns,lf=l_ns,gf=UN,pre=c,maxsteps=300,prec=1.e-12 ).Do()

            UOld.vec.data = UN.vec
            Redraw (blocking=False)

    err_u, err_p, err_div = CalcL2Error(UOld)
    store.append ( (X.ndof, mesh.ne, err_u, err_p, err_div) )

    # Solve the Navier-Stoke system with the new mesh
SolveBVP_NavierStokes()

for i in range(No_Refinments):

    # Refine the mesh
    if i != 1:
        mesh.Refine()

    # Update the mixed function space and grid functions if mesh has changed
    X.Update()

    UN.Update()
    UOld.Update()


    # Solve the Navier-Stoke system with the new mesh
    SolveBVP_NavierStokes()


# Final print routine to output convergence rates for velocity, pressure, divergence
i = 1
print ("\n\n\n\n\n\n")
print ("-------------------------------------------------------------------")
print (" Cells ||  E_u \t   | rate ||  N/A | rate ||  div")
print ("-------------------------------------------------------------------")
while i < len(store) :
    rate_u   = log(store[i-1][2]/store[i][2])/log(2.0000)
    rate_p   = log(store[i-1][3]/store[i][3])/log(2.0000)
    rate_div = log(store[i-1][4]/store[i][4])/log(2.0000)
    print("%6d ||  %1.1e | %1.1f  ||  %1.1e | %1.1f  ||  %1.1e" % \
          (store[i][1], store[i][2], rate_u, store[i][3], rate_p, store[i][4]))
    i =  i+1
print ("-------------------------------------------------------------------")
print ("\n\n\n")


"""
End of Unit Test.
"""

print("--- %s seconds ---\n\n" % (time.time() - start_time))
