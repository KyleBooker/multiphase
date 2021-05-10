from netgen.geom2d import SplineGeometry
from netgen.geom2d import unit_square
from math import pi
import time

Verbose_Mode        = 1     #   0/1 == Yes/No -- Outputs solution information to terminal
Polynomial_Order    = 3  #   Int           -- Order of approximation polynomials
Initial_Mesh_Size   = 1   #   Float         -- Initial mesh Size
No_Refinments       = 7   #   Int           -- Number of times to refine the mesh
Time_Step           = 1e-10  #   Float         -- Size of the time step to take
No_Time_Solutions   = 1    #   Int           -- Number of transient solutions
nu                  = 10e-5   #   Float         -- Kinematic viscosity

geo = SplineGeometry()
geo.AddRectangle( (0, 0), (1, 1), bcs = ("left", "right", "top", "bottom"))
mesh = Mesh( geo.GenerateMesh(maxh=Initial_Mesh_Size))
# mesh = Mesh(unit_square.GenerateMesh(maxh=Initial_Mesh_Size))
mesh.Curve(3); Draw(mesh)

def Max(A,B):
    return IfPos(A-B,A,B) # If A-B>0 return A; else return B

# A custom function to calculate the L2 Norm error for a given solution
def CalcL2Error(sol):
    err_u = sqrt(Integrate((sol.components[0]-u_exact)**2, mesh))
    p_mean = Integrate(sol.components[1], mesh)
    p_ex_mean = Integrate(p_exact, mesh)
    err_p = sqrt(Integrate((sol.components[1]-p_mean-p_exact+p_ex_mean)**2, mesh))
    print(p_mean)
    print(p_ex_mean)
    # err_p = sqrt(Integrate((sol.components[1]-p_mean-p_exact+p_exact_mean)**2, mesh))
    #err_p = sqrt(Integrate((sol.components[1]-p_exact)**2, mesh))
    err_div = sqrt(Integrate(Trace(Grad(sol.components[0]))**2, mesh))
    return (err_u, err_p, err_div)


# Definition of the outward facing normal for every facet in the domain
n   =   specialcf.normal(mesh.dim) # Normal vector on an interface

# Definition of the individual cell sizes
h   =   specialcf.mesh_size

# A Nitsche penalty parameter is defined in the weak forumation for all facets
alpha   =   1*Polynomial_Order**2/h


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

u_x     =   CoefficientFunction(sin(pi*x)*sin(pi*y))
u_y     =   CoefficientFunction(cos(pi*x)*cos(pi*y))
u_exact =   CoefficientFunction((u_x, u_y))

p_exact = sin(pi*x)*cos(pi*y)

#------------------------------------------------------------------------------#
#   Generation of forcing function, f, that enables exact solution:
#------------------------------------------------------------------------------#

# Helper variables, vector calculus
grad_u          =   CoefficientFunction((   u_x.Diff(x),    u_x.Diff(y),    \
                                            u_y.Diff(x),    u_y.Diff(y)),   \
                                        dims=(2,2))
outerUU         =   OuterProduct(u_exact,u_exact)
div_outerUU     =   CoefficientFunction((outerUU[0,0].Diff(x) + outerUU[0,1].Diff(y),outerUU[1,0].Diff(x) + outerUU[1,1].Diff(y)))

# Forcing function for the Stokes initial condition solution
f_xstokes       =  -nu * (u_exact[0].Diff(x).Diff(x) + u_exact[0].Diff(y).Diff(y))+ p_exact.Diff(x)
f_ystokes       =  -nu *  (u_exact[1].Diff(x).Diff(x) + u_exact[1].Diff(y).Diff(y))+ p_exact.Diff(y)
f_st            =   CoefficientFunction((f_xstokes,f_ystokes))

# Multiplying pressure by the identity tensor
p_I             =   CoefficientFunction((   p_exact,    0,              \
                                            0,          p_exact),       \
                                        dims=(2,2))

# In the weak formulation of Stokes, the Neumann condition requires
# -nu grad(u) + pI projected onto the outward facing normal on the boundary.
# Generated here as h for Stokes.
h_stokes        =   (- nu * grad_u + p_I)*n

# Forcing function for Stokes
f_x             =   u_exact[0].Diff(var_time) - nu * (u_exact[0].Diff(x).Diff(x) \
                  + u_exact[0].Diff(y).Diff(y)) + div_outerUU[0] + p_exact.Diff(x)
f_y             =   u_exact[1].Diff(var_time) - nu * (u_exact[1].Diff(x).Diff(x) \
                  + u_exact[1].Diff(y).Diff(y)) + div_outerUU[1] + p_exact.Diff(y)
force_navier_stokes   = CoefficientFunction((f_x,f_y))

# Neumann boundary condition for INS
h_ins   =  (outerUU - nu*grad_u + p_I)*n - Max(u_exact*n, 0)*u_exact

#------------------------------------------------------------------------------#
#   Setting up the time stepping variables:
#------------------------------------------------------------------------------#

dt      = Time_Step                     #   Time step
t       = 0.0                           #   Initial time
t_final = No_Time_Solutions*Time_Step   #   Final time


V = VectorH1(mesh,order=Polynomial_Order, dirichlet="left|right|top|bottom")
Q = H1(mesh,order=Polynomial_Order-1)
X = FESpace([V,Q])


gfu = GridFunction(X)
velocity = gfu.components[0]
Draw(velocity,mesh,"u")
Draw(gfu.components[1],mesh,"p")

Redraw()

(u,p), (v,q) = X.TnT()

store = []
def Solve_Stokes():

    gfu.components[0].Set(u_exact, definedon=mesh.Boundaries("left|right|top|bottom"))

    a = BilinearForm(X)
    stokes = (nu*InnerProduct(grad(u),grad(v))-div(u)*q-div(v)*p)*dx
    a += stokes
    a.Assemble()

    f = LinearForm(X)
    f += f_st*v*dx
    f.Assemble()

    inv_stokes = a.mat.Inverse(X.FreeDofs())

    res = f.vec.CreateVector()
    res.data = f.vec - a.mat*gfu.vec
    gfu.vec.data += inv_stokes * res

    Redraw()

    err_u, err_p, err_div = CalcL2Error(gfu)
    store.append ( (X.ndof, mesh.ne, err_u, err_p, err_div) )


for i in range(No_Refinments):

    # Refine the mesh
    if i != 1:
        mesh.Refine()

    # Update the mixed function space and grid functions if mesh has changed
    X.Update()
    gfu.Update()
    # Solve the Navier-Stoke system with the new mesh
    Solve_Stokes()


# Final print routine to output convergence rates for velocity, pressure, divergence
i = 1
print ("\n\n\n\n\n\n")
print ("-------------------------------------------------------------------")
print (" Cells ||  E_u \t   | rate ||  E_p     | rate ||  div")
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
