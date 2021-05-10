################################################################################
#The DG BDM Euler-Euler weak formulation using BDM elements.
################################################################################
from ngsolve import *
from netgen.geom2d import SplineGeometry
ngsglobals.msg_level=1 # Set to 1 for more detailed Output/debugging
import numpy as np
import time

def Max(A,B):
    return IfPos(A-B,A,B) # If A-B>0 return A; else return B

def Min(A,B):
    return IfPos(A-B,B,A) # If A-B>0 return B; else return A

def CalcL2Error(approx, exact, mesh):
    return sqrt(Integrate((approx - exact)**2, mesh))

def CalcChange(A, B, mesh):
    return sqrt(Integrate((A - B)**2, mesh))


# Div_a_d_u_d      = CoefficientFunction((grad(A_D_Old)*u_d + A_D_Old*div(u_d))) # Div(a_d*v_d)
# Div_a_c_u_m      = CoefficientFunction((-grad(A_D_Old)*u_m + (1.0 - A_D_Old)*div(u_m))) # Div(a_c*v_m)

def CalcL2Error(sol, sol2):
    u_ex = 1.5*4*y*(0.41-y)/(0.41*0.41)
    u_exact = CoefficientFunction((u_ex, 0.0)) # Dispersed Phase Velocity at Inlet

    err_u = sqrt(Integrate((sol.components[0]-u_exact)**2, mesh))

    err_phase = sqrt(Integrate((A_D - 0.025)**2, mesh))
    #err_p = sqrt(Integrate((sol.components[1]-p_exact)**2, mesh))

    div_d = CoefficientFunction((grad(sol2)*sol.components[0] + A_D_Old*(sol.components[0][0].Diff(x) + sol.components[0][1].Diff(y))))
    div_c = CoefficientFunction((-1.0*grad(sol2)*sol.components[1] + A_D_Old*(sol.components[1][0].Diff(x) + sol.components[1][1].Diff(y))))


    err_div = sqrt(Integrate((div_c + div_d)**2, mesh))
    return (err_u, err_phase, err_div)

################################################################################
# Parameters
################################################################################
dt           = 1e-2 # Time step
final_time   = 1.7 # Final Time
param_t      = Parameter(0.0)
t_0          = 0.625   # Maximum Inlet time

mesh_size    = 2
k            = 2 # Order of approximation polynomials
No_Refinments= 4

################################################################################
# Constants
################################################################################
rho_c    = 1000.0  # Density of Continuous Phase [kg/m^3]
rho_d    = 10.0  # Density of Disperse Phase [kg/m^3]
mu_c     = 5e-3 # Dynamic Viscosity of Continuous [Pa s]
mu_d     = 2e-5 # Dynamic Viscosity of Disperse Phase [Pa s]

x_s      = 0.41  # Characteristic Length scale
v_s      = 1.5 # Characteristic Velocity scale
g_s      = 9.81 # Characteristic Gravity scale
h_s      = 2 # Characteristic Height scale
P_s      = rho_c * g_s * h_s # Characteristic Pressure scale
t_s      = x_s / v_s

bubble_d = 1e-3/x_s # Bubble Size diameter [m]

e        = np.finfo(float).eps # Machine epsilon

Eu_c     = P_s/(rho_c * v_s * v_s) # Continuous Phase Euler Number
Eu_d     = P_s/(rho_d * v_s * v_s) # Continuous Phase Euler Number

Re_c     = 0.001#rho_c * v_s * x_s / mu_c  # Continuous Phase Reynolds number
Re_d     = 0.001 #rho_d * v_s * x_s / mu_d  # Disperse Phase Reynolds number

Fr        = v_s / (sqrt(g_s*x_s))  # Froude number

grav     = CoefficientFunction((0.0, 0.0))

################################################################################
# Mesh Creation
################################################################################
geo = SplineGeometry()
geo.AddRectangle((0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
mesh = Mesh(geo.GenerateMesh(maxh=mesh_size))
mesh.Curve(k)
# mesh.Refine()
# mesh.Refine()
# mesh.Refine()
# mesh.Refine()


# Mesh related functions
h = specialcf.mesh_size # Mesh size
n = specialcf.normal(mesh.dim) # Outward normal vector on element facets

beta = 10.0*(k**2)/h # Penalty Parameter

################################################################################
# Function Spaces
################################################################################

V    = HDiv(mesh, order = k, dirichlet="wall|inlet")
Q    = L2(mesh, order = k-1) # Must be k-1 for stability
X    = FESpace ([V, V, Q], dgjumps = True) # Mixed finite element space
A    = L2(mesh, order = k, dgjumps = True) # Gas Phase Fraction Finite Element Space

################################################################################
# Trial and Test Functions
################################################################################

# u_m: Mixture Velocity Trial Function; v_m: Mixture Velocity Test Function
# u_d: Disperse Phase Velocity Trial Function; v_d: Disperse Phase Velocity Test Function
# p: Pressure Trial Function; q: Pressure Test Function
(u_m, u_d, p), (v_m, v_d, q) = X.TnT()

 # a_d: Disperse Phase Trial Function; z_d: Disperse Phase Test Function
a_d, z = A.TnT()

################################################################################
# Gridfunctions
################################################################################
UN      = GridFunction(X) # Gridfuction for Velocities and Pressure
UOld    = GridFunction(X) # Gridfunction for the solution at previous time step

A_D     = GridFunction(A) # Gridfunction for Disperse Phase Fraction
A_D_Old = GridFunction(A) # Gridfunction for the Disperse Phase Fraction at previous time step

# Solution at PREVIOUS time step
U_M_0   = CoefficientFunction (UOld.components[0]) # Mixture Velocity
U_D_0 = CoefficientFunction(UOld.components[1]) # Disperse Phase Velocity
P_0     = CoefficientFunction (UOld.components[2]) # Pressure
U_C_0 = (U_M_0 - A_D_Old * U_D_0)/(1.0 - A_D_Old) # Continuous Phase Velocity

################################################################################
# Boundary Conditions
################################################################################

#vel_inlet = Min(param_t/t_0, 1.0)*0.0616*exp(-((x/0.025)**2)/(2*((0.1)**2))) # Inlet Velocity: Gaussian Distribution

#ad_inlet  = Min(param_t/t_0, 1.0)*0.026*exp(-((x/0.025)**2)/(2*((0.1)**2)))  # Disperse Phase
ad_inlet = 0.025

# Phase Fraction Boundary Conditions
ad_bnd = CoefficientFunction(ad_inlet) # Dispersed Phase Dirichlet Inlet condition
ad_wall_bnd  = CoefficientFunction(0.025)  # Dispersed Phase Dirichlet Wall condition
force_ad = CoefficientFunction(0.0)

# Dispersed Phase Velocity field
vel_inlet = 1.5*4*y*(0.41-y)/(0.41*0.41)
ud_bnd = CoefficientFunction((vel_inlet, 0.0)) # Dispersed Phase Velocity at Inlet
ud_wall_bnd = CoefficientFunction((0.0, 0.0)) # Dispersed Phase Velocity at Walls

# Mixture Velocity field
um_bnd = CoefficientFunction((vel_inlet, 0.0)) # Mixture Velocity at Inlet
um_wall_bnd = CoefficientFunction((0.0, 0.0)) # Mixture Velocity at Walls

# Neumann boundary condition
neumann_bnd = CoefficientFunction((0.0, 0.0)) # Neumann boundary condition for velocity
disperse_phase_neumannn_bnd = CoefficientFunction((0.0,0.0)) # Neumann boundary condition disperse phase fraction

################################################################################
# Initial Conditions
################################################################################
UN.components[0].Set(CoefficientFunction((0.0,0.0)))
UN.components[1].Set(CoefficientFunction((0.0,0.0)))
A_D.Set(CoefficientFunction(0.025))

################################################################################
# Intermediary Viscous Stress Functions
################################################################################
S_ud            = CoefficientFunction(grad(u_d))
Stress_ud       = CoefficientFunction(0.5*(S_ud + S_ud.trans))
S_ud_O          = CoefficientFunction(grad(u_d.Other()))
Stress_ud_Other = CoefficientFunction(0.5*(S_ud_O + S_ud_O.trans))

S_vd            = CoefficientFunction(grad(v_d))
Stress_vd       = CoefficientFunction(0.5*(S_vd + S_vd.trans))
S_vd_O          = CoefficientFunction(grad(v_d.Other()))
Stress_vd_Other = CoefficientFunction(0.5*(S_vd_O + S_vd_O.trans))

S_vm            = CoefficientFunction(grad(v_m))
Stress_vm       = CoefficientFunction(0.5*(S_vm + S_vm.trans))
S_vm_O          = CoefficientFunction(grad(v_m.Other()))
Stress_vm_Other = CoefficientFunction(0.5*(S_vm_O + S_vm_O.trans))

S_um            = CoefficientFunction(grad(u_m))
Stress_um       = CoefficientFunction(0.5*(S_um + S_um.trans))
S_um_O          = CoefficientFunction(grad(u_m.Other()))
Stress_um_Other = CoefficientFunction(0.5*(S_um_O + S_um_O.trans))

S_um_ac            = CoefficientFunction(((1.0-A_D_Old)*grad(u_m) + OuterProduct(u_m,grad(A_D_Old)))*(1.0/(1.0-A_D_Old)**2))
Stress_um_ac       = CoefficientFunction(0.5*(S_um_ac  + S_um_ac.trans))
S_um_ac_O          = CoefficientFunction((1.0-A_D_Old.Other())*grad(u_m.Other()) + OuterProduct(u_m.Other(),grad(A_D_Old).Other())*(1.0/(1.0-A_D_Old.Other())**2))
Stress_um_ac_Other   = CoefficientFunction(0.5*(S_um_ac_O  + S_um_ac_O.trans))

s_ud_ad_ac    = CoefficientFunction(((1.0-A_D_Old)*(A_D_Old*grad(u_d) + OuterProduct(grad(A_D_Old),u_d)) + OuterProduct(grad(A_D_Old),A_D_Old*u_d))*(1.0/(1.0-A_D_Old)**2))
Stress_ud_ad_ac   = CoefficientFunction(0.5*(s_ud_ad_ac  + s_ud_ad_ac.trans))
s_ud_ad_ac_O    = CoefficientFunction(((1.0-A_D_Old.Other())*(A_D_Old.Other()*grad(u_d.Other()) + OuterProduct(grad(A_D_Old).Other(),u_d.Other())) + OuterProduct(grad(A_D_Old).Other(),A_D_Old.Other()*u_d.Other()))*(1.0/(1.0-A_D_Old.Other())**2))
Stress_ud_ad_ac_Other   = CoefficientFunction(0.5*(s_ud_ad_ac_O  + s_ud_ad_ac_O.trans))

################################################################################
# Jumps and Averages for DG numerical Fluxes
################################################################################

jump_um = u_m - u_m.Other() # [[u_m]]
jump_ud = u_d - u_d.Other() # [[u_d]]
jump_vm = v_m - v_m.Other() # [[v_m]]
jump_vd = v_d - v_d.Other() # [[v_d]]
jump_ud_ad = u_d*A_D_Old - u_d.Other()*A_D_Old.Other() # [[u_d*a_d]]
jump_um_ac = u_m*(1.0/(1.0-A_D_Old)) - u_m.Other()*(1.0/(1.0-A_D_Old.Other())) # [[u_m*a_c]]
jump_ud_ad_ac = u_d*(A_D_Old/(1.0-A_D_Old)) - u_d.Other()*(A_D_Old.Other()/(1.0-A_D_Old.Other())) # [[u_d*a_d/a_c]]

avg_u_m = 0.5*(u_m + u_m.Other()) # {{u_m}}
avg_ud_ad = 0.5*(A_D_Old*u_d + A_D_Old.Other()*u_d.Other()) # {{u_d*a_d}}
avg_ud_ud_ad = 0.5*(A_D_Old*OuterProduct(U_D_0, u_d) + A_D_Old.Other()*OuterProduct(U_D_0.Other(),u_d.Other())) # {{u_d*a_d}}
avg_um_ac = 0.5*(u_m*(1.0/(1.0-A_D_Old)) + u_m.Other()*(1.0/(1.0-A_D_Old.Other()))) # {{u_m*a_c}}
avg_ud_ad_ac = 0.5 * (A_D_Old*u_d*(1.0/(1.0-A_D_Old)) + A_D_Old.Other()*u_d.Other()*(1.0/(1.0-A_D_Old.Other()))) # {{u_d*a_d/a_c}}
avg_um_um_ac = 0.5*((1.0 -A_D_Old)*OuterProduct(U_M_0, u_m) + (1.0 -A_D_Old.Other())*OuterProduct(U_M_0.Other(),u_m.Other())) # {{u_d*a_d}}


avg_Stress_vd   = 0.5*(Stress_vd + Stress_vd_Other)  # {{e(v_d)}}
avg_Stress_vm   = 0.5*(Stress_vm + Stress_vm_Other) # {{e(v_m)}}
avg_ad_Stress_ud = 0.5*(A_D_Old*Stress_ud + A_D_Old.Other()*Stress_ud_Other) # {{a_d* e(u_d)}}
avg_ac_Stress_um = 0.5*((1.0 - A_D_Old)*Stress_um + (1.0 - A_D_Old.Other())*Stress_um_Other) # {{a_d* e(u_d)}}

avg_Stress_um_ac = 0.5*((1.0 - A_D_Old)*Stress_um_ac + (1.0 - A_D_Old.Other())*Stress_um_ac_Other) # {{a_c* e(u_m/a_c)}}
avg_Stress_ud_ad_ac = 0.5*((1.0 - A_D_Old)*Stress_ud_ad_ac + (1.0 - A_D_Old.Other())*Stress_ud_ad_ac_Other)  # {{a_c* e(u_d*a_d/a_c)}}

grad_a_d           = CoefficientFunction(grad(a_d))
# epsilon_a_d       = CoefficientFunction(0.5*(grad_a_d  + grad_a_d.trans))

grad_a_c          = CoefficientFunction(-grad(a_d))
# epsilon_a_c      = CoefficientFunction(0.5*(grad_a_c  + grad_a_c.trans))

################################################################################
# Helper Functions
################################################################################

v_r_0    = (U_D_0 - U_M_0) # Relative velocity: v_d - v_c
Re_var = rho_c*Norm(v_r_0)*bubble_d/mu_d # Reynolds Variable
#C_D = Max((24/(Re_var + e))*(1.0 + 0.15*(Re_var**0.687)), CoefficientFunction(0.44)) # Drag Coefficient
C_D = 0.44

Div_a_d_v_d      = CoefficientFunction((grad(A_D_Old)*v_d + A_D_Old*div(v_d))) # Div(a_d*v_d)
Div_a_c_v_m      = CoefficientFunction((-grad(A_D_Old)*v_m + (1.0 - A_D_Old)*div(v_m))) # Div(a_c*v_m)

Div_a_d_u_d      = CoefficientFunction((grad(A_D_Old)*u_d + A_D_Old*div(u_d))) # Div(a_d*v_d)
Div_a_c_u_m      = CoefficientFunction((-grad(A_D_Old)*u_m + (1.0 - A_D_Old)*div(u_m))) # Div(a_c*v_m)


################################################################################
# Bilinear form
################################################################################
"""
dx : evaluates integral over elements
dx(skeleteon=True) : evaluates integral over interior element boundaries (facets)
ds(skeleteon=True) : evaluates integral over domain boundaries
"""

a_INS = BilinearForm(X)

################################################################################
# Bilinear Form Disperse Phase Momentum Equation
################################################################################

# Change in Momentum
a_INS +=   A_D_Old * u_d * v_d  * dx

# Pressure
a_INS += - dt * Div_a_d_v_d * p * dx

# Advection
a_INS += - dt * InnerProduct(OuterProduct(U_D_0,A_D_Old*u_d), grad(v_d)) * dx
a_INS +=   dt * jump_vd * (avg_ud_ud_ad * n + 0.5 * Norm(U_D_0 * n) * jump_ud_ad) * dx(skeleton=True)
a_INS +=   dt * v_d * (0.5 * A_D_Old * u_d * (U_D_0 * n) + 0.5 * Norm(U_D_0 * n) * A_D_Old * u_d ) * ds(skeleton=True, definedon=mesh.Boundaries("wall|inlet"))
a_INS +=   dt * v_d * (Max(U_D_0*n, 0.0) * A_D_Old * u_d) * ds(skeleton=True, definedon=mesh.Boundaries("outlet"))

# Viscous Stress
a_INS +=   dt * (2.0 * mu_d) * InnerProduct(A_D_Old * Stress_ud, Stress_vd) * dx
a_INS += - dt * (2.0 * mu_d) * InnerProduct(avg_Stress_vd, OuterProduct(jump_ud_ad, n)) * dx(skeleton=True)
a_INS += - dt * (2.0 * mu_d) * InnerProduct(avg_ad_Stress_ud, OuterProduct(jump_vd, n)) * dx(skeleton=True)
a_INS += - dt * (2.0 * mu_d) * InnerProduct(OuterProduct(u_d*A_D_Old, n), Stress_vd) * ds(skeleton=True, definedon=mesh.Boundaries("wall|inlet"))
a_INS += - dt * (2.0 * mu_d) * InnerProduct(OuterProduct(v_d, n), A_D_Old * Stress_ud) * ds(skeleton=True, definedon=mesh.Boundaries("wall|inlet"))
a_INS +=   dt * (2.0 * mu_d) * beta * InnerProduct(jump_ud_ad, jump_vd) * dx(skeleton=True)
a_INS +=   dt * (2.0 * mu_d) * beta * u_d * A_D_Old * v_d * ds(skeleton=True, definedon=mesh.Boundaries("wall|inlet"))

# Drag Force
a_INS +=   dt * (0.75) * (rho_c) * A_D_Old * (C_D/bubble_d) * Norm(v_r_0) * u_d * v_d * dx
a_INS += - dt * (0.75) * (rho_c) * A_D_Old * (C_D/bubble_d) * Norm(v_r_0) * (u_m) * v_d * dx

a_INS += - dt * InnerProduct (OuterProduct(grad_a_d,u_d), Stress_vd) * dx #IS THIS CORRECT?

################################################################################
# Bilinear Form Continuous Phase Momentum Equation
################################################################################

a_INS +=  ((1.0 - A_D_Old ) * u_m * v_m) * dx

# Pressure
a_INS += - dt * Div_a_c_v_m * p * dx

# Advection
a_INS += - dt * InnerProduct(OuterProduct(U_M_0,(1.0 - A_D_Old)*u_m), grad(v_m)) * dx
a_INS +=   dt * jump_vm * (avg_um_um_ac * n + 0.5 * Norm(U_M_0 * n) * jump_um_ac) * dx(skeleton=True)
a_INS +=   dt * v_m * (0.5 * (1.0 -A_D_Old) * u_m * (U_M_0 * n) + 0.5 * Norm(U_M_0 * n) * (1.0 -A_D_Old) * u_m ) * ds(skeleton=True, definedon=mesh.Boundaries("wall|inlet"))
a_INS +=   dt * v_m * (Max(U_M_0*n, 0.0) * (1.0 - A_D_Old) * u_m) * ds(skeleton=True, definedon=mesh.Boundaries("outlet"))

# Viscous Stress
a_INS +=   dt * (2.0 * mu_c) * InnerProduct((1.0 - A_D_Old) * Stress_um, Stress_vm) * dx
a_INS += - dt * (2.0 * mu_c) * InnerProduct(avg_Stress_vm, OuterProduct(jump_um_ac, n)) * dx(skeleton=True)
a_INS += - dt * (2.0 * mu_c) * InnerProduct(avg_ac_Stress_um, OuterProduct(jump_vm, n)) * dx(skeleton=True)
a_INS += - dt * (2.0 * mu_c) * InnerProduct(OuterProduct(u_m*(1.0 - A_D_Old), n), Stress_vm) * ds(skeleton=True, definedon=mesh.Boundaries("wall|inlet"))
a_INS += - dt * (2.0 * mu_c) * InnerProduct(OuterProduct(v_m, n), (1.0 - A_D_Old) * Stress_um) * ds(skeleton=True, definedon=mesh.Boundaries("wall|inlet"))
a_INS +=   dt * (2.0 * mu_c) * beta * InnerProduct(jump_um_ac, jump_vm) * dx(skeleton=True)
a_INS +=   dt * (2.0 * mu_c) * beta * u_m * (1.0 - A_D_Old) * v_m * ds(skeleton=True, definedon=mesh.Boundaries("wall|inlet"))

# Drag Force
a_INS += - dt * (0.75) * (rho_c) * A_D_Old * (C_D/bubble_d) * Norm(v_r_0) * u_d * v_m * dx
a_INS +=   dt * (0.75) * (rho_c) * A_D_Old * (C_D/bubble_d) * Norm(v_r_0) * (u_m) * v_m * dx

a_INS += - dt * InnerProduct (OuterProduct(grad_a_c, u_m), Stress_vm) * dx # FIX

# Mass Convservation
a_INS += - dt * q * Div_a_d_u_d  * dx
a_INS += - dt * q * Div_a_c_u_m  * dx

################################################################################
#  Linear Form Disperse Phase Momentum Equation
################################################################################

f_INS = LinearForm(X)

# Change in Momentum
f_INS +=    A_D_Old * U_D_0*v_d * dx

# Advection
f_INS +=  - dt * v_d * ( 0.5 * U_D_0 * n * (ad_bnd * ud_bnd) - 0.5 * Norm(U_D_0 * n) * (ad_bnd * ud_bnd) ) * ds(skeleton=True, definedon=mesh.Boundaries("inlet"))
f_INS +=  - dt * v_d * ( 0.5 * U_D_0 * n * (ad_wall_bnd * ud_wall_bnd) - 0.5 * Norm(U_D_0 * n) * (ad_wall_bnd * ud_wall_bnd) ) * ds(skeleton=True, definedon=mesh.Boundaries("wall"))
f_INS +=  - dt * v_d * neumann_bnd * ds(skeleton=True, definedon=mesh.Boundaries("outlet"))

# Viscous Stress

f_INS +=    dt * (2.0 * mu_d) * beta * ad_bnd * ud_bnd * v_d * ds(skeleton=True, definedon=mesh.Boundaries("inlet"))
f_INS +=    dt * (2.0 * mu_d) * beta * ad_wall_bnd * ud_wall_bnd * v_d * ds(skeleton=True, definedon=mesh.Boundaries("wall"))

f_INS +=  - dt * (2.0 * mu_d) * InnerProduct(OuterProduct(ad_bnd * ud_bnd, n), Stress_vd) * ds(skeleton=True, definedon=mesh.Boundaries("inlet"))
f_INS +=  - dt * (2.0 * mu_d) * InnerProduct(OuterProduct(ad_wall_bnd * ud_wall_bnd, n), Stress_vd) * ds(skeleton=True, definedon=mesh.Boundaries("wall"))

# Gravity
a_INS +=   -dt * A_D_Old * grav * v_d * dx
# Gravity
a_INS +=  -dt * (1.0 - A_D_Old) * grav * v_m * dx

################################################################################
# Linear Form Continuous Phase Momentum Equation
################################################################################

# Change in Momentum
f_INS +=    (1.0 - A_D_Old) * U_M_0*v_m * dx

# Advection
f_INS +=  - dt * v_m * ( 0.5 * U_M_0 * n * ((1.0 - ad_bnd) * um_bnd) - 0.5 * Norm(U_M_0 * n) * ((1.0 - ad_bnd) * um_bnd) ) * ds(skeleton=True, definedon=mesh.Boundaries("inlet"))
f_INS +=  - dt * v_m * ( 0.5 * U_M_0 * n * ((1.0 - ad_wall_bnd) * um_wall_bnd) - 0.5 * Norm(U_M_0 * n) * ((1.0 - ad_wall_bnd) * um_wall_bnd) ) * ds(skeleton=True, definedon=mesh.Boundaries("wall"))
f_INS +=  - dt * v_m * neumann_bnd * ds(skeleton=True, definedon=mesh.Boundaries("outlet"))

# Viscous Stress

f_INS +=    dt * (2.0 * mu_c) * beta * (1.0 - ad_bnd) * um_bnd * v_m * ds(skeleton=True, definedon=mesh.Boundaries("inlet"))
f_INS +=    dt * (2.0 * mu_c) * beta * (1.0 - ad_wall_bnd) * um_wall_bnd * v_m * ds(skeleton=True, definedon=mesh.Boundaries("wall"))

f_INS +=  - dt * (2.0 * mu_c) * InnerProduct(OuterProduct((1.0 - ad_bnd) * um_bnd, n), Stress_vm) * ds(skeleton=True, definedon=mesh.Boundaries("inlet"))
f_INS +=  - dt * (2.0 * mu_c) * InnerProduct(OuterProduct((1.0 - ad_wall_bnd) * um_wall_bnd, n), Stress_vm) * ds(skeleton=True, definedon=mesh.Boundaries("wall"))

################################################################################
# DG Method for updating Disperse Phase Fraction
################################################################################
a_alpha_d  = BilinearForm(A)

a_alpha_d += a_d*z * dx
a_alpha_d += -dt * (a_d * U_D_0*grad(z)) * dx
a_alpha_d += dt *(z-z.Other())*(U_D_0*n*0.5*(a_d + a_d.Other()) + 0.5*Norm(U_D_0*n)*(a_d - a_d.Other()))* dx(skeleton=True)
a_alpha_d += dt * z * ( 0.5*U_D_0*n*a_d + 0.5*Norm(U_D_0*n)*a_d)* ds(skeleton=True, definedon=mesh.Boundaries("inlet|wall"))
a_alpha_d += dt * z * a_d * Max(U_D_0*n, 0) * ds(skeleton=True, definedon=mesh.Boundaries("outlet"))

# a_alpha_d +=   dt * D * InnerProduct(grad(a_d),  grad(z)) * dx
# a_alpha_d += - dt * D * InnerProduct(0.5*(grad(z) + grad(z).Other()), OuterProduct(a_d - a_d.Other(), n)) * dx(skeleton=True)
# a_alpha_d += - dt * D * InnerProduct(0.5*(grad(a_d) + grad(a_d).Other()), OuterProduct(z- z.Other(), n)) * dx(skeleton=True)
# a_alpha_d += - dt * D * InnerProduct(OuterProduct(a_d, n), grad(z)) * ds(skeleton=True, definedon=mesh.Boundaries("wall|inlet"))
# a_alpha_d += - dt * D * InnerProduct(OuterProduct(z, n), grad(a_d)) * ds(skeleton=True, definedon=mesh.Boundaries("wall|inlet"))
# a_alpha_d +=   dt * D * beta * InnerProduct(a_d - a_d.Other(), z - z.Other()) * dx(skeleton=True)
# a_alpha_d +=   dt * D * beta * a_d * z * ds(skeleton=True, definedon=mesh.Boundaries("wall|inlet"))


f_alpha_d = LinearForm(A)

f_alpha_d += dt*force_ad*z * dx

f_alpha_d += A_D_Old*z * dx
f_alpha_d += - dt * z * (0.5*U_D_0*n*ad_bnd - 0.5*Norm(U_D_0*n)*ad_bnd)* ds(skeleton=True, definedon=mesh.Boundaries("inlet|wall"))
f_alpha_d += - dt * z * disperse_phase_neumannn_bnd * n * ds(skeleton=True, definedon=mesh.Boundaries("outlet"))

# f_alpha_d +=    dt * D * beta * ad_bnd * z * ds(skeleton=True, definedon=mesh.Boundaries("inlet"))
# f_alpha_d +=    dt * D * beta * ad_wall_bnd * z * ds(skeleton=True, definedon=mesh.Boundaries("wall"))
#
# f_alpha_d +=  - dt * D * InnerProduct(OuterProduct(ad_bnd, n),  grad(z)) * ds(skeleton=True, definedon=mesh.Boundaries("inlet"))
# f_alpha_d +=  - dt * D * InnerProduct(OuterProduct(ad_wall_bnd, n), grad(z)) * ds(skeleton=True, definedon=mesh.Boundaries("wall"))


################################################################################
# Implicit Time-stepping
################################################################################
vtk = VTKOutput(ma=mesh,coefs=[UN.components[0][0], UN.components[0][1], UN.components[1][0], UN.components[1][1], UN.components[0], UN.components[1], UN.components[2], A_D ],names=["vel_g_x", "vel_g_y", "vel_l_x", "vel_l_y", "vel_g_mag", "vel_l_mag" "pressure", "alpha_d"],filename="2FF",subdivision=3)

store = []
def solve_tfm():
    with TaskManager():

        Draw(U_C_0[0], mesh, "Continuous_Phase_Velocity_X")
        Draw(U_C_0[1], mesh, "Continuous_Phase_Velocity_Y")
        Draw(U_D_0[0], mesh, "Dispersed_Phase_Velocity_X")
        Draw(U_D_0[1], mesh, "Dispersed_Phase_Velocity_Y")
        Draw(Norm(P_0), mesh, "Pressure")
        Draw(A_D_Old, mesh, "Dispersed_Phase_Fraction")

        pre_INS = Preconditioner(type="direct", bf=a_INS, flags = {"inverse" : "umfpack" } )
        pre_INS_disperse_phase = Preconditioner(type="direct", bf=a_alpha_d, flags = {"inverse" : "umfpack" } )


        UN.components[0].Set(CoefficientFunction((0.0,0.0)))
        UN.components[1].Set(CoefficientFunction((0.0,0.0)))
        A_D.Set(CoefficientFunction(0.025))

        UN.components[0].Set(um_bnd, definedon=mesh.Boundaries("inlet")) # Mixture Velocity Inlet Condition: May depend on time
        UN.components[1].Set(ud_bnd, definedon=mesh.Boundaries("inlet")) # Dispersed Velocity Inlet Condition: May depend on time

        UOld.vec.data    = UN.vec # U^N = U^N+1
        A_D_Old.vec.data = A_D.vec # a_d^N = a_d^N+1

        t       = 0.0 # Initial time
        step    = 0 # Iteration step counter
        # vtk.Do()
        while t <= final_time:
            step += 1
            t += float(dt)
            param_t.Set(t)

            print( 'Time step:  ', step , '  time:  ',t )

            # UN.components[0].Set(um_bnd, definedon=mesh.Boundaries("inlet")) # Mixture Velocity Inlet Condition: May depend on time
            # UN.components[1].Set(ud_bnd, definedon=mesh.Boundaries("inlet")) # Dispersed Velocity Inlet Condition: May depend on time

            #Solve for u_m, u_d, and p
            a_INS.Assemble() # Build mass matrix
            f_INS.Assemble() # Build vector
            pre_INS.Update() # Update preconditioner
            BVP(bf=a_INS,lf=f_INS,gf=UN,pre=pre_INS,maxsteps=5,prec=1e-30).Do() # Solve linear system
            UOld.vec.data  = UN.vec # Update Velocity solution

            #Solve for disperse phase fraction
            a_alpha_d.Assemble() # Build mass matrix
            f_alpha_d.Assemble() # Build vector
            pre_INS_disperse_phase.Update() # Update preconditioner
            BVP(bf=a_alpha_d,lf=f_alpha_d,gf=A_D,pre=pre_INS_disperse_phase,maxsteps=5,prec=1e-30).Do() # Solve linear system
            A_D_Old.vec.data = A_D.vec # Update vector solution

            # if step % 10 == 0:
            #     vtk.Do()   # Output Solution as .vtk file
                # Redraw()    # Revisualize Solution
            Redraw()

    err_u, err_phase, err_div = CalcL2Error(UN, A_D)
    store.append ( (X.ndof, mesh.ne, err_u, err_phase, err_div) )


for i in range(No_Refinments):

    # Refine the mesh
    if i != 1:
        mesh.Refine()

    # Update the mixed function space and grid functions if mesh has changed
    X.Update()
    A.Update()

    UN.Update()
    UOld.Update()

    A_D.Update()
    A_D_Old.Update()


    # Solve the Navier-Stoke system with the new mesh
    solve_tfm()


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

print("--- %s seconds ---\n\n" % (time.time() - start_time))
