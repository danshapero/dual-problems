import numpy as np
import firedrake
from firedrake import Constant, assemble, inner, dot, tr, grad, div, dx, ds
import icepack

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nx", type=int, default=32)
args = parser.parse_args()

# TODO: Fix the units to MPa - m - yr
ρ_I = Constant(icepack.constants.ice_density)
ρ_W = Constant(icepack.constants.water_density)
g = Constant(icepack.constants.gravity)
n = Constant(icepack.constants.glen_flow_law)
T = Constant(260.0)
A = icepack.rate_factor(T)

nx, ny = args.nx, args.nx
Lx, Ly = Constant(20e3), Constant(20e3)
mesh = firedrake.RectangleMesh(nx, ny, float(Lx), float(Ly), diagonal="crossed")
d = mesh.geometric_dimension()

inflow_ids = (1,)
outflow_ids = (2,)
side_wall_ids = (3, 4)

x = firedrake.SpatialCoordinate(mesh)

u_0 = Constant(100.0)
h_0, dh = Constant(500.0), Constant(100.0)
ρ = ρ_I * (1 - ρ_I / ρ_W)
ζ = A * (ρ * g * h_0 / 4)**n
ψ = 1 - (1 - (dh / h_0) * (x[0] / Lx))**(n + 1)
du = ζ * ψ * Lx * (h_0 / dh) / (n + 1)
U_exact = firedrake.as_vector((u_0 + du, 0.0))

cg1 = firedrake.FiniteElement("CG", "triangle", 1)
Q = firedrake.FunctionSpace(mesh, cg1)
V = firedrake.VectorFunctionSpace(mesh, cg1)
b3 = firedrake.FiniteElement("B", "triangle", 3)
Σ = firedrake.TensorFunctionSpace(mesh, cg1 + b3, symmetry=True)

h = firedrake.interpolate(h_0 - dh * x[0] / Lx, Q)

Z = V * Σ
z = firedrake.Function(Z)
u, M = firedrake.split(z)

ν = firedrake.FacetNormal(mesh)

# Picard linearization of the problem
power = h * A / 4 * (inner(M, M) - tr(M)**2 / (d + 1)) * dx
constraint = inner(u, div(h * M) - 0.5 * ρ * g * grad(h**2)) * dx

# TODO: Fix this! Use eqn BCs on the side walls & outflow to fix `M`
inflow_ids = (1, 2, 3, 4)
u_in = U_exact
#u_in = Constant((u_0, 0.0))
boundary = h * inner(dot(M, ν), u_in) * ds(inflow_ids)

L = power + constraint - boundary
F = firedrake.derivative(L, z)
params = {
    "solver_parameters": {
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
}
firedrake.solve(F == 0, z, **params)

# Full nonlinear problem
# FIXME: factors of 2 are probably off
power = h * A / (2**n * (n + 1)) * (inner(M, M) - tr(M)**2 / (d + 1))**((n + 1) / 2) * dx
constraint = inner(u, div(h * M) - 0.5 * ρ * g * grad(h**2)) * dx
boundary = h * inner(dot(M, ν), u_in) * ds(inflow_ids)
L = power + constraint - boundary

F = firedrake.derivative(L, z)
params = {
    "solver_parameters": {
        "snes_type": "newtontr",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
}
firedrake.solve(F == 0, z, **params)

u, M = z.split()
u_exact = firedrake.interpolate(U_exact, V)
error = firedrake.norm(u - u_exact) / firedrake.norm(u_exact)
print(f"Relative error: {error:5.4f}")
