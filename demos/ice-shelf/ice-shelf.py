import firedrake
from firedrake import Constant, assemble, inner, grad, div, dx, ds

# TODO: Fix the units to MPa - m - yr
ρ_I = Constant(917.0)
ρ_W = Constant(1024.0)
g = Constant(9.81)
n = Constant(3.0)
A = Constant(50.0)

nx, ny = 32, 32
Lx, Ly = 20e3, 20e3
mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly, diagonal="crossed")

x = firedrake.SpatialCoordinate(mesh)

cg1 = firedrake.FiniteElement("CG", "triangle", 1)
V = firedrake.VectorFunctionSpace(mesh, cg1)
b3 = firedrake.FiniteElement("B", "triangle", 3)
Σ = firedrake.TensorFunctionSpace(mesh, cg1 + b3, symmetry=True)

Z = V * Σ
z = firedrake.Function(Z)

ν = firedrake.FacetNormal(mesh)

power = h * A / (2**n * (n + 1)) * (inner(M, M) - tr(M)**2 / (d + 1))**((n + 1) / 2) * dx
constraint = inner(u, div(h * M) - ρ_I * g * (1 - ρ_I / ρ_W) * grad(h**2)) * dx
boundary = h * inner(dot(M, ν), u_in) * ds(inflow_ids)
L = power + constraint + boundary
