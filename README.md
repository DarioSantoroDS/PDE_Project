# Fluid–Structure Interaction (FSI) — Monolithic Solver (deal.II)

This project implements a **monolithic solver** for a steady **fluid–structure interaction (FSI)** problem. The computational domain is split into a fluid region and a solid region, coupled along the interface Σ.


This is part of a project-work for Numerical Methods for Partial Differential Equations course [@Polimi](https://www.polimi.it/)

### Authors
Project developed by:
- [Infascelli Riccardo](https://github.com/RiccardoInfascelli)
- [Francesco Rosnati](https://github.com/RosNaviGator)
- [Santoro Dario](https://github.com/DarioSantoroDS)

## Overview

The mathematical model is:
$$
\begin{cases}
-\nu \Delta u + \nabla p = 0 & \text{in } \Omega_{\text{fluid}}, \\
\nabla \cdot u = 0 & \text{in } \Omega_{\text{fluid}}, \\
-\nabla \cdot \sigma(d) = 0 & \text{in } \Omega_{\text{solid}}, \\
u = 0 & \text{on } \Sigma, \\
\sigma(d)n = \nu \nabla un - pn & \text{on } \Sigma.
\end{cases}
$$

Under the small-displacement assumption, the fluid velocity is forced to vanish on the fluid–solid interface.<!-- TO CHECK -->


## Main Features
- hp-FEM discretization with different finite elements for fluid and solid
- One-way coupled monolithic formulation
- Interface integration on faces and subfaces <!-- TO CHECK, SEEMS USELESS -->
- Constraint handling for boundary conditions and fluid–solid continuity 
- Output in **VTU/PVTU** (ParaView)
- Export of system matrix and RHS in **MATLAB** format (via PETSc)
- Direct solver **UMFPACK** (via PETSc)

## Project Structure

- `fsi.cpp` : entry point of the program, sets up and runs the FSI solver.
- `FluidStructureProblem.hpp` : header file defining the `FluidStructureProblem` class and its interface.
- `FluidStructureProblem.cpp` : implementation of the FSI solver including mesh generation, DOF setup, system assembly, interface treatment, solution, and output.


## Build and Run

```bash
mkdir build
cd build
cmake ..
make 
./fsi n_elements_per_edge
```

## Generated Files

- `solution_00.pvtu` : ParaView-readable file with the velocity, pressure, and displacement fields; also present the material_id and subdomain_id fields.
- `system_matrix.m` : ASCII file of the system matrix in MATLAB format.
- `system_rhs.m` : ASCII file of the right-hand side vector in MATLAB format.


## Implementation Notes

- **Fluid equations (Stokes):**

$$
  -\nu \Delta \mathbf{u} + \nabla p = 0, \quad \nabla \cdot \mathbf{u} = 0 \quad \text{in } \Omega_\text{fluid}
$$

- **Solid equations (linear elasticity):**

$$
  - \nabla \cdot \sigma(\mathbf{d}) = 0 \quad \text{in } \Omega_\text{solid}
$$

- **Interface conditions at Σ:**

$$
  \mathbf{u} = 0, \quad \sigma(\mathbf{d}) \mathbf{n} = \nu (\nabla \mathbf{u}) \mathbf{n} - p \mathbf{n}
$$

- **Boundary conditions:**
  - Fluid: Dirichlet on top for inflow and outflow, Neumann on right and left.
  - Solid: Homogeneous Dirichlet.

- **Solution procedure:**
  - hp-Finite Elements to handle different FE spaces in fluid and solid domains.
  - Constraints to enforce zero velocity at the fluid-solid interface.
  - PETSc or Trilinos direct solvers for the linear system.
  - Trilinos iterative solvers for larger problems.
  - Parallel MPI distribution of DOFs and matrices.

- **Output:**
  - Files can be visualized in ParaView.

- **Mesh generation:**
  - Subdivided hypercube from -1 to 1 in both directions, with `n_elements_per_edge` elements per edge.
  - Cells marked with `fluid_domain_id` or `solid_domain_id` based on location.
  - Boundary IDs set for Dirichlet and Neumann boundary conditions.

- **System assembly:**
  - Separate contributions for Stokes and elasticity.
  - Special assembly along the fluid-solid interface.
  - Interface terms ensure correct stress transfer between fluid and solid.


## References

- deal.II step-46 and multiphysics tutorials
- Classical FSI theory for small displacement monolithic coupling


<!-- ## Notes

This is a simplified FSI mini-base project meant to demonstrate a working monolithic solver with parallel capability. It can be extended with:
- Adaptive mesh refinement
- Higher-order elements
- Time-dependent simulations -->

