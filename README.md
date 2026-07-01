# Interactive Immersed Boundary Method

Browser-based immersed boundary (IB) simulations using WebGPU. The app contains:

- A production-oriented **2D IB simulator** with an app-local optimized WebGPU solver.
- An experimental **3D IB simulator** at `/3d` with a raw WebGPU 3D fluid solve and volume rendering.
- The original jax-js 2D solver kept as a reference/fallback path.

**[Live Demo](https://guanhuasun.github.io/ib-sim/)**

An elastic structure is immersed in a viscous, incompressible fluid on a periodic unit domain. The structure exerts elastic forces on the fluid through Peskin regularized delta functions, and the fluid velocity advects the structure. This two-way coupling is the core idea of Peskin's immersed boundary method.

## Solver Architecture

The main hot loops are now app-local WebGPU/WGSL kernels rather than general jax-js array programs.

- **2D optimized path:** `src/routes/webgpu-ib-solver.ts` is the default solver for the 2D page. It keeps simulation state in GPU buffers and runs custom WGSL kernels for force computation, spreading, interpolation, FFT-based fluid solves, mouse forcing, and visualization-field generation.
- **2D reference path:** `src/routes/ib-solver.ts` is the original jax-js implementation. It remains available as a numerical reference/fallback and can be forced with `?solver=reference`.
- **3D path:** `src/routes/ib3d-solver.ts` uses raw WebGPU compute pipelines for a 3D velocity grid, edge-spring structure forces, 3D Peskin coupling, 3D FFT passes, incompressible Fourier projection, and velocity/vorticity scalar-field generation.
- **jax-js role:** jax-js is still in the project. It provides the retained 2D reference solver and is currently used by the pages to initialize/access the WebGPU device. The active 2D optimized solver and 3D solver do not use jax-js arrays or jax-js FFTs in their fluid-solve hot loops.

## Numerical Method

### 2D

The 2D scheme follows the MATLAB implementation in `ib_matlab_2D` (C.S. Peskin):

- **Fluid:** 2-stage IMEX time integration. Advection and forcing are explicit; viscous diffusion is implicit through a Fourier-space solve.
- **Structure:** Elastic spring forces with periodic indexing and a midpoint predictor step.
- **Coupling:** Spread and interpolation use Peskin's 4-point regularized delta function.

### 3D

The 3D page extends the same IB ideas by adding one spatial axis:

- **Fluid:** Periodic 3D velocity grid with a spectral incompressible solve.
- **Structure:** A triangulated sphere with edge-spring forces on the mesh graph.
- **Coupling:** 3D Peskin 4-point interpolation and spreading.
- **Visualization:** Direct WebGPU raymarching of velocity magnitude or vorticity magnitude, with a semi-transparent structure shell so the interior fluid field is visible.

The 3D solver is experimental and intended for local iteration before remote publication.

## Features

- Real-time 2D simulation at 64x64 or 128x128 grid resolution.
- Experimental 3D simulation at 16^3 or 32^3 grid resolution.
- FFT-based incompressible fluid solves on the GPU.
- Interactive mouse/stir forcing.
- Vorticity and velocity-magnitude visualization modes.
- 2D colormaps and optional smoothing.
- 3D raymarched fluid volume rendering.
- Adjustable stiffness, viscosity, timestep, visualization scale, and render sampling.
- jax-js reference/fallback solver retained for comparison.

## Running Locally

```bash
git clone --recurse-submodules https://github.com/guanhuasun/ib-sim.git
cd ib-sim
npm install
npm run dev
```

Open:

- `http://127.0.0.1:5173/` for the 2D simulator.
- `http://127.0.0.1:5173/3d` for the experimental 3D simulator.

Requires a browser with WebGPU support, such as current Chrome or Edge.

## Useful Query Parameters

- `?solver=optimized` - use the default 2D optimized WebGPU solver.
- `?solver=reference` - force the 2D jax-js reference solver.
- `?perf=1` - log 2D solver/render timing information.

## Project Structure

```text
src/routes/
  +page.svelte          # 2D UI, WebGPU rendering, solver selection, animation loop
  webgpu-ib-solver.ts   # Optimized 2D raw WebGPU solver
  ib-solver.ts          # Original jax-js 2D reference/fallback solver
  ib3d-solver.ts        # Experimental 3D raw WebGPU solver
  3d/+page.svelte       # 3D UI, mesh rendering, volume rendering, animation loop
jax-js/                 # Git submodule: jax-js WebGPU array library
```

## Development Notes

- Prefer the optimized WebGPU solver for performance-sensitive work.
- Keep `src/routes/ib-solver.ts` as the reference path unless intentionally changing the 2D numerical baseline.
- The 3D page currently uses edge-spring graph forces on the triangulated shell; future work may add rest-length springs, bending, area, or volume constraints.
- `npm run build` is the primary build check. `npm run check` currently reports an existing Vite/Node typings issue around `node:path` and `__dirname`.

## References

- C.S. Peskin, *The immersed boundary method*, Acta Numerica 11 (2002), 479-517.
- C.S. Peskin, [IB MATLAB 2D code](http://www.math.nyu.edu/~peskin/ib_lecture_notes/)
- E. Zhang, [jax-js](https://github.com/ekzhang/jax-js)
