# Interactive Immersed Boundary Method

Browser-based immersed boundary (IB) simulations using WebGPU. The app contains:

- A production-oriented **2D IB simulator** with an app-local optimized WebGPU solver.
- An experimental **3D IB simulator** at `/3d` with a WebGPU fluid solve, signed vector-field slices, volume rendering, and three.js camera/picking.
- The original jax-js 2D solver kept as a reference/fallback path.

**[Live Demo](https://guanhuasun.github.io/ib-sim/)**

An elastic structure is immersed in a viscous, incompressible fluid on a periodic unit domain. The structure exerts elastic forces on the fluid through Peskin regularized delta functions, and the fluid velocity advects the structure. This two-way coupling is the core idea of Peskin's immersed boundary method.

## Solver Architecture

The main hot loops are now app-local WebGPU/WGSL kernels rather than general jax-js array programs.

- **2D optimized path:** `src/routes/webgpu-ib-solver.ts` is the default solver for the 2D page. It keeps simulation state in GPU buffers and runs custom WGSL kernels for force computation, spreading, interpolation, FFT-based fluid solves, mouse forcing, and visualization-field generation.
- **2D rendering:** `src/routes/webgpu-ib-renderer.ts` draws the scalar field and instanced boundary segments/points, including periodic images. The solver caches both ping-pong dispatch schedules; a displayed frame shares one command encoder for simulation, visualization, and drawing. Unchanged paused frames skip GPU work.
- **2D reference path:** `src/routes/ib-solver.ts` is the original jax-js implementation. It remains available as a numerical reference/fallback and can be forced with `?solver=reference`.
- **3D path:** `src/routes/ib3d-solver.ts` uses raw WebGPU compute pipelines for a 3D velocity grid, edge-spring structure forces, 3D Peskin coupling, 3D FFT passes, incompressible Fourier projection, and full velocity/vorticity vector fields. `src/routes/3d/renderer.ts` uses three.js OrbitControls, camera matrices and ray picking with direct WebGPU drawing of the solver buffers. Only a body pick reads mesh positions back; the frame loop never copies the fluid grid to the CPU.
- **jax-js role:** jax-js is still in the project. It provides the retained 2D reference solver and is currently used by the pages to initialize/access the WebGPU device. The active 2D optimized solver and 3D solver do not use jax-js arrays or jax-js FFTs in their fluid-solve hot loops.

## Numerical Method

### 2D

The 2D scheme follows the MATLAB implementation in `ib_matlab_2D` (C.S. Peskin):

- **Fluid:** 2-stage IMEX time integration. Advection and forcing are explicit; viscous diffusion is implicit through a Fourier-space solve.
- **Structure:** Elastic spring forces `F_k = K(X_(k+1) + X_(k-1) - 2X_k)/Δθ²` with periodic indexing and a midpoint predictor step.
- **Coupling:** Spread and interpolation use Peskin's 4-point regularized delta function. Structural coordinates remain unwrapped; grid indices wrap correctly after repeated crossings in either direction.

### 3D

The 3D page extends the same IB ideas by adding one spatial axis:

- **Fluid:** The same two-stage IMEX scheme as 2D: midpoint velocity, centered skew-symmetric advection, implicit half-step diffusion, and Crank–Nicolson full-step diffusion. The skew stencil divides by `4h` in any dimension (the older MATLAB 3D `6h` coefficient is not used). Projection uses the centered-difference Fourier symbol and the seven-point Laplacian. Density is fixed at one, as in the default 2D setup; there is no extra damping multiplier.
- **Structure:** Midpoint prediction and full-step interpolation follow the 2D sequence. Material coordinates remain unwrapped, including across repeated periodic crossings. The triangulated shell uses the edge-spring energy `K/2 Σ_edges (|X_j-X_i| - L_ij)²`, where each fixed rest length `L_ij` is its edge length on the initial sphere. The initial shell is stress-free; subsequent stretching and compression produce restoring forces. Its nodal forces are integrated forces, unlike the 2D force density per parameter interval. No additional surface-area factor is applied when spreading these nodal forces. This is a surface mesh, so its connectivity and stiffness convention differ from a 1D closed curve.
- **Coupling:** Tensor-product Peskin 4-point interpolation (64 grid neighbors) and spreading with `h^-3`. Only coupling and display coordinates wrap; material edges stay continuous.
- **Visualization:** Select a signed x/y/z component or magnitude of velocity or vorticity. A movable XY/XZ/YZ slice provides a color scale and full 3D vector arrows. Volume mode shows the selected scalar throughout the cube. The shell has adjustable transparency and clipped periodic images; volume/shell compositing is illustrative, not an exact transparency sort for self-intersecting surfaces.
- **Interaction:** Left-drag orbits; right-drag automatically picks/pulls the shell or stirs the visible slice. Shift-right-drag always stirs through the shell. Scroll or middle-drag zooms in every mode; Shift-scroll moves the slice. With the canvas focused, Space plays/pauses, Escape releases the interaction, and Home restores the camera. Optional Auto/Stir fluid/Pull body buttons choose the interaction target; on touch screens, Auto uses one-finger orbit and pinch zoom, with buttons enabling fluid/body interaction. Body pulling uses a capped tether force on a camera-facing plane. The tether enters the usual force/spread/fluid/interpolate sequence and clears on release, cancellation, focus loss, reset, target change, or rebuild.

The 3D solver is experimental and intended for local iteration before remote publication.

The 3D initial velocity is `u = 0.2 (sin(2πz), 0, sin(2πy))`. The default playback advances six steps per displayed frame with `dt = 0.002`; the Steps per frame control adjusts playback speed without changing the time step.

## Features

- Real-time 2D simulation at 64x64 or 128x128 grid resolution.
- Experimental 3D simulation at 64^3 (default) or 32^3 grid resolution.
- FFT-based incompressible fluid solves on the GPU.
- Interactive mouse/stir forcing.
- Vorticity and velocity-magnitude visualization modes.
- 2D colormaps and optional smoothing.
- 3D signed slices, vector arrows, volume rendering, and plane-based fluid/body dragging.
- Adjustable stiffness, viscosity, timestep, visualization scale, and opacity.
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

For a local production preview, run `npm run build` followed by `npm run preview -- --host 127.0.0.1`.

## Useful Query Parameters

- `?solver=optimized` - use the default 2D optimized WebGPU solver.
- `?solver=reference` - force the 2D jax-js reference solver.
- `?perf=1` - log FPS and CPU command-encoding times (not GPU execution times).

## Project Structure

```text
src/routes/
  +page.svelte          # 2D UI, solver selection, animation loop
  webgpu-ib-solver.ts   # Optimized 2D raw WebGPU solver
  webgpu-ib-renderer.ts # Scalar field and periodic instanced boundary rendering
  ib-solver.ts          # Original jax-js 2D reference/fallback solver
  ib3d-solver.ts        # Experimental 3D raw WebGPU solver
  3d/+page.svelte       # 3D UI, gestures, animation loop
  3d/renderer.ts        # Three.js camera/picking, WebGPU slices, vectors, mesh and volume
jax-js/                 # Git submodule: jax-js WebGPU array library
```

## Development Notes

`node scripts/check-2d.mjs` runs Chrome/WebGPU regression checks at N=64/128: repeated periodic crossings, spread/interpolation consistency against jax-js, batch versus individual steps through 1,000 steps, shared-encoder submission, and reset from either ping-pong state. It requires Chrome and Playwright tooling; set `PLAYWRIGHT_MODULE` to an existing Playwright `index.mjs` path when it is installed outside this project. The script starts and closes its own local Vite server.

`node scripts/check-3d.mjs` uses the same browser setup. It compares a z-invariant 3D flow against the unchanged 2D jax-js fluid solver for ten steps at N=8/16; checks analytic vector curl and 100-step viscous decay; checks independent Peskin weights, force conservation and the spread/interpolate adjoint identity after multiple periodic crossings; and checks material forces, 101-step batch parity, divergence, odd/even reset, shared encoding, tether response and a 1,000-step finite-state run at N=16/32/64. It also exercises the browser's orbit, slice stirring, shell picking, grid rebuild and display modes, saving desktop and mobile screenshots under ignored `output/playwright/`. These regression checks do not establish mesh convergence or stability for every selectable parameter combination.

Local before/after measurements using completed GPU work (five alternating, warmed runs) reduced median 100-step batch time from 14.4 to 7.7 ms at N=64 and 19.6 to 11.1 ms at N=128. At a 512x512 canvas, rendering fell from about 0.30/0.57 ms to 0.10 ms per frame. These are machine-specific measurements, not performance guarantees; the page still advances one step per animation frame. Solver states matched the previous optimized implementation exactly through 1,000 steps in these tests.

- Prefer the optimized WebGPU solver for performance-sensitive work.
- Keep `src/routes/ib-solver.ts` as the reference path unless intentionally changing the 2D numerical baseline.
- The 3D page currently uses rest-length edge springs on the triangulated shell; future work may add bending, area, or volume constraints.
- Run `npm run check` and `npm run build` for TypeScript/Svelte and production-build validation.

## References

- C.S. Peskin, *The immersed boundary method*, Acta Numerica 11 (2002), 479-517.
- C.S. Peskin, [IB MATLAB 2D code](http://www.math.nyu.edu/~peskin/ib_lecture_notes/)
- E. Zhang, [jax-js](https://github.com/ekzhang/jax-js)
