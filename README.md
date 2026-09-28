# Interactive 2D Immersed Boundary Method

Interactive simulation of the 2D immersed boundary (IB) method running entirely in the browser via WebGPU.

**[Live Demo](https://guanhuasun.github.io/ib-sim/)**

An elastic membrane (closed curve) is immersed in a viscous, incompressible fluid on a periodic domain $[0,1]^2$. The fluid and structure are coupled through regularized delta functions: the membrane exerts elastic forces on the fluid, and the fluid velocity advects the membrane. This two-way coupling is the core idea of Peskin's immersed boundary method.

## Method

The numerical scheme follows the MATLAB implementation in `ib_matlab_2D` (C.S. Peskin):

- **Fluid solver**: 2-stage IMEX (implicit-explicit) time integration. Advection and forcing are treated explicitly; viscous diffusion is handled implicitly via a Fourier-space solve. The 2D FFT is composed from two 1D FFTs.
- **Structure solver**: Elastic spring forces $F_k = K(X_{k+1} + X_{k-1} - 2X_k)/\Delta\theta^2$ with periodic indexing. A midpoint predictor step improves temporal accuracy.
- **Coupling**: Spread (structure $\to$ fluid) and interpolation (fluid $\to$ structure) use Peskin's 4-point regularized delta function. Structural coordinates remain unwrapped; grid indices wrap correctly after repeated crossings in either direction.

The default solver uses app-local WebGPU/WGSL kernels for coupling, FFT-based fluid solves, and visualization. [jax-js](https://github.com/ekzhang/jax-js) provides device initialization and the retained 2D reference/fallback solver; the optimized simulation does not use jax-js array operations in its hot loop.

The solver caches both ping-pong dispatch schedules and shares one command encoder for simulation, visualization, and drawing. The renderer draws the field and instanced boundary segments/points with periodic copies. Unchanged paused frames skip GPU work.

## Features

- Real-time simulation at 64x64 or 128x128 grid resolution
- FFT-based implicit viscosity (unconditionally stable for diffusion)
- Interactive mouse drag to apply localized body forces
- Multiple diverging colormaps (Cyan-Magenta, Teal-Orange, Red-Blue, Purple-Orange, Brown-Teal, Coolwarm)
- Vorticity and velocity magnitude visualization modes
- Optional bilinear upsampling for smoother rendering
- Periodic boundary wrapping for the immersed structure
- Adjustable parameters: stiffness $K$, viscosity $\mu$, time step $\Delta t$

## Running locally

```bash
git clone --recurse-submodules https://github.com/guanhuasun/ib-sim.git
cd ib-sim
npm install
npm run dev
```

Requires a WebGPU-capable browser such as Chrome or Edge.

For a local production preview, run `npm run build` followed by `npm run preview -- --host 127.0.0.1`. Nothing is published by these commands.

## Comparison and Checks

- `?solver=optimized` selects the default WebGPU solver.
- `?solver=reference` forces the jax-js reference solver.
- `?perf=1` logs FPS and CPU command-encoding times, not GPU execution times.

`node scripts/check-2d.mjs` runs Chrome/WebGPU regression checks at N=64/128: repeated periodic crossings, spread/interpolation consistency against jax-js, batch versus individual steps through 1,000 steps, shared-encoder submission, and reset from either ping-pong state. It requires Chrome and Playwright tooling; set `PLAYWRIGHT_MODULE` to an existing Playwright `index.mjs` path when it is installed outside this project. The script starts and closes its own local Vite server.

Local before/after measurements using completed GPU work (five alternating, warmed runs) reduced median 100-step batch time from 14.4 to 7.7 ms at N=64 and 19.6 to 11.1 ms at N=128. At a 512x512 canvas, rendering fell from about 0.30/0.57 ms to 0.10 ms per frame. These are machine-specific measurements, not performance guarantees; the page still advances one step per animation frame. Solver states matched the previous optimized implementation exactly through 1,000 steps in these tests.

`npm run build` passes. `npm run check` currently reports existing Node typing errors for `node:path` and `__dirname` in `vite.config.ts`.

## Project structure

```
src/routes/
  +page.svelte          # UI, solver selection, animation loop
  webgpu-ib-solver.ts   # Optimized raw WebGPU solver
  webgpu-ib-renderer.ts # Field and periodic instanced boundary rendering
  ib-solver.ts          # jax-js reference/fallback solver
scripts/check-2d.mjs     # GPU regression checks
jax-js/           # git submodule — jax-js WebGPU array library
```

## References

- C.S. Peskin, *The immersed boundary method*, Acta Numerica 11 (2002), 479-517.
- C.S. Peskin, [IB MATLAB 2D code](http://www.math.nyu.edu/~peskin/ib_lecture_notes/)
- E. Zhang, [jax-js](https://github.com/ekzhang/jax-js)
