# 2D Immersed Boundary Method

Interactive simulation of the 2D immersed boundary (IB) method running entirely in the browser via WebGPU.

**[Live Demo](https://guanhuasun.github.io/ib-sim/)**

An elastic membrane (closed curve) is immersed in a viscous, incompressible fluid on a periodic domain $[0,1]^2$. The fluid and structure are coupled through regularized delta functions: the membrane exerts elastic forces on the fluid, and the fluid velocity advects the membrane. This two-way coupling is the core idea of Peskin's immersed boundary method.

## Method

The numerical scheme follows the MATLAB implementation in `ib_matlab_2D` (C.S. Peskin):

- **Fluid solver**: 2-stage IMEX (implicit-explicit) time integration. Advection and forcing are treated explicitly; viscous diffusion is handled implicitly via a Fourier-space solve. The 2D FFT is composed from two 1D FFTs.
- **Structure solver**: Elastic spring forces $F_k = K(X_{k+1} + X_{k-1} - 2X_k)/\Delta\theta^2$ with periodic indexing. A midpoint predictor step improves temporal accuracy.
- **Coupling**: Spread (structure $\to$ fluid) and interpolation (fluid $\to$ structure) use the 4-point regularized delta function of Peskin, implemented via vectorized broadcasting over an $[N, N, N_b]$ tensor.

All computation runs on the GPU through [jax-js](https://github.com/ekzhang/jax-js), a JAX-like array library targeting WebGPU.

## Features

- Real-time simulation at 64x64 or 128x128 grid resolution
- FFT-based implicit viscosity (unconditionally stable for diffusion)
- Interactive mouse drag to apply localized body forces
- Multiple diverging colormaps (Cyan-Magenta, Teal-Orange, Blue-Yellow, Green-Purple, Red-Blue, Twilight)
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

Requires a browser with WebGPU support (Chrome 113+, Edge 113+, or Firefox Nightly).

## Project structure

```
src/routes/
  +page.svelte    # UI, WebGPU rendering, animation loop
  ib-solver.ts    # Numerical solver (FFT, spread/interp, time stepping)
jax-js/           # git submodule — jax-js WebGPU array library
```

## References

- C.S. Peskin, *The immersed boundary method*, Acta Numerica 11 (2002), 479-517.
- C.S. Peskin, [IB MATLAB 2D code](http://www.math.nyu.edu/~peskin/ib_lecture_notes/)
- E. Zhang, [jax-js](https://github.com/ekzhang/jax-js)
