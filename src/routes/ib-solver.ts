// 2D Immersed Boundary Method solver using jax-js
// Port of ib_matlab_2D

import { numpy as np, DType } from "@jax-js/jax";

// --- Types ---

export type IBParams = {
  N: number;
  Nb: number;
  L: number;
  h: number;
  dt: number;
  rho: number;
  mu: number;
  K: number;
  dtheta: number;
};

export type IBState = {
  u: np.Array; // [N, N, 2]
  X: np.Array; // [Nb, 2]
};

export type Precomputed = {
  a: np.Array; // [N, N, 2, 2] real Fourier operator
};

// --- Initialization ---

export function createParams(N: number, K: number, mu: number): IBParams {
  const L = 1.0;
  const h = L / N;
  const Nb = Math.ceil(Math.PI * (L / 2) / (h / 2));
  const dtheta = (2 * Math.PI) / Nb;
  return { N, Nb, L, h, dt: 0.01, rho: 1, mu, K, dtheta };
}

export function initState(p: IBParams): IBState {
  const { N, Nb, L, h, dtheta } = p;

  // Boundary: circle at center, radius L/4
  const Xdata = new Float32Array(Nb * 2);
  for (let k = 0; k < Nb; k++) {
    const theta = k * dtheta;
    Xdata[k * 2] = L / 2 + (L / 4) * Math.cos(theta);
    Xdata[k * 2 + 1] = L / 2 + (L / 4) * Math.sin(theta);
  }
  const X = np.array(Xdata, { shape: [Nb, 2] });

  // Velocity: u_y = sin(2*pi*x/L), u_x = 0
  const uData = new Float32Array(N * N * 2);
  for (let j1 = 0; j1 < N; j1++) {
    const val = Math.sin((2 * Math.PI * j1 * h) / L);
    for (let j2 = 0; j2 < N; j2++) {
      uData[(j1 * N + j2) * 2 + 1] = val;
    }
  }
  const u = np.array(uData, { shape: [N, N, 2] });
  return { u, X };
}

export function initFourierOperator(p: IBParams): Precomputed {
  const { N, h, dt, mu, rho } = p;
  const data = new Float32Array(N * N * 4);

  // Identity
  for (let m1 = 0; m1 < N; m1++) {
    for (let m2 = 0; m2 < N; m2++) {
      const b = (m1 * N + m2) * 4;
      data[b] = 1;
      data[b + 3] = 1;
    }
  }

  // Subtract Leray-Helmholtz projection for non-singular modes
  for (let m1 = 0; m1 < N; m1++) {
    for (let m2 = 0; m2 < N; m2++) {
      if (!((m1 === 0 || m1 === N / 2) && (m2 === 0 || m2 === N / 2))) {
        const s1 = Math.sin(((2 * Math.PI) / N) * m1);
        const s2 = Math.sin(((2 * Math.PI) / N) * m2);
        const ss = s1 * s1 + s2 * s2;
        const b = (m1 * N + m2) * 4;
        data[b] -= (s1 * s1) / ss;
        data[b + 1] -= (s1 * s2) / ss;
        data[b + 2] -= (s2 * s1) / ss;
        data[b + 3] -= (s2 * s2) / ss;
      }
    }
  }

  // Divide by implicit viscosity factor
  for (let m1 = 0; m1 < N; m1++) {
    for (let m2 = 0; m2 < N; m2++) {
      const s1 = Math.sin((Math.PI / N) * m1);
      const s2 = Math.sin((Math.PI / N) * m2);
      const factor =
        1 + (dt / 2) * (mu / rho) * (4 / (h * h)) * (s1 * s1 + s2 * s2);
      const b = (m1 * N + m2) * 4;
      data[b] /= factor;
      data[b + 1] /= factor;
      data[b + 2] /= factor;
      data[b + 3] /= factor;
    }
  }

  return { a: np.array(data, { shape: [N, N, 2, 2] }) };
}

// --- FFT2D helpers ---

type ComplexPair = { real: np.Array; imag: np.Array };

function fft2d(real: np.Array, imag: np.Array): ComplexPair {
  const step1 = np.fft.fft({ real, imag }, 0);
  return np.fft.fft(step1, 1);
}

function ifft2d(real: np.Array, imag: np.Array): ComplexPair {
  const step1 = np.fft.ifft({ real, imag }, 0);
  return np.fft.ifft(step1, 1);
}

// Apply real 2x2 operator a to complex vector w: a[N,N,2,2] * w[N,N,2]
function applyOperator(
  a: np.Array,
  wR: np.Array,
  wI: np.Array,
): ComplexPair {
  const outR = np.einsum("ijkl,ijl->ijk", a.ref, wR);
  const outI = np.einsum("ijkl,ijl->ijk", a, wI);
  return { real: outR, imag: outI };
}

// --- Stencil operations ---

// Periodic shift along axis 0 or 1 (works for any ndim >= 1)
function shift(field: np.Array, axis: number, amount: number): np.Array {
  const n = field.shape[axis];
  if (axis === 0) {
    if (amount === 1) {
      return np.concatenate([field.ref.slice([1, n]), field.slice([0, 1])], 0);
    } else {
      return np.concatenate([field.ref.slice([n - 1, n]), field.slice([0, n - 1])], 0);
    }
  } else {
    if (amount === 1) {
      return np.concatenate([field.ref.slice([], [1, n]), field.slice([], [0, 1])], 1);
    } else {
      return np.concatenate([field.ref.slice([], [n - 1, n]), field.slice([], [0, n - 1])], 1);
    }
  }
}

// Laplacian: (u[i+1]+u[i-1]+u[j+1]+u[j-1]-4u) / h^2, u: [N,N,2]
function laplacianOp(u: np.Array, h: number): np.Array {
  const uip = shift(u.ref, 0, 1);
  const uim = shift(u.ref, 0, -1);
  const ujp = shift(u.ref, 1, 1);
  const ujm = shift(u.ref, 1, -1);
  return uip.add(uim).add(ujp).add(ujm).sub(u.mul(4)).mul(1 / (h * h));
}

// sk(u, g): conservative advection of [N,N] scalar g by [N,N,2] velocity u
function sk(u: np.Array, g: np.Array, h: number): np.Array {
  const u1 = u.ref.slice([], [], 0); // u_x [N,N]
  const u2 = u.slice([], [], 1); // u_y [N,N]

  const u1_ip = shift(u1.ref, 0, 1);
  const u1_im = shift(u1.ref, 0, -1);
  const u2_jp = shift(u2.ref, 1, 1);
  const u2_jm = shift(u2.ref, 1, -1);

  const g_ip = shift(g.ref, 0, 1);
  const g_im = shift(g.ref, 0, -1);
  const g_jp = shift(g.ref, 1, 1);
  const g_jm = shift(g, 1, -1);

  // ((u1_ip+u1)*g_ip - (u1_im+u1)*g_im + (u2_jp+u2)*g_jp - (u2_jm+u2)*g_jm) / (4h)
  const t1 = u1_ip.add(u1.ref).mul(g_ip);
  const t2 = u1_im.add(u1).mul(g_im);
  const t3 = u2_jp.add(u2.ref).mul(g_jp);
  const t4 = u2_jm.add(u2).mul(g_jm);
  return t1.sub(t2).add(t3.sub(t4)).mul(1 / (4 * h));
}

// skew(u) = [sk(u, u[:,:,0]), sk(u, u[:,:,1])]
function skewOp(u: np.Array, h: number): np.Array {
  const u0 = u.ref.slice([], [], 0);
  const u1 = u.ref.slice([], [], 1);
  const s0 = sk(u.ref, u0, h);
  const s1 = sk(u, u1, h);
  return np.stack([s0, s1], 2);
}

// --- Fluid solver: 2-stage IMEX ---

export function fluidSolve(
  u: np.Array,
  ff: np.Array,
  a: np.Array,
  p: IBParams,
): [np.Array, np.Array] {
  const { dt, rho, mu, h, N } = p;

  // Stage 1: w = u - (dt/2)*skew(u) + (dt/(2*rho))*ff
  const sk1 = skewOp(u.ref, h);
  const w1 = u.ref.sub(sk1.mul(dt / 2)).add(ff.ref.mul(dt / (2 * rho)));

  const W1 = fft2d(w1, np.zeros([N, N, 2]));
  const UU = applyOperator(a.ref, W1.real, W1.imag);
  const uh = ifft2d(UU.real, UU.imag);
  const uHalf = uh.real;
  uh.imag.dispose();

  // Stage 2: w = u - dt*skew(uHalf) + (dt/rho)*ff + (dt/2)*(mu/rho)*laplacian(u)
  const sk2 = skewOp(uHalf.ref, h);
  const lap = laplacianOp(u.ref, h);
  const w2 = u
    .sub(sk2.mul(dt))
    .add(ff.mul(dt / rho))
    .add(lap.mul((dt / 2) * (mu / rho)));

  const W2 = fft2d(w2, np.zeros([N, N, 2]));
  const UUU = applyOperator(a, W2.real, W2.imag);
  const uf = ifft2d(UUU.real, UUU.imag);
  const uFull = uf.real;
  uf.imag.dispose();

  return [uFull, uHalf];
}

// --- Delta function, spread, interpolation ---

// 4-point regularized delta: r: [Nb] -> [Nb, 4] for offsets [-1, 0, +1, +2]
function computePhi(r: np.Array): np.Array {
  const q = np.sqrt(r.ref.mul(-1).add(1).mul(r.ref).mul(4).add(1));
  const r2 = r.mul(2);
  const w0 = r2.ref.mul(-1).add(3).sub(q.ref).mul(1 / 8); // (3-2r-q)/8
  const w1 = r2.ref.mul(-1).add(3).add(q.ref).mul(1 / 8); // (3-2r+q)/8
  const w2 = r2.ref.add(1).add(q.ref).mul(1 / 8); // (1+2r+q)/8
  const w3 = r2.add(1).sub(q).mul(1 / 8); // (1+2r-q)/8
  return np.stack([w0, w1, w2, w3], 1);
}

// Build gridWeight[N, N, Nb] for spread/interp via broadcasting
function buildGridWeight(X: np.Array, p: IBParams): np.Array {
  const { N, Nb, h } = p;

  const s = X.mul(1 / h); // [Nb, 2]
  const iFloor = np.floor(s.ref).astype(DType.Int32); // [Nb, 2]
  const r = s.sub(iFloor.ref.astype(DType.Float32)); // [Nb, 2]

  const r0 = r.ref.slice([], 0); // [Nb]
  const r1 = r.slice([], 1);
  const if0 = iFloor.ref.slice([], 0); // [Nb] int32
  const if1 = iFloor.slice([], 1);

  const phi0 = computePhi(r0); // [Nb, 4]
  const phi1 = computePhi(r1);

  const offsets = np.array(new Int32Array([-1, 0, 1, 2]), { shape: [4] });

  // Indices: (floor + offset + N) % N
  const i1 = if0.reshape([-1, 1]).add(offsets.ref).add(N).mod(N); // [Nb, 4]
  const i2 = if1.reshape([-1, 1]).add(offsets).add(N).mod(N);

  const gridIdx = np.arange(N).astype(DType.Int32); // [N]

  // matchRows[j, k, p] = (j == i1[k, p]) -> [N, Nb, 4]
  const matchRows = gridIdx.ref
    .reshape([N, 1, 1])
    .equal(i1.reshape([1, Nb, 4]))
    .astype(DType.Float32);
  const rowContrib = matchRows.mul(phi0.reshape([1, Nb, 4])).sum(2); // [N, Nb]

  const matchCols = gridIdx
    .reshape([N, 1, 1])
    .equal(i2.reshape([1, Nb, 4]))
    .astype(DType.Float32);
  const colContrib = matchCols.mul(phi1.reshape([1, Nb, 4])).sum(2); // [N, Nb]

  // gridWeight[j1, j2, k] = rowContrib[j1, k] * colContrib[j2, k]
  return rowContrib.reshape([N, 1, Nb]).mul(colContrib.reshape([1, N, Nb]));
}

// Spread boundary forces to grid: F[Nb,2], X[Nb,2] -> ff[N,N,2]
export function spread(F: np.Array, X: np.Array, p: IBParams): np.Array {
  const c = p.dtheta / (p.h * p.h);
  const gw = buildGridWeight(X, p);
  return np.einsum("ijk,kd->ijd", gw, F).mul(c);
}

// Interpolate grid velocity to boundary: u[N,N,2], X[Nb,2] -> U[Nb,2]
export function interp(u: np.Array, X: np.Array, p: IBParams): np.Array {
  const gw = buildGridWeight(X, p);
  return np.einsum("ijk,ijd->kd", gw, u);
}

// --- Force computation ---

export function computeForce(X: np.Array, p: IBParams): np.Array {
  const { Nb, K, dtheta } = p;
  const kpData = new Int32Array(Nb);
  const kmData = new Int32Array(Nb);
  for (let k = 0; k < Nb; k++) {
    kpData[k] = (k + 1) % Nb;
    kmData[k] = (k - 1 + Nb) % Nb;
  }
  const kp = np.array(kpData, { shape: [Nb] });
  const km = np.array(kmData, { shape: [Nb] });
  const Xkp = X.ref.slice(kp); // [Nb, 2]
  const Xkm = X.ref.slice(km);
  return Xkp.add(Xkm).sub(X.mul(2)).mul(K / (dtheta * dtheta));
}

// --- Time step ---

export function timeStep(
  state: IBState,
  pre: Precomputed,
  p: IBParams,
): IBState {
  const { u, X } = state;
  const { dt } = p;

  // 1. Interpolate velocity to boundary, predict midpoint
  const U0 = interp(u.ref, X.ref, p);
  const XX = X.ref.add(U0.mul(dt / 2));

  // 2. Compute forces, spread to grid
  const F = computeForce(XX.ref, p);
  const ff = spread(F, XX.ref, p);

  // 3. Solve fluid
  const [uNew, uHalf] = fluidSolve(u, ff, pre.a.ref, p);

  // 4. Update boundary using half-step velocity
  const U1 = interp(uHalf, XX, p);
  const XNew = X.add(U1.mul(dt));

  return { u: uNew, X: XNew };
}

// --- Vorticity ---

export function computeVorticity(u: np.Array, p: IBParams): np.Array {
  const { h } = p;
  const uy = u.ref.slice([], [], 1); // [N,N]
  const ux = u.slice([], [], 0);

  const uy_ip = shift(uy.ref, 0, 1);
  const uy_im = shift(uy, 0, -1);
  const ux_jp = shift(ux.ref, 1, 1);
  const ux_jm = shift(ux, 1, -1);

  // (duy/dx - dux/dy) / (2h)
  return uy_ip.sub(uy_im).sub(ux_jp.sub(ux_jm)).mul(1 / (2 * h));
}

// Velocity magnitude: sqrt(ux^2 + uy^2) -> [N,N]
export function computeVelMagnitude(u: np.Array): np.Array {
  const ux = u.ref.slice([], [], 0);
  const uy = u.slice([], [], 1);
  return np.sqrt(ux.ref.mul(ux).add(uy.ref.mul(uy)));
}

// CPU-side bilinear upsample from Float32Array [N*N] -> [M*M]
export function upsampleCPU(src: Float32Array, N: number, M: number): Float32Array {
  const dst = new Float32Array(M * M);
  for (let mi = 0; mi < M; mi++) {
    for (let mj = 0; mj < M; mj++) {
      const fi = (mi / M) * N - 0.5;
      const fj = (mj / M) * N - 0.5;
      const i0 = Math.floor(fi);
      const j0 = Math.floor(fj);
      const i1 = i0 + 1;
      const j1 = j0 + 1;
      const wi = fi - i0;
      const wj = fj - j0;
      // Periodic wrapping
      const ci0 = ((i0 % N) + N) % N;
      const ci1 = ((i1 % N) + N) % N;
      const cj0 = ((j0 % N) + N) % N;
      const cj1 = ((j1 % N) + N) % N;
      dst[mi * M + mj] =
        (1 - wi) * (1 - wj) * src[ci0 * N + cj0] +
        wi * (1 - wj) * src[ci1 * N + cj0] +
        (1 - wi) * wj * src[ci0 * N + cj1] +
        wi * wj * src[ci1 * N + cj1];
    }
  }
  return dst;
}
