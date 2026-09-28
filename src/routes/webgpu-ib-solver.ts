import type { IBParams } from "./ib-solver";

export type VizMode = "vorticity" | "velocity";

type GpuBuffers = {
  u: GPUBuffer;
  uHalf: GPUBuffer;
  uNext: GPUBuffer;
  w: GPUBuffer;
  zA: GPUBuffer;
  zB: GPUBuffer;
  X: GPUBuffer;
  XMid: GPUBuffer;
  XNext: GPUBuffer;
  F: GPUBuffer;
  ff: GPUBuffer;
  field: GPUBuffer;
};

type PipelineKey =
  | "predict"
  | "force"
  | "spread"
  | "w1"
  | "w2"
  | "updateX"
  | "mouse"
  | "viz"
  | "fft0Real"
  | "fft1Op"
  | "ifft1"
  | "ifft0Real";

const PARAM_FLOATS = 6;
const TWO_PI = Math.PI * 2;
type Dispatch = { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup; count: number };
const PIPELINE_BINDINGS: Record<PipelineKey, { inputs: number[]; outputs: number[] }> = {
  predict: { inputs: [0, 1, 2], outputs: [3] },
  force: { inputs: [0, 4], outputs: [5] },
  spread: { inputs: [0, 6, 7], outputs: [8] },
  w1: { inputs: [0, 9, 10], outputs: [11] },
  w2: { inputs: [0, 12, 13, 14], outputs: [15] },
  updateX: { inputs: [0, 16, 17, 18], outputs: [19] },
  mouse: { inputs: [0, 20], outputs: [21] },
  viz: { inputs: [0, 22, 23], outputs: [24] },
  fft0Real: { inputs: [25], outputs: [26] },
  fft1Op: { inputs: [0, 27], outputs: [28] },
  ifft1: { inputs: [29], outputs: [30] },
  ifft0Real: { inputs: [31], outputs: [32] },
};

export class WebGpuIbSolver {
  readonly fieldBuffer: GPUBuffer;

  #buffers: GpuBuffers;
  #pipelines: Map<PipelineKey, GPUComputePipeline>;
  #paramBuffer: GPUBuffer;
  #mouseBuffer: GPUBuffer;
  #vizModeBuffer: GPUBuffer;
  #params: IBParams;
  #destroyed = false;
  #phase = 0;
  #steps: Dispatch[][] = [];
  #mouse: Dispatch[] = [];
  #viz: Dispatch[] = [];
  #vizMode: VizMode | undefined;

  private constructor(
    readonly device: GPUDevice,
    params: IBParams,
    buffers: GpuBuffers,
    paramBuffer: GPUBuffer,
    mouseBuffer: GPUBuffer,
    vizModeBuffer: GPUBuffer,
    pipelines: Map<PipelineKey, GPUComputePipeline>,
  ) {
    this.#params = { ...params };
    this.#buffers = buffers;
    this.fieldBuffer = buffers.field;
    this.#paramBuffer = paramBuffer;
    this.#mouseBuffer = mouseBuffer;
    this.#vizModeBuffer = vizModeBuffer;
    this.#pipelines = pipelines;
    this.#writeParams();
    // Prebind both ping-pong states; no bind groups or schedules are allocated per step.
    for (let phase = 0; phase < 2; phase++) {
      this.#steps.push(this.#createStep());
      const grid = Math.ceil(params.N * params.N / 256);
      this.#mouse.push(this.#command("mouse", [paramBuffer, mouseBuffer], [buffers.u], grid));
      this.#viz.push(this.#command("viz", [paramBuffer, vizModeBuffer, buffers.u], [buffers.field], grid));
      this.#swap();
    }
  }

  static async init(device: GPUDevice, params: IBParams) {
    if (params.N !== 64 && params.N !== 128) {
      throw new Error(`Optimized WebGPU solver supports N=64 or N=128, got ${params.N}`);
    }

    // Compile before allocating state so a shader failure can fall back without leaking buffers.
    const pipelines = await createPipelines(device, params);
    const buffers = createBuffers(device, params);
    const paramBuffer = device.createBuffer({
      size: PARAM_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const mouseBuffer = device.createBuffer({
      size: 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const vizModeBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    return new WebGpuIbSolver(
      device,
      params,
      buffers,
      paramBuffer,
      mouseBuffer,
      vizModeBuffer,
      pipelines,
    );
  }

  get boundaryBuffer() {
    return this.#buffers.X;
  }

  setParams(params: IBParams) {
    this.#assertAlive();
    if (params.N !== this.#params.N || params.Nb !== this.#params.Nb) {
      throw new Error("Create a new optimized solver to resize the grid or boundary");
    }
    this.#params = { ...params };
    this.#writeParams();
  }

  reset(params: IBParams) {
    this.#assertAlive();
    this.setParams(params);

    const { u, X } = initialData(params);
    this.device.queue.writeBuffer(this.#buffers.u, 0, u);
    this.device.queue.writeBuffer(this.#buffers.X, 0, X);
  }

  applyMouseForce(x: number, y: number, dx: number, dy: number, encoder?: GPUCommandEncoder) {
    this.#assertAlive();
    const data = new Float32Array([x, y, dx, dy]);
    this.device.queue.writeBuffer(this.#mouseBuffer, 0, data);
    this.#encode(this.#mouse[this.#phase], encoder);
  }

  stepBatch(count = 1, encoder?: GPUCommandEncoder) {
    this.#assertAlive();
    const commands = encoder ?? this.device.createCommandEncoder();
    const pass = commands.beginComputePass();
    for (let i = 0; i < count; i++) {
      for (const command of this.#steps[this.#phase]) dispatch(pass, command);
      this.#swap();
    }
    pass.end();
    if (!encoder) this.device.queue.submit([commands.finish()]);
  }

  renderField(vizMode: VizMode, encoder?: GPUCommandEncoder) {
    this.#assertAlive();
    if (vizMode !== this.#vizMode) {
      this.device.queue.writeBuffer(this.#vizModeBuffer, 0, new Uint32Array([vizMode === "vorticity" ? 0 : 1]));
      this.#vizMode = vizMode;
    }
    this.#encode(this.#viz[this.#phase], encoder);
    return this.#buffers.field;
  }

  destroy() {
    if (this.#destroyed) return;
    this.#destroyed = true;
    for (const buffer of Object.values(this.#buffers)) buffer.destroy();
    this.#paramBuffer.destroy();
    this.#mouseBuffer.destroy();
    this.#vizModeBuffer.destroy();
    this.#steps = [];
    this.#mouse = [];
    this.#viz = [];
  }

  #swap() {
    const b = this.#buffers;
    [b.u, b.uNext] = [b.uNext, b.u];
    [b.X, b.XNext] = [b.XNext, b.X];
    this.#phase ^= 1;
  }

  #createStep(): Dispatch[] {
    const { N, Nb } = this.#params;
    const b = this.#buffers, p = this.#paramBuffer;
    const grid = Math.ceil(N * N / 256), boundary = Math.ceil(Nb / 128);
    const fluid = (output: GPUBuffer) => [
      this.#command("fft0Real", [b.w], [b.zA], N),
      this.#command("fft1Op", [p, b.zA], [b.zB], N),
      this.#command("ifft1", [b.zB], [b.zA], N),
      this.#command("ifft0Real", [b.zA], [output], N),
    ];
    return [
      this.#command("predict", [p, b.u, b.X], [b.XMid], boundary),
      this.#command("force", [p, b.XMid], [b.F], boundary),
      this.#command("spread", [p, b.XMid, b.F], [b.ff], grid),
      this.#command("w1", [p, b.u, b.ff], [b.w], grid),
      ...fluid(b.uHalf),
      this.#command("w2", [p, b.u, b.uHalf, b.ff], [b.w], grid),
      ...fluid(b.uNext),
      this.#command("updateX", [p, b.uHalf, b.X, b.XMid], [b.XNext], boundary),
    ];
  }

  #command(
    key: PipelineKey,
    inputs: GPUBuffer[],
    outputs: GPUBuffer[],
    count: number,
  ): Dispatch {
    const pipeline = this.#pipelines.get(key)!;
    const bindings = PIPELINE_BINDINGS[key];
    const entries: GPUBindGroupEntry[] = [];
    for (let i = 0; i < inputs.length; i++) {
      entries.push({ binding: bindings.inputs[i], resource: { buffer: inputs[i] } });
    }
    for (let i = 0; i < outputs.length; i++) {
      entries.push({ binding: bindings.outputs[i], resource: { buffer: outputs[i] } });
    }

    return {
      pipeline, count,
      bindGroup: this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
      }),
    };
  }

  #encode(command: Dispatch, encoder?: GPUCommandEncoder) {
    const commands = encoder ?? this.device.createCommandEncoder();
    const pass = commands.beginComputePass();
    dispatch(pass, command);
    pass.end();
    if (!encoder) this.device.queue.submit([commands.finish()]);
  }

  #writeParams() {
    const p = this.#params;
    const data = new Float32Array([
      p.dt,
      p.rho,
      p.mu,
      p.K,
      p.h,
      p.dtheta,
    ]);
    this.device.queue.writeBuffer(this.#paramBuffer, 0, data);
  }

  #assertAlive() {
    if (this.#destroyed) throw new Error("Optimized WebGPU solver has been destroyed");
  }
}

function dispatch(pass: GPUComputePassEncoder, { pipeline, bindGroup, count }: Dispatch) {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(count);
}

function createBuffers(device: GPUDevice, params: IBParams): GpuBuffers {
  const { N, Nb } = params;
  const gridBytes = N * N * 2 * 4;
  const complexBytes = N * N * 4 * 4;
  const scalarBytes = N * N * 4;
  const boundaryBytes = Nb * 2 * 4;

  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const buffers: GpuBuffers = {
    u: device.createBuffer({ size: gridBytes, usage: storage }),
    uHalf: device.createBuffer({ size: gridBytes, usage: storage }),
    uNext: device.createBuffer({ size: gridBytes, usage: storage }),
    w: device.createBuffer({ size: gridBytes, usage: storage }),
    zA: device.createBuffer({ size: complexBytes, usage: storage }),
    zB: device.createBuffer({ size: complexBytes, usage: storage }),
    X: device.createBuffer({ size: boundaryBytes, usage: storage }),
    XMid: device.createBuffer({ size: boundaryBytes, usage: storage }),
    XNext: device.createBuffer({ size: boundaryBytes, usage: storage }),
    F: device.createBuffer({ size: boundaryBytes, usage: storage }),
    ff: device.createBuffer({ size: gridBytes, usage: storage }),
    field: device.createBuffer({ size: scalarBytes, usage: storage }),
  };

  const { u, X } = initialData(params);
  device.queue.writeBuffer(buffers.u, 0, u);
  device.queue.writeBuffer(buffers.X, 0, X);
  return buffers;
}

function initialData(p: IBParams) {
  const { N, Nb, L, h, dtheta } = p;
  const X = new Float32Array(Nb * 2);
  for (let k = 0; k < Nb; k++) {
    const theta = k * dtheta;
    X[k * 2] = L / 2 + (L / 4) * Math.cos(theta);
    X[k * 2 + 1] = L / 2 + (L / 4) * Math.sin(theta);
  }

  const u = new Float32Array(N * N * 2);
  for (let j1 = 0; j1 < N; j1++) {
    const val = Math.sin((TWO_PI * j1 * h) / L);
    for (let j2 = 0; j2 < N; j2++) {
      u[(j1 * N + j2) * 2 + 1] = val;
    }
  }
  return { u, X };
}

async function createPipelines(device: GPUDevice, params: IBParams) {
  const shader = solverShader(params);
  const module = device.createShaderModule({ code: shader });
  const entries: [PipelineKey, string][] = [
    ["predict", "predict_midpoint"],
    ["force", "compute_force"],
    ["spread", "spread_force"],
    ["w1", "build_w1"],
    ["w2", "build_w2"],
    ["updateX", "update_boundary"],
    ["mouse", "apply_mouse_force"],
    ["viz", "compute_viz"],
    ["fft0Real", "fft_axis0_real"],
    ["fft1Op", "fft_axis1_project"],
    ["ifft1", "ifft_axis1"],
    ["ifft0Real", "ifft_axis0_real"],
  ];
  return new Map(await Promise.all(entries.map(async ([key, entryPoint]) => [
    key,
    await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint },
    }),
  ] as const)));
}

function solverShader({ N, Nb }: IBParams) {
  const logN = Math.round(Math.log2(N));
  return /* wgsl */ `
const N: u32 = ${N}u;
const NB: u32 = ${Nb}u;
const LOG_N: u32 = ${logN}u;
const PI: f32 = 3.141592653589793;
const TWO_PI: f32 = 6.283185307179586;

@group(0) @binding(0) var<storage, read> params: array<f32>;

fn dt() -> f32 { return params[0]; }
fn rho() -> f32 { return params[1]; }
fn mu() -> f32 { return params[2]; }
fn stiffness() -> f32 { return params[3]; }
fn h() -> f32 { return params[4]; }
fn dtheta() -> f32 { return params[5]; }

fn grid_idx(i: u32, j: u32) -> u32 {
  return (i * N + j) * 2u;
}

fn scalar_idx(i: u32, j: u32) -> u32 {
  return i * N + j;
}

fn wrap_i(v: i32) -> u32 {
  let n = i32(N);
  return u32(((v % n) + n) % n);
}

fn cmul(a: vec4f, wr: f32, wi: f32) -> vec4f {
  return vec4f(
    a.x * wr - a.y * wi,
    a.x * wi + a.y * wr,
    a.z * wr - a.w * wi,
    a.z * wi + a.w * wr,
  );
}

fn bit_reverse(x0: u32) -> u32 {
  var x = x0;
  var y = 0u;
  for (var i = 0u; i < LOG_N; i++) {
    y = (y << 1u) | (x & 1u);
    x = x >> 1u;
  }
  return y;
}

fn phi(r: f32, offset: i32) -> f32 {
  let q = sqrt(4.0 * r * (1.0 - r) + 1.0);
  if (offset == -1) { return (3.0 - 2.0 * r - q) * 0.125; }
  if (offset == 0) { return (3.0 - 2.0 * r + q) * 0.125; }
  if (offset == 1) { return (1.0 + 2.0 * r + q) * 0.125; }
  if (offset == 2) { return (1.0 + 2.0 * r - q) * 0.125; }
  return 0.0;
}

fn weight_for_grid(j: u32, s: f32) -> f32 {
  let base = i32(floor(s));
  let r = s - f32(base);
  let d = i32(wrap_i(i32(j) - base));
  if (d == i32(N) - 1) { return phi(r, -1); }
  if (d == 0) { return phi(r, 0); }
  if (d == 1) { return phi(r, 1); }
  if (d == 2) { return phi(r, 2); }
  return 0.0;
}

fn interp_at(x: vec2f, u: ptr<storage, array<f32>, read>) -> vec2f {
  let sx = x.x / h();
  let sy = x.y / h();
  let bx = i32(floor(sx));
  let by = i32(floor(sy));
  let rx = sx - f32(bx);
  let ry = sy - f32(by);
  var out = vec2f(0.0);
  for (var ox = -1; ox <= 2; ox++) {
    let wx = phi(rx, ox);
    let i = wrap_i(bx + ox);
    for (var oy = -1; oy <= 2; oy++) {
      let wy = phi(ry, oy);
      let j = wrap_i(by + oy);
      let g = grid_idx(i, j);
      out += wx * wy * vec2f((*u)[g], (*u)[g + 1u]);
    }
  }
  return out;
}

fn skew_component(i: u32, j: u32, comp: u32, u: ptr<storage, array<f32>, read>) -> f32 {
  let ip = (i + 1u) % N;
  let im = (i + N - 1u) % N;
  let jp = (j + 1u) % N;
  let jm = (j + N - 1u) % N;
  let u1_ip = (*u)[grid_idx(ip, j)];
  let u1_im = (*u)[grid_idx(im, j)];
  let u1 = (*u)[grid_idx(i, j)];
  let u2_jp = (*u)[grid_idx(i, jp) + 1u];
  let u2_jm = (*u)[grid_idx(i, jm) + 1u];
  let u2 = (*u)[grid_idx(i, j) + 1u];
  let g_ip = (*u)[grid_idx(ip, j) + comp];
  let g_im = (*u)[grid_idx(im, j) + comp];
  let g_jp = (*u)[grid_idx(i, jp) + comp];
  let g_jm = (*u)[grid_idx(i, jm) + comp];
  return ((u1_ip + u1) * g_ip - (u1_im + u1) * g_im + (u2_jp + u2) * g_jp - (u2_jm + u2) * g_jm) / (4.0 * h());
}

fn lap_component(i: u32, j: u32, comp: u32, u: ptr<storage, array<f32>, read>) -> f32 {
  let ip = (i + 1u) % N;
  let im = (i + N - 1u) % N;
  let jp = (j + 1u) % N;
  let jm = (j + N - 1u) % N;
  return (
    (*u)[grid_idx(ip, j) + comp] +
    (*u)[grid_idx(im, j) + comp] +
    (*u)[grid_idx(i, jp) + comp] +
    (*u)[grid_idx(i, jm) + comp] -
    4.0 * (*u)[grid_idx(i, j) + comp]
  ) / (h() * h());
}

var<workgroup> fft_data: array<vec4f, ${N}>;

fn fft_shared(tid: u32, inverse: bool) {
  workgroupBarrier();
  for (var stage = 1u; stage <= LOG_N; stage++) {
    let len = 1u << stage;
    let half = len >> 1u;
    if (tid < N / 2u) {
      let group = tid / half;
      let j = tid % half;
      let i0 = group * len + j;
      let i1 = i0 + half;
      let sign = select(-1.0, 1.0, inverse);
      let angle = sign * TWO_PI * f32(j) / f32(len);
      let w = vec2f(cos(angle), sin(angle));
      let a = fft_data[i0];
      let b = cmul(fft_data[i1], w.x, w.y);
      fft_data[i0] = a + b;
      fft_data[i1] = a - b;
    }
    workgroupBarrier();
  }
}

@group(0) @binding(1) var<storage, read> u_predict: array<f32>;
@group(0) @binding(2) var<storage, read> x_predict: array<f32>;
@group(0) @binding(3) var<storage, read_write> x_mid_out: array<f32>;

@compute @workgroup_size(128)
fn predict_midpoint(@builtin(global_invocation_id) id: vec3u) {
  let k = id.x;
  if (k >= NB) { return; }
  let x = vec2f(x_predict[k * 2u], x_predict[k * 2u + 1u]);
  let vel = interp_at(x, &u_predict);
  x_mid_out[k * 2u] = x.x + 0.5 * dt() * vel.x;
  x_mid_out[k * 2u + 1u] = x.y + 0.5 * dt() * vel.y;
}

@group(0) @binding(4) var<storage, read> x_force: array<f32>;
@group(0) @binding(5) var<storage, read_write> force_out: array<f32>;

@compute @workgroup_size(128)
fn compute_force(@builtin(global_invocation_id) id: vec3u) {
  let k = id.x;
  if (k >= NB) { return; }
  let kp = (k + 1u) % NB;
  let km = (k + NB - 1u) % NB;
  let xk = vec2f(x_force[k * 2u], x_force[k * 2u + 1u]);
  let xkp = vec2f(x_force[kp * 2u], x_force[kp * 2u + 1u]);
  let xkm = vec2f(x_force[km * 2u], x_force[km * 2u + 1u]);
  let f = (xkp + xkm - 2.0 * xk) * (stiffness() / (dtheta() * dtheta()));
  force_out[k * 2u] = f.x;
  force_out[k * 2u + 1u] = f.y;
}

@group(0) @binding(6) var<storage, read> x_spread: array<f32>;
@group(0) @binding(7) var<storage, read> force_in: array<f32>;
@group(0) @binding(8) var<storage, read_write> ff_out: array<f32>;

@compute @workgroup_size(256)
fn spread_force(@builtin(global_invocation_id) id: vec3u) {
  let idx = id.x;
  if (idx >= N * N) { return; }
  let i = idx / N;
  let j = idx % N;
  var f = vec2f(0.0);
  for (var k = 0u; k < NB; k++) {
    let sx = x_spread[k * 2u] / h();
    let wx = weight_for_grid(i, sx);
    if (wx == 0.0) { continue; }
    let sy = x_spread[k * 2u + 1u] / h();
    let w = wx * weight_for_grid(j, sy);
    f += w * vec2f(force_in[k * 2u], force_in[k * 2u + 1u]);
  }
  f *= dtheta() / (h() * h());
  ff_out[idx * 2u] = f.x;
  ff_out[idx * 2u + 1u] = f.y;
}

@group(0) @binding(9) var<storage, read> u_w1: array<f32>;
@group(0) @binding(10) var<storage, read> ff_w1: array<f32>;
@group(0) @binding(11) var<storage, read_write> w1_out: array<f32>;

@compute @workgroup_size(256)
fn build_w1(@builtin(global_invocation_id) id: vec3u) {
  let idx = id.x;
  if (idx >= N * N) { return; }
  let i = idx / N;
  let j = idx % N;
  let g = idx * 2u;
  let sk0 = skew_component(i, j, 0u, &u_w1);
  let sk1 = skew_component(i, j, 1u, &u_w1);
  w1_out[g] = u_w1[g] - 0.5 * dt() * sk0 + 0.5 * dt() / rho() * ff_w1[g];
  w1_out[g + 1u] = u_w1[g + 1u] - 0.5 * dt() * sk1 + 0.5 * dt() / rho() * ff_w1[g + 1u];
}

@group(0) @binding(12) var<storage, read> u_w2: array<f32>;
@group(0) @binding(13) var<storage, read> u_half_w2: array<f32>;
@group(0) @binding(14) var<storage, read> ff_w2: array<f32>;
@group(0) @binding(15) var<storage, read_write> w2_out: array<f32>;

@compute @workgroup_size(256)
fn build_w2(@builtin(global_invocation_id) id: vec3u) {
  let idx = id.x;
  if (idx >= N * N) { return; }
  let i = idx / N;
  let j = idx % N;
  let g = idx * 2u;
  let sk0 = skew_component(i, j, 0u, &u_half_w2);
  let sk1 = skew_component(i, j, 1u, &u_half_w2);
  let lap0 = lap_component(i, j, 0u, &u_w2);
  let lap1 = lap_component(i, j, 1u, &u_w2);
  w2_out[g] = u_w2[g] - dt() * sk0 + dt() / rho() * ff_w2[g] + 0.5 * dt() * mu() / rho() * lap0;
  w2_out[g + 1u] = u_w2[g + 1u] - dt() * sk1 + dt() / rho() * ff_w2[g + 1u] + 0.5 * dt() * mu() / rho() * lap1;
}

@group(0) @binding(16) var<storage, read> u_update: array<f32>;
@group(0) @binding(17) var<storage, read> x_old_update: array<f32>;
@group(0) @binding(18) var<storage, read> x_mid_update: array<f32>;
@group(0) @binding(19) var<storage, read_write> x_next_out: array<f32>;

@compute @workgroup_size(128)
fn update_boundary(@builtin(global_invocation_id) id: vec3u) {
  let k = id.x;
  if (k >= NB) { return; }
  let xmid = vec2f(x_mid_update[k * 2u], x_mid_update[k * 2u + 1u]);
  let vel = interp_at(xmid, &u_update);
  x_next_out[k * 2u] = x_old_update[k * 2u] + dt() * vel.x;
  x_next_out[k * 2u + 1u] = x_old_update[k * 2u + 1u] + dt() * vel.y;
}

@group(0) @binding(20) var<storage, read> mouse: array<f32>;
@group(0) @binding(21) var<storage, read_write> u_mouse: array<f32>;

@compute @workgroup_size(256)
fn apply_mouse_force(@builtin(global_invocation_id) id: vec3u) {
  let idx = id.x;
  if (idx >= N * N) { return; }
  let i = idx / N;
  let j = idx % N;
  let x = f32(i) * h();
  let y = f32(j) * h();
  let rx = x - mouse[0];
  let ry = y - mouse[1];
  let sigma = 0.07;
  let g = 45.0 * exp(-(rx * rx + ry * ry) / (2.0 * sigma * sigma));
  let base = idx * 2u;
  u_mouse[base] += g * mouse[2] * dt();
  u_mouse[base + 1u] += g * mouse[3] * dt();
}

@group(0) @binding(22) var<storage, read> viz_mode: array<u32>;
@group(0) @binding(23) var<storage, read> u_viz: array<f32>;
@group(0) @binding(24) var<storage, read_write> field_out: array<f32>;

@compute @workgroup_size(256)
fn compute_viz(@builtin(global_invocation_id) id: vec3u) {
  let idx = id.x;
  if (idx >= N * N) { return; }
  let i = idx / N;
  let j = idx % N;
  let g = idx * 2u;
  if (viz_mode[0] == 1u) {
    field_out[idx] = sqrt(u_viz[g] * u_viz[g] + u_viz[g + 1u] * u_viz[g + 1u]);
  } else {
    let ip = (i + 1u) % N;
    let im = (i + N - 1u) % N;
    let jp = (j + 1u) % N;
    let jm = (j + N - 1u) % N;
    let duy_dx = u_viz[grid_idx(ip, j) + 1u] - u_viz[grid_idx(im, j) + 1u];
    let dux_dy = u_viz[grid_idx(i, jp)] - u_viz[grid_idx(i, jm)];
    field_out[idx] = (duy_dx - dux_dy) / (2.0 * h());
  }
}

@group(0) @binding(25) var<storage, read> real_fft_in: array<f32>;
@group(0) @binding(26) var<storage, read_write> z_fft0_out: array<vec4f>;

@compute @workgroup_size(${N})
fn fft_axis0_real(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let tid = lid.x;
  let col = wg.x;
  let src_i = bit_reverse(tid);
  let g = grid_idx(src_i, col);
  fft_data[tid] = vec4f(real_fft_in[g], 0.0, real_fft_in[g + 1u], 0.0);
  fft_shared(tid, false);
  z_fft0_out[scalar_idx(tid, col)] = fft_data[tid];
}

@group(0) @binding(27) var<storage, read> z_fft1_in: array<vec4f>;
@group(0) @binding(28) var<storage, read_write> z_fft1_out: array<vec4f>;

@compute @workgroup_size(${N})
fn fft_axis1_project(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let tid = lid.x;
  let row = wg.x;
  let src_j = bit_reverse(tid);
  fft_data[tid] = z_fft1_in[scalar_idx(row, src_j)];
  fft_shared(tid, false);

  let m1 = row;
  let m2 = tid;
  var a00 = 1.0;
  var a01 = 0.0;
  var a10 = 0.0;
  var a11 = 1.0;
  if (!((m1 == 0u || m1 == N / 2u) && (m2 == 0u || m2 == N / 2u))) {
    let s1 = sin(TWO_PI * f32(m1) / f32(N));
    let s2 = sin(TWO_PI * f32(m2) / f32(N));
    let ss = s1 * s1 + s2 * s2;
    a00 -= s1 * s1 / ss;
    a01 -= s1 * s2 / ss;
    a10 -= s2 * s1 / ss;
    a11 -= s2 * s2 / ss;
  }
  let d1 = sin(PI * f32(m1) / f32(N));
  let d2 = sin(PI * f32(m2) / f32(N));
  let factor = 1.0 + 0.5 * dt() * mu() / rho() * 4.0 / (h() * h()) * (d1 * d1 + d2 * d2);
  a00 /= factor;
  a01 /= factor;
  a10 /= factor;
  a11 /= factor;

  let z = fft_data[tid];
  z_fft1_out[scalar_idx(row, tid)] = vec4f(
    a00 * z.x + a01 * z.z,
    a00 * z.y + a01 * z.w,
    a10 * z.x + a11 * z.z,
    a10 * z.y + a11 * z.w,
  );
}

@group(0) @binding(29) var<storage, read> z_ifft1_in: array<vec4f>;
@group(0) @binding(30) var<storage, read_write> z_ifft1_out: array<vec4f>;

@compute @workgroup_size(${N})
fn ifft_axis1(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let tid = lid.x;
  let row = wg.x;
  let src_j = bit_reverse(tid);
  fft_data[tid] = z_ifft1_in[scalar_idx(row, src_j)];
  fft_shared(tid, true);
  z_ifft1_out[scalar_idx(row, tid)] = fft_data[tid];
}

@group(0) @binding(31) var<storage, read> z_ifft0_in: array<vec4f>;
@group(0) @binding(32) var<storage, read_write> real_ifft0_out: array<f32>;

@compute @workgroup_size(${N})
fn ifft_axis0_real(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let tid = lid.x;
  let col = wg.x;
  let src_i = bit_reverse(tid);
  fft_data[tid] = z_ifft0_in[scalar_idx(src_i, col)];
  fft_shared(tid, true);
  let scale = 1.0 / f32(N * N);
  let g = grid_idx(tid, col);
  real_ifft0_out[g] = fft_data[tid].x * scale;
  real_ifft0_out[g + 1u] = fft_data[tid].z * scale;
}
`;
}
