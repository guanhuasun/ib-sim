export type IB3DParams = {
  N: number;
  dt: number;
  K: number;
  mu: number;
  refinement: number;
  radius: number;
};

export type IB3DFieldMode = "velocity" | "vorticity";

type Mesh = {
  positions: Float32Array;
  triangles: Uint32Array;
  edges: Uint32Array;
  starts: Uint32Array;
  neighbors: Uint32Array;
  restLengths: Float32Array;
  nb: number;
  nt: number;
  ne: number;
};

type Buffers = {
  u: GPUBuffer;
  uHalf: GPUBuffer;
  uNext: GPUBuffer;
  w: GPUBuffer;
  zrA: GPUBuffer;
  ziA: GPUBuffer;
  zrB: GPUBuffer;
  ziB: GPUBuffer;
  X: GPUBuffer;
  XMid: GPUBuffer;
  XNext: GPUBuffer;
  F: GPUBuffer;
  ff: GPUBuffer;
  triangles: GPUBuffer;
  edges: GPUBuffer;
  starts: GPUBuffer;
  neighbors: GPUBuffer;
  restLengths: GPUBuffer;
  field: GPUBuffer;
};

type PipelineKey =
  | "predict"
  | "force"
  | "spread"
  | "rhsHalf"
  | "rhsFull"
  | "fftXReal"
  | "fftY"
  | "fftZ"
  | "project"
  | "idftZ"
  | "idftY"
  | "idftXReal"
  | "updateX"
  | "impulse"
  | "field";

const PARAM_BYTES = 48;
const IMPULSE_BYTES = 32;
const WORKGROUP = 128;
type CachedBinding = { buffers: GPUBuffer[]; group: GPUBindGroup };

export class WebGpuIb3DSolver {
  readonly mesh: Mesh;
  #buffers: Buffers;
  #params: IB3DParams;
  #paramBuffer: GPUBuffer;
  #impulseBuffer: GPUBuffer;
  #fieldModeBuffer: GPUBuffer;
  #grabBuffer: GPUBuffer;
  #pipelines = new Map<PipelineKey, GPUComputePipeline>();
  #bindings = new Map<PipelineKey, CachedBinding[]>();
  #fieldMode: IB3DFieldMode | undefined;
  #destroyed = false;

  private constructor(
    readonly device: GPUDevice,
    params: IB3DParams,
    mesh: Mesh,
    buffers: Buffers,
    paramBuffer: GPUBuffer,
    impulseBuffer: GPUBuffer,
    fieldModeBuffer: GPUBuffer,
    pipelines: Map<PipelineKey, GPUComputePipeline>,
  ) {
    this.#params = { ...params };
    this.mesh = mesh;
    this.#buffers = buffers;
    this.#paramBuffer = paramBuffer;
    this.#impulseBuffer = impulseBuffer;
    this.#fieldModeBuffer = fieldModeBuffer;
    this.#grabBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.#pipelines = pipelines;
    this.#writeParams();
  }

  static async init(device: GPUDevice, params: IB3DParams) {
    validateParams(params);
    // Compile before allocating simulation state so shader failures do not leak buffers.
    const pipelines = await createPipelines(device, params);
    const mesh = createSphereMesh(params.refinement, params.radius);
    const buffers = createBuffers(device, params, mesh);
    const paramBuffer = device.createBuffer({
      size: PARAM_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const impulseBuffer = device.createBuffer({
      size: IMPULSE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const fieldModeBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    return new WebGpuIb3DSolver(
      device,
      params,
      mesh,
      buffers,
      paramBuffer,
      impulseBuffer,
      fieldModeBuffer,
      pipelines,
    );
  }

  get vertexBuffer() {
    return this.#buffers.X;
  }

  get velocityBuffer() {
    return this.#buffers.u;
  }

  /** A temporary tether force on one material point, spread through the usual IB coupling. */
  setGrab(
    vertex = -1,
    target: [number, number, number] = [0, 0, 0],
    strength = 0.02,
  ) {
    this.#assertAlive();
    const data = new Float32Array([
      target[0],
      target[1],
      target[2],
      strength,
      vertex,
      0,
      0,
      0,
    ]);
    this.device.queue.writeBuffer(this.#grabBuffer, 0, data);
  }

  async readPositions() {
    this.#assertAlive();
    const source = this.vertexBuffer;
    const staging = this.device.createBuffer({
      size: source.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(source, 0, staging, 0, source.size);
      this.device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      return new Float32Array(staging.getMappedRange()).slice();
    } finally {
      staging.destroy();
    }
  }

  get triangleBuffer() {
    return this.#buffers.triangles;
  }

  get edgeBuffer() {
    return this.#buffers.edges;
  }

  get fieldBuffer() {
    return this.#buffers.field;
  }

  get params() {
    return this.#params;
  }

  reset(params = this.#params) {
    this.#assertAlive();
    validateParams(params);
    if (
      params.N !== this.#params.N ||
      params.refinement !== this.#params.refinement ||
      params.radius !== this.#params.radius
    ) {
      throw new Error("IB3D reset cannot resize grid or mesh");
    }
    this.#params = { ...params };
    this.setGrab();
    this.#writeParams();
    this.device.queue.writeBuffer(
      this.#buffers.u,
      0,
      gpuData(initialVelocity(params)),
    );
    this.device.queue.writeBuffer(
      this.#buffers.X,
      0,
      gpuData(this.mesh.positions),
    );
  }

  setParams(params: IB3DParams) {
    this.#assertAlive();
    validateParams(params);
    if (
      params.N !== this.#params.N ||
      params.refinement !== this.#params.refinement ||
      params.radius !== this.#params.radius
    ) {
      throw new Error("Rebuild the 3D solver to change its grid or mesh");
    }
    this.#params = { ...params };
    this.#writeParams();
  }

  applyImpulse(
    center: [number, number, number],
    force: [number, number, number],
  ) {
    this.#assertAlive();
    this.device.queue.writeBuffer(
      this.#impulseBuffer,
      0,
      gpuData(
        new Float32Array([
          center[0],
          center[1],
          center[2],
          0.08,
          force[0],
          force[1],
          force[2],
          0,
        ]),
      ),
    );
    const encoder = this.device.createCommandEncoder();
    this.#dispatch(
      encoder,
      "impulse",
      [this.#paramBuffer, this.#impulseBuffer, this.#buffers.u],
      [Math.ceil(this.#params.N ** 3 / WORKGROUP)],
    );
    this.device.queue.submit([encoder.finish()]);
  }

  renderField(mode: IB3DFieldMode) {
    this.#assertAlive();
    const encoder = this.device.createCommandEncoder();
    const field = this.encodeField(encoder, mode);
    this.device.queue.submit([encoder.finish()]);
    return field;
  }

  encodeField(encoder: GPUCommandEncoder, mode: IB3DFieldMode) {
    this.#assertAlive();
    if (mode !== this.#fieldMode) {
      this.device.queue.writeBuffer(
        this.#fieldModeBuffer,
        0,
        gpuData(new Uint32Array([mode === "vorticity" ? 1 : 0])),
      );
      this.#fieldMode = mode;
    }
    this.#dispatch(
      encoder,
      "field",
      [
        this.#paramBuffer,
        this.#fieldModeBuffer,
        this.#buffers.u,
        this.#buffers.field,
      ],
      [Math.ceil(this.#params.N ** 3 / WORKGROUP)],
    );
    return this.#buffers.field;
  }

  stepBatch(count = 1, externalEncoder?: GPUCommandEncoder) {
    this.#assertAlive();
    const encoder = externalEncoder ?? this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (let i = 0; i < count; i++) {
      this.#encodeStep(pass);
      [this.#buffers.u, this.#buffers.uNext] = [
        this.#buffers.uNext,
        this.#buffers.u,
      ];
      [this.#buffers.X, this.#buffers.XNext] = [
        this.#buffers.XNext,
        this.#buffers.X,
      ];
    }
    pass.end();
    if (!externalEncoder) this.device.queue.submit([encoder.finish()]);
  }

  destroy() {
    if (this.#destroyed) return;
    this.#destroyed = true;
    for (const buffer of Object.values(this.#buffers)) buffer.destroy();
    this.#paramBuffer.destroy();
    this.#impulseBuffer.destroy();
    this.#fieldModeBuffer.destroy();
    this.#grabBuffer.destroy();
    this.#bindings.clear();
  }

  #encodeStep(encoder: GPUComputePassEncoder) {
    const grid = [Math.ceil(this.#params.N ** 3 / WORKGROUP)];
    const bd = [Math.ceil(this.mesh.nb / WORKGROUP)];
    this.#dispatch(
      encoder,
      "predict",
      [this.#paramBuffer, this.#buffers.u, this.#buffers.X, this.#buffers.XMid],
      bd,
    );
    this.#dispatch(
      encoder,
      "force",
      [
        this.#paramBuffer,
        this.#buffers.XMid,
        this.#buffers.starts,
        this.#buffers.neighbors,
        this.#buffers.F,
        this.#grabBuffer,
        this.#buffers.restLengths,
      ],
      bd,
    );
    this.#dispatch(
      encoder,
      "spread",
      [
        this.#paramBuffer,
        this.#buffers.XMid,
        this.#buffers.F,
        this.#buffers.ff,
      ],
      grid,
    );
    this.#dispatch(
      encoder,
      "rhsHalf",
      [
        this.#paramBuffer,
        this.#buffers.u,
        this.#buffers.u,
        this.#buffers.ff,
        this.#buffers.w,
      ],
      grid,
    );
    this.#encodeFluidSolve(encoder, this.#buffers.w, this.#buffers.uHalf);
    this.#dispatch(
      encoder,
      "rhsFull",
      [
        this.#paramBuffer,
        this.#buffers.u,
        this.#buffers.uHalf,
        this.#buffers.ff,
        this.#buffers.w,
      ],
      grid,
    );
    this.#encodeFluidSolve(encoder, this.#buffers.w, this.#buffers.uNext);
    this.#dispatch(
      encoder,
      "updateX",
      [
        this.#paramBuffer,
        this.#buffers.uHalf,
        this.#buffers.X,
        this.#buffers.XMid,
        this.#buffers.XNext,
      ],
      bd,
    );
  }

  #encodeFluidSolve(
    encoder: GPUComputePassEncoder,
    input: GPUBuffer,
    output: GPUBuffer,
  ) {
    const axisLines = [this.#params.N * this.#params.N];
    this.#dispatch(
      encoder,
      "fftXReal",
      [this.#paramBuffer, input, this.#buffers.zrA, this.#buffers.ziA],
      axisLines,
    );
    this.#dispatch(
      encoder,
      "fftY",
      [
        this.#paramBuffer,
        this.#buffers.zrA,
        this.#buffers.ziA,
        this.#buffers.zrB,
        this.#buffers.ziB,
      ],
      axisLines,
    );
    this.#dispatch(
      encoder,
      "fftZ",
      [
        this.#paramBuffer,
        this.#buffers.zrB,
        this.#buffers.ziB,
        this.#buffers.zrA,
        this.#buffers.ziA,
      ],
      axisLines,
    );
    this.#dispatch(
      encoder,
      "project",
      [
        this.#paramBuffer,
        this.#buffers.zrA,
        this.#buffers.ziA,
        this.#buffers.zrB,
        this.#buffers.ziB,
      ],
      [Math.ceil(this.#params.N ** 3 / WORKGROUP)],
    );
    this.#dispatch(
      encoder,
      "idftZ",
      [
        this.#paramBuffer,
        this.#buffers.zrB,
        this.#buffers.ziB,
        this.#buffers.zrA,
        this.#buffers.ziA,
      ],
      axisLines,
    );
    this.#dispatch(
      encoder,
      "idftY",
      [
        this.#paramBuffer,
        this.#buffers.zrA,
        this.#buffers.ziA,
        this.#buffers.zrB,
        this.#buffers.ziB,
      ],
      axisLines,
    );
    this.#dispatch(
      encoder,
      "idftXReal",
      [this.#paramBuffer, this.#buffers.zrB, this.#buffers.ziB, output],
      axisLines,
    );
  }

  #dispatch(
    encoder: GPUCommandEncoder | GPUComputePassEncoder,
    key: PipelineKey,
    buffers: GPUBuffer[],
    workgroups: number[],
  ) {
    const pipeline = this.#pipelines.get(key);
    if (!pipeline) throw new Error(`Missing IB3D pipeline ${key}`);
    const cached = this.#bindings.get(key) ?? [];
    let binding = cached.find((entry) =>
      entry.buffers.every((buffer, i) => buffer === buffers[i]),
    );
    if (!binding) {
      binding = {
        buffers,
        group: this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({
            binding,
            resource: { buffer },
          })),
        }),
      };
      cached.push(binding);
      this.#bindings.set(key, cached);
    }
    const standalone = "beginComputePass" in encoder;
    const pass = standalone ? encoder.beginComputePass() : encoder;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, binding.group);
    pass.dispatchWorkgroups(
      workgroups[0],
      workgroups[1] ?? 1,
      workgroups[2] ?? 1,
    );
    if (standalone) pass.end();
  }

  #writeParams() {
    const p = this.#params;
    const data = new ArrayBuffer(PARAM_BYTES);
    const view = new DataView(data);
    view.setUint32(0, p.N, true);
    view.setUint32(4, this.mesh.nb, true);
    view.setUint32(8, this.mesh.ne, true);
    view.setUint32(12, this.mesh.nt, true);
    view.setFloat32(16, 1 / p.N, true);
    view.setFloat32(20, p.dt, true);
    view.setFloat32(24, p.K, true);
    view.setFloat32(28, p.mu, true);
    view.setFloat32(36, p.radius, true);
    this.device.queue.writeBuffer(this.#paramBuffer, 0, data);
  }

  #assertAlive() {
    if (this.#destroyed) throw new Error("IB3D solver has been destroyed");
  }
}

function validateParams(p: IB3DParams) {
  if (!Number.isInteger(p.N) || p.N < 4 || p.N > 128 || (p.N & (p.N - 1)) !== 0)
    throw new Error("Grid must be a power of two from 4 to 128");
  if (
    ![p.dt, p.K, p.mu, p.radius].every(Number.isFinite) ||
    p.dt <= 0 ||
    p.K < 0 ||
    p.mu < 0 ||
    p.radius <= 0 ||
    p.radius >= 0.5
  )
    throw new Error("Invalid 3D physical parameters");
  if (!Number.isInteger(p.refinement) || p.refinement < 0 || p.refinement > 4)
    throw new Error("Invalid mesh refinement");
}

function createBuffers(
  device: GPUDevice,
  params: IB3DParams,
  mesh: Mesh,
): Buffers {
  const n3 = params.N ** 3;
  const gridBytes = n3 * 16;
  const bdBytes = mesh.nb * 16;
  return {
    u: dataBuffer(
      device,
      initialVelocity(params),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    ),
    uHalf: emptyBuffer(device, gridBytes, GPUBufferUsage.STORAGE),
    uNext: emptyBuffer(
      device,
      gridBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    ),
    w: emptyBuffer(device, gridBytes, GPUBufferUsage.STORAGE),
    zrA: emptyBuffer(device, gridBytes, GPUBufferUsage.STORAGE),
    ziA: emptyBuffer(device, gridBytes, GPUBufferUsage.STORAGE),
    zrB: emptyBuffer(device, gridBytes, GPUBufferUsage.STORAGE),
    ziB: emptyBuffer(device, gridBytes, GPUBufferUsage.STORAGE),
    X: dataBuffer(
      device,
      mesh.positions,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    ),
    XMid: emptyBuffer(device, bdBytes, GPUBufferUsage.STORAGE),
    XNext: emptyBuffer(
      device,
      bdBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    ),
    F: emptyBuffer(device, bdBytes, GPUBufferUsage.STORAGE),
    ff: emptyBuffer(device, gridBytes, GPUBufferUsage.STORAGE),
    triangles: dataBuffer(device, mesh.triangles, GPUBufferUsage.STORAGE),
    edges: dataBuffer(device, mesh.edges, GPUBufferUsage.STORAGE),
    starts: dataBuffer(device, mesh.starts, GPUBufferUsage.STORAGE),
    neighbors: dataBuffer(device, mesh.neighbors, GPUBufferUsage.STORAGE),
    restLengths: dataBuffer(device, mesh.restLengths, GPUBufferUsage.STORAGE),
    field: emptyBuffer(device, n3 * 16, GPUBufferUsage.STORAGE),
  };
}

function emptyBuffer(
  device: GPUDevice,
  size: number,
  usage: GPUBufferUsageFlags,
) {
  return device.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_SRC });
}

function dataBuffer(
  device: GPUDevice,
  data: Float32Array | Uint32Array,
  usage: GPUBufferUsageFlags,
) {
  const buffer = device.createBuffer({
    size: align4(data.byteLength),
    usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buffer, 0, gpuData(data));
  return buffer;
}

function gpuData(data: Float32Array | Uint32Array): GPUAllowSharedBufferSource {
  return data as unknown as GPUAllowSharedBufferSource;
}

function align4(n: number) {
  return Math.ceil(n / 4) * 4;
}

function initialVelocity({ N }: IB3DParams) {
  const u = new Float32Array(N ** 3 * 4);
  const amp = 0.2;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      for (let k = 0; k < N; k++) {
        const p = ((i * N + j) * N + k) * 4;
        u[p] = amp * Math.sin((2 * Math.PI * k) / N);
        u[p + 1] = 0;
        u[p + 2] = amp * Math.sin((2 * Math.PI * j) / N);
      }
    }
  }
  return u;
}

function createSphereMesh(refinement: number, radius: number): Mesh {
  const theta = (2 * Math.PI) / 5;
  const z = Math.cos(theta) / (1 - Math.cos(theta));
  const r = Math.sqrt(1 - z * z);
  let vertices: number[][] = [[0, 0, 1]];
  for (let j = 1; j <= 5; j++)
    vertices.push([r * Math.cos(j * theta), r * Math.sin(j * theta), z]);
  for (let j = 1; j <= 5; j++) {
    const k = j - 0.5;
    vertices.push([r * Math.cos(k * theta), r * Math.sin(k * theta), -z]);
  }
  vertices.push([0, 0, -1]);
  let tris = [
    [0, 1, 2],
    [0, 2, 3],
    [0, 3, 4],
    [0, 4, 5],
    [0, 5, 1],
    [1, 7, 2],
    [2, 8, 3],
    [3, 9, 4],
    [4, 10, 5],
    [5, 6, 1],
    [7, 1, 6],
    [8, 2, 7],
    [9, 3, 8],
    [10, 4, 9],
    [6, 5, 10],
    [6, 11, 7],
    [7, 11, 8],
    [8, 11, 9],
    [9, 11, 10],
    [10, 11, 6],
  ];
  for (let n = 0; n < refinement; n++) {
    const midpoint = new Map<string, number>();
    const next: number[][] = [];
    const mid = (a: number, b: number) => {
      const key = a < b ? `${a}:${b}` : `${b}:${a}`;
      const cached = midpoint.get(key);
      if (cached !== undefined) return cached;
      const p = normalize([
        vertices[a][0] + vertices[b][0],
        vertices[a][1] + vertices[b][1],
        vertices[a][2] + vertices[b][2],
      ]);
      vertices.push(p);
      midpoint.set(key, vertices.length - 1);
      return vertices.length - 1;
    };
    for (const [a, b, c] of tris) {
      const ab = mid(a, b),
        bc = mid(b, c),
        ca = mid(c, a);
      next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    }
    tris = next;
  }
  const edgeMap = new Map<string, [number, number]>();
  for (const [a, b, c] of tris)
    for (const [u, v] of [
      [a, b],
      [b, c],
      [c, a],
    ]) {
      const key = u < v ? `${u}:${v}` : `${v}:${u}`;
      if (!edgeMap.has(key)) edgeMap.set(key, u < v ? [u, v] : [v, u]);
    }
  const edges = [...edgeMap.values()];
  const adj = Array.from({ length: vertices.length }, () => [] as number[]);
  for (const [a, b] of edges) {
    adj[a].push(b);
    adj[b].push(a);
  }
  const starts = new Uint32Array(vertices.length * 2);
  const neighbors = new Uint32Array(adj.reduce((s, a) => s + a.length, 0));
  let offset = 0;
  adj.forEach((items, i) => {
    starts[i * 2] = offset;
    starts[i * 2 + 1] = items.length;
    neighbors.set(items, offset);
    offset += items.length;
  });
  const positions = new Float32Array(vertices.length * 4);
  vertices.forEach((p, i) => {
    positions[i * 4] = 0.5 + radius * p[0];
    positions[i * 4 + 1] = 0.5 + radius * p[1];
    positions[i * 4 + 2] = 0.5 + radius * p[2];
    positions[i * 4 + 3] = 1;
  });
  // Each material edge is relaxed in the initial geometry; never reset these
  // lengths during deformation or wrap material edges across the periodic box.
  const restLengths = new Float32Array(neighbors.length);
  adj.forEach((items, i) => {
    items.forEach((j, a) => {
      restLengths[starts[i * 2] + a] = Math.hypot(
        positions[j * 4] - positions[i * 4],
        positions[j * 4 + 1] - positions[i * 4 + 1],
        positions[j * 4 + 2] - positions[i * 4 + 2],
      );
    });
  });
  return {
    positions,
    triangles: new Uint32Array(tris.flat()),
    edges: new Uint32Array(edges.flat()),
    starts,
    neighbors,
    restLengths,
    nb: vertices.length,
    nt: tris.length,
    ne: edges.length,
  };
}

function normalize(p: number[]) {
  const s = Math.hypot(p[0], p[1], p[2]);
  return [p[0] / s, p[1] / s, p[2] / s];
}

async function createPipelines(device: GPUDevice, params: IB3DParams) {
  const pipelines = new Map<PipelineKey, GPUComputePipeline>();
  const shaders: Record<PipelineKey, string> = {
    predict: predictShader(params),
    force: forceShader(params),
    spread: spreadShader(params),
    rhsHalf: rhsShader(params, true),
    rhsFull: rhsShader(params, false),
    fftXReal: fftXRealShader(params),
    fftY: fftAxisShader(params, "y", false),
    fftZ: fftAxisShader(params, "z", false),
    project: projectShader(params),
    idftZ: fftAxisShader(params, "z", true),
    idftY: fftAxisShader(params, "y", true),
    idftXReal: idftXRealShader(params),
    updateX: updateShader(params),
    impulse: impulseShader(params),
    field: fieldShader(params),
  };
  for (const [key, code] of Object.entries(shaders) as [
    PipelineKey,
    string,
  ][]) {
    pipelines.set(
      key,
      await device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module: device.createShaderModule({ code }),
          entryPoint: "main",
        },
      }),
    );
  }
  return pipelines;
}

function common({ N }: IB3DParams) {
  const logN = Math.round(Math.log2(N));
  return /* wgsl */ `
const N: u32 = ${N}u;
const LOG_N: u32 = ${logN}u;
const PI: f32 = 3.141592653589793;
const TWO_PI: f32 = 6.283185307179586;

struct Params {
  n: u32, nb: u32, ne: u32, nt: u32,
  h: f32, dt: f32, k: f32, mu: f32,
  reserved: f32, radius: f32, pad0: f32, pad1: f32,
}

fn wrapi(i: i32) -> u32 {
  let ni = i32(N);
  return u32(((i % ni) + ni) % ni);
}
fn idx3(i: u32, j: u32, k: u32) -> u32 {
  return ((i * N + j) * N + k);
}
fn wrap01(x: vec3f) -> vec3f {
  return x - floor(x);
}
fn periodicDelta(a: vec3f, b: vec3f) -> vec3f {
  var d = a - b;
  d = d - round(d);
  return d;
}
fn phi(r: f32) -> f32 {
  let x = abs(r);
  if (x < 1.0) { return (3.0 - 2.0 * x + sqrt(1.0 + 4.0 * x - 4.0 * x * x)) * 0.125; }
  if (x < 2.0) { return (5.0 - 2.0 * x - sqrt(-7.0 + 12.0 * x - 4.0 * x * x)) * 0.125; }
  return 0.0;
}
fn gridDelta(cell: u32, s: f32) -> f32 {
  var d = f32(cell) - s;
  let nf = f32(N);
  d = d - round(d / nf) * nf;
  return d;
}
fn getv(u: ptr<storage, array<vec4f>, read>, i: u32, j: u32, k: u32) -> vec3f {
  return (*u)[idx3(i, j, k)].xyz;
}
fn pick(v: vec3f, c: u32) -> f32 {
  if (c == 0u) { return v.x; }
  if (c == 1u) { return v.y; }
  return v.z;
}
fn interp(u: ptr<storage, array<vec4f>, read>, p0: vec3f) -> vec3f {
  let s = wrap01(p0) * f32(N);
  let base = vec3i(floor(s));
  var out = vec3f(0.0);
  for (var di = -1; di <= 2; di++) {
    let ii = wrapi(base.x + di);
    let wx = phi(f32(base.x + di) - s.x);
    for (var dj = -1; dj <= 2; dj++) {
      let jj = wrapi(base.y + dj);
      let wy = phi(f32(base.y + dj) - s.y);
      for (var dk = -1; dk <= 2; dk++) {
        let kk = wrapi(base.z + dk);
        let wz = phi(f32(base.z + dk) - s.z);
        out += getv(u, ii, jj, kk) * (wx * wy * wz);
      }
    }
  }
  return out;
}

fn skewComponent(i: u32, j: u32, k: u32, c: u32, u: ptr<storage, array<vec4f>, read>) -> f32 {
  let ip = (i + 1u) % N;
  let im = (i + N - 1u) % N;
  let jp = (j + 1u) % N;
  let jm = (j + N - 1u) % N;
  let kp = (k + 1u) % N;
  let km = (k + N - 1u) % N;
  let uc = getv(u, i, j, k);
  let vip = getv(u, ip, j, k);
  let vim = getv(u, im, j, k);
  let vjp = getv(u, i, jp, k);
  let vjm = getv(u, i, jm, k);
  let vkp = getv(u, i, j, kp);
  let vkm = getv(u, i, j, km);
  return (
    (vip.x + uc.x) * pick(vip, c) - (vim.x + uc.x) * pick(vim, c) +
    (vjp.y + uc.y) * pick(vjp, c) - (vjm.y + uc.y) * pick(vjm, c) +
    (vkp.z + uc.z) * pick(vkp, c) - (vkm.z + uc.z) * pick(vkm, c)
  ) / (4.0 * p_h()); // Half of (advective + conservative), in any dimension.
}

fn skewVec(i: u32, j: u32, k: u32, u: ptr<storage, array<vec4f>, read>) -> vec3f {
  return vec3f(skewComponent(i, j, k, 0u, u), skewComponent(i, j, k, 1u, u), skewComponent(i, j, k, 2u, u));
}

fn lapVec(i: u32, j: u32, k: u32, u: ptr<storage, array<vec4f>, read>) -> vec3f {
  let ip = (i + 1u) % N;
  let im = (i + N - 1u) % N;
  let jp = (j + 1u) % N;
  let jm = (j + N - 1u) % N;
  let kp = (k + 1u) % N;
  let km = (k + N - 1u) % N;
  return (
    getv(u, ip, j, k) + getv(u, im, j, k) +
    getv(u, i, jp, k) + getv(u, i, jm, k) +
    getv(u, i, j, kp) + getv(u, i, j, km) -
    6.0 * getv(u, i, j, k)
  ) / (p_h() * p_h());
}

fn bitReverse(x0: u32) -> u32 {
  var x = x0;
  var y = 0u;
  for (var i = 0u; i < LOG_N; i++) {
    y = (y << 1u) | (x & 1u);
    x = x >> 1u;
  }
  return y;
}

fn p_h() -> f32 { return params0.h; }
`;
}

function predictShader(params: IB3DParams) {
  return (
    common(params) +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> params0: Params;
@group(0) @binding(1) var<storage, read> u: array<vec4f>;
@group(0) @binding(2) var<storage, read> x: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> xMid: array<vec4f>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  if (id >= params0.nb) { return; }
  let v = interp(&u, x[id].xyz);
  xMid[id] = vec4f(x[id].xyz + 0.5 * params0.dt * v, 1.0);
}`
  );
}

function forceShader(params: IB3DParams) {
  return (
    common(params) +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> params0: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4f>;
@group(0) @binding(2) var<storage, read> starts: array<vec2u>;
@group(0) @binding(3) var<storage, read> neighbors: array<u32>;
@group(0) @binding(4) var<storage, read_write> f: array<vec4f>;
@group(0) @binding(5) var<storage, read> grab: array<vec4f>;
@group(0) @binding(6) var<storage, read> restLengths: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  if (id >= params0.nb) { return; }
  let sc = starts[id];
  let xi = x[id].xyz;
  var force = vec3f(0.0);
  for (var a = 0u; a < sc.y; a++) {
    let j = neighbors[sc.x + a];
    let edge = x[j].xyz - xi;
    let len = length(edge);
    // A collapsed edge has no defined direction; avoid a division by zero.
    if (len > 1e-8) {
      force += params0.k * (len - restLengths[sc.x + a]) * (edge / len);
    }
  }
  if (grab[0].w > 0.0 && i32(id) == i32(grab[1].x)) {
    let pull = grab[0].w * (grab[0].xyz - xi);
    force += pull * min(1.0, 0.01 / max(length(pull), 1e-8));
  }
  f[id] = vec4f(force, 0.0);
}`
  );
}

function spreadShader(params: IB3DParams) {
  return (
    common(params) +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> params0: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4f>;
@group(0) @binding(2) var<storage, read> f: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> ff: array<vec4f>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  let n3 = N * N * N;
  if (id >= n3) { return; }
  let i = id / (N * N);
  let j = (id / N) % N;
  let k = id % N;
  var out = vec3f(0.0);
  for (var q = 0u; q < params0.nb; q++) {
    let s = wrap01(x[q].xyz) * f32(N);
    let wx = phi(gridDelta(i, s.x));
    if (wx == 0.0) { continue; }
    let wy = phi(gridDelta(j, s.y));
    if (wy == 0.0) { continue; }
    let wz = phi(gridDelta(k, s.z));
    out += f[q].xyz * (wx * wy * wz);
  }
  ff[id] = vec4f(out / (params0.h * params0.h * params0.h), 0.0);
}`
  );
}

function rhsShader(params: IB3DParams, halfStep: boolean) {
  const scale = halfStep ? "0.5" : "1.0";
  const lap = halfStep
    ? ""
    : "w += 0.5 * params0.dt * params0.mu * lapVec(i, j, k, &base);";
  return (
    common(params) +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> params0: Params;
@group(0) @binding(1) var<storage, read> base: array<vec4f>;
@group(0) @binding(2) var<storage, read> adv: array<vec4f>;
@group(0) @binding(3) var<storage, read> ff: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> outW: array<vec4f>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  let n3 = N * N * N;
  if (id >= n3) { return; }
  let i = id / (N * N);
  let j = (id / N) % N;
  let k = id % N;
  var w = base[id].xyz - ${scale} * params0.dt * skewVec(i, j, k, &adv) + ${scale} * params0.dt * ff[id].xyz;
  ${lap}
  outW[id] = vec4f(w, 0.0);
}`
  );
}

function fftCommon(params: IB3DParams) {
  return (
    common(params) +
    /* wgsl */ `
var<workgroup> fftRe: array<vec4f, N>;
var<workgroup> fftIm: array<vec4f, N>;

fn fftShared(tid: u32, inverse: bool) {
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
      let wr = cos(angle);
      let wi = sin(angle);
      let ar = fftRe[i0];
      let ai = fftIm[i0];
      let br0 = fftRe[i1];
      let bi0 = fftIm[i1];
      let br = br0 * wr - bi0 * wi;
      let bi = br0 * wi + bi0 * wr;
      fftRe[i0] = ar + br;
      fftIm[i0] = ai + bi;
      fftRe[i1] = ar - br;
      fftIm[i1] = ai - bi;
    }
    workgroupBarrier();
  }
}
`
  );
}

function fftXRealShader(params: IB3DParams) {
  return (
    fftCommon(params) +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> params0: Params;
@group(0) @binding(1) var<storage, read> realIn: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> zrOut: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> ziOut: array<vec4f>;
@compute @workgroup_size(${params.N})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let tid = lid.x;
  if (params0.n == 0u) { return; }
  let line = wg.x;
  let j = line / N;
  let k = line % N;
  let src = bitReverse(tid);
  fftRe[tid] = vec4f(realIn[idx3(src, j, k)].xyz, 0.0);
  fftIm[tid] = vec4f(0.0);
  fftShared(tid, false);
  zrOut[idx3(tid, j, k)] = fftRe[tid];
  ziOut[idx3(tid, j, k)] = fftIm[tid];
}`
  );
}

function fftAxisShader(params: IB3DParams, axis: "y" | "z", inverse: boolean) {
  const load =
    axis === "y"
      ? "let i = line / N; let k = line % N; let src = bitReverse(tid); let dst = tid; let readIdx = idx3(i, src, k); let writeIdx = idx3(i, dst, k);"
      : "let i = line / N; let j = line % N; let src = bitReverse(tid); let dst = tid; let readIdx = idx3(i, j, src); let writeIdx = idx3(i, j, dst);";
  return (
    fftCommon(params) +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> params0: Params;
@group(0) @binding(1) var<storage, read> zrIn: array<vec4f>;
@group(0) @binding(2) var<storage, read> ziIn: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> zrOut: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> ziOut: array<vec4f>;
@compute @workgroup_size(${params.N})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let tid = lid.x;
  if (params0.n == 0u) { return; }
  let line = wg.x;
  ${load}
  fftRe[tid] = zrIn[readIdx];
  fftIm[tid] = ziIn[readIdx];
  fftShared(tid, ${inverse ? "true" : "false"});
  zrOut[writeIdx] = fftRe[tid];
  ziOut[writeIdx] = fftIm[tid];
}`
  );
}

function projectShader(params: IB3DParams) {
  return (
    common(params) +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> params0: Params;
@group(0) @binding(1) var<storage, read> zrIn: array<vec4f>;
@group(0) @binding(2) var<storage, read> ziIn: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> zrOut: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> ziOut: array<vec4f>;
fn applyOp(v: vec3f, s: vec3f, factor: f32, special: bool) -> vec3f {
  if (special) { return v / factor; }
  return (v - s * (dot(s, v) / dot(s, s))) / factor;
}
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  let n3 = N * N * N;
  if (id >= n3) { return; }
  let m1 = id / (N * N);
  let m2 = (id / N) % N;
  let m3 = id % N;
  let special =
    (m1 == 0u || m1 == N / 2u) &&
    (m2 == 0u || m2 == N / 2u) &&
    (m3 == 0u || m3 == N / 2u);
  let s = vec3f(
    sin(TWO_PI * f32(m1) / f32(N)),
    sin(TWO_PI * f32(m2) / f32(N)),
    sin(TWO_PI * f32(m3) / f32(N)),
  );
  let d = vec3f(
    sin(PI * f32(m1) / f32(N)),
    sin(PI * f32(m2) / f32(N)),
    sin(PI * f32(m3) / f32(N)),
  );
  let factor = 1.0 + 0.5 * params0.dt * params0.mu * 4.0 / (params0.h * params0.h) * dot(d, d);
  zrOut[id] = vec4f(applyOp(zrIn[id].xyz, s, factor, special), 0.0);
  ziOut[id] = vec4f(applyOp(ziIn[id].xyz, s, factor, special), 0.0);
}`
  );
}

function idftXRealShader(params: IB3DParams) {
  return (
    fftCommon(params) +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> params0: Params;
@group(0) @binding(1) var<storage, read> zrIn: array<vec4f>;
@group(0) @binding(2) var<storage, read> ziIn: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outU: array<vec4f>;
@compute @workgroup_size(${params.N})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let tid = lid.x;
  let line = wg.x;
  let j = line / N;
  let k = line % N;
  let src = bitReverse(tid);
  fftRe[tid] = zrIn[idx3(src, j, k)];
  fftIm[tid] = ziIn[idx3(src, j, k)];
  fftShared(tid, true);
  let scale = 1.0 / f32(params0.n * params0.n * params0.n);
  outU[idx3(tid, j, k)] = vec4f(fftRe[tid].xyz * scale, 0.0);
}`
  );
}

function updateShader(params: IB3DParams) {
  return (
    common(params) +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> params0: Params;
@group(0) @binding(1) var<storage, read> uHalf: array<vec4f>;
@group(0) @binding(2) var<storage, read> x: array<vec4f>;
@group(0) @binding(3) var<storage, read> xMid: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> xNext: array<vec4f>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  if (id >= params0.nb) { return; }
  let v = interp(&uHalf, xMid[id].xyz);
  xNext[id] = vec4f(x[id].xyz + params0.dt * v, 1.0);
}`
  );
}

function impulseShader(params: IB3DParams) {
  return (
    common(params) +
    /* wgsl */ `
struct Impulse { center: vec4f, force: vec4f }
@group(0) @binding(0) var<storage, read> params0: Params;
@group(0) @binding(1) var<storage, read> impulse: Impulse;
@group(0) @binding(2) var<storage, read_write> u: array<vec4f>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  let n3 = N * N * N;
  if (id >= n3) { return; }
  let i = id / (N * N);
  let j = (id / N) % N;
  let k = id % N;
  let pos = vec3f(f32(i), f32(j), f32(k)) * params0.h;
  let d = periodicDelta(pos, impulse.center.xyz);
  let g = exp(-dot(d, d) / (2.0 * impulse.center.w * impulse.center.w));
  u[id] = vec4f(u[id].xyz + params0.dt * impulse.force.xyz * g, 0.0);
}`
  );
}

function fieldShader(params: IB3DParams) {
  return (
    common(params) +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> params0: Params;
@group(0) @binding(1) var<storage, read> mode: array<u32>;
@group(0) @binding(2) var<storage, read> u: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> field: array<vec4f>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  let n3 = N * N * N;
  if (id >= n3) { return; }
  let i = id / (N * N);
  let j = (id / N) % N;
  let k = id % N;
  if (mode[0] == 0u) {
    field[id] = vec4f(u[id].xyz, length(u[id].xyz));
    return;
  }
  let ip = (i + 1u) % N;
  let im = (i + N - 1u) % N;
  let jp = (j + 1u) % N;
  let jm = (j + N - 1u) % N;
  let kp = (k + 1u) % N;
  let km = (k + N - 1u) % N;
  let dvz_dy = (getv(&u, i, jp, k).z - getv(&u, i, jm, k).z) / (2.0 * params0.h);
  let dvy_dz = (getv(&u, i, j, kp).y - getv(&u, i, j, km).y) / (2.0 * params0.h);
  let dvx_dz = (getv(&u, i, j, kp).x - getv(&u, i, j, km).x) / (2.0 * params0.h);
  let dvz_dx = (getv(&u, ip, j, k).z - getv(&u, im, j, k).z) / (2.0 * params0.h);
  let dvy_dx = (getv(&u, ip, j, k).y - getv(&u, im, j, k).y) / (2.0 * params0.h);
  let dvx_dy = (getv(&u, i, jp, k).x - getv(&u, i, jm, k).x) / (2.0 * params0.h);
  let curl = vec3f(dvz_dy - dvy_dz, dvx_dz - dvz_dx, dvy_dx - dvx_dy);
  field[id] = vec4f(curl, length(curl));
}`
  );
}
