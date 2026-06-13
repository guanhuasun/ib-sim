<script lang="ts">
  import { defaultDevice, getWebGPUDevice, init } from "@jax-js/jax";
  import { onMount } from "svelte";

  import { WebGpuIb3DSolver, type IB3DFieldMode, type IB3DParams } from "../ib3d-solver";

  let canvas: HTMLCanvasElement;
  let device: GPUDevice;
  let context: GPUCanvasContext;
  let format: GPUTextureFormat;
  let volumePipeline: GPURenderPipeline;
  let surfacePipeline: GPURenderPipeline;
  let edgePipeline: GPURenderPipeline;
  let uniformBuffer: GPUBuffer;
  let depthTexture: GPUTexture | null = null;
  let depthView: GPUTextureView | null = null;
  let solver: WebGpuIb3DSolver | null = null;
  let running = false;
  let pointerDown = false;
  let lastX = 0;
  let lastY = 0;
  let yaw = $state(-0.65);
  let pitch = $state(0.35);
  let step = $state(0);
  let fps = $state(0);
  let simTime = $state(0);
  let paused = $state(false);
  let errorMsg = $state("");
  let dragMode = $state<"rotate" | "stir">("rotate");
  let paramN = $state(32);
  let paramRefine = $state(2);
  let paramK = $state(0.006);
  let paramMu = $state(0.01);
  let paramDt = $state(0.002);
  let stepsPerFrame = $state(1);
  let volumeMode = $state<IB3DFieldMode | "off">("velocity");
  let volumeOpacity = $state(0.055);
  let volumeThreshold = $state(0.08);
  let volumeScale = $state(0.08);
  let volumeSamples = $state(96);
  let meshStats = $state("");

  function makeParams(): IB3DParams {
    return {
      N: paramN,
      dt: paramDt,
      K: paramK,
      mu: paramMu,
      damping: 1.0,
      refinement: paramRefine,
      radius: 0.28,
    };
  }

  async function initWebGPU() {
    await init("webgpu");
    defaultDevice("webgpu");
    device = getWebGPUDevice();
    device.addEventListener("uncapturederror", (event) => {
      const message = event.error?.message ?? String(event.error);
      console.error("IB3D WebGPU error:", message);
      errorMsg = message;
    });
    context = canvas.getContext("webgpu") as GPUCanvasContext;
    format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "opaque" });
    uniformBuffer = device.createBuffer({
      size: 24 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const volumeModule = device.createShaderModule({ code: volumeShader() });
    volumePipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: volumeModule, entryPoint: "vs" },
      fragment: {
        module: volumeModule,
        entryPoint: "fs",
        targets: [{
          format,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
          },
        }],
      },
      primitive: { topology: "triangle-list" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "always" },
    });
    const surfaceModule = device.createShaderModule({ code: surfaceShader() });
    surfacePipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: surfaceModule, entryPoint: "vs" },
      fragment: {
        module: surfaceModule,
        entryPoint: "fs",
        targets: [{
          format,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
          },
        }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
    const edgeModule = device.createShaderModule({ code: edgeShader() });
    edgePipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: edgeModule, entryPoint: "vs" },
      fragment: { module: edgeModule, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "line-list" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less-equal" },
    });
  }

  async function rebuildSolver() {
    solver?.destroy();
    solver = await WebGpuIb3DSolver.init(device, makeParams());
    meshStats = `${solver.mesh.nb} vertices / ${solver.mesh.nt} triangles / ${solver.mesh.ne} edges`;
    step = 0;
    simTime = 0;
  }

  function syncParams() {
    solver?.setParams(makeParams());
  }

  function resetSim() {
    solver?.reset(makeParams());
    step = 0;
    simTime = 0;
    paused = false;
  }

  async function changeShape() {
    running = false;
    await new Promise((r) => requestAnimationFrame(r));
    await rebuildSolver();
    simulate();
  }

  function ensureDepth() {
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(320, Math.floor(canvas.clientWidth * dpr));
    const height = Math.max(280, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width === width && canvas.height === height && depthTexture) return;
    canvas.width = width;
    canvas.height = height;
    depthTexture?.destroy();
    depthTexture = device.createTexture({
      size: [width, height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    depthView = depthTexture.createView();
  }

  function renderFrame() {
    if (!solver) return;
    ensureDepth();
    const aspect = canvas.width / canvas.height;
    const uniforms = new Float32Array(24);
    uniforms.set(rotationMatrix(yaw, pitch), 0);
    uniforms[16] = aspect;
    uniforms[17] = 1.95;
    uniforms[18] = volumeOpacity;
    uniforms[19] = volumeThreshold;
    uniforms[20] = volumeSamples;
    uniforms[21] = volumeScale;
    uniforms[22] = paramN;
    uniforms[23] = volumeMode === "vorticity" ? 1 : 0;
    device.queue.writeBuffer(uniformBuffer, 0, uniforms);
    const view = context.getCurrentTexture().createView();
    const encoder = device.createCommandEncoder();
    const field = volumeMode === "off" ? null : solver.encodeField(encoder, volumeMode);
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue: { r: 0.965, g: 0.965, b: 0.95, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthView!,
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    if (field) {
      pass.setPipeline(volumePipeline);
      pass.setBindGroup(0, bind(volumePipeline, [uniformBuffer, field]));
      pass.draw(3);
    }
    pass.setPipeline(surfacePipeline);
    pass.setBindGroup(0, bind(surfacePipeline, [uniformBuffer, solver.vertexBuffer, solver.triangleBuffer]));
    pass.draw(solver.mesh.nt * 3);
    pass.setPipeline(edgePipeline);
    pass.setBindGroup(0, bind(edgePipeline, [uniformBuffer, solver.vertexBuffer, solver.edgeBuffer]));
    pass.draw(solver.mesh.ne * 2);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  function bind(pipeline: GPURenderPipeline, buffers: GPUBuffer[]) {
    return device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  }

  async function simulate() {
    running = true;
    let last = performance.now();
    let frames = 0;
    while (running) {
      renderFrame();
      frames++;
      if (!paused && solver) {
        solver.stepBatch(stepsPerFrame);
        step += stepsPerFrame;
        simTime += stepsPerFrame * paramDt;
      }
      const now = performance.now();
      if (now - last > 1000) {
        fps = Math.round((frames * 1000) / (now - last));
        frames = 0;
        last = now;
      }
      await new Promise((r) => requestAnimationFrame(r));
    }
  }

  function onPointerDown(e: PointerEvent) {
    pointerDown = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  }

  function onPointerMove(e: PointerEvent) {
    if (!pointerDown) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    if (dragMode === "stir" || e.shiftKey) {
      stir(e, dx, dy);
    } else {
      yaw += dx * 0.006;
      pitch = clamp(pitch + dy * 0.006, -1.25, 1.25);
    }
  }

  function stir(e: PointerEvent, dx: number, dy: number) {
    if (!solver || Math.hypot(dx, dy) < 0.5) return;
    const rect = canvas.getBoundingClientRect();
    const sx = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
    const sy = (0.5 - (e.clientY - rect.top) / rect.height) * 2;
    const basis = cameraBasis(yaw, pitch);
    const aspect = rect.width / rect.height;
    const center = add([0.5, 0.5, 0.5], scale(add(scale(basis.right, sx * aspect), scale(basis.up, sy)), 0.28));
    const force = add(scale(basis.right, dx * 0.55), scale(basis.up, -dy * 0.55));
    solver.applyImpulse(wrapVec(center), force);
  }

  function onPointerUp(e: PointerEvent) {
    pointerDown = false;
    canvas.releasePointerCapture(e.pointerId);
  }

  onMount(() => {
    startup();
    return () => {
      running = false;
      solver?.destroy();
      depthTexture?.destroy();
    };
  });

  async function startup() {
    try {
      await initWebGPU();
      await rebuildSolver();
      simulate();
    } catch (e) {
      console.error(e);
      errorMsg = String(e);
    }
  }

  function rotationMatrix(yaw: number, pitch: number) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    return [
      cy, sy * sp, sy * cp, 0,
      0, cp, -sp, 0,
      -sy, cy * sp, cy * cp, 0,
      0, 0, 0, 1,
    ];
  }

  function cameraBasis(yaw: number, pitch: number) {
    const m = rotationMatrix(yaw, pitch);
    return {
      right: [m[0], m[1], m[2]] as [number, number, number],
      up: [m[4], m[5], m[6]] as [number, number, number],
    };
  }

  function add(a: number[], b: number[]) {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]] as [number, number, number];
  }
  function scale(a: number[], s: number) {
    return [a[0] * s, a[1] * s, a[2] * s] as [number, number, number];
  }
  function wrapVec(a: number[]) {
    return a.map((x) => x - Math.floor(x)) as [number, number, number];
  }
  function clamp(x: number, lo: number, hi: number) {
    return Math.max(lo, Math.min(hi, x));
  }

  function surfaceShader() {
    return /* wgsl */ `
const STRUCTURE_ALPHA: f32 = 0.58;
struct Uniforms { m: mat4x4f, params0: vec4f, params1: vec4f }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> x: array<vec4f>;
@group(0) @binding(2) var<storage, read> tri: array<u32>;
struct Out { @builtin(position) pos: vec4f, @location(0) color: vec3f }
fn project(p: vec3f) -> vec4f {
  let q = (u.m * vec4f(p - vec3f(0.5), 1.0)).xyz;
  return vec4f(q.x * u.params0.y / u.params0.x, q.y * u.params0.y, 0.45 - q.z * 0.5, 1.0);
}
@vertex fn vs(@builtin(vertex_index) vi: u32) -> Out {
  let t = vi / 3u;
  let corner = vi % 3u;
  let ia = tri[t * 3u];
  let ib = tri[t * 3u + 1u];
  let ic = tri[t * 3u + 2u];
  let idx = select(select(ia, ib, corner == 1u), ic, corner == 2u);
  let a = x[ia].xyz - vec3f(0.5);
  let b = x[ib].xyz - vec3f(0.5);
  let c = x[ic].xyz - vec3f(0.5);
  let n = normalize((u.m * vec4f(normalize(cross(b - a, c - a)), 0.0)).xyz);
  let light = normalize(vec3f(0.25, 0.45, 0.85));
  let shade = 0.38 + 0.62 * max(dot(n, light), 0.0);
  var out: Out;
  out.pos = project(x[idx].xyz);
  out.color = mix(vec3f(0.08, 0.37, 0.62), vec3f(0.0, 0.72, 0.68), shade);
  return out;
}
@fragment fn fs(in: Out) -> @location(0) vec4f {
  return vec4f(in.color, STRUCTURE_ALPHA);
}`;
  }

  function edgeShader() {
    return /* wgsl */ `
struct Uniforms { m: mat4x4f, params0: vec4f, params1: vec4f }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> x: array<vec4f>;
@group(0) @binding(2) var<storage, read> edge: array<u32>;
fn project(p: vec3f) -> vec4f {
  let q = (u.m * vec4f(p - vec3f(0.5), 1.0)).xyz;
  return vec4f(q.x * u.params0.y / u.params0.x, q.y * u.params0.y, 0.44 - q.z * 0.5, 1.0);
}
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return project(x[edge[vi]].xyz);
}
@fragment fn fs() -> @location(0) vec4f {
  return vec4f(0.02, 0.08, 0.12, 0.75);
}`;
  }

  function volumeShader() {
    return /* wgsl */ `
struct Uniforms { m: mat4x4f, params0: vec4f, params1: vec4f }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> field: array<f32>;
struct Out { @builtin(position) pos: vec4f, @location(0) ndc: vec2f }
fn invRot(v: vec3f) -> vec3f {
  return vec3f(
    u.m[0].x * v.x + u.m[0].y * v.y + u.m[0].z * v.z,
    u.m[1].x * v.x + u.m[1].y * v.y + u.m[1].z * v.z,
    u.m[2].x * v.x + u.m[2].y * v.y + u.m[2].z * v.z,
  );
}
fn boxHit(ro: vec3f, rd: vec3f) -> vec2f {
  let inv = 1.0 / rd;
  let lo = (vec3f(0.0) - ro) * inv;
  let hi = (vec3f(1.0) - ro) * inv;
  let near = min(lo, hi);
  let far = max(lo, hi);
  return vec2f(max(max(near.x, near.y), near.z), min(min(far.x, far.y), far.z));
}
fn idx(i: u32, j: u32, k: u32, n: u32) -> u32 {
  return (i * n + j) * n + k;
}
fn texel(i: u32, j: u32, k: u32, n: u32) -> f32 {
  return field[idx(min(i, n - 1u), min(j, n - 1u), min(k, n - 1u), n)];
}
fn sampleField(p: vec3f, n: u32) -> f32 {
  let q = clamp(p, vec3f(0.0), vec3f(0.9999)) * f32(n);
  let b = vec3u(floor(q));
  let f = fract(q);
  let i1 = min(b.x + 1u, n - 1u);
  let j1 = min(b.y + 1u, n - 1u);
  let k1 = min(b.z + 1u, n - 1u);
  let c00 = mix(texel(b.x, b.y, b.z, n), texel(i1, b.y, b.z, n), f.x);
  let c10 = mix(texel(b.x, j1, b.z, n), texel(i1, j1, b.z, n), f.x);
  let c01 = mix(texel(b.x, b.y, k1, n), texel(i1, b.y, k1, n), f.x);
  let c11 = mix(texel(b.x, j1, k1, n), texel(i1, j1, k1, n), f.x);
  return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}
@vertex fn vs(@builtin(vertex_index) vi: u32) -> Out {
  let p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0))[vi];
  var out: Out;
  out.pos = vec4f(p, 0.0, 1.0);
  out.ndc = p;
  return out;
}
@fragment fn fs(in: Out) -> @location(0) vec4f {
  let aspect = u.params0.x;
  let scale = u.params0.y;
  let opacity = u.params0.z;
  let threshold = u.params0.w;
  let samples = max(u32(u.params1.x + 0.5), 16u);
  let fieldScale = max(u.params1.y, 0.0001);
  let n = max(u32(u.params1.z + 0.5), 2u);
  let isVorticity = u.params1.w > 0.5;
  let xy = vec2f(in.ndc.x * aspect / scale, in.ndc.y / scale);
  let ro = vec3f(0.5) + invRot(vec3f(xy, -1.05));
  let rd = normalize(invRot(vec3f(0.0, 0.0, 1.0)));
  let hit = boxHit(ro, rd);
  if (hit.x > hit.y || hit.y < 0.0) { discard; }
  let t0 = max(hit.x, 0.0);
  let dt = (hit.y - t0) / f32(samples);
  var acc = vec4f(0.0);
  for (var s = 0u; s < 192u; s++) {
    if (s >= samples || acc.a > 0.96) { break; }
    let p = ro + rd * (t0 + (f32(s) + 0.5) * dt);
    let v = clamp(sampleField(p, n) / fieldScale, 0.0, 1.25);
    let a = smoothstep(threshold, 1.0, v) * opacity;
    let cold = select(vec3f(0.04, 0.23, 0.65), vec3f(0.23, 0.06, 0.38), isVorticity);
    let hot = select(vec3f(0.0, 0.82, 0.82), vec3f(1.0, 0.48, 0.08), isVorticity);
    let color = mix(cold, hot, min(v, 1.0));
    acc = vec4f(acc.rgb + (1.0 - acc.a) * a * color, acc.a + (1.0 - acc.a) * a);
  }
  return acc;
}`;
  }
</script>

<svelte:head>
  <title>Interactive 3D Immersed Boundary Method</title>
</svelte:head>

<main class="ib3d">
  <nav class="breadcrumb">
    <a href="../">&larr; 2D IB simulation</a>
  </nav>

  {#if errorMsg}
    <pre class="error-msg">{errorMsg}</pre>
  {/if}

  <header class="page-header">
    <h1><span>Interactive</span> 3D Immersed Boundary Method</h1>
    <p>
      Triangulated elastic sphere coupled to a periodic 3D velocity grid with
      Peskin 4-point interpolation and spreading.
    </p>
  </header>

  <section class="stage">
    <canvas
      bind:this={canvas}
      onpointerdown={onPointerDown}
      onpointermove={onPointerMove}
      onpointerup={onPointerUp}
      onpointercancel={onPointerUp}
    ></canvas>
    <div class="hud">
      <b>t = {simTime.toFixed(2)} s</b>
      <span>step {step} / {fps} FPS</span>
      <span>{meshStats}</span>
    </div>
  </section>

  <section class="controls">
    <label>
      <span>Drag</span>
      <select bind:value={dragMode}>
        <option value="rotate">Rotate</option>
        <option value="stir">Stir</option>
      </select>
    </label>
    <label>
      <span>Grid</span>
      <select bind:value={paramN} onchange={() => void changeShape()}>
        <option value={16}>16^3</option>
        <option value={32}>32^3</option>
      </select>
    </label>
    <label>
      <span>Mesh</span>
      <select bind:value={paramRefine} onchange={() => void changeShape()}>
        <option value={1}>80 tris</option>
        <option value={2}>320 tris</option>
        <option value={3}>1280 tris</option>
      </select>
    </label>
    <label>
      <span>K</span>
      <input type="range" min="0.001" max="0.04" step="0.001" bind:value={paramK} oninput={syncParams} />
      <output>{paramK.toFixed(3)}</output>
    </label>
    <label>
      <span>mu</span>
      <input type="range" min="0.002" max="0.04" step="0.001" bind:value={paramMu} oninput={syncParams} />
      <output>{paramMu.toFixed(3)}</output>
    </label>
    <label>
      <span>dt</span>
      <input type="range" min="0.0005" max="0.004" step="0.0005" bind:value={paramDt} oninput={syncParams} />
      <output>{paramDt.toFixed(3)}</output>
    </label>
    <label>
      <span>Field</span>
      <select bind:value={volumeMode}>
        <option value="velocity">|u|</option>
        <option value="vorticity">|curl u|</option>
        <option value="off">Off</option>
      </select>
    </label>
    <label>
      <span>Alpha</span>
      <input type="range" min="0.01" max="0.12" step="0.005" bind:value={volumeOpacity} />
      <output>{volumeOpacity.toFixed(3)}</output>
    </label>
    <label>
      <span>Cutoff</span>
      <input type="range" min="0.01" max="0.6" step="0.01" bind:value={volumeThreshold} />
      <output>{volumeThreshold.toFixed(2)}</output>
    </label>
    <label>
      <span>Scale</span>
      <input type="range" min="0.02" max="0.6" step="0.01" bind:value={volumeScale} />
      <output>{volumeScale.toFixed(2)}</output>
    </label>
    <label>
      <span>Samples</span>
      <input type="range" min="32" max="160" step="16" bind:value={volumeSamples} />
      <output>{volumeSamples}</output>
    </label>
    <label>
      <span>Steps</span>
      <input type="range" min="1" max="6" step="1" bind:value={stepsPerFrame} />
      <output>{stepsPerFrame}</output>
    </label>
    <button onclick={resetSim}>Reset</button>
    <button class="primary" onclick={() => (paused = !paused)}>{paused ? "Play" : "Pause"}</button>
  </section>

  <p class="note">
    Shift-drag stirs from rotate mode. The 3D path uses edge-spring forces, 3D
    Peskin coupling, a spectral incompressible fluid solve, and direct WebGPU
    raymarching for the velocity or vorticity volume field.
  </p>
</main>

<style>
  .ib3d {
    max-width: 1080px;
    margin: 0 auto;
    padding: 24px;
    color: var(--color-text);
    font-family: var(--font-sans);
  }
  .breadcrumb {
    font-family: var(--font-mono);
    font-size: 12px;
    margin-bottom: 18px;
  }
  .breadcrumb a {
    color: var(--color-text-meta);
  }
  .error-msg {
    padding: 12px;
    border: 1px solid var(--color-warn);
    background: var(--color-danger-bg);
    color: var(--color-warn);
    white-space: pre-wrap;
  }
  .page-header {
    margin-bottom: 14px;
  }
  h1 {
    font-family: var(--font-serif);
    font-size: 24px;
    line-height: 1.2;
    margin: 0 0 6px;
  }
  h1 span {
    color: var(--color-link);
    font-style: italic;
    font-weight: 700;
  }
  .page-header p,
  .note {
    margin: 0;
    max-width: 760px;
    color: var(--color-text-meta);
    line-height: 1.5;
  }
  .stage {
    position: relative;
    border: 1px solid var(--color-border);
    background: #f6f6f2;
    margin: 18px 0 14px;
    height: min(68vh, 680px);
    min-height: 420px;
  }
  canvas {
    width: 100%;
    height: 100%;
    display: block;
    touch-action: none;
    cursor: grab;
  }
  canvas:active {
    cursor: grabbing;
  }
  .hud {
    position: absolute;
    left: 14px;
    top: 14px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    align-items: center;
    max-width: calc(100% - 28px);
    padding: 8px 10px;
    background: rgb(255 255 255 / 82%);
    border: 1px solid var(--color-border);
    font-family: var(--font-mono);
    font-size: 12px;
  }
  .controls {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
    align-items: end;
    margin-bottom: 12px;
  }
  label {
    display: grid;
    gap: 5px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--color-text-meta);
  }
  select,
  input,
  button {
    min-height: 34px;
    font: inherit;
  }
  output {
    color: var(--color-text);
  }
  button {
    border: 1px solid var(--color-text);
    background: transparent;
    color: var(--color-text);
    cursor: pointer;
  }
  button.primary {
    background: var(--color-text);
    color: var(--color-bg);
  }
  .note {
    font-size: 13px;
  }
</style>
