<script lang="ts">
  import {
    defaultDevice,
    getWebGPUDevice,
    init,
    numpy as np,
  } from "@jax-js/jax";
  import { onMount } from "svelte";

  import {
    createParams,
    initState,
    initFourierOperator,
    timeStep,
    computeVorticity,
    computeVelMagnitude,
    upsampleCPU,
    type IBParams,
    type IBState,
    type Precomputed,
  } from "./ib-solver";

  // --- Parameters ---
  let paramK = $state(1.0);
  let paramMu = $state(0.01);
  let paramN = $state(64);
  let paramDt = $state(0.01);
  let vizMode = $state<"vorticity" | "velocity">("vorticity");
  let colormap = $state(0); // index into WGSL colormap array
  let smooth = $state(true); // bilinear upsample toggle
  const colormapNames = [
    "Cyan-Magenta",
    "Teal-Orange",
    "Blue-Yellow",
    "Green-Purple",
    "Red-Blue",
    "Twilight",
  ];
  let N = $state(64);
  let M = $derived(N * 2); // upsample resolution
  const PAUSE_INTERVAL = 1000;
  let dtMin = $derived(N === 64 ? 0.002 : 0.001);
  let dtMax = $derived(N === 64 ? 0.02 : 0.01);
  let dtStep = $derived(N === 64 ? 0.001 : 0.0005);
  let dtDefault = $derived(N === 64 ? 0.01 : 0.005);

  let canvas: HTMLCanvasElement;
  let running = false;
  let frameCount = $state(0);
  let fps = $state(0);
  let paused = $state(false);
  let autoPaused = $state(false); // true when auto-paused at interval
  let errorMsg = $state("");

  let params: IBParams;
  let sim: IBState;
  let pre: Precomputed;

  // Derived
  let simTime = $derived(frameCount * paramDt);

  // Mouse state
  let mouseDown = $state(false);
  let mouseX = 0;
  let mouseY = 0;
  let lastMouseX = 0;
  let lastMouseY = 0;

  // --- WebGPU Rendering ---
  let gpuDevice: GPUDevice;
  let renderPipeline: GPURenderPipeline;
  let renderBindGroupLayout: GPUBindGroupLayout;
  let canvasContext: GPUCanvasContext;
  let uniformBuffer: GPUBuffer;
  let fieldBuffer: GPUBuffer;

  async function initRenderer() {
    gpuDevice = getWebGPUDevice();
    canvasContext = canvas.getContext("webgpu") as GPUCanvasContext;
    const format = navigator.gpu.getPreferredCanvasFormat();
    canvasContext.configure({ device: gpuDevice, format, alphaMode: "opaque" });

    const shaderCode = /* wgsl */ `
      struct Uniforms {
        texSize: vec2f,
        fieldScale: f32,
        nb: u32,
        pointRadius: f32,
        lineWidth: f32,
        vizMode: u32,
        cmap: u32,
      };

      @group(0) @binding(0) var<uniform> u: Uniforms;
      @group(0) @binding(1) var<storage, read> field: array<f32>;
      @group(0) @binding(2) var<storage, read> boundary: array<f32>;

      struct VertexOutput {
        @builtin(position) pos: vec4f,
        @location(0) uv: vec2f,
      };

      @vertex fn vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
        var pos = array<vec2f, 3>(
          vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3)
        );
        var out: VertexOutput;
        out.pos = vec4f(pos[vi], 0, 1);
        out.uv = (pos[vi] + 1) * 0.5;
        return out;
      }

      // Diverging colormaps: v in [-1,1], black at center
      fn cmapCyanMagenta(v: f32) -> vec3f {
        let t = clamp(v, -1.0, 1.0);
        if (t > 0.0) { return mix(vec3f(0,0,0), vec3f(1.0, 0.2, 0.8), t); }
        else { return mix(vec3f(0,0,0), vec3f(0.0, 0.9, 0.9), -t); }
      }
      fn cmapTealOrange(v: f32) -> vec3f {
        let t = clamp(v, -1.0, 1.0);
        if (t > 0.0) { return mix(vec3f(0,0,0), vec3f(1.0, 0.55, 0.1), t); }
        else { return mix(vec3f(0,0,0), vec3f(0.0, 0.7, 0.65), -t); }
      }
      fn cmapBlueYellow(v: f32) -> vec3f {
        let t = clamp(v, -1.0, 1.0);
        if (t > 0.0) { return mix(vec3f(0,0,0), vec3f(1.0, 0.95, 0.2), t); }
        else { return mix(vec3f(0,0,0), vec3f(0.15, 0.3, 1.0), -t); }
      }
      fn cmapGreenPurple(v: f32) -> vec3f {
        let t = clamp(v, -1.0, 1.0);
        if (t > 0.0) { return mix(vec3f(0,0,0), vec3f(0.7, 0.2, 1.0), t); }
        else { return mix(vec3f(0,0,0), vec3f(0.1, 0.85, 0.3), -t); }
      }
      fn cmapRedBlue(v: f32) -> vec3f {
        let t = clamp(v, -1.0, 1.0);
        if (t > 0.0) { return mix(vec3f(0,0,0), vec3f(1.0, 0.2, 0.1), t); }
        else { return mix(vec3f(0,0,0), vec3f(0.1, 0.4, 1.0), -t); }
      }
      fn cmapTwilight(v: f32) -> vec3f {
        let t = clamp(v, -1.0, 1.0);
        if (t > 0.0) {
          if (t < 0.5) { return mix(vec3f(0,0,0), vec3f(0.8, 0.15, 0.3), t * 2.0); }
          else { return mix(vec3f(0.8, 0.15, 0.3), vec3f(1.0, 0.7, 0.5), (t - 0.5) * 2.0); }
        } else {
          let s = -t;
          if (s < 0.5) { return mix(vec3f(0,0,0), vec3f(0.2, 0.15, 0.6), s * 2.0); }
          else { return mix(vec3f(0.2, 0.15, 0.6), vec3f(0.5, 0.6, 1.0), (s - 0.5) * 2.0); }
        }
      }

      fn divergingColor(v: f32, cm: u32) -> vec3f {
        switch cm {
          case 0u: { return cmapCyanMagenta(v); }
          case 1u: { return cmapTealOrange(v); }
          case 2u: { return cmapBlueYellow(v); }
          case 3u: { return cmapGreenPurple(v); }
          case 4u: { return cmapRedBlue(v); }
          case 5u: { return cmapTwilight(v); }
          default: { return cmapCyanMagenta(v); }
        }
      }

      fn distToSeg(p: vec2f, a: vec2f, b: vec2f) -> f32 {
        let ab = b - a;
        let ap = p - a;
        let t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
        return length(ap - ab * t);
      }

      // Periodic distance from point p to point q in [0,1]^2 domain
      fn periodicPointDist(p: vec2f, q: vec2f) -> f32 {
        var d = p - q;
        d = d - round(d);
        return length(d);
      }

      // Periodic distance from point p to segment a-b, using nearest periodic image
      fn periodicSegDist(p: vec2f, a: vec2f, b: vec2f) -> f32 {
        // Shift a to nearest image of p
        var da = a - p;
        da = da - round(da);
        let a2 = p + da;
        // Shift b relative to a using shortest periodic offset
        var db = b - a;
        db = db - round(db);
        let b2 = a2 + db;
        return distToSeg(p, a2, b2);
      }

      @fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
        let M = u32(u.texSize.x);
        let wx = in.uv.x;
        let wy = in.uv.y;

        // Node-centered sampling: pixel px corresponds to source index px at
        // UV px/M, so round-to-nearest (not floor) for correct alignment with
        // boundary points drawn at their true physical coordinates.
        let px = min(u32(round(wx * f32(M))), M - 1u);
        let py = min(u32(round(wy * f32(M))), M - 1u);
        let idx = px * M + py;
        let v = field[idx] / u.fieldScale;

        // Both modes use selected colormap (magnitude uses positive branch)
        var color = divergingColor(v, u.cmap);

        let nb = u.nb;
        let worldPt = vec2f(wx, wy);

        // Draw segments with periodic wrapping
        for (var k = 0u; k < nb; k++) {
          let a = vec2f(boundary[k * 2u], boundary[k * 2u + 1u]);
          let nk = (k + 1u) % nb;
          let b = vec2f(boundary[nk * 2u], boundary[nk * 2u + 1u]);
          let d = periodicSegDist(worldPt, a, b);
          if (d < u.lineWidth) {
            let alpha = smoothstep(u.lineWidth, u.lineWidth * 0.3, d);
            color = mix(color, vec3f(0.9, 0.9, 0.9), alpha);
          }
        }

        // Draw points with periodic wrapping
        for (var k = 0u; k < nb; k++) {
          let bpt = vec2f(boundary[k * 2u], boundary[k * 2u + 1u]);
          let dist = periodicPointDist(worldPt, bpt);
          if (dist < u.pointRadius) {
            let alpha = smoothstep(u.pointRadius, u.pointRadius * 0.3, dist);
            color = mix(color, vec3f(1.0, 1.0, 1.0), alpha);
          }
        }

        return vec4f(color, 1.0);
      }
    `;

    const shaderModule = gpuDevice.createShaderModule({ code: shaderCode });

    renderBindGroupLayout = gpuDevice.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
      ],
    });

    renderPipeline = gpuDevice.createRenderPipeline({
      layout: gpuDevice.createPipelineLayout({
        bindGroupLayouts: [renderBindGroupLayout],
      }),
      vertex: { module: shaderModule },
      fragment: { module: shaderModule, targets: [{ format }] },
    });

    uniformBuffer = gpuDevice.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    reallocFieldBuffer();
  }

  function reallocFieldBuffer() {
    if (fieldBuffer) fieldBuffer.destroy();
    fieldBuffer = gpuDevice.createBuffer({
      size: M * M * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  function renderFrame(fieldData: Float32Array, boundBuffer: GPUBuffer, texRes: number = M) {
    const pointRadius = 0.6 / (64 * 2);
    const lineWidth = 0.4 / (64 * 2);
    const uniformData = new Float32Array([
      texRes, texRes,
      vizMode === "vorticity" ? 3.0 : 1.0,
      0, pointRadius, lineWidth, 0, 0,
    ]);
    new Uint32Array(uniformData.buffer, 12, 1)[0] = params.Nb;
    new Uint32Array(uniformData.buffer, 24, 1)[0] = vizMode === "vorticity" ? 0 : 1;
    new Uint32Array(uniformData.buffer, 28, 1)[0] = colormap;
    gpuDevice.queue.writeBuffer(uniformBuffer, 0, uniformData);
    gpuDevice.queue.writeBuffer(fieldBuffer, 0, fieldData as unknown as ArrayBuffer);

    const bindGroup = gpuDevice.createBindGroup({
      layout: renderBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: fieldBuffer } },
        { binding: 2, resource: { buffer: boundBuffer } },
      ],
    });

    const encoder = gpuDevice.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: canvasContext.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    pass.setPipeline(renderPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    gpuDevice.queue.submit([encoder.finish()]);
  }

  // --- Apply localized Gaussian force from mouse ---
  function applyMouseForce() {
    if (!mouseDown || !sim) return;

    const { N: n, h } = params;
    const cx = mouseX;
    const cy = 1 - mouseY; // screen coords → physical coords

    const magnitude = 30.0;
    const sigma = 0.05;
    const sigma2 = sigma * sigma;

    // Drag direction in physical coords
    let fx = (mouseX - lastMouseX) * 80;
    let fy = -(mouseY - lastMouseY) * 80; // flip y for screen → physical

    if (Math.abs(fx) < 0.01 && Math.abs(fy) < 0.01) return;

    const forceData = new Float32Array(n * n * 2);
    for (let j1 = 0; j1 < n; j1++) {
      for (let j2 = 0; j2 < n; j2++) {
        const x = j1 * h;
        const y = j2 * h;
        const rx = x - cx;
        const ry = y - cy;
        const r2 = rx * rx + ry * ry;
        const g = magnitude * Math.exp(-r2 / (2 * sigma2));
        forceData[(j1 * n + j2) * 2] = g * fx;
        forceData[(j1 * n + j2) * 2 + 1] = g * fy;
      }
    }
    const force = np.array(forceData, { shape: [n, n, 2] });
    sim.u = sim.u.add(force.mul(params.dt));
  }

  // --- Main loop ---
  async function simulate() {
    running = true;
    let lastTime = performance.now();
    let frames = 0;

    while (running) {
      if (!paused) {
        applyMouseForce();
        sim = timeStep(sim, pre, params);
        frameCount++;
        frames++;

        // Auto-pause every PAUSE_INTERVAL steps
        if (frameCount % PAUSE_INTERVAL === 0) {
          paused = true;
          autoPaused = true;
        }
      }

      // Compute visualization field, read to CPU, upsample
      let viz: np.Array;
      if (vizMode === "vorticity") {
        viz = computeVorticity(sim.u.ref, params);
      } else {
        viz = computeVelMagnitude(sim.u.ref);
      }
      const rawData = (await viz.data()) as Float32Array;
      const fieldData = smooth ? upsampleCPU(rawData, N, M) : rawData;
      const texRes = smooth ? M : N;

      const boundBuf = await sim.X.ref.gpuBuffer();
      renderFrame(fieldData, boundBuf, texRes);

      // FPS counter
      const now = performance.now();
      if (now - lastTime > 1000) {
        fps = Math.round((frames * 1000) / (now - lastTime));
        frames = 0;
        lastTime = now;
      }

      await new Promise((r) => requestAnimationFrame(r));
    }
  }

  function togglePause() {
    paused = !paused;
    autoPaused = false;
  }

  function resetSim() {
    if (sim) {
      sim.u.dispose();
      sim.X.dispose();
    }
    params = createParams(N, paramK, paramMu);
    params.dt = paramDt;
    sim = initState(params);
    pre = initFourierOperator(params);
    frameCount = 0;
    paused = false;
    autoPaused = false;
  }

  function changeN(newN: number) {
    N = newN;
    paramDt = dtDefault;
    reallocFieldBuffer();
    resetSim();
  }

  async function startup() {
    await init("webgpu");
    defaultDevice("webgpu");

    canvas.width = 512;
    canvas.height = 512;

    N = paramN;
    paramDt = dtDefault;
    params = createParams(N, paramK, paramMu);
    params.dt = paramDt;
    sim = initState(params);
    pre = initFourierOperator(params);

    await initRenderer();
    simulate();
  }

  onMount(() => {
    startup().catch((e) => {
      console.error(e);
      errorMsg = String(e);
    });
    return () => {
      running = false;
    };
  });

  // --- Mouse handlers ---
  function getMousePos(e: MouseEvent) {
    const rect = canvas.getBoundingClientRect();
    return [
      (e.clientX - rect.left) / rect.width,
      (e.clientY - rect.top) / rect.height,
    ];
  }

  function onMouseDown(e: MouseEvent) {
    const [mx, my] = getMousePos(e);
    mouseX = mx;
    mouseY = my;
    lastMouseX = mx;
    lastMouseY = my;
    mouseDown = true;
  }

  function onMouseMove(e: MouseEvent) {
    lastMouseX = mouseX;
    lastMouseY = mouseY;
    [mouseX, mouseY] = getMousePos(e);
  }

  function onMouseUp() {
    mouseDown = false;
  }

  function onTouchStart(e: TouchEvent) {
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    mouseX = (touch.clientX - rect.left) / rect.width;
    mouseY = (touch.clientY - rect.top) / rect.height;
    lastMouseX = mouseX;
    lastMouseY = mouseY;
    mouseDown = true;
  }

  function onTouchMove(e: TouchEvent) {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    lastMouseX = mouseX;
    lastMouseY = mouseY;
    mouseX = (touch.clientX - rect.left) / rect.width;
    mouseY = (touch.clientY - rect.top) / rect.height;
  }
</script>

<main class="ib-sim">
  {#if errorMsg}
    <pre class="error-msg">{errorMsg}</pre>
  {/if}

  <div class="header">
    <h1>2D Immersed Boundary Method</h1>
    <p>
      Elastic membrane coupled to incompressible fluid via regularized delta
      functions. Click and drag to apply force.
    </p>
    <p>
      Using <a href="https://github.com/ekzhang/jax-js">jax-js</a> on WebGPU. N={N}, FFT-based IMEX solver. &Delta;t={paramDt.toFixed(4)}
    </p>
  </div>

  <div class="sim-area">
    <div class="spacer"></div>
    <div class="canvas-wrap">
      <canvas
        bind:this={canvas}
        onmousemove={onMouseMove}
        onmousedown={onMouseDown}
        onmouseup={onMouseUp}
        onmouseleave={onMouseUp}
        ontouchmove={onTouchMove}
        ontouchstart={onTouchStart}
        ontouchend={onMouseUp}
        ontouchcancel={onMouseUp}
      ></canvas>
      <div class="hud">
        <div class="hud-time">t = {simTime.toFixed(2)} s</div>
        <div class="hud-detail">step {frameCount} &middot; {fps} FPS</div>
      </div>
    </div>

    <div class="param-panel">
      <div class="param-group">
        <div class="param-label">K &mdash; Stiffness</div>
        <div class="param-desc">Elastic spring constant of the membrane [N/m]</div>
        <div class="param-row">
          <input
            type="range"
            min="0.2"
            max="5"
            step="0.01"
            bind:value={paramK}
            oninput={() => { if (params) params.K = paramK; }}
          />
          <span class="param-val">{paramK.toFixed(2)}</span>
        </div>
      </div>

      <div class="param-group">
        <div class="param-label">&mu; &mdash; Viscosity</div>
        <div class="param-desc">Dynamic viscosity of the fluid [Pa&middot;s]</div>
        <div class="param-row">
          <input
            type="range"
            min="0.001"
            max="0.1"
            step="0.001"
            bind:value={paramMu}
            oninput={() => {
              if (params) {
                params.mu = paramMu;
                pre = initFourierOperator(params);
              }
            }}
          />
          <span class="param-val">{paramMu.toFixed(3)}</span>
        </div>
      </div>

      <div class="param-group">
        <div class="param-label">&Delta;t &mdash; Time step</div>
        <div class="param-desc">Integration step size [s]. Reduce if unstable.</div>
        <div class="param-row">
          <input
            type="range"
            min={dtMin}
            max={dtMax}
            step={dtStep}
            bind:value={paramDt}
            oninput={() => {
              if (params) {
                params.dt = paramDt;
                pre = initFourierOperator(params);
              }
            }}
          />
          <span class="param-val">{paramDt.toFixed(3)}</span>
        </div>
      </div>
    </div>
  </div>

  <div class="toolbar">
    {#if autoPaused}
      <span class="auto-pause-msg">Paused at step {frameCount}.</span>
    {/if}
    <label class="toolbar-label">
      Grid:
      <select value={paramN} onchange={(e) => changeN(Number(e.currentTarget.value))}>
        <option value={64}>64×64</option>
        <option value={128}>128×128</option>
      </select>
    </label>
    <label class="toolbar-label">
      View:
      <select bind:value={vizMode}>
        <option value="vorticity">Vorticity</option>
        <option value="velocity">|Velocity|</option>
      </select>
    </label>
    <label class="toolbar-label">
      Color:
      <select bind:value={colormap}>
        {#each colormapNames as name, i}
          <option value={i}>{name}</option>
        {/each}
      </select>
    </label>
    <label class="toolbar-label">
      <input type="checkbox" bind:checked={smooth} />
      Smooth
    </label>
    <button onclick={togglePause}>
      {paused ? (autoPaused ? "Continue" : "Play") : "Pause"}
    </button>
    <button onclick={resetSim}>Reset</button>
  </div>
</main>

<svelte:head>
  <style>
    body {
      background: #000;
    }
  </style>
</svelte:head>

<style>
  .ib-sim {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    height: 100vh;
    overflow: hidden;
    color: #eee;
    font-family: system-ui, sans-serif;
    padding: 8px;
  }

  .error-msg {
    color: red;
    background: #300;
    padding: 12px;
    max-width: 600px;
    overflow: auto;
    font-size: 0.8rem;
    border-radius: 4px;
  }

  .header {
    text-align: center;
    padding-bottom: 6px;
    color: #bbb;
  }

  .header h1 {
    font-size: 1.15rem;
    font-weight: 600;
    color: #ddd;
  }

  .header p {
    font-size: 0.8rem;
    color: #777;
    margin-top: 2px;
  }

  .header a {
    color: rgba(253, 224, 71, 0.8);
    text-decoration: underline;
  }

  /* --- Main simulation area: spacer + canvas + right panel --- */

  .sim-area {
    display: flex;
    gap: 16px;
    align-items: flex-start;
  }

  .spacer {
    width: 180px;
    flex-shrink: 0;
  }

  .canvas-wrap {
    position: relative;
    flex-shrink: 0;
  }

  canvas {
    width: 512px;
    height: 512px;
    cursor: crosshair;
    border: 1px solid #222;
    display: block;
  }

  .hud {
    position: absolute;
    top: 8px;
    left: 10px;
    text-align: left;
    pointer-events: none;
  }

  .hud-time {
    font-size: 1.1rem;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.85);
    text-shadow: 0 1px 4px rgba(0, 0, 0, 0.9);
  }

  .hud-detail {
    font-size: 0.8rem;
    color: rgba(255, 255, 255, 0.5);
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.9);
  }

  /* --- Parameter panel (right side) --- */

  .param-panel {
    display: flex;
    flex-direction: column;
    gap: 16px;
    width: 180px;
    padding-top: 4px;
  }

  .param-group {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .param-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #ccc;
  }

  .param-desc {
    font-size: 0.7rem;
    color: #666;
    line-height: 1.3;
  }

  .param-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 2px;
  }

  .param-row input[type="range"] {
    width: 120px;
    accent-color: #e8c44a;
  }

  .param-val {
    font-size: 0.8rem;
    color: #aaa;
    font-variant-numeric: tabular-nums;
    min-width: 3.5em;
    text-align: right;
  }

  /* --- Centered toolbar --- */

  .toolbar {
    display: flex;
    gap: 12px;
    align-items: center;
    justify-content: center;
    margin-top: 8px;
    font-size: 0.85rem;
    color: #ccc;
  }

  .toolbar-label {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .toolbar select {
    background: #222;
    border: 1px solid #444;
    color: #eee;
    border-radius: 4px;
    padding: 2px 4px;
  }

  .toolbar button {
    padding: 4px 14px;
    background: #222;
    border: 1px solid #444;
    color: #eee;
    border-radius: 4px;
    cursor: pointer;
  }

  .toolbar button:hover {
    background: #333;
  }

  .auto-pause-msg {
    font-size: 0.8rem;
    color: #e8c44a;
  }
</style>
