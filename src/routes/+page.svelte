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
  let invertBg = $state(true); // false: black-centered; true: white-centered
  const colormapNames = [
    "Cyan-Magenta",
    "Teal-Orange",
    "Red-Blue",
    "Purple-Orange",
    "Brown-Teal",
    "Coolwarm",
  ];
  // Endpoint colors for the UI swatch previews.
  // Order: [negative (left), positive (right)] — matches WGSL convention.
  const colormapEnds: [string, string][] = [
    ["rgb(0,230,230)", "rgb(255,51,204)"],
    ["rgb(0,179,166)", "rgb(255,140,26)"],
    ["rgb(26,102,255)", "rgb(255,51,26)"],
    ["rgb(84,39,136)", "rgb(230,97,1)"],
    ["rgb(140,81,10)", "rgb(1,102,94)"],
    ["rgb(59,76,192)", "rgb(180,4,38)"],
  ];
  let colormapGradients = $derived.by(() => {
    const center = invertBg ? "#fff" : "#000";
    return colormapEnds.map(
      ([neg, pos]) => `linear-gradient(to right, ${neg}, ${center}, ${pos})`,
    );
  });
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
  let vizLabel = $derived(vizMode === "vorticity" ? "vort" : "|vel|");
  let cmapLabel = $derived(colormapNames[colormap].toLowerCase());

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
        invertBg: u32,
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

      // Diverging colormaps: v in [-1,1], center color is c (either black or white).
      fn cmapCyanMagenta(v: f32, c: vec3f) -> vec3f {
        let t = clamp(v, -1.0, 1.0);
        if (t > 0.0) { return mix(c, vec3f(1.0, 0.2, 0.8), t); }
        else { return mix(c, vec3f(0.0, 0.9, 0.9), -t); }
      }
      fn cmapTealOrange(v: f32, c: vec3f) -> vec3f {
        let t = clamp(v, -1.0, 1.0);
        if (t > 0.0) { return mix(c, vec3f(1.0, 0.55, 0.1), t); }
        else { return mix(c, vec3f(0.0, 0.7, 0.65), -t); }
      }
      fn cmapRedBlue(v: f32, c: vec3f) -> vec3f {
        let t = clamp(v, -1.0, 1.0);
        if (t > 0.0) { return mix(c, vec3f(1.0, 0.2, 0.1), t); }
        else { return mix(c, vec3f(0.1, 0.4, 1.0), -t); }
      }
      // ColorBrewer PuOr (purple <-> orange)
      fn cmapPurpleOrange(v: f32, c: vec3f) -> vec3f {
        let t = clamp(v, -1.0, 1.0);
        if (t > 0.0) { return mix(c, vec3f(0.902, 0.380, 0.004), t); }
        else { return mix(c, vec3f(0.329, 0.153, 0.533), -t); }
      }
      // ColorBrewer BrBG (brown <-> blue-green)
      fn cmapBrownTeal(v: f32, c: vec3f) -> vec3f {
        let t = clamp(v, -1.0, 1.0);
        if (t > 0.0) { return mix(c, vec3f(0.004, 0.400, 0.369), t); }
        else { return mix(c, vec3f(0.549, 0.318, 0.039), -t); }
      }
      // Matplotlib coolwarm endpoints
      fn cmapCoolwarm(v: f32, c: vec3f) -> vec3f {
        let t = clamp(v, -1.0, 1.0);
        if (t > 0.0) { return mix(c, vec3f(0.706, 0.016, 0.149), t); }
        else { return mix(c, vec3f(0.231, 0.298, 0.753), -t); }
      }

      fn divergingColor(v: f32, cm: u32, c: vec3f) -> vec3f {
        switch cm {
          case 0u: { return cmapCyanMagenta(v, c); }
          case 1u: { return cmapTealOrange(v, c); }
          case 2u: { return cmapRedBlue(v, c); }
          case 3u: { return cmapPurpleOrange(v, c); }
          case 4u: { return cmapBrownTeal(v, c); }
          case 5u: { return cmapCoolwarm(v, c); }
          default: { return cmapCyanMagenta(v, c); }
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

        let centerColor = select(vec3f(0.0), vec3f(1.0), u.invertBg == 1u);
        let segColor    = select(vec3f(0.9), vec3f(0.12), u.invertBg == 1u);
        let ptColor     = select(vec3f(1.0), vec3f(0.0),  u.invertBg == 1u);

        // Both modes use selected colormap (magnitude uses positive branch)
        var color = divergingColor(v, u.cmap, centerColor);

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
            color = mix(color, segColor, alpha);
          }
        }

        // Draw points with periodic wrapping
        for (var k = 0u; k < nb; k++) {
          let bpt = vec2f(boundary[k * 2u], boundary[k * 2u + 1u]);
          let dist = periodicPointDist(worldPt, bpt);
          if (dist < u.pointRadius) {
            let alpha = smoothstep(u.pointRadius, u.pointRadius * 0.3, dist);
            color = mix(color, ptColor, alpha);
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
      size: 48,
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
    const uniformData = new Float32Array(12); // 48 bytes
    uniformData[0] = texRes;
    uniformData[1] = texRes;
    uniformData[2] = vizMode === "vorticity" ? 3.0 : 1.0;
    uniformData[4] = pointRadius;
    uniformData[5] = lineWidth;
    new Uint32Array(uniformData.buffer, 12, 1)[0] = params.Nb;
    new Uint32Array(uniformData.buffer, 24, 1)[0] = vizMode === "vorticity" ? 0 : 1;
    new Uint32Array(uniformData.buffer, 28, 1)[0] = colormap;
    new Uint32Array(uniformData.buffer, 32, 1)[0] = invertBg ? 1 : 0;
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
  <nav class="breadcrumb">
    <a href="https://guanhuasun.github.io/">&larr; guanhuasun.github.io</a>
  </nav>

  {#if errorMsg}
    <pre class="error-msg">{errorMsg}</pre>
  {/if}

  <header class="page-header">
    <h1>
      <a
        href="https://math.nyu.edu/~peskin/ib_lecture_notes/index.html"
        class="title-link"
        target="_blank"
        rel="noopener noreferrer">2D Immersed Boundary Method</a
      >
    </h1>
    <p class="lede">
      Elastic membrane coupled to incompressible fluid via regularized delta
      functions. Click and drag to apply force.
    </p>
    <p class="tech-line">
      Using <a href="https://github.com/ekzhang/jax-js">jax-js</a> on WebGPU
      &middot; N={N} &middot; FFT-based IMEX solver &middot; &Delta;t={paramDt.toFixed(4)}
    </p>
  </header>

  <hr />

  <section class="sim-area">
    <div class="spacer"></div>

    <div class="canvas-block">
      <div class="canvas-caption">
        N={N} &middot; &Delta;t={paramDt.toFixed(4)} &middot; {vizLabel} &middot; {cmapLabel}
      </div>
      <div class="canvas-matte">
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
        <div class="hud hud-pill">
          <div class="hud-time">t = {simTime.toFixed(2)} s</div>
          <div class="hud-detail">step {frameCount} &middot; {fps} FPS</div>
        </div>
      </div>
    </div>

    <aside class="param-panel">
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

      <div class="param-group">
        <div class="param-label">Colormap</div>
        <div class="param-desc">Diverging palette for field visualization</div>
        <div class="swatch-row" role="radiogroup" aria-label="Colormap">
          {#each colormapNames as name, i}
            <button
              type="button"
              class="swatch"
              class:selected={colormap === i}
              style:background={colormapGradients[i]}
              role="radio"
              aria-checked={colormap === i}
              aria-label={name}
              title={name}
              onclick={() => (colormap = i)}
            ></button>
          {/each}
        </div>
      </div>
    </aside>
  </section>

  <section class="toolbar">
    <div class="toolbar-left">
      {#if autoPaused}
        <span class="auto-pause-msg">Paused at step {frameCount}.</span>
      {/if}
    </div>

    <div class="toolbar-right">
      <label class="toolbar-field">
        <span class="toolbar-label">Grid</span>
        <div class="select-wrap">
          <select value={paramN} onchange={(e) => changeN(Number(e.currentTarget.value))}>
            <option value={64}>64 &times; 64</option>
            <option value={128}>128 &times; 128</option>
          </select>
        </div>
      </label>

      <label class="toolbar-field">
        <span class="toolbar-label">View</span>
        <div class="select-wrap">
          <select bind:value={vizMode}>
            <option value="vorticity">Vorticity</option>
            <option value="velocity">|Velocity|</option>
          </select>
        </div>
      </label>

      <label class="toolbar-field inline">
        <input type="checkbox" bind:checked={smooth} />
        <span class="toolbar-label">Smooth</span>
      </label>

      <button class="btn" onclick={resetSim}>Reset</button>
      <button class="btn btn-primary" onclick={togglePause}>
        {paused ? (autoPaused ? "Continue" : "Play") : "Pause"}
      </button>
    </div>
  </section>
</main>

<style>
  .ib-sim {
    max-width: 960px;
    margin: 0 auto;
    padding: 28px 24px 48px;
    color: var(--color-text);
    font-family: var(--font-serif);
  }

  .breadcrumb {
    font-family: var(--font-mono);
    font-size: 12px;
    margin-bottom: 20px;
  }
  .breadcrumb a {
    color: var(--color-text-meta);
  }
  .breadcrumb a:hover {
    color: var(--color-text);
  }

  .error-msg {
    background: var(--color-danger-bg);
    color: var(--color-warn);
    border: 1px solid var(--color-warn);
    padding: 12px 14px;
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 1.4;
    margin-bottom: 16px;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .page-header {
    margin-bottom: 16px;
  }
  .page-header h1 {
    font-family: var(--font-serif);
    font-size: 22px;
    font-weight: 600;
    line-height: 1.25;
    letter-spacing: -0.01em;
    margin: 0 0 6px 0;
    color: var(--color-text);
  }
  .title-link {
    color: inherit;
    text-decoration: none;
    transition: color 0.15s ease;
  }
  .title-link:hover {
    color: var(--color-link);
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  .lede {
    font-family: var(--font-sans);
    font-size: 14px;
    line-height: 1.55;
    color: var(--color-text-meta);
    margin: 0 0 6px 0;
    max-width: 680px;
  }
  .tech-line {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--color-text-caption);
    margin: 0;
  }

  hr {
    margin: 16px 0 24px;
  }

  /* --- Simulation area: spacer + canvas + right panel --- */

  .sim-area {
    display: flex;
    gap: 24px;
    align-items: flex-start;
  }
  .spacer {
    flex: 1 1 auto;
    min-width: 0;
  }
  .canvas-block {
    flex: 0 0 auto;
  }
  .canvas-caption {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--color-text-caption);
    margin-bottom: 6px;
    letter-spacing: 0.01em;
  }
  .canvas-matte {
    position: relative;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    padding: 12px;
    display: inline-block;
    line-height: 0;
  }
  canvas {
    width: 512px;
    height: 512px;
    cursor: crosshair;
    display: block;
    border: 1px solid var(--color-border);
  }

  .hud {
    position: absolute;
    left: 20px;
    bottom: 20px;
    line-height: 1.2;
  }
  .hud-time {
    font-family: var(--font-mono);
    font-size: 14px;
    font-weight: 500;
    color: var(--color-text);
  }
  .hud-detail {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--color-text-meta);
    margin-top: 1px;
  }

  /* --- Parameter panel --- */

  .param-panel {
    display: flex;
    flex-direction: column;
    gap: 20px;
    width: 220px;
    flex: 0 0 220px;
    padding-top: 18px; /* align with canvas matte below caption */
  }
  .param-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .param-label {
    font-family: var(--font-sans);
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.01em;
    color: var(--color-text);
  }
  .param-desc {
    font-family: var(--font-serif);
    font-size: 12px;
    color: var(--color-text-meta);
    line-height: 1.4;
  }
  .param-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 4px;
  }
  .param-row input[type="range"] {
    flex: 1 1 auto;
    width: auto;
    accent-color: var(--color-border-strong);
    cursor: pointer;
  }
  .param-val {
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 500;
    color: var(--color-text);
    font-variant-numeric: tabular-nums;
    min-width: 3.2em;
    text-align: right;
  }

  /* Range slider: flat track + soft-cornered handle */
  .param-row input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    height: 2px;
    background: var(--color-border);
    border: 0;
    padding: 0;
  }
  .param-row input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 14px;
    height: 14px;
    background: var(--color-border-strong);
    border-radius: 2px;
    border: 0;
    cursor: pointer;
    transition: background 0.15s ease;
  }
  .param-row input[type="range"]::-webkit-slider-thumb:hover {
    background: var(--color-link);
  }
  .param-row input[type="range"]::-moz-range-thumb {
    width: 14px;
    height: 14px;
    background: var(--color-border-strong);
    border-radius: 2px;
    border: 0;
    cursor: pointer;
    transition: background 0.15s ease;
  }
  .param-row input[type="range"]::-moz-range-thumb:hover {
    background: var(--color-link);
  }
  .param-row input[type="range"]::-moz-range-track {
    height: 2px;
    background: var(--color-border);
    border: 0;
  }

  /* Colormap swatches */
  .swatch-row {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 4px;
    margin-top: 6px;
  }
  .swatch {
    height: 18px;
    width: 100%;
    border: 1px solid var(--color-border);
    padding: 0;
    background-clip: padding-box;
    cursor: pointer;
    transition: border-color 0.15s ease, transform 0.15s ease;
  }
  .swatch:hover {
    border-color: var(--color-border-strong);
  }
  .swatch.selected {
    border: 2px solid var(--color-border-strong);
    height: 18px;
  }

  /* --- Toolbar --- */

  .toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin-top: 28px;
    padding-top: 16px;
    border-top: 1px solid var(--color-border);
    flex-wrap: wrap;
  }
  .toolbar-left {
    display: flex;
    align-items: center;
    min-height: 32px;
  }
  .toolbar-right {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }
  .toolbar-field {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .toolbar-field.inline {
    flex-direction: row;
    align-items: center;
    gap: 6px;
    padding-top: 18px; /* align with selects */
  }
  .toolbar-label {
    font-family: var(--font-sans);
    font-size: 11px;
    font-weight: 500;
    color: var(--color-text-meta);
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .toolbar-right .btn {
    align-self: flex-end;
  }

  .select-wrap {
    position: relative;
  }
  .select-wrap::after {
    content: "";
    position: absolute;
    right: 10px;
    top: 50%;
    width: 8px;
    height: 8px;
    border-right: 1px solid var(--color-text-meta);
    border-bottom: 1px solid var(--color-text-meta);
    transform: translateY(-75%) rotate(45deg);
    pointer-events: none;
  }
  .toolbar select {
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    color: var(--color-text);
    font-family: var(--font-sans);
    font-size: 13px;
    font-weight: 500;
    padding: 6px 28px 6px 10px;
    border-radius: 0;
    cursor: pointer;
    transition: border-color 0.15s ease;
    min-width: 110px;
  }
  .toolbar select:hover {
    border-color: var(--color-border-strong);
  }

  /* Checkbox */
  .toolbar-field.inline input[type="checkbox"] {
    -webkit-appearance: none;
    appearance: none;
    width: 14px;
    height: 14px;
    background: var(--color-bg);
    border: 1px solid var(--color-border-strong);
    border-radius: 2px;
    cursor: pointer;
    position: relative;
    margin: 0;
    transition: background 0.15s ease;
  }
  .toolbar-field.inline input[type="checkbox"]:checked {
    background: var(--color-border-strong);
  }
  .toolbar-field.inline input[type="checkbox"]:checked::after {
    content: "";
    position: absolute;
    left: 3px;
    top: 0px;
    width: 5px;
    height: 9px;
    border-right: 2px solid var(--color-bg);
    border-bottom: 2px solid var(--color-bg);
    transform: rotate(45deg);
  }

  .auto-pause-msg {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--color-warn);
  }
</style>
