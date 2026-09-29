<script lang="ts">
  import {
    defaultDevice,
    getWebGPUDevice,
    init,
    numpy as np,
  } from "@jax-js/jax";
  import { onMount } from "svelte";
  import SimulationNav from "$lib/SimulationNav.svelte";

  import {
    createParams,
    initState,
    initFourierOperator,
    createStep,
    computeVorticity,
    computeVelMagnitude,
    type IBParams,
    type IBState,
    type Precomputed,
  } from "./ib-solver";
  import { WebGpuIbSolver } from "./webgpu-ib-solver";
  import { WebGpuIbRenderer } from "./webgpu-ib-renderer";

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
  const PAUSE_INTERVAL_SECONDS = 20;
  let dtMin = $derived(N === 64 ? 0.002 : 0.001);
  let dtMax = $derived(N === 64 ? 0.02 : 0.01);
  let dtStep = $derived(N === 64 ? 0.001 : 0.0005);
  let dtDefault = $derived(N === 64 ? 0.01 : 0.005);

  let canvas: HTMLCanvasElement;
  let running = false;
  let frameCount = $state(0);
  let simTime = $state(0);
  let nextAutoPauseTime = PAUSE_INTERVAL_SECONDS;
  let fps = $state(0);
  let paused = $state(false);
  let autoPaused = $state(false); // true when auto-paused at a simulation-time mark
  let errorMsg = $state("");
  let solverMode = $state<"optimized" | "reference">("reference");
  let perfMode = false;

  let params: IBParams;
  let sim: IBState;
  let pre: Precomputed;
  let stepFn: ReturnType<typeof createStep>;
  let optimizedSolver: WebGpuIbSolver | null = null;

  // Derived
  let vizLabel = $derived(vizMode === "vorticity" ? "vort" : "|vel|");
  let cmapLabel = $derived(colormapNames[colormap].toLowerCase());

  // Mouse state
  let mouseDown = $state(false);
  let mouseX = 0;
  let mouseY = 0;
  let lastMouseX = 0;
  let lastMouseY = 0;

  let gpuDevice: GPUDevice;
  let renderer: WebGpuIbRenderer;

  // --- Apply localized Gaussian force from mouse ---
  function applyMouseForce(encoder?: GPUCommandEncoder) {
    if (!mouseDown || !params) return;

    const { N: n, h } = params;
    const cx = mouseX;
    const cy = 1 - mouseY; // screen coords → physical coords

    const magnitude = 45.0;
    const sigma = 0.07;
    const sigma2 = sigma * sigma;

    // Drag direction in physical coords
    const fx = (mouseX - lastMouseX) * 110;
    const fy = -(mouseY - lastMouseY) * 110; // flip y for screen → physical

    if (Math.abs(fx) < 0.01 && Math.abs(fy) < 0.01) return;

    if (optimizedSolver) {
      optimizedSolver.applyMouseForce(cx, cy, fx, fy, encoder);
      return;
    }
    if (!sim) return;

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
    let perfSteps = 0;
    let perfStepMs = 0;
    let perfRenderMs = 0;
    let lastView = "";

    while (running) {
      let encoder: GPUCommandEncoder | undefined;
      const advance = !paused;
      if (advance) {
        if (optimizedSolver) encoder = gpuDevice.createCommandEncoder();
        applyMouseForce(encoder);
        try {
          const t0 = perfMode ? performance.now() : 0;
          if (optimizedSolver) {
            optimizedSolver.stepBatch(1, encoder);
          } else {
            const [u2, X2] = stepFn(sim.u, sim.X, pre.a.ref, params);
            sim = { u: u2, X: X2 };
          }
          if (perfMode) perfStepMs += performance.now() - t0;
        } catch (e) {
          console.error(e);
          errorMsg = String(e);
          running = false;
          break;
        }
        frameCount++;
        frames++;
        simTime += params.dt;

        // Auto-pause every fixed amount of simulated time, independent of dt.
        if (simTime >= nextAutoPauseTime - 1e-9) {
          paused = true;
          autoPaused = true;
          while (nextAutoPauseTime <= simTime + 1e-9) {
            nextAutoPauseTime += PAUSE_INTERVAL_SECONDS;
          }
        }
      }

      // Paused frames need no GPU work unless a display control changes.
      const view = `${frameCount}:${N}:${vizMode}:${colormap}:${invertBg}:${smooth}`;
      if (advance || view !== lastView) {
        lastView = view;
        const renderT0 = perfMode ? performance.now() : 0;
        if (optimizedSolver) {
          encoder ??= gpuDevice.createCommandEncoder();
          const fieldBuf = optimizedSolver.renderField(vizMode, encoder);
          renderer.render(fieldBuf, optimizedSolver.boundaryBuffer, params, { vizMode, colormap, invertBg, smooth }, encoder);
        } else {
          let viz: np.Array;
          if (vizMode === "vorticity") {
            viz = computeVorticity(sim.u.ref, params);
          } else {
            viz = computeVelMagnitude(sim.u.ref);
          }
          // Keep jax-js arrays alive until rendering is submitted.
          const fieldBuf = viz.ref.gpuBufferSync();
          const boundBuf = await sim.X.ref.gpuBuffer();
          if (!running) { viz.dispose(); break; }
          renderer.render(fieldBuf, boundBuf, params, { vizMode, colormap, invertBg, smooth });
          viz.dispose();
        }
        if (perfMode) {
          perfRenderMs += performance.now() - renderT0;
          perfSteps++;
        }
      }

      // FPS counter
      const now = performance.now();
      if (now - lastTime > 1000) {
        fps = Math.round((frames * 1000) / (now - lastTime));
        if (perfMode && perfSteps > 0) {
          console.info(
            `[ib-sim perf] solver=${solverMode} N=${N} fps=${fps} stepEncodeCPU=${(perfStepMs / perfSteps).toFixed(3)}ms renderEncodeCPU=${(perfRenderMs / perfSteps).toFixed(3)}ms`,
          );
          perfSteps = 0;
          perfStepMs = 0;
          perfRenderMs = 0;
        }
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

  function resetClock() {
    frameCount = 0;
    simTime = 0;
    nextAutoPauseTime = PAUSE_INTERVAL_SECONDS;
  }

  function updateSolverParams() {
    if (!params) return;
    params.K = paramK;
    params.mu = paramMu;
    params.dt = paramDt;
    if (optimizedSolver) {
      optimizedSolver.setParams(params);
    } else {
      pre?.a.dispose();
      pre = initFourierOperator(params);
    }
  }

  function resetSim() {
    params = createParams(N, paramK, paramMu);
    params.dt = paramDt;
    if (optimizedSolver) {
      optimizedSolver.reset(params);
    } else {
      sim?.u.dispose();
      sim?.X.dispose();
      pre?.a.dispose();
      sim = initState(params);
      pre = initFourierOperator(params);
    }
    resetClock();
    paused = false;
    autoPaused = false;
  }

  async function changeN(newN: number) {
    running = false;
    await new Promise((r) => requestAnimationFrame(r));
    N = newN;
    paramN = newN;
    paramDt = dtDefault;
    if (optimizedSolver) {
      optimizedSolver.destroy();
      optimizedSolver = null;
      try {
        await setupOptimizedSolver();
      } catch (e) {
        console.warn("Optimized solver unavailable after grid change; using reference path.", e);
        setupReferenceSolver();
      }
    } else {
      resetSim();
    }
    simulate();
  }

  function setupReferenceSolver() {
    solverMode = "reference";
    optimizedSolver = null;
    stepFn = createStep();
    resetSim();
  }

  async function setupOptimizedSolver() {
    solverMode = "optimized";
    params = createParams(N, paramK, paramMu);
    params.dt = paramDt;
    optimizedSolver = await WebGpuIbSolver.init(gpuDevice, params);
    resetClock();
    paused = false;
    autoPaused = false;
  }

  async function startup() {
    const url = new URL(window.location.href);
    const preferredSolver = url.searchParams.get("solver") === "reference" ? "reference" : "optimized";
    perfMode = url.searchParams.get("perf") === "1";

    await init("webgpu");
    defaultDevice("webgpu");

    canvas.width = 512;
    canvas.height = 512;

    N = paramN;
    paramDt = dtDefault;

    gpuDevice = getWebGPUDevice();
    renderer = new WebGpuIbRenderer(gpuDevice, canvas);
    if (preferredSolver === "optimized") {
      try {
        await setupOptimizedSolver();
      } catch (e) {
        console.warn("Optimized solver unavailable; using reference path.", e);
        setupReferenceSolver();
      }
    } else {
      setupReferenceSolver();
    }
    simulate();
  }

  onMount(() => {
    startup().catch((e) => {
      console.error(e);
      errorMsg = String(e);
    });
    return () => {
      running = false;
      optimizedSolver?.destroy();
      sim?.u.dispose();
      sim?.X.dispose();
      pre?.a.dispose();
      renderer?.destroy();
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

<svelte:head>
  <title>Interactive 2D Immersed Boundary Method</title>
</svelte:head>

<main class="ib-sim">
  <nav class="breadcrumb">
    <a href="https://guanhuasun.github.io/">&larr; guanhuasun.github.io</a>
  </nav>

  {#if errorMsg}
    <pre class="error-msg">{errorMsg}</pre>
  {/if}

  <header class="page-header">
    <h1>
      <span class="title-emphasis">Interactive</span>
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
      WebGPU {solverMode} solver &middot; N={N} &middot; FFT-based IMEX solver
      &middot; &Delta;t={paramDt.toFixed(4)}
    </p>
    <SimulationNav current="2d" />
  </header>

  <hr />

  <section class="sim-area">
    <div class="spacer"></div>

    <div class="canvas-block">
      <div class="canvas-caption">
        {solverMode} &middot; N={N} &middot; &Delta;t={paramDt.toFixed(4)} &middot; {vizLabel} &middot; {cmapLabel}
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
            oninput={updateSolverParams}
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
            oninput={updateSolverParams}
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
            oninput={updateSolverParams}
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
        <span class="auto-pause-msg">Paused at t = {simTime.toFixed(2)} s.</span>
      {/if}
    </div>

    <div class="toolbar-right">
      <label class="toolbar-field">
        <span class="toolbar-label">Grid</span>
        <div class="select-wrap">
          <select value={paramN} onchange={(e) => void changeN(Number(e.currentTarget.value))}>
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

<footer class="copyright">&copy; 2026 Guanhua Sun</footer>

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
  .title-emphasis {
    color: var(--color-link);
    font-style: italic;
    font-weight: 700;
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

  .copyright {
    position: fixed;
    right: 16px;
    bottom: 12px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--color-text-caption);
    pointer-events: none;
  }
  @media (max-width: 860px) {
    .sim-area { flex-direction: column; }
    .spacer { display: none; }
    .canvas-block { width: 100%; max-width: 538px; }
    .canvas-matte { width: 100%; box-sizing: border-box; }
    canvas { width: 100%; height: auto; aspect-ratio: 1; }
    .param-panel { width: 100%; flex: auto; padding-top: 0; }
    .copyright { position: static; text-align: right; margin: 16px; }
  }
</style>
