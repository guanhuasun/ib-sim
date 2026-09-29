<script lang="ts">
  import { defaultDevice, getWebGPUDevice, init } from "@jax-js/jax";
  import { onMount } from "svelte";
  import SimulationHeader from "$lib/SimulationHeader.svelte";
  import { MOUSE, Plane, Vector3 } from "three";
  import {
    WebGpuIb3DSolver,
    type IB3DFieldMode,
    type IB3DParams,
  } from "../ib3d-solver";
  import { Ib3DRenderer, type ViewOptions } from "./renderer";

  let canvas: HTMLCanvasElement;
  let device: GPUDevice;
  let renderer: Ib3DRenderer | null = null;
  let solver: WebGpuIb3DSolver | null = null;
  let disposed = false;
  let frameId = 0;
  let revision = 0;
  let gesture = 0;
  let pointerId: number | null = null;
  let activeDrag: "picking" | "body" | "fluid" | null = null;
  let latestPointer: { x: number; y: number } | null = null;
  let plane = new Plane();
  let previous: Vector3 | null = null;
  let cursor: Vector3 | null = null;
  let grabbed = -1;
  let grabOffset = new Vector3();
  let lastFrame = 0;
  let frames = 0;
  let busy = $state(true);
  let paused = $state(false);
  let step = $state(0);
  let simTime = $state(0);
  let fps = $state(0);
  let errorMsg = $state("");
  let status = $state("Preparing the simulation…");
  let mode = $state<"auto" | "fluid" | "body">("auto");
  let field = $state<IB3DFieldMode>("vorticity");
  let view = $state<ViewOptions["view"]>("slice");
  let component = $state(0);
  let axis = $state(2);
  let depth = $state(0.5);
  let fieldScale = $state(0.5);
  let opacity = $state(0.85);
  let surface = $state(0.6);
  let arrows = $state(true);
  let paramN = $state(64);
  let paramRefine = $state(2);
  let paramK = $state(0.006);
  let paramMu = $state(0.01);
  let paramDt = $state(0.002);
  let stepsPerFrame = $state(6);
  let meshStats = $state("");
  const axes = ["x", "y", "z"];
  const inDomain = (p: Vector3) => p.toArray().every((v) => v >= 0 && v <= 1);
  function makeParams(): IB3DParams {
    return {
      N: paramN,
      dt: paramDt,
      K: paramK,
      mu: paramMu,
      refinement: paramRefine,
      radius: 0.28,
    };
  }
  function syncParams() {
    if (!busy) solver?.setParams(makeParams());
  }
  function endGesture() {
    gesture++;
    const id = pointerId;
    pointerId = null;
    if (id !== null && canvas.hasPointerCapture(id))
      canvas.releasePointerCapture(id);
    grabbed = -1;
    activeDrag = null;
    latestPointer = null;
    previous = null;
    cursor = null;
    solver?.setGrab();
    status = idleHint();
  }
  function idleHint() {
    const interaction =
      mode === "auto"
        ? "pull shell / stir slice"
        : mode === "fluid"
          ? "stir slice"
          : "pull shell";
    return `Left-drag: orbit · right-drag: ${interaction} · scroll: zoom`;
  }
  function changeMode() {
    endGesture();
    if (mode === "fluid") view = "slice";
  }
  function sliceChanged() {
    endGesture();
  }
  function viewChanged() {
    endGesture();
  }
  function fieldChanged() {
    fieldScale = field === "vorticity" ? 0.5 : 0.08;
  }
  function resetSim() {
    if (busy) return;
    endGesture();
    solver?.reset(makeParams());
    step = 0;
    simTime = 0;
  }
  async function rebuild() {
    endGesture();
    busy = true;
    const token = ++revision;
    try {
      const next = await WebGpuIb3DSolver.init(device, makeParams());
      if (disposed || token !== revision) {
        next.destroy();
        return;
      }
      solver?.destroy();
      solver = next;
      step = 0;
      simTime = 0;
      meshStats = `${next.mesh.nb} vertices · ${next.mesh.nt} triangles`;
      errorMsg = "";
    } catch (e) {
      errorMsg = String(e);
      if (solver) {
        paramN = solver.params.N;
        paramRefine = solver.params.refinement;
      }
    } finally {
      if (token === revision) busy = false;
    }
  }
  function frame(now: number) {
    if (disposed) return;
    try {
      if (solver && renderer && !busy) {
        const encoder = device.createCommandEncoder();
        if (!paused) {
          solver.stepBatch(stepsPerFrame, encoder);
          step += stepsPerFrame;
          simTime += stepsPerFrame * paramDt;
        }
        if (view !== "off") solver.encodeField(encoder, field);
        renderer.render(
          encoder,
          solver,
          {
            view,
            component,
            scale: fieldScale,
            axis,
            depth,
            opacity,
            surface,
            arrows,
          },
          cursor,
        );
        device.queue.submit([encoder.finish()]);
        frames++;
      }
      if (now - lastFrame > 1000) {
        fps = Math.round((frames * 1000) / (now - lastFrame));
        frames = 0;
        lastFrame = now;
      }
    } catch (e) {
      errorMsg = String(e);
      paused = true;
    }
    frameId = requestAnimationFrame(frame);
  }
  function slicePlane() {
    const n = new Vector3();
    n.setComponent(axis, 1);
    return new Plane(n, -depth);
  }
  function startFluid(clientX: number, clientY: number) {
    if (!renderer) return;
    view = "slice";
    plane = slicePlane();
    previous = renderer.planePoint(clientX, clientY, plane);
    if (!previous || !inDomain(previous)) {
      endGesture();
      status =
        "Start inside the highlighted slice · left-drag to change the view";
      return;
    }
    activeDrag = "fluid";
    cursor = previous.clone();
    status = "Stirring fluid on the slice · release to let go";
  }
  // Capture before OrbitControls' native pointer listener: each drag has one owner.
  async function onPointerDown(e: PointerEvent) {
    if (busy || !solver || !renderer) {
      e.stopImmediatePropagation();
      return;
    }
    canvas.focus({ preventScroll: true });
    const orbit =
      (e.button === 0 && e.pointerType !== "touch") ||
      e.button === 1 ||
      e.altKey ||
      (e.pointerType === "touch" && mode === "auto");
    if (orbit) {
      endGesture();
      // OrbitControls swaps rotate/pan with Ctrl/Meta/Shift. Keep camera gestures
      // rotating even when a modifier is held; panning remains disabled.
      const rotate =
        e.ctrlKey || e.metaKey || e.shiftKey ? MOUSE.PAN : MOUSE.ROTATE;
      renderer.controls.mouseButtons.LEFT = rotate;
      renderer.controls.mouseButtons.RIGHT = rotate;
      return;
    }
    e.preventDefault();
    e.stopImmediatePropagation();
    if (e.button !== (e.pointerType === "touch" ? 0 : 2) || !e.isPrimary)
      return;
    endGesture();
    const token = gesture;
    const automatic = mode === "auto";
    pointerId = e.pointerId;
    canvas.setPointerCapture(e.pointerId);
    if (mode === "fluid" || e.shiftKey) {
      startFluid(e.clientX, e.clientY);
    } else {
      activeDrag = "picking";
      status = "Picking the shell…";
      // Read only on a pick; the frame loop keeps the full fluid grid on the GPU.
      const active = solver;
      try {
        const x = await active.readPositions();
        if (token !== gesture || active !== solver || disposed) return;
        const ray = renderer.ray(e.clientX, e.clientY).clone();
        const triangles = active.mesh.triangles;
        let nearest = Infinity,
          hit: Vector3 | null = null,
          vertex = -1,
          offset = new Vector3();
        const a = new Vector3(),
          b = new Vector3(),
          c = new Vector3(),
          p = new Vector3();
        for (let t = 0; t < triangles.length; t += 3) {
          const ids = [triangles[t], triangles[t + 1], triangles[t + 2]];
          a.fromArray(x, ids[0] * 4);
          b.fromArray(x, ids[1] * 4);
          c.fromArray(x, ids[2] * 4);
          const center = a
            .clone()
            .add(b)
            .add(c)
            .multiplyScalar(1 / 3)
            .floor();
          for (let i = -1; i <= 1; i++)
            for (let j = -1; j <= 1; j++)
              for (let k = -1; k <= 1; k++) {
                const delta = new Vector3(i, j, k).sub(center);
                const av = a.clone().add(delta),
                  bv = b.clone().add(delta),
                  cv = c.clone().add(delta);
                if (
                  !ray.intersectTriangle(av, bv, cv, false, p) ||
                  !inDomain(p)
                )
                  continue;
                const distance = ray.origin.distanceToSquared(p);
                if (distance >= nearest) continue;
                nearest = distance;
                hit = p.clone();
                const points = [av, bv, cv];
                let corner = 0;
                for (let q = 1; q < 3; q++)
                  if (
                    points[q].distanceToSquared(p) <
                    points[corner].distanceToSquared(p)
                  )
                    corner = q;
                vertex = ids[corner];
                offset = new Vector3().fromArray(x, vertex * 4).sub(p);
              }
        }
        if (!hit) {
          if (automatic) {
            startFluid(e.clientX, e.clientY);
            if (latestPointer)
              moveInteraction(latestPointer.x, latestPointer.y);
          } else {
            endGesture();
            status =
              "No shell at that point — try a visible triangle or Shift-right-drag to stir";
          }
          return;
        }
        activeDrag = "body";
        grabbed = vertex;
        grabOffset = offset;
        cursor = hit;
        plane.setFromNormalAndCoplanarPoint(
          renderer.camera.getWorldDirection(new Vector3()),
          hit,
        );
        active.setGrab(
          vertex,
          hit.clone().add(offset).toArray() as [number, number, number],
        );
        status = `Pulling material point ${vertex} · release to let go`;
        if (latestPointer) moveInteraction(latestPointer.x, latestPointer.y);
      } catch (e) {
        if (token === gesture) {
          errorMsg = String(e);
          endGesture();
        }
      }
    }
  }
  function onPointerMove(e: PointerEvent) {
    if (busy || !renderer || !solver) return;
    if (pointerId === null) {
      cursor =
        (mode === "fluid" || e.shiftKey) && e.buttons === 0
          ? renderer.planePoint(e.clientX, e.clientY, slicePlane())
          : null;
      if (cursor && !inDomain(cursor)) cursor = null;
      return;
    }
    if (pointerId !== e.pointerId) return;
    latestPointer = { x: e.clientX, y: e.clientY };
    if (activeDrag !== "picking") moveInteraction(e.clientX, e.clientY);
  }
  function moveInteraction(clientX: number, clientY: number) {
    if (!renderer || !solver) return;
    const p = renderer.planePoint(clientX, clientY, plane);
    if (!p) return;
    if (activeDrag === "body" && grabbed >= 0) {
      cursor = p;
      solver.setGrab(
        grabbed,
        p.clone().add(grabOffset).toArray() as [number, number, number],
      );
    } else if (activeDrag === "fluid") {
      if (!inDomain(p)) {
        previous = null;
        cursor = null;
        return;
      }
      if (previous) {
        const force = p.clone().sub(previous).multiplyScalar(180);
        if (force.length() > 12) force.setLength(12);
        solver.applyImpulse(
          p.toArray() as [number, number, number],
          force.toArray() as [number, number, number],
        );
      }
      previous = p;
      cursor = p;
    }
  }
  function onPointerUp(e: PointerEvent) {
    if (pointerId === e.pointerId) {
      endGesture();
    }
  }
  function onWheel(e: WheelEvent) {
    endGesture();
    if (!e.shiftKey || busy) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    view = "slice";
    const delta =
      e.deltaY *
      (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? canvas.clientHeight : 1);
    depth =
      Math.round(
        Math.max(
          0,
          Math.min(1, depth + Math.max(-0.05, Math.min(0.05, delta * 0.001))),
        ) * 100,
      ) / 100;
  }
  function onKeyDown(e: KeyboardEvent) {
    if (e.target !== canvas || e.ctrlKey || e.metaKey || e.altKey || e.repeat)
      return;
    if (e.code === "Space" && !busy) {
      e.preventDefault();
      endGesture();
      paused = !paused;
    } else if (e.key === "Escape") {
      e.preventDefault();
      endGesture();
    } else if (e.key === "Home") {
      e.preventDefault();
      endGesture();
      renderer?.controls.reset();
    }
  }
  function onGpuError(event: GPUUncapturedErrorEvent) {
    errorMsg = event.error.message;
    paused = true;
  }
  onMount(() => {
    canvas.addEventListener("pointerdown", onPointerDown, true);
    canvas.addEventListener("wheel", onWheel, {
      capture: true,
      passive: false,
    });
    window.addEventListener("blur", endGesture);
    void (async () => {
      try {
        await init("webgpu");
        defaultDevice("webgpu");
        if (disposed) return;
        device = getWebGPUDevice();
        device.addEventListener("uncapturederror", onGpuError);
        renderer = new Ib3DRenderer(device, canvas);
        await rebuild();
        if (disposed) return;
        changeMode();
        lastFrame = performance.now();
        frameId = requestAnimationFrame(frame);
      } catch (e) {
        errorMsg = String(e);
        busy = false;
      }
    })();
    return () => {
      disposed = true;
      revision++;
      endGesture();
      canvas.removeEventListener("pointerdown", onPointerDown, true);
      canvas.removeEventListener("wheel", onWheel, true);
      window.removeEventListener("blur", endGesture);
      cancelAnimationFrame(frameId);
      renderer?.destroy();
      solver?.destroy();
      device?.removeEventListener("uncapturederror", onGpuError);
    };
  });
</script>

<svelte:head><title>Interactive 3D Immersed Boundary Method</title></svelte:head
>
<main>
  <SimulationHeader
    dimension="3d"
    description="An elastic shell in an incompressible fluid. Drag to orbit, stir the fluid, or pull the shell."
    grid={paramN}
    dt={paramDt}
  />
  {#if errorMsg}<pre class="error" role="alert">{errorMsg}</pre>{/if}
  <div class="toolbar">
    <div class="modes" aria-label="Interaction mode">
      {#each [["auto", "Auto"], ["fluid", "Stir fluid"], ["body", "Pull body"]] as [value, label]}
        <button
          class:active={mode === value}
          aria-pressed={mode === value}
          disabled={busy}
          onclick={() => {
            mode = value as typeof mode;
            changeMode();
          }}>{label}</button
        >
      {/each}
    </div>
    <div class="actions">
      <button onclick={() => renderer?.controls.reset()}>Home view</button>
      <button disabled={busy} onclick={resetSim}>Reset</button>
      <button
        class="primary"
        disabled={busy}
        onclick={() => {
          endGesture();
          paused = !paused;
        }}>{paused ? "Play" : "Pause"}</button
      >
    </div>
  </div>
  <div class="sim-area">
    <div class="canvas-block">
      <div class="canvas-caption">
        {field === "vorticity" ? "Vorticity" : "Velocity"} · {view === "slice"
          ? "Vector slice"
          : view === "volume"
            ? "Volume"
            : "Shell only"} · {meshStats}
      </div>
      <section class="stage" aria-label="3D immersed boundary simulation">
        <canvas
          bind:this={canvas}
          tabindex="0"
          aria-label="Interactive 3D fluid and elastic shell"
          aria-describedby="interaction-help"
          onkeydown={onKeyDown}
          onpointermove={onPointerMove}
          onpointerup={onPointerUp}
          onpointercancel={onPointerUp}
          onlostpointercapture={onPointerUp}
          onpointerleave={() => {
            if (pointerId === null) cursor = null;
          }}
        ></canvas>
        <div class="hud">
          <b>t = {simTime.toFixed(3)}</b><span>{step} steps · {fps} FPS</span
          ><span>{paramN}³ · {meshStats}</span>
        </div>
        <div class="legend">
          <b
            >{field === "vorticity" ? "Vorticity ω" : "Velocity u"}{component <
            3
              ? ` · ${axes[component]}`
              : " · magnitude"}</b
          >
          <div class:signed={component < 3} class="ramp"></div>
          <div class="ticks">
            <span>{component < 3 ? `−${fieldScale}` : "0"}</span
            >{#if component < 3}<span>0</span>{/if}<span>{fieldScale}</span>
          </div>
          <small
            >{arrows && view === "slice"
              ? "Arrows show the full 3D vector"
              : "Colors saturate at the displayed range"}</small
          >
        </div>
        <div class="axis-key">
          <span class="x">x</span><span class="y">y</span><span class="z"
            >z</span
          >
        </div>
        {#if busy}<div class="loading">Preparing grid and shell…</div>{/if}
        <div class="status" aria-live="polite">
          {status}{paused ? " · PAUSED" : ""}
        </div>
      </section>
    </div>
    <aside class="param-panel" aria-label="Simulation parameters">
      <fieldset disabled={busy}>
        <legend>Simulation</legend>
        <label
          >Grid<select
            aria-label="Grid"
            bind:value={paramN}
            onchange={() => void rebuild()}
            ><option value={32}>32³</option><option value={64}>64³</option
            ></select
          ></label
        >
        <label
          >Surface mesh<select
            aria-label="Surface mesh"
            bind:value={paramRefine}
            onchange={() => void rebuild()}
            ><option value={1}>80 triangles</option><option value={2}
              >320 triangles</option
            ><option value={3}>1280 triangles</option></select
          ></label
        >
        <label
          >Stiffness K <output>{paramK.toFixed(3)}</output><input
            type="range"
            min="0.001"
            max="0.04"
            step="0.001"
            bind:value={paramK}
            oninput={syncParams}
          /></label
        >
        <label
          >Viscosity μ <output>{paramMu.toFixed(3)}</output><input
            type="range"
            min="0.002"
            max="0.04"
            step="0.001"
            bind:value={paramMu}
            oninput={syncParams}
          /></label
        >
        <label
          >Time step Δt <output>{paramDt.toFixed(4)}</output><input
            type="range"
            min="0.0005"
            max="0.004"
            step="0.0005"
            bind:value={paramDt}
            oninput={syncParams}
          /></label
        >
        <label
          >Steps per frame <output>{stepsPerFrame}</output><input
            type="range"
            min="1"
            max="6"
            step="1"
            bind:value={stepsPerFrame}
          /></label
        >
      </fieldset>
    </aside>
  </div>
  <div class="panels">
    <fieldset>
      <legend>Flow display</legend>
      <label
        >Field<select
          aria-label="Field"
          bind:value={field}
          onchange={fieldChanged}
          ><option value="vorticity">Vorticity ω</option><option
            value="velocity">Velocity u</option
          ></select
        ></label
      >
      <label
        >View<select aria-label="View" bind:value={view} onchange={viewChanged}
          ><option value="slice">Slice</option><option value="volume"
            >Volume</option
          ><option value="off">Shell only</option></select
        ></label
      >
      <label
        >Color<select aria-label="Color" bind:value={component}
          ><option value={0}>x component</option><option value={1}
            >y component</option
          ><option value={2}>z component</option><option value={3}
            >Magnitude</option
          ></select
        ></label
      >
      <label
        >Color range<input
          aria-label="Color range"
          type="number"
          min="0.001"
          max="100"
          step="0.01"
          bind:value={fieldScale}
        /></label
      >
      <label class="check"
        ><input type="checkbox" bind:checked={arrows} />Vector arrows on slice</label
      >
      <label
        >Field opacity <output>{opacity.toFixed(2)}</output><input
          type="range"
          min="0.05"
          max="1"
          step="0.05"
          bind:value={opacity}
        /></label
      >
      <label
        >Shell opacity <output>{surface.toFixed(2)}</output><input
          type="range"
          min="0"
          max="0.8"
          step="0.02"
          bind:value={surface}
        /></label
      >
    </fieldset>
    <fieldset>
      <legend>Slice & interaction</legend>
      <label
        >Slice plane<select
          aria-label="Slice plane"
          bind:value={axis}
          onchange={sliceChanged}
          ><option value={0}>YZ · normal x</option><option value={1}
            >XZ · normal y</option
          ><option value={2}>XY · normal z</option></select
        ></label
      >
      <label
        >Depth {axes[axis]} <output>{depth.toFixed(2)}</output><input
          aria-label="Slice depth"
          type="range"
          min="0"
          max="1"
          step="0.01"
          bind:value={depth}
          oninput={sliceChanged}
        /></label
      >
      <p id="interaction-help">
        Left-drag to orbit. Right-drag the shell to pull it, or right-drag
        elsewhere to stir the slice. Shift-right-drag always stirs, including
        through the shell. Scroll or middle-drag zooms. Shift-scroll moves the
        slice.
      </p>
      <p>
        Pull body picks a material point and applies a spring force in a plane
        facing you. Hold and move while playing; release to let go.
      </p>
      <p>
        With the canvas focused: Space plays/pauses, Escape releases the body,
        Home restores the camera. Buttons choose the right-drag target;
        left-drag and scroll work in every mode. On touch screens, Auto rotates
        with one finger and zooms with a pinch; use the buttons to stir or pull.
      </p>
    </fieldset>
  </div>
  <p class="footnote">
    The slice colors show a signed component or magnitude; arrows retain
    direction. The shell uses edge springs relaxed at the initial sphere, with
    midpoint coupling and the same two-stage fluid scheme as the 2D simulation.
  </p>
</main>

<footer class="copyright">© 2026 Guanhua Sun</footer>

<style>
  main {
    max-width: 960px;
    margin: auto;
    padding: 28px 24px 48px;
    color: var(--color-text);
    font-family: var(--font-sans);
  }
  .toolbar,
  .modes,
  .actions {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .toolbar {
    justify-content: space-between;
    margin: 0 0 16px;
    min-height: 44px;
    flex-wrap: wrap;
  }
  button,
  select,
  input {
    font: 13px var(--font-sans);
  }
  button,
  select,
  input[type="number"] {
    border: 1px solid var(--color-border);
    background: var(--color-bg);
    color: var(--color-text);
    padding: 8px 12px;
    border-radius: 0;
  }
  button {
    cursor: pointer;
    min-height: 36px;
    border-color: var(--color-border-strong);
    font-weight: 500;
  }
  button:hover {
    background: var(--color-surface);
  }
  button.primary {
    background: var(--color-link);
    color: white;
    border-color: var(--color-link);
    min-width: 70px;
  }
  button.primary:hover {
    background: var(--color-link-hover);
  }
  button.active {
    background: var(--color-text);
    color: var(--color-bg);
  }
  button:disabled {
    opacity: 0.45;
    cursor: wait;
  }
  .stage {
    position: relative;
    aspect-ratio: 1;
    padding: 12px;
    box-sizing: border-box;
    border: 1px solid var(--color-border);
    background: var(--color-surface);
  }
  canvas {
    width: 100%;
    height: 100%;
    display: block;
    touch-action: none;
    cursor: grab;
    border: 1px solid var(--color-border);
  }
  canvas:focus-visible {
    outline: 2px solid var(--color-link);
    outline-offset: -2px;
  }
  .hud,
  .legend,
  .status,
  .axis-key {
    position: absolute;
    pointer-events: none;
    font: 11px var(--font-mono);
    background: rgb(255 255 255 / 90%);
    padding: 9px 12px;
    color: var(--color-text-meta);
  }
  .hud {
    left: 20px;
    top: 20px;
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    max-width: calc(100% - 48px);
  }
  .legend {
    left: 20px;
    bottom: 62px;
    width: 190px;
  }
  .legend b {
    font-weight: 500;
  }
  .ramp {
    height: 8px;
    background: linear-gradient(90deg, #eff7ed, #058082);
    margin-top: 10px;
  }
  .ramp.signed {
    background: linear-gradient(90deg, #1f5cc2, #f7f7ed, #d4331a);
  }
  .ticks {
    display: flex;
    justify-content: space-between;
    margin: 5px 0;
  }
  .legend small {
    font: 10px var(--font-mono);
  }
  .status {
    bottom: 13px;
    left: 13px;
    right: 13px;
    border-top: 1px solid var(--color-border);
    padding: 10px 14px;
  }
  .axis-key {
    right: 20px;
    bottom: 60px;
    display: flex;
    gap: 12px;
  }
  .x {
    color: #cc3829;
  }
  .y {
    color: #268c45;
  }
  .z {
    color: #2959cc;
  }
  .loading {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    background: rgb(244 244 244 / 80%);
  }
  .panels {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 20px;
  }
  fieldset {
    min-width: 0;
    border: 0;
    border-top: 1px solid var(--color-border);
    padding: 16px 0 0;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  legend {
    font: 600 13px var(--font-sans);
    padding: 0 8px 0 0;
  }
  label {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 6px;
    font: 500 13px var(--font-sans);
    color: var(--color-text);
  }
  label select,
  label input[type="range"],
  label input[type="number"] {
    grid-column: 1/-1;
    width: 100%;
    box-sizing: border-box;
  }
  label.check {
    display: flex;
    align-items: center;
  }
  output {
    font: 12px var(--font-mono);
    color: var(--color-text);
  }
  fieldset p,
  .footnote {
    font-size: 12px;
    line-height: 1.65;
    color: var(--color-text-meta);
    margin: 0;
  }
  .footnote {
    margin-top: 16px;
    max-width: 950px;
  }
  .error {
    white-space: pre-wrap;
    background: #ffe5dc;
    padding: 16px;
    color: #8b2e17;
  }
  .sim-area {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 220px;
    gap: 24px;
    align-items: start;
  }
  .canvas-block {
    min-width: 0;
  }
  .canvas-caption {
    font: 11px/1.6 var(--font-mono);
    color: var(--color-text-meta);
    margin-bottom: 6px;
  }
  .param-panel fieldset {
    gap: 20px;
  }
  .copyright {
    max-width: 960px;
    margin: 0 auto;
    padding: 0 24px 20px;
    text-align: right;
    font: 11px var(--font-mono);
    color: var(--color-text-caption);
  }
  input[type="range"] {
    appearance: none;
    height: 2px;
    background: var(--color-border);
    padding: 0;
    margin: 10px 0;
    cursor: pointer;
  }
  input[type="range"]::-webkit-slider-thumb {
    appearance: none;
    width: 14px;
    height: 14px;
    background: var(--color-border-strong);
    border: 0;
    border-radius: 2px;
  }
  input[type="range"]::-moz-range-thumb {
    width: 14px;
    height: 14px;
    background: var(--color-border-strong);
    border: 0;
    border-radius: 2px;
  }
  input[type="range"]::-webkit-slider-thumb:hover {
    background: var(--color-link);
  }
  input[type="checkbox"] {
    accent-color: var(--color-border-strong);
  }
  @media (max-width: 760px) {
    main {
      padding: 24px 16px 32px;
    }
    .sim-area,
    .panels {
      grid-template-columns: 1fr;
    }
    .stage {
      aspect-ratio: auto;
      height: 440px;
    }
    .hud {
      gap: 6px;
    }
    .actions {
      margin-left: auto;
    }
  }
</style>
