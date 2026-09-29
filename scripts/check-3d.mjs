// Run like check-2d.mjs; PLAYWRIGHT_MODULE may point to an external Playwright index.mjs.
import { readFileSync, mkdirSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { isAbsolute } from "node:path";
import ts from "typescript";
const modulePath = process.env.PLAYWRIGHT_MODULE || "playwright";
const { chromium } = await import(
  isAbsolute(modulePath) ? pathToFileURL(modulePath).href : modulePath
);
const { createServer } = await import("vite");
const server = await createServer({
  server: { host: "127.0.0.1", port: 0, strictPort: true },
});
let browser;
try {
  await server.listen();
  browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    args: ["--enable-unsafe-webgpu"],
  });
  const page = await browser.newPage({
    viewport: { width: 1280, height: 1080 },
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  await page.route("**/__check3d", (r) =>
    r.fulfill({
      contentType: "text/html",
      body: "<title>3D numerical checks</title>",
      headers: {
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
      },
    }),
  );
  const url = server.resolvedUrls.local[0];
  await page.goto(url + "__check3d");
  const source =
    readFileSync("src/routes/ib3d-solver.ts", "utf8") +
    "\nexport { predictShader, forceShader, spreadShader };";
  const code = ts.transpileModule(source, {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.ESNext,
    },
  }).outputText;
  const results = await page.evaluate(async (code) => {
    const { WebGpuIb3DSolver, predictShader, forceShader, spreadShader } =
      await import(
        URL.createObjectURL(new Blob([code], { type: "text/javascript" }))
      );
    const {
      numpy: np,
      init,
      defaultDevice,
      getWebGPUDevice,
    } = await import("/jax-js/src/index.ts");
    const reference = await import("/src/routes/ib-solver.ts");
    await init("webgpu");
    defaultDevice("webgpu");
    const device = getWebGPUDevice();
    device.pushErrorScope("validation");
    const report = {};
    const assert = (ok, message) => {
      if (!ok) throw new Error(message);
    };
    const maxDiff = (a, b) =>
      a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
    const read = async (source) => {
      const target = device.createBuffer({
        size: source.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const e = device.createCommandEncoder();
      e.copyBufferToBuffer(source, 0, target, 0, source.size);
      device.queue.submit([e.finish()]);
      await target.mapAsync(GPUMapMode.READ);
      const a = new Float32Array(target.getMappedRange()).slice();
      target.destroy();
      return a;
    };
    const params = (N) => ({
      N,
      dt: 0.002,
      K: 0.006,
      mu: 0.01,
      refinement: 1,
      radius: 0.28,
    });
    // Differential test: every z plane must follow the unmodified 2D jax-js fluid solver.
    for (const N of [8, 16]) {
      const p = { ...params(N), K: 0 },
        p2 = { ...reference.createParams(N, 0, p.mu), dt: p.dt };
      const sim = await WebGpuIb3DSolver.init(device, p);
      const seed2 = new Float32Array(N * N * 2),
        seed3 = new Float32Array(N ** 3 * 4);
      for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++) {
          const x = (2 * Math.PI * i) / N,
            y = (2 * Math.PI * j) / N;
          const a = 0.14 * Math.sin(y) + 0.06 * Math.cos(x) * Math.sin(y),
            b = 0.11 * Math.sin(x) - 0.06 * Math.sin(x) * Math.cos(y);
          seed2[(i * N + j) * 2] = a;
          seed2[(i * N + j) * 2 + 1] = b;
          for (let k = 0; k < N; k++)
            seed3.set([a, b, 0, 0], ((i * N + j) * N + k) * 4);
        }
      device.queue.writeBuffer(sim.velocityBuffer, 0, seed3);
      let u = np.array(seed2, { shape: [N, N, 2] });
      const { a } = reference.initFourierOperator(p2);
      for (let step = 0; step < 10; step++) {
        const [next, half] = reference.fluidSolve(
          u,
          np.zeros([N, N, 2]),
          a.ref,
          p2,
        );
        half.dispose();
        u = next;
      }
      sim.stepBatch(10);
      const actual = await read(sim.velocityBuffer),
        expected = await u.ref.data();
      let error = 0;
      for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++)
          for (let k = 0; k < N; k++)
            for (let c = 0; c < 3; c++)
              error = Math.max(
                error,
                Math.abs(
                  actual[((i * N + j) * N + k) * 4 + c] -
                    (c < 2 ? expected[(i * N + j) * 2 + c] : 0),
                ),
              );
      if (error >= 2e-5) {
        const validation = await device.popErrorScope();
        throw new Error(
          `2D extrusion N=${N}: ${error}, GPU ${Array.from(actual.slice(N * 4, N * 4 + 4))}, ref ${Array.from(expected.slice(2, 6))}, validation ${validation?.message}`,
        );
      }
      report[`extrusion${N}`] = error;
      u.dispose();
      a.dispose();
      sim.destroy();
    }
    // Analytic curl and Crank-Nicolson decay exercise the added axis independently.
    {
      const N = 16,
        p = { ...params(N), K: 0 },
        sim = await WebGpuIb3DSolver.init(device, p),
        seed = new Float32Array(N ** 3 * 4);
      for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++)
          for (let k = 0; k < N; k++)
            seed.set(
              [
                Math.sin((2 * Math.PI * k) / N),
                Math.sin((2 * Math.PI * i) / N),
                Math.sin((2 * Math.PI * j) / N),
                0,
              ],
              ((i * N + j) * N + k) * 4,
            );
      device.queue.writeBuffer(sim.velocityBuffer, 0, seed);
      const curl = await read(sim.renderField("vorticity"));
      let error = 0;
      const derivative = N * Math.sin((2 * Math.PI) / N);
      for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++)
          for (let k = 0; k < N; k++) {
            const c = [
              derivative * Math.cos((2 * Math.PI * j) / N),
              derivative * Math.cos((2 * Math.PI * k) / N),
              derivative * Math.cos((2 * Math.PI * i) / N),
            ];
            const q = ((i * N + j) * N + k) * 4;
            for (let d = 0; d < 4; d++)
              error = Math.max(
                error,
                Math.abs(curl[q + d] - (d < 3 ? c[d] : Math.hypot(...c))),
              );
          }
      assert(error < 2e-5, `curl: ${error}`);
      report.curl = error;
      for (let q = 0; q < seed.length; q += 4) {
        seed[q + 1] = 0;
        seed[q + 2] = 0;
      }
      device.queue.writeBuffer(sim.velocityBuffer, 0, seed);
      sim.stepBatch(100);
      const decayed = await read(sim.velocityBuffer);
      const lambda = 4 * N * N * Math.sin(Math.PI / N) ** 2;
      const ratio =
        ((1 - 0.5 * p.dt * p.mu * lambda) / (1 + 0.5 * p.dt * p.mu * lambda)) **
        100;
      error = maxDiff(
        decayed,
        seed.map((v) => v * ratio),
      );
      assert(error < 3e-5, `diffusion: ${error}`);
      report.diffusion = error;
      sim.destroy();
    }
    // Independent tensor-product Peskin weights: partition, spread/interpolation adjointness,
    // and invariance after multiple crossings in all three directions.
    {
      const N = 8,
        p = params(N),
        nb = 3;
      const raw = new ArrayBuffer(48),
        dv = new DataView(raw);
      dv.setUint32(0, N, true);
      dv.setUint32(4, nb, true);
      dv.setFloat32(16, 1 / N, true);
      dv.setFloat32(20, 0.5, true);
      dv.setFloat32(24, p.K, true);
      const owned = [];
      const buffer = (data) => {
        const b = device.createBuffer({
          size: data.byteLength,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
        });
        device.queue.writeBuffer(b, 0, data);
        owned.push(b);
        return b;
      };
      const pb = buffer(raw),
        positions = new Float32Array([
          3.98, -2.02, 1.03, 1, -4.01, 2.47, 0.99, 1, 1.34, -3.71, 5.12, 1,
        ]);
      const force = new Float32Array([
        0.2, -0.3, 0.4, 0, -0.6, 0.1, 0.2, 0, 0.1, 0.5, -0.2, 0,
      ]);
      const u = new Float32Array(N ** 3 * 4);
      for (let i = 0; i < u.length; i++)
        if (i % 4 < 3) u[i] = Math.sin(i * 1.23) * 0.3;
      const xb = buffer(positions),
        fb = buffer(force),
        ub = buffer(u),
        spread = buffer(new Float32Array(u.length)),
        mid = buffer(new Float32Array(positions.length));
      const dispatch = async (code, buffers, count) => {
        const pipeline = await device.createComputePipelineAsync({
          layout: "auto",
          compute: {
            module: device.createShaderModule({ code }),
            entryPoint: "main",
          },
        });
        const e = device.createCommandEncoder(),
          pass = e.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(
          0,
          device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: buffers.map((b, binding) => ({
              binding,
              resource: { buffer: b },
            })),
          }),
        );
        pass.dispatchWorkgroups(count);
        pass.end();
        device.queue.submit([e.finish()]);
      };
      await dispatch(
        spreadShader(p),
        [pb, xb, fb, spread],
        Math.ceil(N ** 3 / 128),
      );
      await dispatch(predictShader(p), [pb, ub, xb, mid], 1);
      const actual = await read(spread),
        xm = await read(mid);
      const phi = (r) => {
        const x = Math.abs(r);
        return x < 1
          ? (3 - 2 * x + Math.sqrt(1 + 4 * x - 4 * x * x)) / 8
          : x < 2
            ? (5 - 2 * x - Math.sqrt(-7 + 12 * x - 4 * x * x)) / 8
            : 0;
      };
      let error = 0,
        lhs = 0,
        rhs = 0;
      const mass = [0, 0, 0];
      for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++)
          for (let k = 0; k < N; k++)
            for (let c = 0; c < 3; c++) {
              let expected = 0;
              for (let q = 0; q < nb; q++) {
                const weights = [i, j, k].map((g, d) => {
                  let delta = g - positions[q * 4 + d] * N;
                  delta -= Math.round(delta / N) * N;
                  return phi(delta);
                });
                expected +=
                  force[q * 4 + c] *
                  weights.reduce((a, b) => a * b, 1) *
                  N ** 3;
              }
              const index = ((i * N + j) * N + k) * 4 + c;
              error = Math.max(
                error,
                Math.abs(actual[index] - expected) / N ** 3,
              );
              mass[c] += actual[index] / N ** 3;
              lhs += (actual[index] * u[index]) / N ** 3;
            }
      for (let q = 0; q < nb; q++)
        for (let c = 0; c < 3; c++)
          rhs +=
            (force[q * 4 + c] * (xm[q * 4 + c] - positions[q * 4 + c])) / 0.25;
      assert(
        error < 2e-6 && Math.abs(lhs - rhs) < 5e-6,
        `coupling ${error} adjoint ${lhs - rhs}`,
      );
      for (let c = 0; c < 3; c++)
        assert(
          Math.abs(mass[c] - force[c] - force[4 + c] - force[8 + c]) < 2e-6,
          "spread conservation",
        );
      report.coupling = error;
      report.adjoint = Math.abs(lhs - rhs);
      // Surface forces must use continuous material coordinates, not minimum-image edges.
      const starts = buffer(new Uint32Array([0, 2, 2, 2, 4, 2])),
        neighbors = buffer(new Uint32Array([1, 2, 0, 2, 0, 1])),
        out = buffer(new Float32Array(12)),
        grab = buffer(new Float32Array(8));
      const length = (x, i, j) =>
        Math.hypot(...[0, 1, 2].map((c) => x[j * 4 + c] - x[i * 4 + c]));
      const rest = buffer(
        new Float32Array([
          length(positions, 0, 1),
          length(positions, 0, 2),
          length(positions, 1, 0),
          length(positions, 1, 2),
          length(positions, 2, 0),
          length(positions, 2, 1),
        ]),
      );
      for (const scale of [1, 0.75, 1.25]) {
        const deformed = positions.map((v, i) => (i % 4 < 3 ? scale * v : v));
        device.queue.writeBuffer(xb, 0, deformed);
        await dispatch(
          forceShader(p),
          [pb, xb, starts, neighbors, out, grab, rest],
          1,
        );
        const f = await read(out);
        // Independent energy-gradient check includes both compression and stretch.
        const energy = (x) => {
          let e = 0;
          for (let i = 0; i < 3; i++)
            for (let j = i + 1; j < 3; j++)
              e += 0.5 * p.K * (length(x, i, j) - length(positions, i, j)) ** 2;
          return e;
        };
        for (let i = 0; i < 3; i++)
          for (let c = 0; c < 3; c++) {
            const plus = Array.from(deformed),
              minus = Array.from(deformed),
              eps = 1e-4;
            plus[i * 4 + c] += eps;
            minus[i * 4 + c] -= eps;
            const expected = -(energy(plus) - energy(minus)) / (2 * eps);
            assert(
              Math.abs(f[i * 4 + c] - expected) < 1e-7,
              `rest-length material edge force, scale ${scale}`,
            );
          }
        for (let c = 0; c < 3; c++)
          assert(
            Math.abs(f[c] + f[4 + c] + f[8 + c]) < 1e-7,
            "internal force balance",
          );
        if (scale === 1)
          assert(
            f.every((v) => Math.abs(v) < 1e-7),
            "relaxed edges have no force",
          );
      }
      const pSphere = { ...params(16), refinement: 2 },
        sphere = await WebGpuIb3DSolver.init(device, pSphere);
      const mesh = sphere.mesh;
      dv.setUint32(4, mesh.nb, true);
      device.queue.writeBuffer(pb, 0, raw);
      const sphereForce = buffer(new Float32Array(mesh.nb * 4));
      await dispatch(
        forceShader(pSphere),
        [
          pb,
          buffer(mesh.positions),
          buffer(mesh.starts),
          buffer(mesh.neighbors),
          sphereForce,
          grab,
          buffer(mesh.restLengths),
        ],
        Math.ceil(mesh.nb / 128),
      );
      const initialForce = await read(sphereForce);
      report.initialSphereForce = Math.max(...initialForce.map(Math.abs));
      assert(report.initialSphereForce < 1e-8, "initial sphere is stress-free");
      device.queue.writeBuffer(
        sphere.velocityBuffer,
        0,
        new Float32Array(pSphere.N ** 3 * 4),
      );
      sphere.stepBatch(100);
      report.quiescentSphereDrift = maxDiff(
        await sphere.readPositions(),
        mesh.positions,
      );
      const quietVelocity = await read(sphere.velocityBuffer);
      assert(
        report.quiescentSphereDrift < 1e-6 &&
          quietVelocity.every((v) => Math.abs(v) < 1e-6),
        "relaxed sphere in still fluid remains stationary",
      );
      sphere.destroy();
      owned.forEach((b) => b.destroy());
    }
    // Nonlinear evolution, projection, shared encoder, both ping-pong reset phases and tether.
    for (const N of [16, 32, 64]) {
      const p = params(N),
        a = await WebGpuIb3DSolver.init(device, p),
        b = await WebGpuIb3DSolver.init(device, p);
      const e = device.createCommandEncoder();
      a.stepBatch(101, e);
      a.encodeField(e, "vorticity");
      device.queue.submit([e.finish()]);
      for (let i = 0; i < 101; i++) b.stepBatch();
      const av = await read(a.velocityBuffer),
        bv = await read(b.velocityBuffer);
      assert(
        av.every(Number.isFinite) && maxDiff(av, bv) === 0,
        "batch parity",
      );
      let divergence = 0;
      const at = (i, j, k, c) =>
        av[((((i + N) % N) * N + ((j + N) % N)) * N + ((k + N) % N)) * 4 + c];
      for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++)
          for (let k = 0; k < N; k++)
            divergence = Math.max(
              divergence,
              Math.abs(
                ((at(i + 1, j, k, 0) -
                  at(i - 1, j, k, 0) +
                  at(i, j + 1, k, 1) -
                  at(i, j - 1, k, 1) +
                  at(i, j, k + 1, 2) -
                  at(i, j, k - 1, 2)) *
                  N) /
                  2,
              ),
            );
      assert(divergence < 1e-4, `divergence ${divergence}`);
      report[`divergence${N}`] = divergence;
      b.stepBatch();
      a.reset();
      b.reset();
      a.stepBatch(3);
      b.stepBatch(3);
      assert(
        maxDiff(await read(a.velocityBuffer), await read(b.velocityBuffer)) ===
          0,
        "odd/even reset",
      );
      const x = await a.readPositions();
      a.setGrab(0, [x[0] + 0.15, x[1], x[2]]);
      a.stepBatch(10);
      b.stepBatch(10);
      assert(
        maxDiff(await read(a.velocityBuffer), await read(b.velocityBuffer)) >
          1e-6,
        "tether must affect fluid",
      );
      b.stepBatch(987); // 1,000 steps after reset at the default physical parameters.
      assert(
        (await read(b.velocityBuffer)).every(Number.isFinite),
        "1000-step finite velocity",
      );
      assert(
        (await b.readPositions()).every(Number.isFinite),
        "1000-step finite shell",
      );
      a.destroy();
      b.destroy();
    }
    const validation = await device.popErrorScope();
    assert(!validation, validation?.message);
    return report;
  }, code);
  console.log("3D numerical checks:", JSON.stringify(results));
  await page.goto(url + "3d");
  await page.getByRole("button", { name: "Pause", exact: true }).waitFor();
  await page.waitForFunction(
    () => !document.querySelector("fieldset[disabled]"),
  );
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  if (await page.locator("[role=alert]").count())
    throw new Error(await page.locator("[role=alert]").innerText());
  console.log(
    "3D controls:",
    (await page.locator(".toolbar").ariaSnapshot()).replace(/\n/g, " "),
  );
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  const box = await page.locator("canvas").boundingBox();
  const x = box.x + box.width / 2,
    y = box.y + box.height / 2;
  const canvas = page.locator("canvas");
  const assertUI = (ok, message) => {
    if (!ok) throw new Error(message);
  };
  const waitStatus = (text) =>
    page.waitForFunction(
      (text) => document.querySelector(".status")?.textContent?.includes(text),
      text,
      { timeout: 10000 },
    );
  const canvasImage = async () => {
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.mouse.move(5, 5);
    await page.evaluate(
      () =>
        new Promise((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(resolve)),
        ),
    );
    // Compare the scene interior, excluding the live FPS/status overlays and focus ring.
    return page.screenshot({
      clip: { x: x - 140, y: y - 130, width: 280, height: 260 },
    });
  };
  assertUI(
    (await page
      .getByRole("button", { name: "Auto", exact: true })
      .getAttribute("aria-pressed")) === "true",
    "Auto must be the default",
  );
  await canvas.focus();
  const initialImage = await canvasImage();
  // The default right gesture must pick the shell without rotating the camera.
  await page.mouse.move(x, y);
  await page.mouse.down({ button: "right" });
  await waitStatus("Pulling material point");
  await page.mouse.move(x + 35, y - 20, { steps: 5 });
  await page.mouse.up({ button: "right" });
  assertUI(
    initialImage.equals(await canvasImage()),
    "Pulling a paused shell must not orbit or move it directly",
  );
  // Empty parts of the slice automatically stir; Shift bypasses a shell hit.
  await page.mouse.move(x - 30, y - 155);
  await page.mouse.down({ button: "right" });
  await waitStatus("Stirring fluid");
  await page.mouse.move(x - 15, y - 150, { steps: 3 });
  await page.mouse.up({ button: "right" });
  await page.keyboard.down("Shift");
  await page.mouse.move(x, y);
  await page.mouse.down({ button: "right" });
  await waitStatus("Stirring fluid");
  await page.keyboard.up("Shift"); // Releasing a modifier must not change an active gesture.
  await page.mouse.move(x + 30, y - 10, { steps: 4 });
  await waitStatus("Stirring fluid");
  await page.mouse.up({ button: "right" });
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await page.getByRole("button", { name: "Pull body", exact: true }).click();
  await canvas.focus();
  const beforeOrbit = await canvasImage();
  await page.mouse.move(x, y);
  await page.mouse.down({ button: "left" });
  await page.mouse.move(x + 60, y + 25, { steps: 6 });
  await page.mouse.up({ button: "left" });
  assertUI(
    !beforeOrbit.equals(await canvasImage()),
    "Left-drag must orbit in Pull body mode",
  );
  await page.keyboard.press("Home");
  assertUI(
    beforeOrbit.equals(await canvasImage()),
    "Home must restore the camera",
  );
  await page.keyboard.down("Alt");
  await page.mouse.move(x, y);
  await page.mouse.down({ button: "right" });
  await page.mouse.move(x + 50, y + 20, { steps: 5 });
  await page.mouse.up({ button: "right" });
  await page.keyboard.up("Alt");
  assertUI(
    !beforeOrbit.equals(await canvasImage()),
    "Alt-right-drag must orbit",
  );
  await page.keyboard.press("Home");
  await page.mouse.move(x, y);
  await page.mouse.wheel(0, -150);
  assertUI(
    !beforeOrbit.equals(await canvasImage()),
    "Wheel must zoom in Pull body mode",
  );
  await page.keyboard.press("Home");
  await page.mouse.move(x, y);
  await page.mouse.down({ button: "middle" });
  await page.mouse.move(x, y + 60, { steps: 5 });
  await page.mouse.up({ button: "middle" });
  assertUI(!beforeOrbit.equals(await canvasImage()), "Middle-drag must zoom");
  await page.keyboard.press("Home");
  await page.keyboard.down("Shift");
  await page.mouse.move(x, y);
  await page.mouse.wheel(0, 100);
  await page.keyboard.up("Shift");
  await page.waitForFunction(
    () =>
      Number(document.querySelector('input[aria-label="Slice depth"]').value) >
      0.5,
  );
  await page.getByLabel("Slice depth", { exact: true }).fill("0.5");
  await canvas.focus();
  assertUI(
    beforeOrbit.equals(await canvasImage()),
    "Shift-wheel must move the slice without zooming",
  );
  await page.mouse.move(x, y);
  await page.mouse.down({ button: "right" });
  await waitStatus("Pulling material point");
  await page.keyboard.press("Escape");
  await page.mouse.up({ button: "right" });
  assertUI(
    !(await page.locator(".status").innerText()).includes(
      "Pulling material point",
    ),
    "Escape must release the tether",
  );
  // Shortcuts are local to the canvas: Space should not steal input from controls.
  await canvas.press("Space");
  await page.getByRole("button", { name: "Pause", exact: true }).waitFor();
  await canvas.press("Space");
  await page.getByRole("button", { name: "Play", exact: true }).waitFor();
  await page.getByLabel("Color range", { exact: true }).focus();
  await page.keyboard.press("Space");
  assertUI(
    (await page.getByRole("button", { name: "Play", exact: true }).count()) ===
      1,
    "Typing outside the canvas must not resume the simulation",
  );
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  console.log(
    "Auto picking/stirring, left/Alt orbit, wheel/middle zoom, slice wheel and keyboard shortcuts passed.",
  );
  await page.getByRole("button", { name: "Stir fluid", exact: true }).click();
  await page.mouse.move(x, y);
  await page.mouse.down({ button: "right" });
  await page.mouse.move(x + 60, y - 20, { steps: 8 });
  await page.mouse.up({ button: "right" });
  await page.getByRole("button", { name: "Pull body", exact: true }).click();
  await page.mouse.move(x, y);
  await page.mouse.down({ button: "right" });
  mkdirSync("output/playwright", { recursive: true });
  await page.screenshot({
    path: "output/playwright/3d-pick.png",
    fullPage: true,
  });
  console.log("Pick status:", await page.locator(".status").innerText());
  await page.waitForFunction(
    () =>
      document
        .querySelector(".status")
        ?.textContent?.includes("Pulling material point"),
    null,
    { timeout: 10000 },
  );
  await page.mouse.move(x + 50, y - 35, { steps: 8 });
  await page.mouse.up({ button: "right" });
  await page.getByLabel("View", { exact: true }).selectOption("volume");
  await page.getByLabel("Color", { exact: true }).selectOption("3");
  await page.screenshot({
    path: "output/playwright/3d-volume.png",
    fullPage: true,
  });
  await page.getByLabel("Slice plane", { exact: true }).selectOption("0");
  await page.getByLabel("Slice depth", { exact: true }).fill("0.35");
  await page.getByLabel("Grid", { exact: true }).selectOption("64");
  await page.waitForFunction(
    () => !document.querySelector("fieldset[disabled]"),
  );
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await page.waitForFunction(
    () =>
      Number(document.querySelector(".hud b")?.textContent?.split("=")[1]) >=
      0.12,
  );
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  assertUI(
    (await page.locator(".hud").innerText()).includes("64³"),
    "64³ grid must run and render",
  );
  await page.getByLabel("Grid", { exact: true }).selectOption("32");
  await page.waitForFunction(
    () => !document.querySelector("fieldset[disabled]"),
  );
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await page.getByLabel("View", { exact: true }).selectOption("slice");
  await page.getByLabel("Color", { exact: true }).selectOption("0");
  await page.getByLabel("Slice plane", { exact: true }).selectOption("2");
  await page.getByLabel("Slice depth", { exact: true }).fill("0.5");
  await page.getByRole("button", { name: "Auto", exact: true }).click();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.mouse.move(x, y);
  await page.mouse.down({ button: "left" });
  await page.mouse.move(x + 35, y + 20, { steps: 5 });
  await page.mouse.up({ button: "left" });
  await page.getByRole("button", { name: "Home view", exact: true }).click();
  await page.getByRole("button", { name: "Auto", exact: true }).click();
  mkdirSync("output/playwright", { recursive: true });
  await page.screenshot({
    path: "output/playwright/3d-slice.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: "output/playwright/3d-mobile.png",
    fullPage: true,
  });
  if (
    await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth,
    )
  )
    throw new Error("Mobile horizontal overflow");
  if (await page.locator("[role=alert]").count())
    throw new Error(await page.locator("[role=alert]").innerText());
  if (errors.length) throw new Error(errors.join("\n"));
  console.log(
    "3D browser interactions passed; screenshot: output/playwright/3d-slice.png",
  );
} finally {
  await browser?.close();
  await server.close();
}
