import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { isAbsolute } from 'node:path';
import ts from 'typescript';
const playwright = process.env.PLAYWRIGHT_MODULE || 'playwright';
const { chromium } = await import(isAbsolute(playwright) ? pathToFileURL(playwright).href : playwright);

(async () => {
  const { createServer } = await import('vite');
  const server = await createServer({ server: { host: '127.0.0.1', port: 0, strictPort: true } });
  let browser;
  try {
    await server.listen();
    browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('console', m => { if (m.type() === 'error') errors.push(m.text() + ' ' + m.location().url); });
    await page.route('**/__periodic_check', route => route.fulfill({ contentType: 'text/html', body: '<title>Periodic coupling check</title>', headers: { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' } }));
    await page.goto(`${server.resolvedUrls.local[0]}__periodic_check`);
    const code = ts.transpileModule(readFileSync('src/routes/webgpu-ib-solver.ts', 'utf8') + '\nexport { solverShader };', { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText;
    const results = await page.evaluate(async code => {
      const { solverShader, WebGpuIbSolver } = await import(URL.createObjectURL(new Blob([code], { type: 'text/javascript' })));
      const { numpy: np, init, defaultDevice, getWebGPUDevice } = await import('/jax-js/src/index.ts');
      const reference = await import('/src/routes/ib-solver.ts');
      await init('webgpu');
      defaultDevice('webgpu');
      const device = getWebGPUDevice();
      device.pushErrorScope('validation');
      const summaries = [];
      const read = async source => {
        const target = device.createBuffer({ size: source.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(source, 0, target, 0, source.size);
        device.queue.submit([encoder.finish()]);
        await target.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(target.getMappedRange()).slice();
        target.unmap(); target.destroy();
        return data;
      };
      for (const N of [64, 128]) {
        const p = { ...reference.createParams(N, 1, 0.01), Nb: 1, dtheta: 1, dt: 0.01 };
        const module = device.createShaderModule({ code: solverShader(p) });
        const spread = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'spread_force' } });
        const predict = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'predict_midpoint' } });
        const buffers = [];
        function buffer(size, data) {
          const b = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
          buffers.push(b);
          if (data) device.queue.writeBuffer(b, 0, new Float32Array(data));
          return b;
        }
        const params = buffer(32, [p.dt, p.rho, p.mu, p.K, p.h, p.dtheta, N, 1]);
        const x = buffer(8), force = buffer(8, [1, 1]), ff = buffer(N*N*8), mid = buffer(8);
        const velocity = buffer(N*N*8, new Float32Array(N*N*2).fill(1));
        const readback = device.createBuffer({ size: N*N*8+8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const bind = (pipeline, entries) => device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
        const sb = bind(spread, [[0, params], [6, x], [7, force], [8, ff]]);
        const ib = bind(predict, [[0, params], [1, velocity], [2, x], [3, mid]]);
        const locations = [-3.125, -1.01, -1, -0.0001, 0, 0.9999, 1, 1+1/N-0.000001, 1+1/N, 1+1.25/N, 2.125, 3.125];
        let maxMassError = 0, maxInterpolationError = 0, maxParityError = 0, cases = 0;
        for (const a of locations) for (const point of [[a, 0.375], [0.375, a], [a, a]]) {
          const coords = new Float32Array(point);
          device.queue.writeBuffer(x, 0, coords);
          const encoder = device.createCommandEncoder();
          for (const [pipeline, group, count] of [[spread, sb, Math.ceil(N*N/256)], [predict, ib, 1]]) {
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(count); pass.end();
          }
          encoder.copyBufferToBuffer(ff, 0, readback, 0, N*N*8);
          encoder.copyBufferToBuffer(mid, 0, readback, N*N*8, 8);
          device.queue.submit([encoder.finish()]);
          await readback.mapAsync(GPUMapMode.READ);
          const actual = new Float32Array(readback.getMappedRange()).slice();
          readback.unmap();
          const refSpread = reference.spread(np.ones([1, 2]), np.array(coords, { shape: [1, 2] }), p);
          const refInterp = reference.interp(np.ones([N, N, 2]), np.array(coords, { shape: [1, 2] }), p);
          const expected = await refSpread.ref.data(), interpolated = await refInterp.ref.data();
          let mass = 0, refMass = 0;
          for (let i=0; i<N*N*2; i+=2) { mass += actual[i]*p.h*p.h; refMass += expected[i]*p.h*p.h; maxParityError = Math.max(maxParityError, Math.abs(actual[i]-expected[i])*p.h*p.h); }
          maxMassError = Math.max(maxMassError, Math.abs(mass-1), Math.abs(refMass-1));
          for (let c=0; c<2; c++) maxInterpolationError = Math.max(maxInterpolationError, Math.abs(interpolated[c]-1), Math.abs((actual[N*N*2+c]-coords[c])/(p.dt/2)-1));
          refSpread.dispose(); refInterp.dispose(); cases++;
        }
        summaries.push({ N, cases, maxMassError, maxInterpolationError, maxParityError });
        if (maxMassError > 1e-5 || maxInterpolationError > 1e-4 || maxParityError > 1e-5) throw new Error(JSON.stringify(summaries));
        readback.destroy(); buffers.forEach(b => b.destroy());

        const simulationParams = { ...reference.createParams(N, 1, 0.01), dt: 0.001 };
        const batched = await WebGpuIbSolver.init(device, simulationParams);
        const singles = await WebGpuIbSolver.init(device, simulationParams);
        for (const count of [1, 9, 90, 900]) {
          const encoder = device.createCommandEncoder();
          batched.stepBatch(count, encoder);
          batched.renderField('vorticity', encoder);
          device.queue.submit([encoder.finish()]);
          for (let i = 0; i < count; i++) singles.stepBatch();
          for (const mode of ['vorticity', 'velocity']) {
            const a = await read(batched.renderField(mode));
            const b = await read(singles.renderField(mode));
            if (a.some((v, i) => !Number.isFinite(v) || v !== b[i])) throw new Error(`Batch mismatch: N=${N}, ${mode}`);
          }
          const a = await read(batched.boundaryBuffer), b = await read(singles.boundaryBuffer);
          if (a.some((v, i) => v !== b[i])) throw new Error(`Boundary mismatch: N=${N}`);
        }
        singles.stepBatch(); // Reset from opposite ping-pong phases and change parameters.
        simulationParams.dt = 0.002; simulationParams.K = 0.5;
        batched.reset(simulationParams); singles.reset(simulationParams);
        batched.stepBatch(10); singles.stepBatch(10);
        const a = await read(batched.renderField('velocity')), b = await read(singles.renderField('velocity'));
        if (a.some((v, i) => !Number.isFinite(v) || v !== b[i])) throw new Error(`Reset mismatch: N=${N}`);
        batched.destroy(); singles.destroy();
      }
      const error = await device.popErrorScope();
      if (error) throw new Error(error.message);
      return summaries;
    }, code);
    console.log('GPU periodic coupling:', JSON.stringify(results));
    if (errors.length) throw new Error(errors.join('\n'));
    console.log('GPU periodic coupling checks passed.');
  } finally {
    await browser?.close();
    await server.close();
  }
})().catch(e => { console.error(e); process.exitCode = 1; });
