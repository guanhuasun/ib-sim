import type { IBParams } from "./ib-solver";
import type { VizMode } from "./webgpu-ib-solver";

type View = { vizMode: VizMode; colormap: number; invertBg: boolean; smooth: boolean };

export class WebGpuIbRenderer {
  #context: GPUCanvasContext;
  #layout: GPUBindGroupLayout;
  #field: GPURenderPipeline;
  #lines: GPURenderPipeline;
  #points: GPURenderPipeline;
  #uniform: GPUBuffer;
  #data = new Float32Array(8);
  #ints = new Uint32Array(this.#data.buffer);
  #groups = new WeakMap<GPUBuffer, WeakMap<GPUBuffer, GPUBindGroup>>();

  constructor(private device: GPUDevice, canvas: HTMLCanvasElement) {
    this.#context = canvas.getContext("webgpu")!;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.#context.configure({ device, format, alphaMode: "opaque" });
    this.#uniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.#layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ] });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.#layout] });
    const module = device.createShaderModule({ code: renderShader });
    const pipeline = (vertex: string, fragment: string, blend?: GPUBlendState) => device.createRenderPipeline({
      layout,
      vertex: { module, entryPoint: vertex },
      fragment: { module, entryPoint: fragment, targets: [{ format, blend }] },
    });
    const blend: GPUBlendState = {
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
    };
    this.#field = pipeline("fieldVertex", "fieldFragment");
    this.#lines = pipeline("lineVertex", "boundaryFragment", blend);
    this.#points = pipeline("pointVertex", "boundaryFragment", blend);
  }

  render(field: GPUBuffer, boundary: GPUBuffer, p: IBParams, view: View, encoder = this.device.createCommandEncoder()) {
    this.#ints[0] = p.N;
    this.#ints[1] = p.Nb;
    this.#data[2] = view.vizMode === "vorticity" ? 3 : 1;
    this.#ints[3] = view.colormap;
    this.#data[4] = 0.6 / 128;
    this.#data[5] = 0.4 / 128;
    this.#ints[6] = Number(view.invertBg);
    this.#ints[7] = Number(view.smooth);
    this.device.queue.writeBuffer(this.#uniform, 0, this.#data);
    let groups = this.#groups.get(field);
    if (!groups) this.#groups.set(field, groups = new WeakMap());
    let group = groups.get(boundary);
    if (!group) {
      group = this.device.createBindGroup({ layout: this.#layout, entries: [
        { binding: 0, resource: { buffer: this.#uniform } },
        { binding: 1, resource: { buffer: field } },
        { binding: 2, resource: { buffer: boundary } },
      ] });
      groups.set(boundary, group);
    }
    const pass = encoder.beginRenderPass({ colorAttachments: [{
      view: this.#context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store",
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
    }] });
    pass.setBindGroup(0, group);
    pass.setPipeline(this.#field);
    pass.draw(3);
    // Nine images cover faces and corners; rasterization clips them to the periodic box.
    pass.setPipeline(this.#lines);
    pass.draw(6, p.Nb * 9);
    pass.setPipeline(this.#points);
    pass.draw(6, p.Nb * 9);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  destroy() {
    this.#uniform.destroy();
    this.#context.unconfigure();
    this.#groups = new WeakMap();
  }
}

const renderShader = /* wgsl */ `
struct Uniforms {
  n: u32, nb: u32, fieldScale: f32, cmap: u32,
  pointRadius: f32, lineWidth: f32, invertBg: u32, bilinear: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read> boundary: array<vec2f>;

const NEGATIVE = array<vec3f, 6>(
  vec3f(0.0, 0.9, 0.9), vec3f(0.0, 0.7, 0.65), vec3f(0.1, 0.4, 1.0),
  vec3f(0.329, 0.153, 0.533), vec3f(0.549, 0.318, 0.039), vec3f(0.231, 0.298, 0.753),
);
const POSITIVE = array<vec3f, 6>(
  vec3f(1.0, 0.2, 0.8), vec3f(1.0, 0.55, 0.1), vec3f(1.0, 0.2, 0.1),
  vec3f(0.902, 0.380, 0.004), vec3f(0.004, 0.400, 0.369), vec3f(0.706, 0.016, 0.149),
);
struct FieldOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
@vertex fn fieldVertex(@builtin(vertex_index) vi: u32) -> FieldOut {
  let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3))[vi];
  return FieldOut(vec4f(p, 0, 1), (p + 1.0) * 0.5);
}
fn texel(i: u32, j: u32) -> f32 { return field[(i % u.n) * u.n + (j % u.n)]; }
@fragment fn fieldFragment(in: FieldOut) -> @location(0) vec4f {
  let q = in.uv * f32(u.n);
  let b = vec2u(floor(q));
  var raw: f32;
  if (u.bilinear == 1u) {
    let w = fract(q);
    raw = mix(mix(texel(b.x, b.y), texel(b.x + 1u, b.y), w.x),
              mix(texel(b.x, b.y + 1u), texel(b.x + 1u, b.y + 1u), w.x), w.y);
  } else {
    let p = vec2u(round(q));
    raw = texel(p.x, p.y);
  }
  let v = clamp(raw / u.fieldScale, -1.0, 1.0);
  let center = select(vec3f(0), vec3f(1), u.invertBg == 1u);
  let color = select(NEGATIVE[u.cmap], POSITIVE[u.cmap], v > 0.0);
  return vec4f(mix(center, color, abs(v)), 1);
}

struct BoundaryOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec2f,
  @location(1) @interpolate(flat) a: vec2f,
  @location(2) @interpolate(flat) b: vec2f,
  @location(3) @interpolate(flat) point: u32,
}
fn boundaryVertex(vi: u32, instance: u32, point: u32) -> BoundaryOut {
  let k = instance / 9u;
  let tile = instance % 9u;
  let shift = vec2f(f32(tile % 3u) - 1.0, f32(tile / 3u) - 1.0);
  let a = fract(boundary[k]) + shift;
  let delta = boundary[(k + 1u) % u.nb] - boundary[k];
  let b = a + select(delta - round(delta), vec2f(0), point == 1u);
  let radius = select(u.lineWidth, u.pointRadius, point == 1u);
  let corner = array<vec2f, 6>(vec2f(0,0), vec2f(1,0), vec2f(0,1), vec2f(0,1), vec2f(1,0), vec2f(1,1))[vi];
  let world = mix(min(a, b) - radius, max(a, b) + radius, corner);
  return BoundaryOut(vec4f(world * 2.0 - 1.0, 0, 1), world, a, b, point);
}
@vertex fn lineVertex(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> BoundaryOut {
  return boundaryVertex(vi, ii, 0u);
}
@vertex fn pointVertex(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> BoundaryOut {
  return boundaryVertex(vi, ii, 1u);
}
@fragment fn boundaryFragment(in: BoundaryOut) -> @location(0) vec4f {
  let ab = in.b - in.a;
  let ap = in.world - in.a;
  let t = clamp(dot(ap, ab) / max(dot(ab, ab), 1e-20), 0.0, 1.0);
  let d = length(ap - ab * t);
  let radius = select(u.lineWidth, u.pointRadius, in.point == 1u);
  let alpha = 1.0 - smoothstep(radius * 0.3, radius, d);
  let line = select(0.9, 0.12, u.invertBg == 1u);
  let point = select(1.0, 0.0, u.invertBg == 1u);
  return vec4f(vec3f(select(line, point, in.point == 1u)), alpha);
}
`;
