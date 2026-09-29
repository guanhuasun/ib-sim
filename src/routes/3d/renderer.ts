import {
  Matrix4,
  MOUSE,
  OrthographicCamera,
  Plane,
  Raycaster,
  Vector2,
  Vector3,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { WebGpuIb3DSolver } from "../ib3d-solver";

export type ViewOptions = {
  view: "slice" | "volume" | "off";
  component: number;
  scale: number;
  axis: number;
  depth: number;
  opacity: number;
  surface: number;
  arrows: boolean;
};
const common = /* wgsl */ `
struct Uniforms { vp: mat4x4f, inv: mat4x4f, field: vec4f, slice: vec4f, style: vec4f, eye: vec4f }
@group(0) @binding(0) var<uniform> u: Uniforms;
fn project(p: vec3f) -> vec4f {
  let q = u.vp * vec4f(p, 1.0);
  return vec4f(q.xy, (q.z + q.w) * 0.5, q.w);
}
fn outside(p: vec3f) -> bool { return any(p < vec3f(-0.001)) || any(p > vec3f(1.001)); }
fn shift(ii: u32) -> vec3f { return vec3f(f32(ii / 9u), f32((ii / 3u) % 3u), f32(ii % 3u)) - 1.0; }
fn planePoint(a: f32, b: f32) -> vec3f {
  if (u.slice.x < 0.5) { return vec3f(u.slice.y, a, b); }
  if (u.slice.x < 1.5) { return vec3f(a, u.slice.y, b); }
  return vec3f(a, b, u.slice.y);
}
fn scalar(v: vec4f) -> f32 {
  let c = u32(u.field.y);
  if (c < 3u) { return v[c]; }
  return v.w;
}
fn color(v: f32) -> vec3f {
  let t = clamp(v / u.field.z, -1.0, 1.0);
  if (u.field.y > 2.5) { return mix(vec3f(0.94,0.97,0.93), vec3f(0.02,0.5,0.51), max(t,0.0)); }
  let end = select(vec3f(0.12,0.36,0.76), vec3f(0.83,0.20,0.10), t >= 0.0);
  return mix(vec3f(0.97,0.97,0.93), end, abs(t));
}
`;
const sampling = /* wgsl */ `
@group(0) @binding(1) var<storage, read> field: array<vec4f>;
fn texel(i: vec3u, n: u32) -> vec4f { let q = i % vec3u(n); return field[(q.x * n + q.y) * n + q.z]; }
fn sampleField(p: vec3f) -> vec4f {
  let n = u32(u.field.x);
  let q = fract(p + vec3f(2.0)) * f32(n);
  let b = vec3u(floor(q)); let f = fract(q);
  let c00 = mix(texel(b,n),texel(b+vec3u(1,0,0),n),f.x);
  let c10 = mix(texel(b+vec3u(0,1,0),n),texel(b+vec3u(1,1,0),n),f.x);
  let c01 = mix(texel(b+vec3u(0,0,1),n),texel(b+vec3u(1,0,1),n),f.x);
  let c11 = mix(texel(b+vec3u(0,1,1),n),texel(b+vec3u(1,1,1),n),f.x);
  return mix(mix(c00,c10,f.y),mix(c01,c11,f.y),f.z);
}
`;
const fieldShader =
  common +
  sampling +
  /* wgsl */ `
struct Out { @builtin(position) pos: vec4f, @location(0) ndc: vec2f }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> Out {
  let p = array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3))[vi];
  var o: Out; o.pos=vec4f(p,0,1); o.ndc=p; return o;
}
@fragment fn fs(o: Out) -> @location(0) vec4f {
  let near = u.inv * vec4f(o.ndc,-1,1); let far = u.inv * vec4f(o.ndc,1,1);
  let ro = near.xyz / near.w; let rd = normalize(far.xyz / far.w - ro);
  if (u.field.w < 1.5) {
    let axis = u32(u.slice.x);
    if (abs(rd[axis]) < 1e-6) { discard; }
    let t = (u.slice.y-ro[axis])/rd[axis]; let p=ro+rd*t;
    if (t < 0.0 || outside(p)) { discard; }
    return vec4f(color(scalar(sampleField(p))),u.slice.z);
  }
  let inv = 1.0 / select(vec3f(1e-7),rd,abs(rd)>vec3f(1e-7));
  let a = -ro*inv; let b = (vec3f(1)-ro)*inv;
  let lo = min(a,b); let hi = max(a,b);
  let t0=max(max(max(lo.x,lo.y),lo.z),0.0); let t1=min(min(hi.x,hi.y),hi.z);
  if (t0>=t1) { discard; }
  let ds=(t1-t0)/96.0; var acc=vec4f(0);
  for(var i=0u;i<96u;i++) {
    let v=scalar(sampleField(ro+rd*(t0+(f32(i)+0.5)*ds)));
    let strength=clamp(abs(v)/u.field.z,0.0,1.0);
    let alpha=1.0-exp(-smoothstep(0.06,1.0,strength)*u.slice.z*ds*9.0);
    acc=vec4f(acc.rgb+(1.0-acc.a)*alpha*color(v),acc.a+(1.0-acc.a)*alpha);
    if (acc.a>0.98) { break; }
  }
  // The render target uses ordinary alpha blending, so return unpremultiplied RGB.
  return vec4f(acc.rgb/max(acc.a,1e-6),acc.a);
}
`;
const meshShader =
  common +
  /* wgsl */ `
@group(0) @binding(1) var<storage,read> x: array<vec4f>;
@group(0) @binding(2) var<storage,read> tri: array<u32>;
struct Out { @builtin(position) pos: vec4f, @location(0) world: vec3f, @location(1) normal: vec3f }
@vertex fn vs(@builtin(vertex_index) vi:u32,@builtin(instance_index) ii:u32) -> Out {
  let t=vi/3u; let a=x[tri[3u*t]].xyz; let b=x[tri[3u*t+1u]].xyz; let c=x[tri[3u*t+2u]].xyz;
  let p=x[tri[vi]].xyz-floor((a+b+c)/3.0)+shift(ii);
  var o:Out; o.pos=project(p); o.world=p; o.normal=normalize(cross(b-a,c-a)); return o;
}
@fragment fn fs(o:Out) -> @location(0) vec4f {
  if(outside(o.world)) { discard; }
  let light=0.4+0.6*abs(dot(o.normal,normalize(vec3f(0.4,0.8,1))));
  return vec4f(mix(vec3f(0.12,0.3,0.35),vec3f(0.55,0.76,0.72),light),u.style.x);
}
`;
const edgeShader =
  common +
  /* wgsl */ `
@group(0) @binding(1) var<storage,read> x: array<vec4f>;
@group(0) @binding(2) var<storage,read> edges: array<u32>;
struct Out { @builtin(position) pos:vec4f,@location(0) world:vec3f }
@vertex fn vs(@builtin(vertex_index) vi:u32,@builtin(instance_index) ii:u32) -> Out {
  let a=x[edges[(vi/2u)*2u]].xyz; let b=x[edges[(vi/2u)*2u+1u]].xyz;
  let p=x[edges[vi]].xyz-floor((a+b)*0.5)+shift(ii);
  var o:Out; o.pos=project(p); o.world=p; return o;
}
@fragment fn fs(o:Out) -> @location(0) vec4f { if(outside(o.world)) { discard; } return vec4f(0.12,0.22,0.24,0.42); }
`;
// A square 17×17 lattice gives approximately twice the former 12×12 density.
const ARROW_GRID = 17;
const arrowShader =
  common +
  sampling +
  /* wgsl */ `
@vertex fn vs(@builtin(vertex_index) vi:u32,@builtin(instance_index) ii:u32) -> @builtin(position) vec4f {
  let p=planePoint((f32(ii/${ARROW_GRID}u)+0.5)/${ARROW_GRID}.0,(f32(ii%${ARROW_GRID}u)+0.5)/${ARROW_GRID}.0);
  let v=sampleField(p).xyz; let size=length(v); let dir=v/max(size,1e-8);
  let len=${0.068 * 12 / ARROW_GRID}*clamp(size/u.field.z,0.0,1.0);
  let side=normalize(cross(dir,select(vec3f(0,0,1),vec3f(0,1,0),abs(dir.z)>0.8))+vec3f(1e-8));
  let tip=p+dir*len*0.5; let tail=p-dir*len*0.5;
  var q=tail;
  if(vi==1u||vi==2u||vi==4u) { q=tip; }
  if(vi==3u) { q=tip-dir*len*0.3+side*len*0.2; }
  if(vi==5u) { q=tip-dir*len*0.3-side*len*0.2; }
  return project(q);
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(0.08,0.12,0.16,0.9); }
`;
const guideShader =
  common +
  /* wgsl */ `
@vertex fn vs(@location(0) p:vec3f,@location(1) c:vec3f) -> Guide {
  var o:Guide; o.pos=project(p); o.color=c; return o;
}
struct Guide { @builtin(position) pos:vec4f,@location(0) color:vec3f }
@fragment fn fs(o:Guide) -> @location(0) vec4f { return vec4f(o.color,0.85); }
`;

/** Three.js owns camera/picking; direct WebGPU consumes solver buffers without readback. */
export class Ib3DRenderer {
  readonly camera = new OrthographicCamera(-1, 1, 1, -1, 0.01, 20);
  readonly controls: OrbitControls;
  readonly raycaster = new Raycaster();
  #context: GPUCanvasContext;
  #uniform: GPUBuffer;
  #guides: GPUBuffer;
  #pipelines: Record<string, GPURenderPipeline>;
  #bindings = new Map<string, WeakMap<GPUBuffer, GPUBindGroup>>();
  #vp = new Matrix4();
  #inv = new Matrix4();
  constructor(
    readonly device: GPUDevice,
    readonly canvas: HTMLCanvasElement,
  ) {
    this.#context = canvas.getContext("webgpu")!;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.#context.configure({ device, format, alphaMode: "opaque" });
    this.#uniform = device.createBuffer({
      size: 192,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.#guides = device.createBuffer({
      size: 8192,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.camera.position.set(1.8, 1.3, 2.0);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0.5, 0.5, 0.5);
    this.controls.enablePan = false;
    // The page captures right-drag for physics; camera controls own left and middle.
    this.controls.mouseButtons.RIGHT = null;
    this.controls.mouseButtons.MIDDLE = MOUSE.DOLLY;
    this.controls.minZoom = 0.6;
    this.controls.maxZoom = 4;
    this.controls.update();
    this.controls.saveState();
    const pipeline = (
      code: string,
      topology: GPUPrimitiveTopology = "triangle-list",
      cullMode: GPUCullMode = "none",
      guides = false,
    ) => {
      const module = device.createShaderModule({ code });
      return device.createRenderPipeline({
        layout: "auto",
        vertex: {
          module,
          entryPoint: "vs",
          ...(guides
            ? {
                buffers: [
                  {
                    arrayStride: 24,
                    attributes: [
                      {
                        shaderLocation: 0,
                        offset: 0,
                        format: "float32x3" as GPUVertexFormat,
                      },
                      {
                        shaderLocation: 1,
                        offset: 12,
                        format: "float32x3" as GPUVertexFormat,
                      },
                    ],
                  },
                ],
              }
            : {}),
        },
        fragment: {
          module,
          entryPoint: "fs",
          targets: [
            {
              format,
              blend: {
                color: {
                  srcFactor: "src-alpha",
                  dstFactor: "one-minus-src-alpha",
                },
                alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
              },
            },
          ],
        },
        primitive: { topology, cullMode },
      });
    };
    this.#pipelines = {
      back: pipeline(meshShader, "triangle-list", "front"),
      front: pipeline(meshShader, "triangle-list", "back"),
      field: pipeline(fieldShader),
      edges: pipeline(edgeShader, "line-list"),
      arrows: pipeline(arrowShader, "line-list"),
      guides: pipeline(guideShader, "line-list", "none", true),
    };
  }
  ray(clientX: number, clientY: number) {
    const r = this.canvas.getBoundingClientRect();
    this.camera.updateMatrixWorld();
    this.raycaster.setFromCamera(
      new Vector2(
        (2 * (clientX - r.left)) / r.width - 1,
        1 - (2 * (clientY - r.top)) / r.height,
      ),
      this.camera,
    );
    return this.raycaster.ray;
  }
  planePoint(clientX: number, clientY: number, plane: Plane) {
    return this.ray(clientX, clientY).intersectPlane(plane, new Vector3());
  }
  render(
    encoder: GPUCommandEncoder,
    solver: WebGpuIb3DSolver,
    o: ViewOptions,
    cursor: Vector3 | null,
  ) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(this.canvas.clientWidth * dpr)),
      height = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    const aspect = width / height;
    const halfHeight = 0.82 / Math.min(aspect, 1);
    this.camera.left = -halfHeight * aspect;
    this.camera.right = halfHeight * aspect;
    this.camera.top = halfHeight;
    this.camera.bottom = -halfHeight;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.camera.updateMatrixWorld();
    this.#vp.multiplyMatrices(
      this.camera.projectionMatrix,
      this.camera.matrixWorldInverse,
    );
    this.#inv.copy(this.#vp).invert();
    const data = new Float32Array(48);
    data.set(this.#vp.elements);
    data.set(this.#inv.elements, 16);
    const scale = Number.isFinite(o.scale) ? Math.max(1e-5, o.scale) : 0.5;
    data.set(
      [
        solver.params.N,
        o.component,
        scale,
        o.view === "volume" ? 2 : 1,
        o.axis,
        o.depth,
        o.opacity,
        0,
        o.surface,
        0,
        0,
        0,
      ],
      32,
    );
    this.device.queue.writeBuffer(this.#uniform, 0, data);
    const lines: number[] = [];
    const line = (a: number[], b: number[], color = [0.55, 0.61, 0.6]) =>
      lines.push(...a, ...color, ...b, ...color);
    for (let axis = 0; axis < 3; axis++)
      for (let a = 0; a < 2; a++)
        for (let b = 0; b < 2; b++) {
          const p = [0, 0, 0];
          p[(axis + 1) % 3] = a;
          p[(axis + 2) % 3] = b;
          const q = [...p];
          q[axis] = 1;
          line(p, q);
        }
    const point = (a: number, b: number) =>
      o.axis === 0
        ? [o.depth, a, b]
        : o.axis === 1
          ? [a, o.depth, b]
          : [a, b, o.depth];
    if (o.view === "slice") {
      const corners = [point(0, 0), point(1, 0), point(1, 1), point(0, 1)];
      for (let i = 0; i < 4; i++)
        line(corners[i], corners[(i + 1) % 4], [0.75, 0.45, 0.14]);
    }
    // Small colored axes anchored at the domain origin.
    line([0, 0, 0], [0.18, 0, 0], [0.8, 0.22, 0.16]);
    line([0, 0, 0], [0, 0.18, 0], [0.15, 0.55, 0.27]);
    line([0, 0, 0], [0, 0, 0.18], [0.16, 0.35, 0.8]);
    if (cursor)
      for (let axis = 0; axis < 3; axis++) {
        const a = cursor.toArray(),
          b = cursor.toArray();
        a[axis] -= 0.025;
        b[axis] += 0.025;
        line(a, b, [0.82, 0.24, 0.08]);
      }
    this.device.queue.writeBuffer(this.#guides, 0, new Float32Array(lines));
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.#context.getCurrentTexture().createView(),
          clearValue: { r: 0.97, g: 0.97, b: 0.97, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    const draw = (
      key: string,
      buffers: GPUBuffer[],
      count: number,
      instances = 1,
    ) => {
      const pipeline = this.#pipelines[key];
      const cache =
        this.#bindings.get(key) ?? new WeakMap<GPUBuffer, GPUBindGroup>();
      const identity = buffers[0] ?? this.#uniform;
      let group = cache.get(identity);
      if (!group) {
        group = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [this.#uniform, ...buffers].map((buffer, binding) => ({
            binding,
            resource: { buffer },
          })),
        });
        cache.set(identity, group);
        this.#bindings.set(key, cache);
      }
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.draw(count, instances);
    };
    if (o.surface > 0)
      draw(
        "back",
        [solver.vertexBuffer, solver.triangleBuffer],
        solver.mesh.nt * 3,
        27,
      );
    if (o.view !== "off") draw("field", [solver.fieldBuffer], 3);
    if (o.surface > 0)
      draw(
        "front",
        [solver.vertexBuffer, solver.triangleBuffer],
        solver.mesh.nt * 3,
        27,
      );
    draw(
      "edges",
      [solver.vertexBuffer, solver.edgeBuffer],
      solver.mesh.ne * 2,
      27,
    );
    if (o.arrows && o.view === "slice")
      draw("arrows", [solver.fieldBuffer], 6, ARROW_GRID * ARROW_GRID);
    pass.setVertexBuffer(0, this.#guides);
    draw("guides", [], lines.length / 6);
    pass.end();
  }
  destroy() {
    this.controls.dispose();
    this.#bindings.clear();
    this.#uniform.destroy();
    this.#guides.destroy();
    this.#context.unconfigure();
  }
}
