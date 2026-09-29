<script lang="ts">
  import SimulationNav from "./SimulationNav.svelte";

  let {
    dimension,
    description,
    grid,
    dt,
  }: {
    dimension: "2d" | "3d";
    description: string;
    grid: number;
    dt: number;
  } = $props();
</script>

<nav class="breadcrumb" aria-label="Site">
  <a href="https://guanhuasun.github.io/">← guanhuasun.github.io</a>
</nav>
<header>
  <h1>
    <em>Interactive</em>{" "}<a
      href="https://math.nyu.edu/~peskin/ib_lecture_notes/index.html"
      target="_blank"
      rel="noopener noreferrer"
      >{dimension.toUpperCase()} Immersed Boundary Method</a
    >
  </h1>
  <p class="lede">{description}</p>
  <p class="tech-line">
    WebGPU · {grid}{dimension === "2d" ? "²" : "³"} grid · FFT-based IMEX · Δt={dt.toFixed(
      4,
    )}
  </p>
  <SimulationNav current={dimension} />
</header>

<style>
  .breadcrumb {
    font: 12px var(--font-mono);
    margin-bottom: 20px;
  }
  .breadcrumb a {
    color: var(--color-text-meta);
  }
  header {
    border-bottom: 1px solid var(--color-border);
    padding-bottom: 20px;
    margin-bottom: 20px;
  }
  h1 {
    font: 600 22px/1.25 var(--font-serif);
    letter-spacing: -0.01em;
    margin: 0 0 8px;
  }
  h1 em {
    color: var(--color-link);
    font-weight: 700;
    margin-right: 0.12em;
  }
  h1 a {
    color: inherit;
  }
  h1 a:hover {
    color: var(--color-link);
  }
  .lede {
    font: 14px/1.55 var(--font-sans);
    color: var(--color-text-meta);
    margin: 0 0 6px;
  }
  .tech-line {
    font: 12px/1.6 var(--font-mono);
    color: var(--color-text-meta);
    margin: 0;
  }
</style>
