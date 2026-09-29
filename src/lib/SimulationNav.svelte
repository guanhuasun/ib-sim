<script lang="ts">
  import { base } from "$app/paths";

  let { current }: { current: "2d" | "3d" } = $props();
</script>

<nav aria-label="Simulation dimension">
  <a href={`${base}/`} aria-current={current === "2d" ? "page" : undefined}>
    2D Simulation
  </a>
  <a href={`${base}/3d`} aria-current={current === "3d" ? "page" : undefined}>
    3D Simulation
  </a>
</nav>

<style>
  nav {
    display: flex;
    width: fit-content;
    max-width: 100%;
    margin-top: 18px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--color-border);
    background: var(--color-surface);
    font: 600 14px var(--font-sans);
  }
  a {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 44px;
    padding: 0 22px;
    color: var(--color-text-meta);
    text-decoration: none;
  }
  a:hover {
    background: var(--color-bg);
    color: var(--color-link);
  }
  a[aria-current="page"] {
    background: var(--color-link);
    color: white;
  }
  a:focus-visible {
    outline: 2px solid var(--color-focus);
    outline-offset: 2px;
  }
  @media (max-width: 600px) {
    nav {
      width: 100%;
    }
    a {
      flex: 1;
      padding: 0 8px;
    }
  }
</style>
