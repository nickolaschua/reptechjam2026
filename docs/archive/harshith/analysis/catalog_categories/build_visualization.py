#!/usr/bin/env python3
"""Build the inline, zoomable catalogue category hierarchy visualization."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = Path("/Users/gsharsh/.codex/visualizations/2026/08/28/01a048f9-0a8a-77c2-b4e4-f50bfb084822/catalog-category-hierarchy.html")


def main() -> None:
    tree = json.loads((HERE / "category_tree.json").read_text(encoding="utf-8"))
    data = json.dumps(tree, ensure_ascii=False, separators=(",", ":"))
    fragment = r'''<div id="catalog-category-hierarchy">
  <h2>Catalogue category hierarchy</h2>
  <div class="viz-row cch-toolbar">
    <button type="button" class="btn btn-ghost" id="cch-back" disabled aria-label="Go up one category"><i data-lucide="arrow-left" aria-hidden="true"></i> Up</button>
    <div id="cch-selection" class="card" aria-live="polite"></div>
  </div>
  <div id="cch-chart"></div>
  <div class="viz-row text-small text-muted cch-legend"><span><span class="cch-swatch cch-current"></span>Selected branch</span><span><span class="cch-swatch cch-child"></span>Descendants</span><span>Width = products in category path</span></div>
  <div class="sr-only" id="cch-description">A zoomable hierarchy of all catalogue category paths. Select a rectangle to focus on that branch and use Up to return.</div>
</div>
<style>
  #catalog-category-hierarchy { width: 100%; color: var(--foreground); }
  #catalog-category-hierarchy h2 { margin-bottom: 0.75rem; }
  #catalog-category-hierarchy .cch-toolbar { align-items: stretch; margin-bottom: 0.75rem; }
  #catalog-category-hierarchy #cch-selection { flex: 1 1 28rem; padding: 0.65rem 0.8rem; min-width: 0; }
  #catalog-category-hierarchy .cch-selection-path { overflow-wrap: anywhere; }
  #catalog-category-hierarchy .cch-selection-count { margin-left: 0.55rem; color: var(--muted-foreground); white-space: nowrap; }
  #catalog-category-hierarchy #cch-chart { width: 100%; min-height: 400px; }
  #catalog-category-hierarchy .cch-svg { display: block; width: 100%; }
  #catalog-category-hierarchy .cch-node { cursor: pointer; }
  #catalog-category-hierarchy .cch-node rect { fill: color-mix(in srgb, var(--viz-series-1) 16%, transparent); }
  #catalog-category-hierarchy .cch-node[data-depth="1"] rect { fill: color-mix(in srgb, var(--viz-series-1) 30%, transparent); }
  #catalog-category-hierarchy .cch-node[data-selected="true"] rect { fill: color-mix(in srgb, var(--viz-series-2) 40%, transparent); }
  #catalog-category-hierarchy .cch-node:hover rect { fill: color-mix(in srgb, var(--accent) 72%, transparent); }
  #catalog-category-hierarchy .cch-node text { fill: var(--foreground); font-size: 12px; pointer-events: none; }
  #catalog-category-hierarchy .cch-node .cch-count { fill: var(--muted-foreground); }
  #catalog-category-hierarchy .cch-column-label { fill: var(--muted-foreground); font-size: 11px; }
  #catalog-category-hierarchy .cch-divider { stroke: var(--background); stroke-width: 1px; pointer-events: none; }
  #catalog-category-hierarchy .cch-legend { display: flex; flex-wrap: wrap; gap: 0.45rem 1rem; margin-top: 0.55rem; }
  #catalog-category-hierarchy .cch-legend > span { display: inline-flex; align-items: center; gap: 0.35rem; }
  #catalog-category-hierarchy .cch-swatch { width: 0.75rem; height: 0.75rem; display: inline-block; }
  #catalog-category-hierarchy .cch-current { background: color-mix(in srgb, var(--viz-series-2) 40%, transparent); }
  #catalog-category-hierarchy .cch-child { background: color-mix(in srgb, var(--viz-series-1) 16%, transparent); }
  @media (max-width: 520px) {
    #catalog-category-hierarchy #cch-selection { flex-basis: 100%; }
    #catalog-category-hierarchy #cch-chart { min-height: 480px; }
    #catalog-category-hierarchy .cch-legend > span:last-child { flex: 1 0 100%; }
  }
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {
  const rootEl = document.getElementById('catalog-category-hierarchy');
  if (!rootEl || typeof d3 === 'undefined') return;
  const raw = __TREE_DATA__;
  const chartEl = rootEl.querySelector('#cch-chart');
  const selectionEl = rootEl.querySelector('#cch-selection');
  const backButton = rootEl.querySelector('#cch-back');
  const format = d3.format(',');
  const hierarchy = d3.hierarchy(raw).sum(d => d.direct_count || 0).sort((a, b) => b.value - a.value || d3.ascending(a.data.name, b.data.name));
  let focus = hierarchy;

  function pathOf(node) {
    return node.ancestors().reverse().slice(1).map(d => d.data.name);
  }

  function updateSelection() {
    const path = pathOf(focus);
    const label = path.length ? path.join(' › ') : 'All products';
    const percent = (focus.value / hierarchy.value * 100).toFixed(focus === hierarchy ? 0 : 1);
    selectionEl.innerHTML = `<span class="cch-selection-path">${escapeHtml(label)}</span><span class="cch-selection-count tabular-nums">${format(focus.value)} products · ${percent}%</span>`;
    backButton.disabled = !focus.parent;
  }

  function escapeHtml(value) {
    const temp = document.createElement('span');
    temp.textContent = value;
    return temp.innerHTML;
  }

  function draw() {
    updateSelection();
    chartEl.replaceChildren();
    const width = Math.max(320, Math.floor(chartEl.getBoundingClientRect().width));
    const narrow = width < 520;
    const height = narrow ? 480 : 410;
    const header = 24;
    const visibleDepth = Math.max(1, Math.min(5, hierarchy.height - focus.depth + 1));
    const descendants = focus.descendants().filter(d => d.depth <= focus.depth + visibleDepth - 1);
    const localRoot = d3.hierarchy(focus.data).sum(d => d.direct_count || 0).sort((a, b) => b.value - a.value || d3.ascending(a.data.name, b.data.name));
    d3.partition().size([height - header, visibleDepth])(localRoot);
    const columnWidth = width / visibleDepth;
    const svg = d3.select(chartEl).append('svg')
      .attr('class', 'cch-svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('height', height)
      .attr('role', 'tree')
      .attr('aria-labelledby', 'cch-description');

    svg.selectAll('.cch-column-label')
      .data(d3.range(visibleDepth))
      .join('text')
      .attr('class', 'cch-column-label')
      .attr('x', d => d * columnWidth + 5)
      .attr('y', 15)
      .text(d => d === 0 ? (focus === hierarchy ? 'Catalogue' : 'Selected') : `Level ${focus.depth + d}`);

    const nodes = svg.append('g').attr('transform', `translate(0,${header})`)
      .selectAll('g')
      .data(localRoot.descendants().filter(d => d.depth < visibleDepth && d.y1 > d.y0))
      .join('g')
      .attr('class', 'cch-node')
      .attr('data-depth', d => d.depth)
      .attr('data-selected', d => d.depth === 0 ? 'true' : 'false')
      .attr('role', 'treeitem')
      .attr('aria-label', d => `${d.data.name}: ${format(d.value)} products`)
      .attr('transform', d => `translate(${d.depth * columnWidth},${d.x0})`)
      .on('click', (_, d) => zoomToLocal(d))
      .on('keydown', (event, d) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); zoomToLocal(d); } });

    nodes.append('rect')
      .attr('width', Math.max(1, columnWidth - 1))
      .attr('height', d => Math.max(1, d.x1 - d.x0 - 1))
      .attr('data-tooltip', d => `${pathOf(resolveOriginal(d)).join(' › ') || 'All products'} · ${format(d.value)} products`);

    nodes.append('line').attr('class', 'cch-divider')
      .attr('x1', columnWidth - 1).attr('x2', columnWidth - 1)
      .attr('y1', 0).attr('y2', d => Math.max(1, d.x1 - d.x0));

    nodes.each(function(d) {
      const h = d.x1 - d.x0;
      const w = columnWidth - 10;
      if (h < 17 || w < 34) return;
      const group = d3.select(this);
      const name = d.data.name;
      const maxChars = Math.max(3, Math.floor(w / 7));
      const shown = name.length > maxChars ? name.slice(0, Math.max(1, maxChars - 1)) + '…' : name;
      group.append('text').attr('x', 5).attr('y', Math.min(14, h - 4)).text(shown);
      if (h >= 31) group.append('text').attr('class', 'cch-count').attr('x', 5).attr('y', 27).text(format(d.value));
    });
  }

  function resolveOriginal(localNode) {
    let original = focus;
    for (const part of localNode.ancestors().reverse().slice(1)) {
      original = original.children.find(child => child.data === part.data || child.data.name === part.data.name);
    }
    return original;
  }

  function zoomToLocal(localNode) {
    const original = resolveOriginal(localNode);
    if (!original || original === focus || !original.children) return;
    focus = original;
    draw();
  }

  backButton.addEventListener('click', () => {
    if (focus.parent) { focus = focus.parent; draw(); }
  });

  let resizeTimer;
  new ResizeObserver(() => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(draw, 80);
  }).observe(chartEl);
  draw();
})();
</script>
'''.replace("__TREE_DATA__", data)
    OUTPUT.write_text(fragment, encoding="utf-8")


if __name__ == "__main__":
    main()
