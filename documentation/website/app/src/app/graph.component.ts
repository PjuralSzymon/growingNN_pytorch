import { Component } from '@angular/core';
import { CONTENT_PAGES, GRAPH_STATS } from './generated/content';

@Component({
  selector: 'app-graph',
  template: `
    <main class="graph-page">
      <header class="graph-header">
        <div><span>Obsidian vault</span><h1>Knowledge graph</h1>
          <p>Drag nodes to rearrange them. Scroll to zoom. Double-click a node to open its documentation page.</p>
        </div>
        <div class="graph-stats"><span><b>{{ stats.nodes }}</b> pages</span><span><b>{{ stats.edges }}</b> links</span></div>
      </header>
      <div class="graph-toolbar">
        <div class="graph-legend">
          @for (category of categories; track category; let index = $index) {
            <span><i [style.background]="colors[index % colors.length]"></i>{{ category }}</span>
          }
        </div>
        <span>Powered by PyVis</span>
      </div>
      <div class="graph-stage">
        <iframe class="knowledge-graph-frame" src="/assets/knowledge-graph.html" title="Interactive documentation knowledge graph"></iframe>
        <div class="graph-hint">Hover to inspect · double-click to open · drag to move</div>
      </div>
    </main>
  `,
})
export class GraphComponent {
  protected readonly stats = GRAPH_STATS;
  protected readonly categories = [
    ...new Set(CONTENT_PAGES.filter((page) => page.section === 'Documentation').map((page) => page.category)),
  ].sort();
  protected readonly colors = ['#7968ee', '#4f8cff', '#35b996', '#e99546', '#db6487', '#9b72cf', '#51a9ba', '#a4a942'];
}
