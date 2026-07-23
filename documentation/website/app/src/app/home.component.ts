import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { CONTENT_PAGES, GRAPH_STATS } from './generated/content';

@Component({
  selector: 'app-home',
  imports: [RouterLink],
  template: `
    <main class="home">
      <section class="hero">
        <div class="hero-copy">
          <div class="eyebrow"><i></i> Dynamic neural architecture research</div>
          <h1>Neural networks that<br /><span>grow as they learn.</span></h1>
          <p>GrowingNN evolves a model during training. It uses SGD for weights and Monte Carlo Tree Search to choose safe changes to the network graph.</p>
          <div class="hero-actions">
            <a class="button primary" routerLink="/guides/algorithm-overview/">Explore the algorithm <span>→</span></a>
            <a class="button ghost" routerLink="/docs/">Read the docs</a>
          </div>
          <div class="hero-stats">
            <span><b>{{ docsCount }}</b> reference pages</span><span><b>FX</b> native graph edits</span><span><b>MCTS</b> architecture search</span>
          </div>
        </div>
        <div class="hero-visual">
          <div class="graph-stage hero-graph-stage">
            <iframe class="knowledge-graph-frame" src="/assets/knowledge-graph.html" title="Interactive documentation knowledge graph"></iframe>
            <a class="hero-graph-link" routerLink="/graph/">Knowledge graph · {{ graphStats.nodes }} pages ↗</a>
          </div>
        </div>
      </section>

      <section class="three-parts">
        <div class="section-heading"><div><span>01 — Start here</span><h2>Three ways to explore</h2></div><p>Move from the core idea to implementation details and measured results.</p></div>
        <div class="feature-grid">
          <a routerLink="/guides/algorithm-overview/" class="feature">
            <span class="feature-icon violet">✦</span><small>CONCEPT</small><h3>How GrowingNN works</h3>
            <p>Understand generations, model training, architecture simulation, and safe graph mutations.</p><b>Learn the algorithm →</b>
          </a>
          <a routerLink="/docs/" class="feature">
            <span class="feature-icon blue">▤</span><small>REFERENCE</small><h3>Technical documentation</h3>
            <p>Browse all {{ docsCount }} Obsidian pages by section. Wiki links remain a connected web of topics.</p><b>Open documentation →</b>
          </a>
          <a routerLink="/experiments/experiment-000-previous-numpy-work/" class="feature">
            <span class="feature-icon mint">⌁</span><small>RESEARCH LOG</small><h3>Experiments and results</h3>
            <p>Follow sequential reports with goals, setup, metrics, findings, and next steps.</p><b>View experiments →</b>
          </a>
        </div>
      </section>

      <section class="experiments-preview">
        <div class="section-heading"><div><span>02 — Latest research</span><h2>Experiment sequence</h2></div><a routerLink="/experiments/experiment-000-previous-numpy-work/">Start from experiment 00 →</a></div>
        <div class="experiment-list">
          @for (page of experiments; track page.slug; let index = $index) {
            <a class="experiment-card" [routerLink]="'/' + page.slug + '/'">
              <span>{{ sequence(index) }}</span><div><h3>{{ page.title }}</h3><p>{{ page.description }}</p></div><b>→</b>
            </a>
          }
        </div>
      </section>
      <footer><a class="brand" routerLink="/"><span class="brand-mark">G</span><span>Growing<span>NN</span></span></a><p>Dynamic neural architecture growth, documented one generation at a time.</p></footer>
    </main>
  `,
})
export class HomeComponent {
  protected readonly docsCount = CONTENT_PAGES.filter((page) => page.section === 'Documentation').length;
  protected readonly experiments = CONTENT_PAGES.filter((page) => page.section === 'Experiments');
  protected readonly graphStats = GRAPH_STATS;

  protected sequence(index: number): string {
    return String(index + 1).padStart(2, '0');
  }
}
