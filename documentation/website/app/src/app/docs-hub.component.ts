import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { CONTENT_PAGES } from './generated/content';

@Component({
  selector: 'app-docs-hub',
  imports: [RouterLink],
  template: `
    <main class="directory-page">
      <div class="directory-hero">
        <div><span>Technical reference</span><h1>Documentation,<br />organized by system.</h1>
          <p>Every page from the Obsidian vault is grouped by its main folder. Use the knowledge graph to explore links between topics.</p>
        </div>
        <a class="graph-promo" routerLink="/graph/"><span>Interactive view</span><strong>Open knowledge graph</strong><b>Explore {{ docs.length }} connected pages →</b></a>
      </div>
      <div class="directory-grid">
        @for (group of groups; track group.category) {
          <section class="doc-section-card">
            <div><small>{{ pageCount(group.pages.length) }} pages</small><h2>{{ group.category }}</h2></div>
            <div class="section-links">
              @for (page of group.pages; track page.slug) {
                <a [routerLink]="'/' + page.slug + '/'"><span>{{ page.title }}</span><b>→</b></a>
              }
            </div>
          </section>
        }
      </div>
    </main>
  `,
})
export class DocsHubComponent {
  protected readonly docs = CONTENT_PAGES.filter((page) => page.section === 'Documentation');
  protected readonly groups = this.groupPages();

  protected pageCount(count: number): string {
    return String(count).padStart(2, '0');
  }

  private groupPages(): { category: string; pages: typeof CONTENT_PAGES }[] {
    const groups = new Map<string, typeof CONTENT_PAGES>();
    this.docs.forEach((page) => groups.set(page.category, [...(groups.get(page.category) ?? []), page]));
    return [...groups.entries()].map(([category, pages]) => ({ category, pages }));
  }
}
