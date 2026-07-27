import { Component, HostListener, computed, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { CONTENT_PAGES } from './generated/content';

@Component({
  selector: 'app-search-dialog',
  imports: [RouterLink],
  template: `
    @if (visible()) {
      <div class="search-dialog open" role="dialog" aria-modal="true" aria-label="Search" (click)="backdrop($event)">
        <div class="search-box">
          <div class="search-input">
            <span>⌕</span>
            <input
              #input
              type="search"
              placeholder="Search GrowingNN documentation…"
              autocomplete="off"
              [value]="query()"
              (input)="query.set(input.value)"
            />
            <button type="button" (click)="close()">Esc</button>
          </div>
          <div class="search-results">
            @if (!query().trim()) {
              <p>Start typing to search every page.</p>
            } @else if (!results().length) {
              <p>No pages match this search.</p>
            } @else {
              @for (page of results(); track page.slug) {
                <a class="search-result" [routerLink]="'/' + page.slug + '/'" (click)="close()">
                  <small>{{ page.section }}</small><strong>{{ page.title }}</strong><span>{{ page.description }}</span>
                </a>
              }
            }
          </div>
        </div>
      </div>
    }
  `,
})
export class SearchDialogComponent {
  readonly visible = signal(false);
  readonly query = signal('');
  readonly results = computed(() => {
    const terms = this.query().trim().toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return [];
    return CONTENT_PAGES.filter((page) => {
      const value = `${page.title} ${page.section} ${page.description}`.toLowerCase();
      return terms.every((term) => value.includes(term));
    }).slice(0, 12);
  });

  @HostListener('document:keydown.escape')
  close(): void {
    this.visible.set(false);
    this.query.set('');
  }

  open(): void {
    this.visible.set(true);
    setTimeout(() => document.querySelector<HTMLInputElement>('.search-input input')?.focus());
  }

  backdrop(event: MouseEvent): void {
    if ((event.target as HTMLElement).classList.contains('search-dialog')) this.close();
  }
}
