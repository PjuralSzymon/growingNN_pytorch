import { isPlatformBrowser } from '@angular/common';
import {
  AfterViewChecked,
  Component,
  ElementRef,
  PLATFORM_ID,
  Renderer2,
  computed,
  inject,
  signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Meta, Title } from '@angular/platform-browser';
import { NavigationEnd, Router, RouterLink } from '@angular/router';
import { filter } from 'rxjs';
import { CONTENT_PAGES, ContentPage } from './generated/content';

@Component({
  selector: 'app-document-page',
  imports: [RouterLink],
  template: `
    @if (page(); as current) {
      <main class="doc-layout">
        <article class="doc-article" #article>
          <div class="breadcrumb"><a routerLink="/">GrowingNN</a><span>/</span><span>{{ current.section }}</span></div>
          <div class="article-content" [innerHTML]="current.html"></div>
          <nav class="page-nav">
            @if (previous(); as previousPage) {
              <a [routerLink]="'/' + previousPage.slug + '/'"><small>Previous</small><span>← {{ previousPage.title }}</span></a>
            } @else { <span></span> }
            @if (next(); as nextPage) {
              <a class="next" [routerLink]="'/' + nextPage.slug + '/'"><small>Next</small><span>{{ nextPage.title }} →</span></a>
            } @else { <span></span> }
          </nav>
        </article>
        <aside class="toc">
          <strong>On this page</strong>
          @for (heading of current.headings; track heading.anchor) {
            @if (heading.level === 2 || heading.level === 3) {
              <a [class]="'level-' + heading.level" [href]="'#' + heading.anchor">{{ heading.title }}</a>
            }
          } @empty {
            <span>No sections</span>
          }
          <a class="edit-link" [href]="editUrl(current)" target="_blank" rel="noreferrer">Edit this page ↗</a>
        </aside>
      </main>
    }
  `,
})
export class DocumentPageComponent implements AfterViewChecked {
  private readonly router = inject(Router);
  private readonly title = inject(Title);
  private readonly meta = inject(Meta);
  private readonly host: ElementRef<HTMLElement> = inject(ElementRef);
  private readonly renderer = inject(Renderer2);
  private readonly browser = isPlatformBrowser(inject(PLATFORM_ID));
  protected readonly page = signal<ContentPage | undefined>(undefined);
  protected readonly previous = computed(() => this.find(this.page()?.previousSlug));
  protected readonly next = computed(() => this.find(this.page()?.nextSlug));

  constructor() {
    this.selectPage(this.router.url);
    this.router.events
      .pipe(filter((event): event is NavigationEnd => event instanceof NavigationEnd), takeUntilDestroyed())
      .subscribe((event) => this.selectPage(event.urlAfterRedirects));
  }

  ngAfterViewChecked(): void {
    if (!this.browser) return;
    const blocks = this.host.nativeElement.querySelectorAll('pre:not([data-copy-ready])') as NodeListOf<HTMLPreElement>;
    blocks.forEach((block) => this.initializeCopyButton(block));
  }

  protected editUrl(page: ContentPage): string {
    return `https://github.com/PjuralSzymon/growingNN_pytorch-2/edit/main/${page.sourcePath}`;
  }

  private selectPage(url: string): void {
    const slug = url.split(/[?#]/)[0].replace(/^\/|\/$/g, '');
    const current = CONTENT_PAGES.find((page) => page.slug === slug);
    this.page.set(current);
    if (current) {
      this.title.setTitle(`${current.title} · GrowingNN`);
      this.meta.updateTag({ name: 'description', content: current.description });
    }
  }

  private find(slug: string | null | undefined): ContentPage | undefined {
    return slug ? CONTENT_PAGES.find((page) => page.slug === slug) : undefined;
  }

  private initializeCopyButton(block: HTMLPreElement): void {
    block.dataset['copyReady'] = 'true';
    const button = this.renderer.createElement('button') as HTMLButtonElement;
    button.className = 'copy-code';
    button.textContent = 'Copy';
    this.renderer.listen(button, 'click', () => this.copyCode(block, button));
    this.renderer.appendChild(block, button);
  }

  private copyCode(block: HTMLPreElement, button: HTMLButtonElement): void {
    const code = block.querySelector('code')?.textContent ?? '';
    void navigator.clipboard.writeText(code).then(() => this.showCopiedState(button));
  }

  private showCopiedState(button: HTMLButtonElement): void {
    button.textContent = 'Copied';
    window.setTimeout(() => {
      button.textContent = 'Copy';
    }, 1200);
  }
}
