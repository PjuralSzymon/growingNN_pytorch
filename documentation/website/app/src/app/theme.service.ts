import { DOCUMENT, isPlatformBrowser } from '@angular/common';
import { Injectable, PLATFORM_ID, inject, signal } from '@angular/core';

export type Theme = 'light' | 'dark';

@Injectable({ providedIn: 'root' })
export class ThemeService {
  private readonly document = inject(DOCUMENT);
  private readonly browser = isPlatformBrowser(inject(PLATFORM_ID));
  readonly current = signal<Theme>('light');

  constructor() {
    if (!this.browser) return;
    const saved = localStorage.getItem('growingnn-theme') as Theme | null;
    const preferred = globalThis.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    this.apply(saved ?? preferred);
  }

  toggle(): void {
    this.apply(this.current() === 'dark' ? 'light' : 'dark');
  }

  private apply(theme: Theme): void {
    this.current.set(theme);
    this.document.documentElement.dataset['theme'] = theme;
    if (this.browser) localStorage.setItem('growingnn-theme', theme);
  }
}
