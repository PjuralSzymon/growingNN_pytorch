import { Component, HostListener, ViewChild, computed, signal } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { CONTENT_PAGES } from './generated/content';
import { SearchDialogComponent } from './search-dialog.component';
import { ThemeService } from './theme.service';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive, SearchDialogComponent],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  @ViewChild(SearchDialogComponent) private searchDialog?: SearchDialogComponent;

  protected readonly sidebarOpen = signal(false);
  protected readonly collapsed = signal<Record<string, boolean>>({});
  protected readonly algorithmPages = CONTENT_PAGES.filter((page) => page.section === 'Algorithm');
  protected readonly experimentPages = CONTENT_PAGES.filter((page) => page.section === 'Experiments');
  protected readonly documentationGroups = computed(() => {
    const groups = new Map<string, typeof CONTENT_PAGES>();
    CONTENT_PAGES.filter((page) => page.section === 'Documentation').forEach((page) => {
      groups.set(page.category, [...(groups.get(page.category) ?? []), page]);
    });
    return [...groups.entries()].map(([category, pages]) => ({ category, pages }));
  });

  constructor(protected readonly theme: ThemeService) {}

  @HostListener('document:keydown', ['$event'])
  protected handleShortcut(event: KeyboardEvent): void {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      this.searchDialog?.open();
    }
  }

  protected toggleGroup(group: string): void {
    this.collapsed.update((state) => ({ ...state, [group]: !state[group] }));
  }

  protected closeSidebar(): void {
    this.sidebarOpen.set(false);
  }
}
