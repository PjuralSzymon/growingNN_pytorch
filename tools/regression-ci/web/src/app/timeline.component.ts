import { Component, inject, signal } from '@angular/core';
import { DecimalPipe, PercentPipe } from '@angular/common';
import { HttpClient } from '@angular/common/http';

export type RunRecord = {
  job: string;
  dataset: string;
  sha: string;
  pr: number | null;
  finished_at: string;
  ok: boolean;
  passed?: boolean;
  error?: string;
  comparison?: {
    mean_val_acc: number;
    mean_param_count: number;
    accuracy: string;
    params: string;
  };
};

@Component({
  selector: 'app-timeline',
  imports: [DecimalPipe, PercentPipe],
  template: `
    <main class="timeline">
      <h1>Regression CI</h1>
      <p>Runs against main, newest first.</p>
      @if (runs().length === 0) {
        <p class="empty">No runs yet.</p>
      } @else {
        <ol>
          @for (run of runs(); track run.finished_at + run.job + run.sha) {
            <li>
              <time>{{ run.finished_at }}</time>
              <span>PR {{ run.pr ?? '—' }}</span>
              <span>{{ run.dataset }}</span>
              @if (run.ok && run.comparison) {
                <span>{{ run.comparison.mean_val_acc | percent: '1.1-1' }} ({{ run.comparison.accuracy }})</span>
                <span>{{ run.comparison.mean_param_count | number: '1.0-0' }} params ({{ run.comparison.params }})</span>
              } @else {
                <span class="failed">{{ run.error || 'failed' }}</span>
              }
            </li>
          }
        </ol>
      }
    </main>
  `,
})
export class TimelineComponent {
  private readonly http = inject(HttpClient);
  protected readonly runs = signal<RunRecord[]>([]);

  constructor() {
    this.http.get<RunRecord[]>('/api/runs').subscribe((rows) => this.runs.set(rows));
  }
}
