import { Component, DestroyRef, ElementRef, inject, signal, viewChild } from '@angular/core';
import { DecimalPipe, PercentPipe } from '@angular/common';
import { HttpClient } from '@angular/common/http';

const LIVE = new Set(['queued', 'pulling', 'running']);
const STEPS = [
  { id: 'queued', label: 'Queued' },
  { id: 'pulling', label: 'Pulling' },
  { id: 'running', label: 'Running' },
  { id: 'done', label: 'Done' },
] as const;

export type JobResult = {
  job: string;
  dataset: string;
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

export type JobRecord = {
  id: string;
  sha: string;
  pr: number | null;
  state: 'queued' | 'pulling' | 'running' | 'done' | 'error';
  step: string;
  queued_at: string;
  updated_at: string;
  pulling_at?: string;
  running_at?: string;
  finished_at?: string;
  error?: string;
  passed?: boolean;
  results?: JobResult[];
  log_tail?: string;
};

function relativeTime(iso: string | undefined, now: number): string {
  if (!iso) {
    return '—';
  }
  const stamp = Date.parse(iso);
  if (Number.isNaN(stamp)) {
    return iso;
  }
  const seconds = Math.max(0, Math.round((now - stamp) / 1000));
  if (seconds < 10) {
    return 'just now';
  }
  if (seconds < 60) {
    return `${seconds}s ago`;
  }
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }
  const hours = Math.round(minutes / 60);
  if (hours < 48) {
    return `${hours}h ago`;
  }
  return iso.slice(0, 10);
}

function pill(job: JobRecord): { label: string; kind: string } {
  if (job.state === 'queued') {
    return { label: 'Queued', kind: 'wait' };
  }
  if (job.state === 'pulling') {
    return { label: 'Pulling', kind: 'run' };
  }
  if (job.state === 'running') {
    return { label: 'Running', kind: 'run' };
  }
  if (job.state === 'error') {
    return { label: 'Failed', kind: 'fail' };
  }
  if (job.passed === false) {
    return { label: 'Below baseline', kind: 'wait' };
  }
  return { label: 'Passed', kind: 'ok' };
}

function stepIndex(job: JobRecord): number {
  if (job.state === 'queued') {
    return 0;
  }
  if (job.state === 'pulling') {
    return 1;
  }
  if (job.state === 'running') {
    return 2;
  }
  if (job.state === 'done') {
    return 3;
  }
  if (job.running_at) {
    return 2;
  }
  if (job.pulling_at) {
    return 1;
  }
  return 0;
}

@Component({
  selector: 'app-timeline',
  imports: [DecimalPipe, PercentPipe],
  template: `
    <main class="page">
      <header class="hero">
        <div>
          <h1>Regression CI</h1>
          <p>Jobs on this worker. Live runs stay at the top.</p>
        </div>
        <div class="hero-status">
          @if (hasLive()) {
            <span class="live-dot" aria-hidden="true"></span>
            <span>Live</span>
          }
          <span class="updated">Updated {{ updatedLabel() }}</span>
        </div>
      </header>

      @if (jobs().length) {
        <div class="counts">
          <span>{{ liveCount() }} live</span>
          <span>{{ queuedCount() }} queued</span>
          <span>{{ failedCount() }} failed</span>
        </div>
      }

      @if (jobs().length === 0) {
        <section class="empty-card">
          <h2>No jobs yet</h2>
          <p>
            The timeline stays empty until you start one. Keep the worker up, then run
            <code>trigger-job.bat</code> with a commit SHA and PR number.
          </p>
        </section>
      } @else {
        @if (nowJobs().length) {
          <section>
            <h2 class="section-label">Now</h2>
            <ol class="jobs">
              @for (job of nowJobs(); track job.id) {
                <li class="job" [class.open]="expandedId() === job.id">
                  <div class="job-head" role="button" tabindex="0" (click)="toggle(job.id)" (keyup.enter)="toggle(job.id)">
                    <span class="pill" [attr.data-kind]="status(job).kind">{{ status(job).label }}</span>
                    <span class="pr">PR {{ job.pr ?? '—' }}</span>
                    <span class="sha">{{ shortSha(job.sha) }}</span>
                    <span class="when">{{ ago(job.updated_at) }}</span>
                    <span class="step-line">{{ job.step }}</span>
                    <ol class="steps" aria-hidden="true">
                      @for (item of steps; track item.id; let i = $index) {
                        <li
                          [class.done]="indexOf(job) > i"
                          [class.current]="indexOf(job) === i && job.state !== 'error'"
                          [class.fail]="job.state === 'error' && indexOf(job) === i"
                        >
                          {{ item.label }}
                        </li>
                      }
                    </ol>
                  </div>
                  @if (expandedId() === job.id) {
                    <div class="job-body">
                      @if (job.error) {
                        <p class="banner-fail">{{ job.error }}</p>
                      }
                      <dl class="meta">
                        <div><dt>Job</dt><dd>{{ job.id }}</dd></div>
                        <div><dt>SHA</dt><dd>{{ job.sha }}</dd></div>
                        <div><dt>PR</dt><dd>{{ job.pr ?? '—' }}</dd></div>
                        <div><dt>Queued</dt><dd>{{ job.queued_at }}</dd></div>
                      </dl>
                      @if (primary(job); as result) {
                        @if (result.comparison) {
                          <div class="metrics">
                            <div>
                              <span class="metric-value">{{ result.comparison.mean_val_acc | percent: '1.1-1' }}</span>
                              <span class="metric-label">val acc ({{ result.comparison.accuracy }})</span>
                            </div>
                            <div>
                              <span class="metric-value">{{ result.comparison.mean_param_count | number: '1.0-0' }}</span>
                              <span class="metric-label">params ({{ result.comparison.params }})</span>
                            </div>
                          </div>
                        }
                      }
                      <div class="log-wrap">
                        <div class="log-head">
                          <span>Log</span>
                          @if (isLive(job)) {
                            <span class="log-live">Live</span>
                          }
                          <button type="button" class="text-btn" (click)="follow.set(!follow()); $event.stopPropagation()">
                            {{ follow() ? 'Following' : 'Follow' }}
                          </button>
                        </div>
                        <pre
                          class="log"
                          #logEl
                          (scroll)="onLogScroll($event)"
                        >{{ logText() || 'Waiting for output…' }}</pre>
                      </div>
                    </div>
                  }
                </li>
              }
            </ol>
          </section>
        }

        @if (earlierJobs().length) {
          <section>
            <h2 class="section-label">Earlier</h2>
            <ol class="jobs">
              @for (job of earlierJobs(); track job.id) {
                <li class="job" [class.open]="expandedId() === job.id">
                  <div class="job-head" role="button" tabindex="0" (click)="toggle(job.id)" (keyup.enter)="toggle(job.id)">
                    <span class="pill" [attr.data-kind]="status(job).kind">{{ status(job).label }}</span>
                    <span class="pr">PR {{ job.pr ?? '—' }}</span>
                    <span class="sha">{{ shortSha(job.sha) }}</span>
                    @if (primary(job); as result) {
                      @if (result.comparison) {
                        <span>{{ result.comparison.mean_val_acc | percent: '1.1-1' }} ({{ result.comparison.accuracy }})</span>
                        <span>{{ result.comparison.mean_param_count | number: '1.0-0' }} params</span>
                      } @else if (result.error) {
                        <span class="failed">{{ result.error }}</span>
                      }
                    }
                    <span class="when">{{ ago(job.finished_at || job.updated_at) }}</span>
                    <ol class="steps" aria-hidden="true">
                      @for (item of steps; track item.id; let i = $index) {
                        <li
                          [class.done]="indexOf(job) > i"
                          [class.current]="indexOf(job) === i && job.state !== 'error'"
                          [class.fail]="job.state === 'error' && indexOf(job) === i"
                        >
                          {{ item.label }}
                        </li>
                      }
                    </ol>
                  </div>
                  @if (expandedId() === job.id) {
                    <div class="job-body">
                      @if (job.error) {
                        <p class="banner-fail">{{ job.error }}</p>
                      }
                      <dl class="meta">
                        <div><dt>Job</dt><dd>{{ job.id }}</dd></div>
                        <div><dt>SHA</dt><dd>{{ job.sha }}</dd></div>
                        <div><dt>PR</dt><dd>{{ job.pr ?? '—' }}</dd></div>
                        <div><dt>Finished</dt><dd>{{ job.finished_at || job.updated_at }}</dd></div>
                      </dl>
                      @if (primary(job); as result) {
                        @if (result.comparison) {
                          <div class="metrics">
                            <div>
                              <span class="metric-value">{{ result.comparison.mean_val_acc | percent: '1.1-1' }}</span>
                              <span class="metric-label">val acc ({{ result.comparison.accuracy }})</span>
                            </div>
                            <div>
                              <span class="metric-value">{{ result.comparison.mean_param_count | number: '1.0-0' }}</span>
                              <span class="metric-label">params ({{ result.comparison.params }})</span>
                            </div>
                          </div>
                        }
                      }
                      <div class="log-wrap">
                        <div class="log-head">
                          <span>Log</span>
                        </div>
                        <pre class="log">{{ logText() || 'No log captured.' }}</pre>
                      </div>
                    </div>
                  }
                </li>
              }
            </ol>
          </section>
        }
      }
    </main>
  `,
})
export class TimelineComponent {
  private readonly http = inject(HttpClient);
  private readonly destroy = inject(DestroyRef);
  private readonly logEl = viewChild<ElementRef<HTMLPreElement>>('logEl');
  private autoExpandDone = false;

  protected readonly steps = STEPS;
  protected readonly jobs = signal<JobRecord[]>([]);
  protected readonly expandedId = signal<string>('');
  protected readonly logTail = signal<string>('');
  protected readonly follow = signal(true);
  protected readonly nowMs = signal(Date.now());

  constructor() {
    this.refresh();
    const timer = setInterval(() => this.refresh(), 2000);
    this.destroy.onDestroy(() => clearInterval(timer));
  }

  protected nowJobs(): JobRecord[] {
    return this.jobs().filter((job) => LIVE.has(job.state));
  }

  protected earlierJobs(): JobRecord[] {
    return this.jobs().filter((job) => !LIVE.has(job.state));
  }

  protected hasLive(): boolean {
    return this.nowJobs().length > 0;
  }

  protected liveCount(): number {
    return this.jobs().filter((job) => job.state === 'pulling' || job.state === 'running').length;
  }

  protected queuedCount(): number {
    return this.jobs().filter((job) => job.state === 'queued').length;
  }

  protected failedCount(): number {
    return this.jobs().filter((job) => job.state === 'error').length;
  }

  protected updatedLabel(): string {
    return 'just now';
  }

  protected status(job: JobRecord): { label: string; kind: string } {
    return pill(job);
  }

  protected shortSha(sha: string): string {
    return sha.slice(0, 7);
  }

  protected ago(iso: string | undefined): string {
    return relativeTime(iso, this.nowMs());
  }

  protected indexOf(job: JobRecord): number {
    return stepIndex(job);
  }

  protected isLive(job: JobRecord): boolean {
    return LIVE.has(job.state);
  }

  protected primary(job: JobRecord): JobResult | undefined {
    const rows = job.results ?? [];
    return rows[rows.length - 1];
  }

  protected logText(): string {
    return this.logTail();
  }

  protected toggle(id: string): void {
    this.autoExpandDone = true;
    this.expandedId.set(this.expandedId() === id ? '' : id);
    this.follow.set(true);
    this.refreshLog();
  }

  protected onLogScroll(event: Event): void {
    const el = event.target as HTMLElement;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    this.follow.set(atBottom);
  }

  private refresh(): void {
    this.nowMs.set(Date.now());
    this.http.get<JobRecord[]>('/api/jobs').subscribe({
      next: (rows) => {
        this.jobs.set(rows);
        if (!this.autoExpandDone) {
          const live = rows.find((job) => LIVE.has(job.state));
          if (live) {
            this.expandedId.set(live.id);
            this.autoExpandDone = true;
          }
        }
        this.refreshLog();
      },
      error: () => undefined,
    });
  }

  private refreshLog(): void {
    const id = this.expandedId();
    if (!id) {
      this.logTail.set('');
      return;
    }
    this.http.get<JobRecord>(`/api/jobs/${id}`).subscribe({
      next: (row) => {
        this.logTail.set(row.log_tail ?? '');
        this.jobs.update((list) => list.map((job) => (job.id === row.id ? { ...job, ...row } : job)));
        this.scrollLog();
      },
      error: () => undefined,
    });
  }

  private scrollLog(): void {
    if (!this.follow()) {
      return;
    }
    queueMicrotask(() => {
      const el = this.logEl()?.nativeElement;
      if (el) {
        el.scrollTop = el.scrollHeight;
      }
    });
  }
}
