import { Component, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  template: `
    <main class="page login">
      <h1>Regression CI</h1>
      <p>Enter the dashboard password.</p>
      <label>
        Password
        <input
          class="password"
          type="password"
          [value]="password()"
          (input)="password.set($any($event.target).value)"
          (keyup.enter)="submit()"
        />
      </label>
      <button type="button" (click)="submit()">Log in</button>
      @if (error()) {
        <p class="error">{{ error() }}</p>
      }
    </main>
  `,
})
export class LoginComponent {
  private readonly http = inject(HttpClient);
  private readonly router = inject(Router);

  protected readonly password = signal('');
  protected readonly error = signal('');

  protected submit(): void {
    this.error.set('');
    this.http.post('/api/login', { password: this.password() }).subscribe({
      next: () => void this.router.navigateByUrl('/'),
      error: () => this.error.set('Wrong password'),
    });
  }
}
