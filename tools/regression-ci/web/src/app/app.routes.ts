import { inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CanActivateFn, Router, Routes } from '@angular/router';
import { catchError, map, of } from 'rxjs';
import { LoginComponent } from './login.component';
import { TimelineComponent } from './timeline.component';

const authGuard: CanActivateFn = () => {
  const http = inject(HttpClient);
  const router = inject(Router);
  return http.get('/api/runs').pipe(
    map(() => true),
    catchError(() => {
      void router.navigateByUrl('/login');
      return of(false);
    }),
  );
};

export const routes: Routes = [
  { path: 'login', component: LoginComponent, title: 'Log in · Regression CI' },
  { path: '', component: TimelineComponent, title: 'Regression CI', canActivate: [authGuard] },
  { path: '**', redirectTo: '' },
];
