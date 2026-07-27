import { Routes } from '@angular/router';
import { CONTENT_ROUTES } from './generated/content';
import { DocsHubComponent } from './docs-hub.component';
import { DocumentPageComponent } from './document-page.component';
import { GraphComponent } from './graph.component';
import { HomeComponent } from './home.component';

export const routes: Routes = [
  { path: '', component: HomeComponent, title: 'Dynamic neural networks · GrowingNN' },
  { path: 'docs', component: DocsHubComponent, title: 'Documentation · GrowingNN' },
  { path: 'graph', component: GraphComponent, title: 'Knowledge graph · GrowingNN' },
  ...CONTENT_ROUTES.map((path) => ({ path, component: DocumentPageComponent })),
  { path: '**', redirectTo: '' },
];
