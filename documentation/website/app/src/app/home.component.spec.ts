import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { HomeComponent } from './home.component';

describe('HomeComponent', () => {
  it('should embed the generated PyVis graph', () => {
    // Arrange
    TestBed.configureTestingModule({ imports: [HomeComponent], providers: [provideRouter([])] });
    const fixture = TestBed.createComponent(HomeComponent);

    // Act
    fixture.detectChanges();
    const frame = (fixture.nativeElement as HTMLElement).querySelector('.knowledge-graph-frame');

    // Assert
    expect(frame?.getAttribute('src')).toBe('/assets/knowledge-graph.html');
  });
});
