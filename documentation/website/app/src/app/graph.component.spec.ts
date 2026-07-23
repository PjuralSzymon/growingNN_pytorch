import { TestBed } from '@angular/core/testing';
import { GraphComponent } from './graph.component';

describe('GraphComponent', () => {
  it('should display graph page and link counts', () => {
    // Arrange
    TestBed.configureTestingModule({ imports: [GraphComponent] });
    const fixture = TestBed.createComponent(GraphComponent);

    // Act
    fixture.detectChanges();
    const stats = (fixture.nativeElement as HTMLElement).querySelector('.graph-stats')?.textContent;

    // Assert
    expect(stats).toContain('45 pages');
    expect(stats).toContain('57 links');
  });
});
