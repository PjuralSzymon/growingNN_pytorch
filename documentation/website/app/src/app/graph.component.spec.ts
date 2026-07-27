import { TestBed } from '@angular/core/testing';
import { GraphComponent } from './graph.component';

describe('GraphComponent', () => {
  it('should display graph page and link counts', () => {
    // The graph header should report the generated node and edge totals.

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

  it('should display categories in locale-aware alphabetical order', () => {
    // The graph legend should have a stable and explicit category order.

    // Arrange
    TestBed.configureTestingModule({ imports: [GraphComponent] });
    const fixture = TestBed.createComponent(GraphComponent);

    // Act
    fixture.detectChanges();
    const categories = Array.from(
      (fixture.nativeElement as HTMLElement).querySelectorAll('.graph-legend span'),
      (element) => element.textContent?.trim() ?? '',
    );

    // Assert
    expect(categories).toEqual([...categories].sort((left, right) => left.localeCompare(right)));
  });
});
