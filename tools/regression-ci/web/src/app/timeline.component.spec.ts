import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { TimelineComponent } from './timeline.component';

describe('TimelineComponent', () => {
  it('should explain how to start a job when none exist', () => {
    // Arrange
    TestBed.configureTestingModule({
      imports: [TimelineComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideZonelessChangeDetection(),
      ],
    });
    const http = TestBed.inject(HttpTestingController);
    const fixture = TestBed.createComponent(TimelineComponent);

    // Act
    const jobs = http.expectOne('/api/jobs');
    jobs.flush([]);
    fixture.detectChanges();
    const root = fixture.nativeElement as HTMLElement;
    fixture.destroy();

    // Assert
    expect(root.textContent).toContain('No jobs yet');
    expect(root.textContent).toContain('trigger-job.bat');
  });
});
