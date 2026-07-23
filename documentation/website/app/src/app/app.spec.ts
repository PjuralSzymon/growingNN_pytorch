import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { App } from './app';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [provideRouter([])],
    }).compileComponents();
  });

  it('should render the GrowingNN brand', async () => {
    // Arrange
    const fixture = TestBed.createComponent(App);
    
    // Act
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    await fixture.whenStable();

    // Assert
    expect(compiled.querySelector('.brand')?.textContent).toContain('GrowingNN');
  });
});
