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
    // The application shell should always show the project brand.

    // Arrange
    const fixture = TestBed.createComponent(App);
    
    // Act
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    await fixture.whenStable();

    // Assert
    expect(compiled.querySelector('.brand')?.textContent).toContain('GrowingNN');
  });

  it('should open search when Control K is pressed', () => {
    // The global shortcut should call the readonly search view query.

    // Arrange
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();

    // Act
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true }));
    fixture.detectChanges();

    // Assert
    expect((fixture.nativeElement as HTMLElement).querySelector('.search-dialog.open')).not.toBeNull();
  });
});
