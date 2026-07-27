import { DOCUMENT } from '@angular/common';
import { TestBed } from '@angular/core/testing';
import { ThemeService } from './theme.service';

describe('ThemeService', () => {
  it('should apply the opposite theme when toggled', () => {
    // Arrange
    const service = TestBed.inject(ThemeService);
    const document = TestBed.inject(DOCUMENT);
    const initial = service.current();

    // Act
    service.toggle();

    // Assert
    expect(document.documentElement.dataset['theme']).toBe(initial === 'dark' ? 'light' : 'dark');
  });
});
