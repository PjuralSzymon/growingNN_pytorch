import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { LoginComponent } from './login.component';

describe('LoginComponent', () => {
  it('should render a password field and a log-in button', () => {
    // Arrange
    TestBed.configureTestingModule({
      imports: [LoginComponent],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideZonelessChangeDetection(),
      ],
    });
    const fixture = TestBed.createComponent(LoginComponent);

    // Act
    fixture.detectChanges();
    const root = fixture.nativeElement as HTMLElement;

    // Assert
    expect(root.querySelector('input.password')?.getAttribute('type')).toBe('password');
    expect(root.querySelector('button')?.textContent?.trim()).toBe('Log in');
  });
});
