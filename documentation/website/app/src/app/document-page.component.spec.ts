import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { DocumentPageComponent } from './document-page.component';

describe('DocumentPageComponent', () => {
  it('adds one copy button and copies the code text', async () => {
    // The rendered code block should remain idempotent and copy its text.

    // Arrange
    let copiedText = '';
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: {
        writeText: (value: string) => {
          copiedText = value;
          return Promise.resolve();
        },
      },
    });
    TestBed.configureTestingModule({
      imports: [DocumentPageComponent],
      providers: [provideRouter([])],
    });
    const fixture = TestBed.createComponent(DocumentPageComponent);
    fixture.detectChanges();
    const pre = document.createElement('pre');
    const code = document.createElement('code');
    code.textContent = 'print("GrowingNN")';
    pre.appendChild(code);
    fixture.nativeElement.appendChild(pre);

    // Act
    fixture.componentInstance.ngAfterViewChecked();
    fixture.componentInstance.ngAfterViewChecked();
    const button = pre.querySelector<HTMLButtonElement>('.copy-code');
    button?.click();
    await Promise.resolve();

    // Assert
    expect(pre.querySelectorAll('.copy-code')).toHaveLength(1);
    expect(copiedText).toBe('print("GrowingNN")');
    expect(button?.textContent).toBe('Copied');
  });
});
