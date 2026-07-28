"""Tests for the documentation content generator."""

from generate_content import MarkdownRenderer, inline_markup


def test_inline_markup_renders_markdown_image() -> None:
    """A Markdown image should become a lazy-loaded HTML image."""

    # Arrange
    markdown = "![MNIST validation accuracy](/assets/experiments/mnist-val-acc.png)"

    # Act
    rendered = inline_markup(markdown, [])

    # Assert
    assert rendered == (
        '<img src="/assets/experiments/mnist-val-acc.png" '
        'alt="MNIST validation accuracy" loading="lazy">'
    )


def test_markdown_renderer_renders_table() -> None:
    """A Markdown table should become semantic responsive HTML."""

    # Arrange
    markdown = "| Scheduler | Accuracy |\n| --- | ---: |\n| Progress Check | `96.53%` |"

    # Act
    rendered, _ = MarkdownRenderer([]).render(markdown)

    # Assert
    assert rendered == "\n".join(
        [
            '<div class="table-wrap"><table><thead><tr>',
            "<th>Scheduler</th>",
            "<th>Accuracy</th>",
            "</tr></thead><tbody>",
            "<tr>",
            "<td>Progress Check</td>",
            "<td><code>96.53%</code></td>",
            "</tr>",
            "</tbody></table></div>",
        ]
    )


def test_markdown_renderer_renders_figure_caption() -> None:
    """A caption callout should become compact figure-caption HTML."""

    # Arrange
    markdown = "> [!CAPTION] Figure 1. Training accuracy by epoch."

    # Act
    rendered, _ = MarkdownRenderer([]).render(markdown)

    # Assert
    assert rendered == '<p class="figure-caption">Figure 1. Training accuracy by epoch.</p>'
