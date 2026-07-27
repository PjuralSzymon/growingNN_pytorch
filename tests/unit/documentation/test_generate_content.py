"""Tests for the documentation content generator."""

from pathlib import Path

from documentation.website.scripts.generate_content import (
    clean_title,
    markdown_to_html,
    strip_frontmatter,
)


def test_strip_frontmatter_removes_complete_lf_block() -> None:
    """A complete LF frontmatter block should be removed."""

    # Arrange
    text = "---\ntitle: Example\n---\n# Heading\n"

    # Act
    result = strip_frontmatter(text)

    # Assert
    assert result == "# Heading\n"


def test_strip_frontmatter_removes_complete_crlf_block() -> None:
    """A complete CRLF frontmatter block should be removed."""

    # Arrange
    text = "---\r\ntitle: Example\r\n---\r\n# Heading\r\n"

    # Act
    result = strip_frontmatter(text)

    # Assert
    assert result == "# Heading\r\n"


def test_strip_frontmatter_preserves_incomplete_block() -> None:
    """Frontmatter without a closing delimiter should remain unchanged."""

    # Arrange
    text = "---\ntitle: Example\n# Heading\n"

    # Act
    result = strip_frontmatter(text)

    # Assert
    assert result == text


def test_clean_title_uses_first_level_one_heading() -> None:
    """The first level-one heading should override the source filename."""

    # Arrange
    path = Path("fallback-name.md")
    text = "## Section\n# Page title\n"

    # Act
    result = clean_title(path, text)

    # Assert
    assert result == "Page title"


def test_clean_title_uses_filename_without_level_one_heading() -> None:
    """The source filename should be used when no level-one heading exists."""

    # Arrange
    path = Path("fallback-name.md")

    # Act
    result = clean_title(path, "## Section\n")

    # Assert
    assert result == "fallback-name"


def test_markdown_to_html_renders_heading_metadata() -> None:
    """A Markdown heading should produce matching HTML and metadata."""

    # Arrange
    markdown = "## Growing `network`\n"

    # Act
    rendered, headings = markdown_to_html(markdown, [])

    # Assert
    assert rendered == '<h2 id="growing-network">Growing <code>network</code></h2>'
    assert headings == [{"level": 2, "title": "Growing network", "anchor": "growing-network"}]


def test_markdown_to_html_switches_list_type() -> None:
    """Changing list marker type should close the old list and open the new one."""

    # Arrange
    markdown = "- first\n1. second\n"

    # Act
    rendered, _ = markdown_to_html(markdown, [])

    # Assert
    assert rendered == "<ul>\n<li>first</li>\n</ul>\n<ol>\n<li>second</li>\n</ol>"


def test_markdown_to_html_closes_fenced_code_block() -> None:
    """A fenced block should render escaped source code."""

    # Arrange
    markdown = "```\na < b\n```\n"

    # Act
    rendered, _ = markdown_to_html(markdown, [])

    # Assert
    assert rendered == "<pre><code>a &lt; b</code></pre>"


def test_markdown_to_html_closes_unfinished_code_block() -> None:
    """An unfinished fenced block should still render its buffered code."""

    # Arrange
    markdown = "```\nvalue\n"

    # Act
    rendered, _ = markdown_to_html(markdown, [])

    # Assert
    assert rendered == "<pre><code>value</code></pre>"


def test_markdown_to_html_renders_callout() -> None:
    """An Obsidian callout should render as one callout element."""

    # Arrange
    markdown = "> [!WARNING] Check this\n> before training\n"

    # Act
    rendered, _ = markdown_to_html(markdown, [])

    # Assert
    assert rendered == '<aside class="callout"><span>Warning</span><p>Check this before training</p></aside>'


def test_markdown_to_html_replaces_missing_image() -> None:
    """An Obsidian image embed should become a visible image-reference note."""

    # Arrange
    markdown = "![[architecture.png]]\n"

    # Act
    rendered, _ = markdown_to_html(markdown, [])

    # Assert
    assert '<span>Note</span>' in rendered
    assert "<code>architecture.png</code>" in rendered
