"""Generate Angular content data and the PyVis graph from Markdown sources."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path


SITE = Path(__file__).parents[1]
REPO = SITE.parents[1]
VAULT = REPO / "documentation" / "obsydian" / "growingNN"
CONTENT = SITE / "content"
ANGULAR = SITE / "app"
GENERATED = ANGULAR / "src" / "app" / "generated"
PUBLIC = ANGULAR / "public"


@dataclass
class Page:
    title: str
    slug: str
    section: str
    source: Path
    text: str
    description: str = ""


def strip_frontmatter(text: str) -> str:
    """Remove a complete YAML frontmatter block from the start of text."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return text
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "".join(lines[index + 1 :])
    return text


def clean_title(path: Path, text: str) -> str:
    """Return the first level-one heading or the source filename."""
    for line in strip_frontmatter(text).splitlines():
        if line[:1] == "#" and line[1:2].isspace():
            return line[2:].strip()
    return path.stem


def description(text: str) -> str:
    """Create a short plain-text page summary."""
    plain = strip_frontmatter(text)
    plain = re.sub(r"!?\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", plain)
    plain = re.sub(r"[#>*_`~\[\]()-]", " ", plain)
    return re.sub(r"\s+", " ", plain).strip()[:180]


def slugify(value: str) -> str:
    """Convert a filesystem or heading value into a URL-safe slug."""
    value = value.replace("\\", "/").lower()
    value = re.sub(r"[^a-z0-9/]+", "-", value)
    return re.sub(r"-+", "-", value).strip("-/")


def load_pages() -> list[Page]:
    """Load all vault, guide, and experiment Markdown pages."""
    pages: list[Page] = []
    for path in sorted(VAULT.rglob("*.md")):
        relative = path.relative_to(VAULT)
        text = path.read_text(encoding="utf-8")
        pages.append(
            Page(
                clean_title(path, text),
                f"docs/{slugify(relative.with_suffix('').as_posix())}",
                "Documentation",
                path,
                text,
                description(text),
            )
        )
    for section_dir, section in (("guides", "Algorithm"), ("experiments", "Experiments")):
        for path in sorted((CONTENT / section_dir).glob("*.md")):
            text = path.read_text(encoding="utf-8")
            pages.append(
                Page(
                    clean_title(path, text),
                    f"{section_dir}/{slugify(path.stem)}",
                    section,
                    path,
                    text,
                    description(text),
                )
            )
    return pages


def resolve_wiki_target(target: str, pages: list[Page]) -> Page | None:
    """Resolve an Obsidian wiki link by path suffix or filename."""
    normalized = target.replace("\\", "/").removesuffix(".md").casefold()
    exact = [page for page in pages if page.source.stem.casefold() == Path(normalized).name]
    path_matches = [
        page
        for page in pages
        if slugify(page.source.with_suffix("").as_posix()).endswith(slugify(normalized))
    ]
    matches = path_matches or exact
    return matches[0] if matches else None


def inline_markup(value: str, pages: list[Page]) -> str:
    """Render inline Markdown and Obsidian wiki links."""
    code_values: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        code_values.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\x00CODE{len(code_values) - 1}\x00"

    value = re.sub(r"`([^`]+)`", stash_code, value)
    value = html.escape(value, quote=False)

    def wiki(match: re.Match[str]) -> str:
        raw, label = match.group(1), match.group(2) or Path(match.group(1)).name
        target = resolve_wiki_target(html.unescape(raw), pages)
        if target:
            return f'<a href="/{target.slug}/">{html.escape(html.unescape(label))}</a>'
        return (
            '<span class="broken-link" title="This Obsidian page was not found">'
            f"{html.escape(html.unescape(label))}</span>"
        )

    value = re.sub(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", wiki, value)
    value = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        lambda match: (
            f'<a href="{html.escape(match.group(2), quote=True)}" target="_blank" '
            f'rel="noreferrer">{match.group(1)}</a>'
        ),
        value,
    )
    value = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda match: f'<a href="{html.escape(match.group(2), quote=True)}">{match.group(1)}</a>',
        value,
    )
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", value)
    for index, code in enumerate(code_values):
        value = value.replace(f"\x00CODE{index}\x00", code)
    return value


class MarkdownRenderer:
    """Render the supported Markdown subset with explicit parser state."""

    def __init__(self, pages: list[Page]) -> None:
        self.pages = pages
        self.output: list[str] = []
        self.headings: list[dict[str, str | int]] = []
        self.paragraph: list[str] = []
        self.list_type: str | None = None
        self.in_code = False
        self.code_lines: list[str] = []
        self.callout: list[str] = []
        self.callout_kind = "NOTE"

    def render(self, markdown: str) -> tuple[str, list[dict[str, str | int]]]:
        """Render Markdown and return HTML with heading metadata."""
        markdown = re.sub(
            r"!\[\[([^\]|]+)(?:\|([^\]]+))?\]\]",
            self._missing_image,
            strip_frontmatter(markdown),
        )
        for raw_line in markdown.splitlines():
            self._render_line(raw_line)
        self._finish()
        return "\n".join(self.output), self.headings

    @staticmethod
    def _missing_image(match: re.Match[str]) -> str:
        """Replace a missing Obsidian image with a visible note."""
        return (
            "\n> [!NOTE] Image reference\n"
            f"> `{match.group(1)}` is referenced by the Obsidian page but is not stored in the repository.\n"
        )

    def _render_line(self, raw_line: str) -> None:
        """Route one source line to the matching block handler."""
        line = raw_line.rstrip()
        if line.startswith("```"):
            self._toggle_code_block()
            return
        if self.in_code:
            self.code_lines.append(line)
            return
        if not line.strip():
            self._handle_blank_line()
            return
        callout_match = re.match(r">\s*\[!(\w+)\]\s*(.*)", line)
        if callout_match:
            self._start_callout(callout_match)
            return
        if line.startswith(">") and self.callout:
            self.callout.append(line.removeprefix(">").strip())
            return
        heading = re.match(r"^(#{1,4})\s+(.+)", line)
        if heading:
            self._handle_heading(heading)
            return
        item = re.match(r"^\s*(?:[-+*]|\d+[.)])\s+(.+)", line)
        if item:
            self._handle_list_item(item, line)
            return
        self.paragraph.append(line.strip())

    def _toggle_code_block(self) -> None:
        """Open or close a fenced code block."""
        self._flush_paragraph()
        self._close_list()
        if self.in_code:
            self._append_code_block()
        self.in_code = not self.in_code

    def _handle_blank_line(self) -> None:
        """Close all blocks that end at a blank line."""
        self._flush_paragraph()
        self._close_list()
        self._flush_callout()

    def _start_callout(self, match: re.Match[str]) -> None:
        """Start an Obsidian callout block."""
        self._flush_paragraph()
        self._close_list()
        self.callout_kind = match.group(1)
        if match.group(2):
            self.callout.append(match.group(2))

    def _handle_heading(self, match: re.Match[str]) -> None:
        """Render a heading and record its navigation metadata."""
        self._flush_paragraph()
        self._close_list()
        level = len(match.group(1))
        title = re.sub(r"`([^`]+)`", r"\1", match.group(2))
        anchor = slugify(title)
        self.headings.append({"level": level, "title": title, "anchor": anchor})
        self.output.append(f'<h{level} id="{anchor}">{inline_markup(match.group(2), self.pages)}</h{level}>')

    def _handle_list_item(self, match: re.Match[str], line: str) -> None:
        """Render one ordered or unordered list item."""
        self._flush_paragraph()
        wanted = "ol" if re.match(r"^\s*\d+", line) else "ul"
        if self.list_type != wanted:
            self._close_list()
            self.output.append(f"<{wanted}>")
            self.list_type = wanted
        self.output.append(f"<li>{inline_markup(match.group(1), self.pages)}</li>")

    def _flush_paragraph(self) -> None:
        """Render and clear the current paragraph buffer."""
        if self.paragraph:
            self.output.append(f"<p>{inline_markup(' '.join(self.paragraph), self.pages)}</p>")
            self.paragraph.clear()

    def _close_list(self) -> None:
        """Close the active list."""
        if self.list_type:
            self.output.append(f"</{self.list_type}>")
            self.list_type = None

    def _flush_callout(self) -> None:
        """Render and clear the current callout buffer."""
        if self.callout:
            self.output.append(
                f'<aside class="callout"><span>{html.escape(self.callout_kind.title())}</span>'
                f"<p>{inline_markup(' '.join(self.callout), self.pages)}</p></aside>"
            )
            self.callout.clear()

    def _append_code_block(self) -> None:
        """Render and clear buffered code lines."""
        self.output.append(f"<pre><code>{html.escape(chr(10).join(self.code_lines))}</code></pre>")
        self.code_lines.clear()

    def _finish(self) -> None:
        """Flush all blocks left open at the end of input."""
        self._flush_paragraph()
        self._close_list()
        self._flush_callout()
        if self.in_code:
            self._append_code_block()


def markdown_to_html(markdown: str, pages: list[Page]) -> tuple[str, list[dict[str, str | int]]]:
    """Render supported Markdown blocks and collect heading metadata."""
    return MarkdownRenderer(pages).render(markdown)


def documentation_category(page: Page) -> str:
    """Return the first vault folder as a readable category."""
    if page.section != "Documentation":
        return page.section
    relative = page.source.relative_to(VAULT)
    if len(relative.parts) == 1:
        return "Overview"
    names = {"references": "References", "tests": "Validation reports", "Utils": "Utilities"}
    return names.get(relative.parts[0], relative.parts[0])


def graph_data(pages: list[Page]) -> dict[str, list[dict[str, str]]]:
    """Create graph nodes and unique edges from resolved wiki links."""
    docs = [page for page in pages if page.section == "Documentation"]
    nodes = [
        {
            "id": page.slug,
            "title": page.title,
            "url": f"/{page.slug}/",
            "category": documentation_category(page),
        }
        for page in docs
    ]
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for page in docs:
        for target_name in re.findall(r"!?\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", page.text):
            target = resolve_wiki_target(target_name, docs)
            if target and target.slug != page.slug:
                edge = tuple(sorted((page.slug, target.slug)))
                if edge not in seen:
                    seen.add(edge)
                    edges.append({"source": edge[0], "target": edge[1]})
    return {"nodes": nodes, "edges": edges}


def create_pyvis_graph(pages: list[Page], output: Path) -> None:
    """Generate the self-contained, theme-aware PyVis iframe."""
    from pyvis.network import Network

    data = graph_data(pages)
    degree = {node["id"]: 0 for node in data["nodes"]}
    for edge in data["edges"]:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1
    categories = sorted({node["category"] for node in data["nodes"]})
    colors = ("#9589f8", "#77a4ff", "#54d3b4", "#f2a65a", "#e77c9c", "#b58add", "#68c0d0", "#bdc25c")
    network = Network(
        height="100%",
        width="100%",
        bgcolor="transparent",
        font_color="#252637",
        directed=False,
        cdn_resources="in_line",
    )
    for node in data["nodes"]:
        network.add_node(
            node["id"],
            label=node["title"],
            title=f'{node["title"]}<br><small>{node["category"]}</small>',
            color=colors[categories.index(node["category"]) % len(colors)],
            size=11 + min(degree[node["id"]], 9),
            font={
                "size": 18,
                "face": "Manrope, Arial",
                "color": "#252637",
                "strokeWidth": 3,
                "strokeColor": "#f7f8fc",
            },
        )
    for edge in data["edges"]:
        network.add_edge(edge["source"], edge["target"], color={"color": "#45485b", "opacity": 0.75})
    network.set_options(
        """{
          "interaction": {"hover": true, "tooltipDelay": 100, "zoomView": true, "dragView": true},
          "nodes": {"borderWidth": 2, "borderWidthSelected": 4},
          "edges": {"width": 1.2, "smooth": {"enabled": true, "type": "dynamic", "roundness": 0.25}},
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -2600,
              "centralGravity": 0.14,
              "springLength": 150,
              "springConstant": 0.025,
              "damping": 0.34,
              "avoidOverlap": 0.4
            },
            "stabilization": {"enabled": true, "iterations": 700, "fit": true},
            "minVelocity": 0.15,
            "maxVelocity": 12
          }
        }"""
    )
    generated = network.generate_html(notebook=False)
    urls = json.dumps({node["id"]: node["url"] for node in data["nodes"]}, ensure_ascii=False)
    behavior = f"""<script>
      const pageUrls = {urls};
      function syncGraphTheme() {{
        const dark = window.parent.document.documentElement.dataset.theme === "dark";
        const color = dark ? "#f0f0f7" : "#252637";
        const strokeColor = dark ? "#11121a" : "#f7f8fc";
        nodes.update(nodes.get().map(function (node) {{
          return {{id: node.id, font: Object.assign({{}}, node.font, {{color, strokeColor}})}};
        }}));
      }}
      syncGraphTheme();
      new MutationObserver(syncGraphTheme).observe(
        window.parent.document.documentElement,
        {{attributes: true, attributeFilter: ["data-theme"]}}
      );
      network.once("stabilizationIterationsDone", function () {{
        network.setOptions({{physics: {{stabilization: false, maxVelocity: 4}}}});
        network.fit({{animation: {{duration: 500, easingFunction: "easeInOutQuad"}}}});
        window.setTimeout(function () {{
          network.moveTo({{
            scale: network.getScale() * 2.4,
            animation: {{duration: 700, easingFunction: "easeInOutQuad"}}
          }});
        }}, 550);
      }});
      network.on("doubleClick", function (params) {{
        if (params.nodes.length) window.parent.location.href = pageUrls[params.nodes[0]];
      }});
    </script>"""
    generated = generated.replace(
        "</head>",
        """<style>
          html, body, .card, .card-body, #mynetwork { width: 100% !important; height: 100% !important; margin: 0 !important; padding: 0 !important; overflow: hidden; }
          html, body, .card, .card-body, #mynetwork { border: 0 !important; background: transparent !important; }
          div.vis-tooltip { background: #f0f0f7; border: 0; border-radius: 7px; color: #171822; font: 13px Arial; padding: 8px 10px; }
        </style></head>""",
    ).replace("</body>", f"{behavior}</body>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated, encoding="utf-8")


def page_records(pages: list[Page]) -> list[dict[str, object]]:
    """Build serializable page records with navigation and rendered content."""
    records: list[dict[str, object]] = []
    for page in pages:
        rendered, headings = markdown_to_html(page.text, pages)
        if not re.search(r"<h1", rendered):
            rendered = f"<h1>{html.escape(page.title)}</h1>\n{rendered}"
        section_pages = [candidate for candidate in pages if candidate.section == page.section]
        index = section_pages.index(page)
        previous = section_pages[index - 1] if index else None
        following = section_pages[index + 1] if index + 1 < len(section_pages) else None
        records.append(
            {
                "title": page.title,
                "slug": page.slug,
                "url": f"/{page.slug}/",
                "section": page.section,
                "category": documentation_category(page),
                "description": page.description,
                "html": rendered,
                "headings": headings,
                "sourcePath": page.source.relative_to(REPO).as_posix(),
                "previousSlug": previous.slug if previous else None,
                "nextSlug": following.slug if following else None,
            }
        )
    return records


def write_angular_content(pages: list[Page], output: Path) -> None:
    """Write typed Angular content and route constants."""
    records = page_records(pages)
    graph = graph_data(pages)
    payload = json.dumps(records, ensure_ascii=False, indent=2)
    source = f"""// Generated by scripts/generate_content.py. Do not edit.
export interface HeadingRecord {{
  level: number;
  title: string;
  anchor: string;
}}

export interface ContentPage {{
  title: string;
  slug: string;
  url: string;
  section: string;
  category: string;
  description: string;
  html: string;
  headings: HeadingRecord[];
  sourcePath: string;
  previousSlug: string | null;
  nextSlug: string | null;
}}

export const CONTENT_PAGES: ContentPage[] = {payload};
export const CONTENT_ROUTES = CONTENT_PAGES.map((page) => page.slug);
export const GRAPH_STATS = {{ nodes: {len(graph["nodes"])}, edges: {len(graph["edges"])} }};
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")


def generate() -> None:
    """Generate every build-time artifact consumed by Angular."""
    pages = load_pages()
    GENERATED.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    write_angular_content(pages, GENERATED / "content.ts")
    create_pyvis_graph(pages, PUBLIC / "assets" / "knowledge-graph.html")
    (PUBLIC / ".htaccess").write_text(
        "DirectoryIndex index.html\n"
        "<IfModule mod_rewrite.c>\n"
        "RewriteEngine On\n"
        "RewriteCond %{REQUEST_FILENAME} !-f\n"
        "RewriteCond %{REQUEST_FILENAME} !-d\n"
        "RewriteRule ^ index.html [L]\n"
        "</IfModule>\n"
        "<IfModule mod_deflate.c>\n"
        "AddOutputFilterByType DEFLATE text/html text/css application/javascript application/json\n"
        "</IfModule>\n",
        encoding="utf-8",
    )
    print(f"Generated Angular content for {len(pages)} pages")


if __name__ == "__main__":
    generate()
