"""Build the GrowingNN documentation as a dependency-free static website."""

from __future__ import annotations

import html
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


SITE = Path(__file__).parent
REPO = SITE.parents[1]
VAULT = REPO / "documentation" / "obsydian" / "growingNN"
CONTENT = SITE / "content"
DIST = SITE / "dist"
ASSETS = SITE / "assets"


@dataclass
class Page:
    title: str
    slug: str
    section: str
    source: Path
    text: str
    description: str = ""


def clean_title(path: Path, text: str) -> str:
    frontmatter_free = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", text, flags=re.S)
    heading = re.search(r"^#\s+(.+)$", frontmatter_free, re.M)
    return heading.group(1).strip() if heading else path.stem


def description(text: str) -> str:
    plain = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", text, flags=re.S)
    plain = re.sub(r"!?\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", plain)
    plain = re.sub(r"[#>*_`~\[\]()-]", " ", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain[:180]


def slugify(value: str) -> str:
    value = value.replace("\\", "/").lower()
    value = re.sub(r"[^a-z0-9/]+", "-", value)
    return re.sub(r"-+", "-", value).strip("-/")


def load_pages() -> list[Page]:
    pages: list[Page] = []
    for path in sorted(VAULT.rglob("*.md")):
        relative = path.relative_to(VAULT)
        text = path.read_text(encoding="utf-8")
        pages.append(
            Page(
                title=clean_title(path, text),
                slug=f"docs/{slugify(relative.with_suffix('').as_posix())}",
                section="Documentation",
                source=path,
                text=text,
                description=description(text),
            )
        )

    for section_dir, section in (("guides", "Algorithm"), ("experiments", "Experiments")):
        for path in sorted((CONTENT / section_dir).glob("*.md")):
            text = path.read_text(encoding="utf-8")
            pages.append(
                Page(
                    title=clean_title(path, text),
                    slug=f"{section_dir}/{slugify(path.stem)}",
                    section=section,
                    source=path,
                    text=text,
                    description=description(text),
                )
            )
    return pages


def resolve_wiki_target(target: str, pages: list[Page]) -> Page | None:
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
        return f'<span class="broken-link" title="This Obsidian page was not found">{html.escape(html.unescape(label))}</span>'

    value = re.sub(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", wiki, value)
    value = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        lambda match: f'<a href="{html.escape(match.group(2), quote=True)}" target="_blank" rel="noreferrer">{match.group(1)}</a>',
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


def markdown_to_html(markdown: str, pages: list[Page]) -> tuple[str, list[tuple[int, str, str]]]:
    markdown = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", markdown, flags=re.S)
    markdown = re.sub(
        r"!\[\[([^\]|]+)(?:\|([^\]]+))?\]\]",
        lambda match: (
            f"\n> [!NOTE] Image reference\n"
            f"> `{match.group(1)}` is referenced by the Obsidian page but is not stored in the repository.\n"
        ),
        markdown,
    )
    output: list[str] = []
    headings: list[tuple[int, str, str]] = []
    paragraph: list[str] = []
    list_type: str | None = None
    in_code = False
    code_lines: list[str] = []
    callout: list[str] = []
    callout_kind = "NOTE"

    def flush_paragraph() -> None:
        if paragraph:
            output.append(f"<p>{inline_markup(' '.join(paragraph), pages)}</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            output.append(f"</{list_type}>")
            list_type = None

    def flush_callout() -> None:
        nonlocal callout
        if callout:
            output.append(
                f'<aside class="callout"><span>{html.escape(callout_kind.title())}</span>'
                f"<p>{inline_markup(' '.join(callout), pages)}</p></aside>"
            )
            callout = []

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            flush_paragraph()
            close_list()
            if in_code:
                output.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines = []
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            flush_paragraph()
            close_list()
            flush_callout()
            continue
        callout_match = re.match(r">\s*\[!(\w+)\]\s*(.*)", line)
        if callout_match:
            flush_paragraph()
            close_list()
            callout_kind = callout_match.group(1)
            if callout_match.group(2):
                callout.append(callout_match.group(2))
            continue
        if line.startswith(">") and callout:
            callout.append(line.removeprefix(">").strip())
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)", line)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            title = re.sub(r"`([^`]+)`", r"\1", heading.group(2))
            anchor = slugify(title)
            headings.append((level, title, anchor))
            output.append(f'<h{level} id="{anchor}">{inline_markup(heading.group(2), pages)}</h{level}>')
            continue
        item = re.match(r"^\s*(?:[-+*]|\d+[.)])\s+(.+)", line)
        if item:
            flush_paragraph()
            ordered = bool(re.match(r"^\s*\d+", line))
            wanted = "ol" if ordered else "ul"
            if list_type != wanted:
                close_list()
                output.append(f"<{wanted}>")
                list_type = wanted
            output.append(f"<li>{inline_markup(item.group(1), pages)}</li>")
            continue
        paragraph.append(line.strip())

    flush_paragraph()
    close_list()
    flush_callout()
    if in_code:
        output.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
    return "\n".join(output), headings


def icon(name: str) -> str:
    icons = {
        "book": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V4H6.5A2.5 2.5 0 0 0 4 6.5v13Z"/><path d="M8 7h8M8 11h6"/>',
        "flask": '<path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.7 3h10.6a2 2 0 0 0 1.7-3l-5-9V3"/><path d="M8 15h8"/>',
        "spark": '<path d="m12 3-1.6 4.4L6 9l4.4 1.6L12 15l1.6-4.4L18 9l-4.4-1.6L12 3Z"/><path d="m5 15-.8 2.2L2 18l2.2.8L5 21l.8-2.2L8 18l-2.2-.8L5 15Z"/>',
    }
    return f'<svg viewBox="0 0 24 24" aria-hidden="true">{icons[name]}</svg>'


def documentation_category(page: Page) -> str:
    """Return the first Obsidian folder as a readable documentation category."""
    if page.section != "Documentation":
        return page.section
    relative = page.source.relative_to(VAULT)
    if len(relative.parts) == 1:
        return "Overview"
    names = {
        "references": "References",
        "tests": "Validation reports",
        "Utils": "Utilities",
    }
    return names.get(relative.parts[0], relative.parts[0])


def sidebar(pages: list[Page], active: str = "") -> str:
    groups: list[str] = []
    for section in ("Algorithm", "Experiments"):
        section_pages = [page for page in pages if page.section == section]
        links = "".join(
            f'<a class="{"active" if page.slug == active else ""}" href="/{page.slug}/">{html.escape(page.title)}</a>'
            for page in section_pages
        )
        groups.append(
            f'<div class="nav-group"><button type="button" aria-expanded="true">'
            f'<span>{section}</span><span class="chevron">⌄</span></button><div>{links}</div></div>'
        )
    docs = [page for page in pages if page.section == "Documentation"]
    categories: list[str] = []
    for category in dict.fromkeys(documentation_category(page) for page in docs):
        links = "".join(
            f'<a class="{"active" if page.slug == active else ""}" href="/{page.slug}/">{html.escape(page.title)}</a>'
            for page in docs
            if documentation_category(page) == category
        )
        categories.append(f'<div class="nav-subgroup"><span>{html.escape(category)}</span>{links}</div>')
    docs_group = (
        '<div class="nav-group"><button type="button" aria-expanded="true"><span>Documentation</span>'
        '<span class="chevron">⌄</span></button><div>'
        '<a class="nav-special" href="/docs/">All sections</a>'
        '<a class="nav-special" href="/graph/">Knowledge graph</a>'
        f'{"".join(categories)}</div></div>'
    )
    return groups[0] + docs_group + groups[1]


def shell(title: str, body: str, pages: list[Page], active: str = "", description_text: str = "") -> str:
    meta = html.escape(description_text or "GrowingNN algorithm documentation", quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{meta}">
  <title>{html.escape(title)} · GrowingNN</title>
  <link rel="stylesheet" href="/assets/site.css">
  <script defer src="/assets/site.js"></script>
</head>
<body>
  <header class="topbar">
    <a class="brand" href="/" aria-label="GrowingNN home"><span class="brand-mark">G</span><span>Growing<span>NN</span></span></a>
    <nav class="topnav" aria-label="Primary navigation">
      <a href="/guides/algorithm-overview/">Algorithm</a>
      <a href="/docs/">Docs</a>
      <a href="/graph/">Graph</a>
      <a href="/experiments/experiment-001-baseline/">Experiments</a>
    </nav>
    <div class="top-actions">
      <button class="search-trigger" type="button" aria-label="Search documentation"><span>⌕</span><span>Search</span><kbd>Ctrl K</kbd></button>
      <button class="theme-toggle" type="button" aria-label="Toggle color theme">◐</button>
      <button class="menu-toggle" type="button" aria-label="Open navigation">☰</button>
    </div>
  </header>
  <div class="page-frame">
    <aside class="sidebar"><div class="sidebar-scroll">{sidebar(pages, active)}</div></aside>
    {body}
  </div>
  <div class="search-dialog" role="dialog" aria-modal="true" aria-label="Search">
    <div class="search-box">
      <div class="search-input"><span>⌕</span><input type="search" placeholder="Search GrowingNN documentation…" autocomplete="off"><button type="button">Esc</button></div>
      <div class="search-results"><p>Start typing to search every page.</p></div>
    </div>
  </div>
</body>
</html>"""


def homepage(pages: list[Page]) -> str:
    docs_count = len([page for page in pages if page.section == "Documentation"])
    experiment_pages = [page for page in pages if page.section == "Experiments"]
    cards = "".join(
        f'<a class="experiment-card" href="/{page.slug}/"><span>{index:02d}</span><div><h3>{html.escape(page.title)}</h3>'
        f'<p>{html.escape(page.description)}</p></div><b>→</b></a>'
        for index, page in enumerate(experiment_pages, 1)
    )
    graph = graph_data(pages)
    graph_preview = f"""
      <div class="graph-stage hero-graph-stage">
        <iframe class="knowledge-graph-frame" src="/assets/knowledge-graph.html" title="Interactive documentation knowledge graph"></iframe>
        <a class="hero-graph-link" href="/graph/">Knowledge graph · {len(graph["nodes"])} pages ↗</a>
      </div>"""
    body = f"""
    <main class="home">
      <section class="hero">
        <div class="hero-copy"><div class="eyebrow"><i></i> Dynamic neural architecture research</div>
          <h1>Neural networks that<br><span>grow as they learn.</span></h1>
          <p>GrowingNN evolves a model during training. It uses SGD for weights and Monte Carlo Tree Search to choose safe changes to the network graph.</p>
          <div class="hero-actions"><a class="button primary" href="/guides/algorithm-overview/">Explore the algorithm <span>→</span></a><a class="button ghost" href="/docs/">Read the docs</a></div>
          <div class="hero-stats"><span><b>{docs_count}</b> reference pages</span><span><b>FX</b> native graph edits</span><span><b>MCTS</b> architecture search</span></div>
        </div>
        <div class="hero-visual">{graph_preview}</div>
      </section>
      <section class="three-parts">
        <div class="section-heading"><div><span>01 — Start here</span><h2>Three ways to explore</h2></div><p>Move from the core idea to implementation details and measured results.</p></div>
        <div class="feature-grid">
          <a href="/guides/algorithm-overview/" class="feature"><span class="feature-icon violet">{icon("spark")}</span><small>CONCEPT</small><h3>How GrowingNN works</h3><p>Understand generations, model training, architecture simulation, and safe graph mutations.</p><b>Learn the algorithm →</b></a>
          <a href="/docs/" class="feature"><span class="feature-icon blue">{icon("book")}</span><small>REFERENCE</small><h3>Technical documentation</h3><p>Browse all {docs_count} Obsidian pages by section. Wiki links are preserved as a connected web of topics.</p><b>Open documentation →</b></a>
          <a href="/experiments/experiment-001-baseline/" class="feature"><span class="feature-icon mint">{icon("flask")}</span><small>RESEARCH LOG</small><h3>Experiments and results</h3><p>Follow sequential experiment reports with goals, setup, metrics, findings, and next steps.</p><b>View experiments →</b></a>
        </div>
      </section>
      <section class="experiments-preview"><div class="section-heading"><div><span>02 — Latest research</span><h2>Experiment sequence</h2></div><a href="/experiments/experiment-001-baseline/">Start from experiment 01 →</a></div><div class="experiment-list">{cards}</div></section>
      <footer><a class="brand" href="/"><span class="brand-mark">G</span><span>Growing<span>NN</span></span></a><p>Dynamic neural architecture growth, documented one generation at a time.</p></footer>
    </main>"""
    return shell("Dynamic neural networks", body, pages)


def graph_data(pages: list[Page]) -> dict[str, list[dict[str, str]]]:
    """Create graph nodes and resolved edges from Obsidian wiki links."""
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
    """Generate a self-contained PyVis graph for embedding in website pages."""
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
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -2600,
              "centralGravity": 0.14,
              "springLength": 150,
              "springConstant": 0.025,
              "damping": 0.34,
              "avoidOverlap": 0.4
            },
            "stabilization": {"enabled": true, "iterations": 700, "updateInterval": 40, "fit": true},
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
            scale: network.getScale() * 1.75,
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
    output.write_text(generated, encoding="utf-8")


def documentation_home(pages: list[Page]) -> str:
    """Render the categorized entry page for all Obsidian documentation."""
    docs = [page for page in pages if page.section == "Documentation"]
    cards: list[str] = []
    for category in dict.fromkeys(documentation_category(page) for page in docs):
        category_pages = [page for page in docs if documentation_category(page) == category]
        links = "".join(
            f'<a href="/{page.slug}/"><span>{html.escape(page.title)}</span><b>→</b></a>'
            for page in category_pages
        )
        cards.append(
            f'<section class="doc-section-card"><div><small>{len(category_pages):02d} pages</small>'
            f'<h2>{html.escape(category)}</h2></div><div class="section-links">{links}</div></section>'
        )
    body = f"""<main class="directory-page">
      <div class="directory-hero"><div><span>Technical reference</span><h1>Documentation,<br>organized by system.</h1>
      <p>Every page from the Obsidian vault is grouped by its main folder. Use the knowledge graph to explore links between topics.</p></div>
      <a class="graph-promo" href="/graph/"><span>Interactive view</span><strong>Open knowledge graph</strong><b>Explore {len(docs)} connected pages →</b></a></div>
      <div class="directory-grid">{"".join(cards)}</div>
    </main>"""
    return shell("Documentation", body, pages, active="docs")


def graph_page(pages: list[Page]) -> str:
    """Render the interactive Obsidian knowledge graph page."""
    data = graph_data(pages)
    categories = sorted({node["category"] for node in data["nodes"]})
    colors = ("#7968ee", "#4f8cff", "#35b996", "#e99546", "#db6487", "#9b72cf", "#51a9ba", "#a4a942")
    legend = "".join(
        f'<span><i style="background:{colors[index % len(colors)]}"></i>{html.escape(category)}</span>'
        for index, category in enumerate(categories)
    )
    body = f"""<main class="graph-page">
      <header class="graph-header"><div><span>Obsidian vault</span><h1>Knowledge graph</h1>
      <p>Drag nodes to rearrange them. Scroll to zoom. Select a node to open its documentation page.</p></div>
      <div class="graph-stats"><span><b>{len(data["nodes"])}</b> pages</span><span><b>{len(data["edges"])}</b> links</span></div></header>
      <div class="graph-toolbar"><div class="graph-legend">{legend}</div><span>Powered by PyVis</span></div>
      <div class="graph-stage"><iframe class="knowledge-graph-frame" src="/assets/knowledge-graph.html" title="Interactive documentation knowledge graph"></iframe>
      <div class="graph-hint">Hover to inspect · double-click to open · drag to move</div></div>
    </main>"""
    return shell("Knowledge graph", body, pages, active="graph")


def render_page(page: Page, pages: list[Page]) -> str:
    rendered, headings = markdown_to_html(page.text, pages)
    if not re.search(r"<h1", rendered):
        rendered = f"<h1>{html.escape(page.title)}</h1>\n{rendered}"
    section_pages = [candidate for candidate in pages if candidate.section == page.section]
    index = section_pages.index(page)
    previous = section_pages[index - 1] if index else None
    following = section_pages[index + 1] if index + 1 < len(section_pages) else None
    toc = "".join(
        f'<a class="level-{level}" href="#{anchor}">{html.escape(title)}</a>'
        for level, title, anchor in headings
        if level in (2, 3)
    )
    prev_link = (
        f'<a href="/{previous.slug}/"><small>Previous</small><span>← {html.escape(previous.title)}</span></a>'
        if previous
        else "<span></span>"
    )
    next_link = (
        f'<a class="next" href="/{following.slug}/"><small>Next</small><span>{html.escape(following.title)} →</span></a>'
        if following
        else "<span></span>"
    )
    body = f"""<main class="doc-layout">
      <article class="doc-article"><div class="breadcrumb"><a href="/">GrowingNN</a><span>/</span><span>{html.escape(page.section)}</span></div>
      <div class="article-content">{rendered}</div>
      <nav class="page-nav">{prev_link}{next_link}</nav></article>
      <aside class="toc"><strong>On this page</strong>{toc or '<span>No sections</span>'}<a class="edit-link" href="https://github.com/PjuralSzymon/growingNN_pytorch-2/edit/main/{quote(page.source.relative_to(REPO).as_posix())}" target="_blank" rel="noreferrer">Edit this page ↗</a></aside>
    </main>"""
    return shell(page.title, body, pages, page.slug, page.description)


def build() -> None:
    pages = load_pages()
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "assets").mkdir(parents=True)
    for asset in ASSETS.iterdir():
        shutil.copy2(asset, DIST / "assets" / asset.name)
    create_pyvis_graph(pages, DIST / "assets" / "knowledge-graph.html")
    (DIST / "index.html").write_text(homepage(pages), encoding="utf-8")
    (DIST / "docs").mkdir()
    (DIST / "docs" / "index.html").write_text(documentation_home(pages), encoding="utf-8")
    (DIST / "graph").mkdir()
    (DIST / "graph" / "index.html").write_text(graph_page(pages), encoding="utf-8")
    for page in pages:
        output_dir = DIST / page.slug
        output_dir.mkdir(parents=True)
        (output_dir / "index.html").write_text(render_page(page, pages), encoding="utf-8")
    search_data = [
        {"title": page.title, "url": f"/{page.slug}/", "section": page.section, "text": description(page.text)}
        for page in pages
    ]
    (DIST / "search-index.json").write_text(json.dumps(search_data, ensure_ascii=False), encoding="utf-8")
    (DIST / ".htaccess").write_text(
        "ErrorDocument 404 /index.html\n<IfModule mod_deflate.c>\nAddOutputFilterByType DEFLATE text/html text/css application/javascript application/json\n</IfModule>\n",
        encoding="utf-8",
    )
    print(f"Built {len(pages)} pages in {DIST}")


if __name__ == "__main__":
    build()
