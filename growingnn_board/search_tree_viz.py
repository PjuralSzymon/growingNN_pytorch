"""Interactive search-tree HTML for GrowingNN Board (pyvis + vis.js)."""

from __future__ import annotations

import html
import json
import math
import re
import textwrap
from typing import Any

_NETWORK_MARKER = "network = new vis.Network(container, data, options);"


def action_short_label(action_str: str | None) -> str:
    if not action_str:
        return "—"
    match = re.search(r"\(\s*([^:(]+)", action_str)
    return match.group(1).strip() if match else action_str[:48]


def tree_from_candidates(
    candidates: list[dict[str, Any]] | None,
    rollouts: int,
) -> dict[str, Any]:
    """Build a flat search tree JSON from candidate rows (legacy runs without searchTree)."""
    children: list[dict[str, Any]] = []
    for index, row in enumerate(candidates or []):
        composite = row.get("compositeScore")
        mean = row.get("score")
        final_score = composite if composite is not None else mean
        children.append(
            {
                "id": f"0-{index}",
                "action": row.get("action"),
                "name": row.get("name") or action_short_label(row.get("action")),
                "depth": 1,
                "visits": row.get("visits", 1),
                "meanScore": mean,
                "ucbScore": row.get("ucbScore"),
                "compositeScore": composite,
                "finalScore": final_score,
                "maxDepthBelow": 1,
                "accuracyAfter": row.get("accuracyAfter"),
                "chosen": bool(row.get("chosen")),
                "children": [],
            }
        )
    return {
        "id": "0",
        "name": "root",
        "depth": 0,
        "visits": rollouts,
        "maxDepthBelow": 1 if children else 0,
        "simMaxDepth": 1 if children else 0,
        "chosen": False,
        "children": children,
    }


def resolve_search_tree(sim: dict[str, Any]) -> dict[str, Any] | None:
    tree = sim.get("searchTree")
    if tree and tree.get("children"):
        return tree
    candidates = sim.get("candidates") or sim.get("candidateActions")
    if not candidates:
        return None
    return tree_from_candidates(candidates, int(sim.get("rollouts") or len(candidates)))


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _node_label(name: str, final_score: Any) -> str:
    short = name if len(name) <= 22 else f"{name[:19]}…"
    score = _fmt(final_score)
    return f"{short}\\n{score}" if score != "—" else short


def _node_tooltip(node: dict[str, Any]) -> str:
    lines = [
        node.get("name") or "—",
        f"Depth: {_fmt(node.get('depth'))}",
        f"Visits: {_fmt(node.get('visits'))}",
        f"Final score: {_fmt(node.get('finalScore'))}",
        f"Mean score: {_fmt(node.get('meanScore'))}",
        f"UCB score: {_fmt(node.get('ucbScore'))}",
        f"Val accuracy: {_fmt(node.get('accuracyAfter'))}",
    ]
    if node.get("chosen"):
        lines.append("Chosen action")
    return "\\n".join(lines)


def _node_color(node: dict[str, Any]) -> str:
    if node.get("id") == "0" or (node.get("depth") == 0 and not node.get("action")):
        return "#e2e8f0"
    try:
        score = float(node.get("finalScore"))
    except (TypeError, ValueError):
        return "#ffffff"
    if not math.isfinite(score):
        return "#ffffff"
    hue = round(max(0.0, min(1.0, score)) * 120)
    return f"hsl({hue}, 70%, 88%)"


def _collect_details(node: dict[str, Any], out: dict[str, dict[str, Any]]) -> None:
    node_id = str(node["id"])
    out[node_id] = {
        "name": node.get("name") or "—",
        "action": node.get("action"),
        "depth": node.get("depth"),
        "visits": node.get("visits"),
        "finalScore": node.get("finalScore"),
        "meanScore": node.get("meanScore"),
        "ucbScore": node.get("ucbScore"),
        "compositeScore": node.get("compositeScore"),
        "accuracyAfter": node.get("accuracyAfter"),
        "maxDepthBelow": node.get("maxDepthBelow"),
        "chosen": bool(node.get("chosen")),
    }
    for child in node.get("children") or []:
        _collect_details(child, out)


def _inject_interaction(html_text: str, node_details: dict[str, dict[str, Any]], summary: dict[str, Any]) -> str:
    panel = textwrap.dedent(
        """
        <div id="gnn-tree-summary">
          Analysis depth: <strong>{max_depth}</strong>
          · Nodes analyzed: <strong>{node_count}</strong>
          · Rollouts: <strong>{rollouts}</strong>
        </div>
        <div id="gnn-node-detail">
          <div id="gnn-node-detail-title">Node details</div>
          <div id="gnn-node-detail-body">Click a node to inspect scores and depth.</div>
        </div>
        """
    ).format(
        max_depth=html.escape(_fmt(summary.get("maxDepth"))),
        node_count=summary.get("nodeCount", 0),
        rollouts=html.escape(_fmt(summary.get("rollouts"))),
    )
    html_text = html_text.replace("<div id=\"mynetwork\"", panel + '\n<div id="mynetwork"', 1)
    style = textwrap.dedent(
        """
        <style>
          #mynetwork { height: calc(100vh - 168px) !important; border-bottom: 1px solid #cbd5e1; }
          #gnn-tree-summary {
            font: 12px/1.4 system-ui, sans-serif;
            color: #64748b;
            padding: 8px 12px;
            border-bottom: 1px solid #e2e8f0;
            background: #f8fafc;
          }
          #gnn-tree-summary strong { color: #0f172a; }
          #gnn-node-detail {
            font: 12px/1.45 system-ui, sans-serif;
            padding: 10px 12px 14px;
            background: #fff;
            min-height: 96px;
          }
          #gnn-node-detail-title { font-weight: 700; color: #0f172a; margin-bottom: 6px; }
          #gnn-node-detail-body { color: #334155; white-space: pre-wrap; }
          #gnn-node-detail-body .term { display: flex; justify-content: space-between; gap: 12px; margin: 2px 0; }
          #gnn-node-detail-body .term span:last-child { font-weight: 600; color: #0f172a; }
        </style>
        """
    )
    html_text = html_text.replace("</head>", style + "\n</head>", 1)
    details_json = json.dumps(node_details)
    script = textwrap.dedent(
        f"""
        {_NETWORK_MARKER}
        var gnnNodeDetails = {details_json};
        function gnnRenderNodeDetail(nodeId) {{
          var body = document.getElementById("gnn-node-detail-body");
          var title = document.getElementById("gnn-node-detail-title");
          var row = gnnNodeDetails[nodeId];
          if (!row) {{
            title.textContent = "Node details";
            body.textContent = "No details for this node.";
            return;
          }}
          title.textContent = row.name || "Node details";
          var lines = [
            ["Depth", row.depth],
            ["Visits", row.visits],
            ["Final score", row.finalScore],
            ["Mean score", row.meanScore],
            ["UCB score", row.ucbScore],
            ["Composite score", row.compositeScore],
            ["Val accuracy", row.accuracyAfter],
            ["Max depth below", row.maxDepthBelow],
            ["Chosen", row.chosen ? "yes" : "no"],
          ];
          body.innerHTML = lines.map(function (pair) {{
            return '<div class="term"><span>' + pair[0] + '</span><span>' + (pair[1] == null ? "—" : pair[1]) + '</span></div>';
          }}).join("");
        }}
        network.on("click", function (params) {{
          if (params.nodes && params.nodes.length) {{
            gnnRenderNodeDetail(String(params.nodes[0]));
          }}
        }});
        network.on("hoverNode", function () {{
          container.style.cursor = "pointer";
        }});
        network.on("blurNode", function () {{
          container.style.cursor = "default";
        }});
        """
    )
    if _NETWORK_MARKER not in html_text:
        raise ValueError("pyvis HTML missing network initialization marker")
    return html_text.replace(_NETWORK_MARKER, script, 1)


def _count_analyzed_nodes(tree: dict[str, Any]) -> int:
    count = 0

    def walk(node: dict[str, Any]) -> None:
        nonlocal count
        for child in node.get("children") or []:
            count += 1
            walk(child)

    walk(tree)
    return count


def _render_search_tree_html_pyvis(
    search_tree: dict[str, Any],
    *,
    rollouts: int | None = None,
    max_depth: int | None = None,
) -> str:
    from pyvis.network import Network

    node_details: dict[str, dict[str, Any]] = {}
    _collect_details(search_tree, node_details)
    summary = {
        "rollouts": rollouts,
        "maxDepth": max_depth if max_depth is not None else search_tree.get("simMaxDepth"),
        "nodeCount": _count_analyzed_nodes(search_tree),
    }

    net = Network(
        height="100%",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#0f172a",
        cdn_resources="in_line",
    )
    net.set_options(
        """
        {
          "nodes": {
            "shape": "box",
            "margin": 12,
            "font": { "size": 13, "face": "system-ui" },
            "widthConstraint": { "minimum": 96, "maximum": 132 },
            "borderWidth": 1,
            "color": { "border": "#94a3b8", "background": "#ffffff", "highlight": { "border": "#2563eb", "background": "#eff6ff" } }
          },
          "edges": {
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
            "color": { "color": "#94a3b8", "highlight": "#2563eb" },
            "smooth": { "type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.35 }
          },
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed",
              "levelSeparation": 155,
              "nodeSpacing": 52,
              "treeSpacing": 40,
              "blockShifting": true,
              "edgeMinimization": false,
              "parentCentralization": false
            }
          },
          "physics": { "enabled": false },
          "interaction": { "hover": true, "tooltipDelay": 80, "navigationButtons": true, "keyboard": true }
        }
        """
    )

    def add_branch(node: dict[str, Any], parent_id: str | None) -> None:
        node_id = str(node["id"])
        is_root = node_id == "0" and not node.get("action")
        depth = int(node.get("depth") or 0)
        label = "Start" if is_root else _node_label(str(node.get("name") or "—"), node.get("finalScore"))
        net.add_node(
            node_id,
            label=label,
            title=_node_tooltip(node),
            color=_node_color(node),
            size=26,
            level=depth,
            borderWidth=2 if node.get("chosen") else 1,
        )
        if parent_id is not None:
            net.add_edge(parent_id, node_id)
        for child in node.get("children") or []:
            add_branch(child, node_id)

    add_branch(search_tree, None)
    html_text = net.generate_html(notebook=False)
    return _inject_interaction(html_text, node_details, summary)


def _group_nodes_by_depth(root: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    layers: dict[int, list[dict[str, Any]]] = {}

    def walk(node: dict[str, Any]) -> None:
        depth = int(node.get("depth") or 0)
        layers.setdefault(depth, []).append(node)
        for child in node.get("children") or []:
            walk(child)

    walk(root)
    return layers


def _render_depth_node_html(node: dict[str, Any]) -> str:
    node_id = html.escape(str(node["id"]))
    is_root = node_id == "0" and not node.get("action")
    name = html.escape(str(node.get("name") or "—"))
    score = html.escape(_fmt(node.get("finalScore")))
    color = _node_color(node)
    chosen = " chosen" if node.get("chosen") else ""
    if is_root:
        return f'<div class="tree-node root" data-id="{node_id}"><div class="title">Start</div></div>'
    return (
        f'<div class="tree-node{chosen}" data-id="{node_id}" style="background:{color}">'
        f'<div class="title">{name}</div><div class="score">{score}</div></div>'
    )


def _render_depth_layers_html(search_tree: dict[str, Any]) -> str:
    layers = _group_nodes_by_depth(search_tree)
    rows: list[str] = []
    for depth in sorted(layers.keys()):
        nodes_html = "".join(_render_depth_node_html(node) for node in layers[depth])
        rows.append(f'<div class="depth-row" data-depth="{depth}">{nodes_html}</div>')
    return "".join(rows)


def _render_tree_node_html(node: dict[str, Any]) -> str:
    node_id = html.escape(str(node["id"]))
    is_root = node_id == "0" and not node.get("action")
    name = html.escape(str(node.get("name") or "—"))
    score = html.escape(_fmt(node.get("finalScore")))
    color = _node_color(node)
    chosen = " chosen" if node.get("chosen") else ""
    body = (
        f'<div class="tree-node root" data-id="{node_id}"><div class="title">Start</div></div>'
        if is_root
        else f'<div class="tree-node{chosen}" data-id="{node_id}" style="background:{color}"><div class="title">{name}</div><div class="score">{score}</div></div>'
    )
    children = node.get("children") or []
    if not children:
        return body
    child_html = "".join(f"<li>{_render_tree_node_html(child)}</li>" for child in children)
    return f"{body}<ul>{child_html}</ul>"


def _render_search_tree_html_fallback(
    search_tree: dict[str, Any],
    *,
    rollouts: int | None = None,
    max_depth: int | None = None,
) -> str:
    """Plain HTML/CSS tree when pyvis is not installed."""
    node_details: dict[str, dict[str, Any]] = {}
    _collect_details(search_tree, node_details)
    summary = {
        "rollouts": rollouts,
        "maxDepth": max_depth if max_depth is not None else search_tree.get("simMaxDepth"),
        "nodeCount": _count_analyzed_nodes(search_tree),
    }
    tree_html = _render_depth_layers_html(search_tree)
    details_json = json.dumps(node_details)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8" /><title>Search tree</title>
<style>
  body {{ margin: 0; font: 12px/1.4 system-ui, sans-serif; color: #0f172a; background: #fff; }}
  #gnn-tree-summary {{ padding: 8px 12px; border-bottom: 1px solid #e2e8f0; background: #f8fafc; color: #64748b; }}
  #gnn-tree-scroll {{ overflow: auto; max-height: calc(100vh - 150px); padding: 12px; }}
  .gnn-tree {{ display: flex; flex-direction: column; gap: 52px; align-items: stretch; }}
  .depth-row {{ display: flex; flex-wrap: wrap; gap: 8px 10px; justify-content: center; align-items: flex-start; padding: 0 8px; border-top: 1px dashed #e2e8f0; padding-top: 14px; }}
  .depth-row:first-child {{ border-top: none; padding-top: 0; }}
  .tree-node {{ min-width: 104px; padding: 8px 10px; border: 1px solid #cbd5e1; border-radius: 6px; background: #fff; text-align: center; cursor: pointer; }}
  .tree-node.root {{ background: #f8fafc; cursor: default; }}
  .tree-node.chosen {{ border-color: #2563eb; background: #eff6ff; }}
  .tree-node.selected {{ outline: 2px solid #2563eb; }}
  .tree-node .title {{ font-weight: 700; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 140px; }}
  .tree-node .score {{ color: #2563eb; font-weight: 700; font-size: 12px; margin-top: 3px; }}
  #gnn-node-detail {{ border-top: 1px solid #e2e8f0; padding: 10px 12px; min-height: 88px; }}
  #gnn-node-detail-title {{ font-weight: 700; margin-bottom: 6px; }}
  .term {{ display: flex; justify-content: space-between; gap: 12px; margin: 2px 0; color: #334155; }}
  .term span:last-child {{ font-weight: 600; color: #0f172a; }}
  .fallback-note {{ color: #92400e; background: #fffbeb; border-bottom: 1px solid #fde68a; padding: 6px 12px; font-size: 11px; }}
</style></head><body>
<div class="fallback-note">Simple tree view (install pyvis for interactive graph: pip install pyvis)</div>
<div id="gnn-tree-summary">Analysis depth: <strong>{html.escape(_fmt(summary.get("maxDepth")))}</strong>
 · Nodes analyzed: <strong>{summary.get("nodeCount", 0)}</strong>
 · Rollouts: <strong>{html.escape(_fmt(summary.get("rollouts")))}</strong></div>
<div id="gnn-tree-scroll"><div class="gnn-tree">{tree_html}</div></div>
<div id="gnn-node-detail"><div id="gnn-node-detail-title">Node details</div><div id="gnn-node-detail-body">Click a node to inspect scores.</div></div>
<script>
const gnnNodeDetails = {details_json};
function gnnRenderNodeDetail(nodeId) {{
  const body = document.getElementById("gnn-node-detail-body");
  const title = document.getElementById("gnn-node-detail-title");
  const row = gnnNodeDetails[nodeId];
  if (!row) return;
  title.textContent = row.name || "Node details";
  const lines = [
    ["Depth", row.depth], ["Visits", row.visits], ["Final score", row.finalScore],
    ["Mean score", row.meanScore], ["UCB score", row.ucbScore], ["Composite score", row.compositeScore],
    ["Val accuracy", row.accuracyAfter], ["Max depth below", row.maxDepthBelow], ["Chosen", row.chosen ? "yes" : "no"],
  ];
  body.innerHTML = lines.map(function (pair) {{
    return '<div class="term"><span>' + pair[0] + '</span><span>' + (pair[1] == null ? "—" : pair[1]) + '</span></div>';
  }}).join("");
}}
document.querySelectorAll(".tree-node:not(.root)").forEach(function (el) {{
  el.onclick = function () {{
    document.querySelectorAll(".tree-node.selected").forEach(function (n) {{ n.classList.remove("selected"); }});
    el.classList.add("selected");
    gnnRenderNodeDetail(el.getAttribute("data-id"));
  }};
}});
</script></body></html>"""


def render_search_tree_html(
    search_tree: dict[str, Any],
    *,
    rollouts: int | None = None,
    max_depth: int | None = None,
) -> str:
    """Return interactive hierarchical search-tree HTML."""
    try:
        return _render_search_tree_html_pyvis(
            search_tree,
            rollouts=rollouts,
            max_depth=max_depth,
        )
    except ImportError:
        return _render_search_tree_html_fallback(
            search_tree,
            rollouts=rollouts,
            max_depth=max_depth,
        )
