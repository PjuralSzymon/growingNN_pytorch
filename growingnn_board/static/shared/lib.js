/** Shared state, API helpers, and app-wide refresh orchestration. */

export const API = "";
export const POLL_MS = 5000;

export const Board = {
  pollTimer: null,
  selectedSimGen: null,
  selectedTrainingGen: null,
  selectedCandidateIndex: null,
  experimentPath: "",
  charts: {},
  useSimplifiedGraph: true,
  refreshHandlers: [],
  snapshots: {},
  lastSearchTreeGen: null,
  lastSimulationPdfPath: "",
};

export function $(id) {
  return document.getElementById(id);
}

export async function api(path, opts = {}) {
  const res = await fetch(`${API}${path}`, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) { /* */ }
    throw new Error(detail);
  }
  return res.json();
}

export function statusClass(status) {
  if (status === "active") return "status-active";
  if (status === "recent") return "status-recent";
  return "status-inactive";
}

export function formatRelativeTime(iso) {
  if (!iso) return "—";
  const then = new Date(iso.replace("Z", "+00:00")).getTime();
  const diffSec = Math.max(0, (Date.now() - then) / 1000);
  if (diffSec < 60) return `${Math.floor(diffSec)} sec ago`;
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)} min ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)} h ${Math.floor((diffSec % 3600) / 60)} min ago`;
  return `${Math.floor(diffSec / 86400)} d ${Math.floor((diffSec % 86400) / 3600)} h ago`;
}

export function formatElapsed(sec) {
  if (sec == null) return "—";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function fmtNum(v, digits = 4) {
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

export function shortActionLabel(action) {
  if (!action) return "—";
  const m = action.match(/\(\s*([^:(]+)/);
  return m ? m[1].trim() : action.slice(0, 40);
}

export function dlRows(rows) {
  return rows.map(([k, v]) => `<dt>${k}</dt><dd>${v ?? "—"}</dd>`).join("");
}

export function formatScoreWeights(weights) {
  if (!weights) return "—";
  const labels = { weight_acc: "acc", weight_loss: "loss", weight_time: "time", weight_countW: "params" };
  return Object.entries(weights)
    .filter(([, v]) => v > 0)
    .map(([k, v]) => `${labels[k] || k}=${v}`)
    .join(", ") || "—";
}

export function scoreBreakdownHtml(breakdown) {
  if (!breakdown?.terms) return "";
  const rows = Object.entries(breakdown.terms).map(([name, t]) =>
    `<div class="score-term"><span>${name}</span><span>w=${fmtNum(t.weight, 2)} × ${fmtNum(t.raw)} = ${fmtNum(t.weighted)}</span></div>`,
  );
  return `<div class="score-breakdown">${rows.join("")}<div class="score-total">Composite: <strong>${fmtNum(breakdown.composite)}</strong></div></div>`;
}

export function structureHtml(structure) {
  if (!structure) return "";
  const mods = (structure.modules || []).slice(0, 4).join(", ");
  return `<div class="structure-summary">
    <div>Modules: <strong>${structure.moduleCount ?? "—"}</strong> · Hidden: <strong>${structure.hiddenModuleCount ?? "—"}</strong></div>
    <div class="structure-modules">${mods || "—"}</div>
  </div>`;
}

export function snapshotKey(value) {
  return JSON.stringify(value);
}

export function snapshotChanged(key, nextValue) {
  const fp = snapshotKey(nextValue);
  if (Board.snapshots[key] === fp) return false;
  Board.snapshots[key] = fp;
  return true;
}

export function resetSnapshots() {
  Board.snapshots = {};
}

export function startPoll() {
  if (Board.pollTimer) clearInterval(Board.pollTimer);
  Board.pollTimer = setInterval(refreshAll, POLL_MS);
}

export function stopPoll() {
  if (Board.pollTimer) clearInterval(Board.pollTimer);
  Board.pollTimer = null;
}

export async function listSimulationGenerations() {
  try {
    const { generations } = await api("/api/simulations");
    return generations || [];
  } catch (_) {
    return [];
  }
}

export async function enrichTimelineWithActions(timeline) {
  if (!timeline?.length) return timeline;
  const simGens = await listSimulationGenerations();
  for (const g of simGens) {
    if (!timeline[g] || timeline[g].actionExecuted) continue;
    try {
      const sim = await api(`/api/simulation/${g}`);
      if (sim.actionChosen) {
        timeline[g].actionExecuted = {
          action: sim.actionChosen,
          shortLabel: shortActionLabel(sim.actionChosen),
        };
      }
    } catch (_) { /* */ }
  }
  return timeline;
}

export async function refreshAll() {
  if (!Board.experimentPath) return;
  try {
    const main = await api("/api/experiment/main");
    let training = null;
    try {
      training = await api("/api/experiment/training");
    } catch (_) { /* */ }
    if (main.generationTimeline?.length) {
      main.generationTimeline = await enrichTimelineWithActions(main.generationTimeline);
    }
    for (const handler of Board.refreshHandlers) {
      await handler(main, training);
    }
  } catch (_) { /* keep last valid UI */ }
}

export async function mountPageHtml(name) {
  const root = $(`view-${name}`);
  if (!root || root.dataset.mounted === "1") return;
  const res = await fetch(`/static/pages/${name}/${name}.html`);
  if (!res.ok) throw new Error(`Failed to load page: ${name}`);
  root.innerHTML = await res.text();
  root.dataset.mounted = "1";
}

export { showView, navigateTo, viewFromLocation, initNavigation } from "./navigation.js";
