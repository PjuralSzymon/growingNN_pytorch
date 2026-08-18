/** Home page: folder picker and recent experiments. */

import {
  API,
  Board,
  $,
  api,
  formatRelativeTime,
  refreshAll,
  resetSnapshots,
  startPoll,
  statusClass,
} from "../../shared/lib.js?v=5";
import { navigateTo } from "../../shared/navigation.js?v=5";

function renderRecentExperiments(experiments) {
  const list = $("recent-list");
  list.innerHTML = "";
  if (!experiments.length) {
    list.innerHTML = `<p class="helper-text">No previous directories found under experiments output.</p>`;
    return;
  }
  for (const exp of experiments) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "recent-row";
    row.innerHTML = `
      <span class="recent-path"><span class="folder-icon">📁</span><span>${exp.path}</span></span>
      <span class="recent-update">${formatRelativeTime(exp.lastUpdate)}</span>
      <span class="recent-status"><span class="status-dot ${statusClass(exp.status)}"></span>${exp.status}</span>`;
    row.onclick = () => loadExperimentPath(exp.path);
    list.appendChild(row);
  }
}

export async function loadRecent() {
  try {
    const data = await api("/api/experiments/recent");
    renderRecentExperiments(data.experiments || []);
  } catch (e) {
    $("recent-list").innerHTML = `<p class="load-status warn">${e.message}</p>`;
  }
}

async function loadExperimentPath(path) {
  $("path-input").value = path;
  await loadExperiment();
}

async function loadExperiment() {
  const path = $("path-input").value.trim();
  if (!path) return;
  $("load-status").textContent = "Loading…";
  $("load-status").classList.remove("warn");
  try {
    const res = await fetch(`${API}/api/experiment/load?path=${encodeURIComponent(path)}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Load failed");
    Board.experimentPath = path;
    resetSnapshots();
    $("load-status").textContent = data.warnings?.length
      ? `Loaded (${data.warnings.join("; ")}) — refreshing every 5s`
      : "Loaded. Refreshing every 5s.";
    startPoll();
    await refreshAll();
    navigateTo("training", { replace: true });
  } catch (e) {
    $("load-status").textContent = e.message;
    $("load-status").classList.add("warn");
  }
}

export function initHome() {
  const loadBtn = $("load-btn");
  const pathInput = $("path-input");
  if (loadBtn) loadBtn.onclick = loadExperiment;
  if (pathInput) {
    pathInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") loadExperiment();
    });
  }
}
