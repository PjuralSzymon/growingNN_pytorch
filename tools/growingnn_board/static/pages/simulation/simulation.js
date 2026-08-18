/** Simulation board: candidates, scores, architecture PDFs. */

import {
  Board,
  $,
  api,
  dlRows,
  fmtNum,
  formatScoreWeights,
  listSimulationGenerations,
  scoreBreakdownHtml,
  shortActionLabel,
  snapshotChanged,
  structureHtml,
} from "../../shared/lib.js?v=5";
import { navigateTo } from "../../shared/navigation.js?v=5";
import { bindPdfToolbar, clearPdfViewer, renderPdfViewer } from "../../shared/pdf.js?v=5";

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function clearSearchTreeFrame(message = "") {
  const frame = $("search-tree-frame");
  const box = $("search-tree-viewport");
  if (frame) {
    frame.removeAttribute("src");
    frame.classList.add("hidden");
  }
  if (box && message) {
    const note = box.querySelector(".search-tree-placeholder");
    if (note) note.textContent = message;
    else {
      const p = document.createElement("p");
      p.className = "helper-text search-tree-placeholder";
      p.textContent = message;
      box.appendChild(p);
    }
  } else if (box) {
    box.querySelectorAll(".search-tree-placeholder").forEach((el) => el.remove());
  }
}

function renderSearchTree(gen, sim, previewWeights = null) {
  const frame = $("search-tree-frame");
  const box = $("search-tree-viewport");
  if (!frame || !box) return;
  box.querySelectorAll(".search-tree-placeholder").forEach((el) => el.remove());
  const hasTree = Boolean(sim?.searchTree?.children?.length || sim?.candidates?.length || sim?.candidateActions?.length);
  if (!hasTree) {
    if (Board.lastSearchTreeGen != null) {
      Board.lastSearchTreeGen = null;
      clearSearchTreeFrame("No search tree data for this generation yet.");
    }
    return;
  }
  const previewKey = previewWeights
    ? `${gen}:${previewWeights.accuracy}:${previewWeights.paramCount}`
    : String(gen);
  if (Board.lastSearchTreeGen === previewKey) return;
  Board.lastSearchTreeGen = previewKey;
  frame.classList.remove("hidden");
  const query = previewWeights
    ? `?accuracy_weight=${encodeURIComponent(previewWeights.accuracy)}&param_count_weight=${encodeURIComponent(previewWeights.paramCount)}`
    : "";
  frame.src = `/api/simulation/${gen}/search-tree${query}`;
}

function simulationSidebarSnapshot(main, sim) {
  const tp = main?.trainingParameters || {};
  const settings = sim?.settings || {};
  const results = sim?.results || {};
  return {
    simMaxTime: settings.simulationMaxTimeSec ?? tp.simulationTimeSec,
    simMaxDepth: settings.simulationMaxDepth ?? tp.simulationEpochs,
    ucb1: settings.ucb1Enabled,
    algorithm: settings.algorithm ?? tp.simulationAlgorithm,
    exploration: settings.explorationConstant,
    rolloutPolicy: settings.rolloutPolicy,
    branching: settings.maxBranchingFactor,
    scoreWeights: sim?.scoreWeights || tp.scoreWeights,
    duration: results.meanSimulationRunTimeSec ?? sim?.durationSec,
    avgScore: results.averageChosenActionScore ?? sim?.scoreChosen,
    chosenTime: results.timeOfChosenSimulationSec ?? sim?.durationSec,
    depth: results.depthReached ?? sim?.maxDepth,
    action: results.chosenAction ?? sim?.actionChosen,
    actionScore: results.chosenActionScore ?? sim?.scoreChosen,
  };
}

function simulationContentSnapshot(sim, gen, candidates) {
  return {
    gen,
    lastUpdate: sim?.lastUpdate,
    candidateCount: candidates.length,
    chosenPdf: sim?.files?.simulationGraphPdf,
    hasSearchTree: Boolean(sim?.searchTree?.children?.length || sim?.candidates?.length),
    actionChosen: sim?.actionChosen,
    candidates: candidates.map((c) => ({
      name: c.name,
      score: c.score,
      accuracyAfter: c.accuracyAfter,
      visits: c.visits,
      isChosen: c.isChosen,
      graphPdf: c.graphPdf,
    })),
  };
}

function simulationGensSnapshot(gens, selected) {
  return { gens, selected };
}

function renderSimulationEmptyState(
  message = "No simulation has run yet. Complete at least one generation with simulation enabled, then return here.",
) {
  const text = escapeHtml(message);
  const sidebar = $("simulation-sidebar");
  if (sidebar) {
    sidebar.innerHTML = `
      <button type="button" class="nav-link-btn" id="goto-training">← Go to the overview board</button>
      <h2>Simulation</h2>
      <p class="helper-text">${text}</p>`;
    $("goto-training").onclick = () => navigateTo("training");
  }
  const candidates = $("candidates");
  if (candidates) candidates.innerHTML = `<p class="helper-text">${text}</p>`;
  const title = $("sim-pdf-title");
  if (title) title.textContent = "Simulation board";
  const pathEl = $("sim-pdf-path");
  if (pathEl) pathEl.textContent = "";
  clearPdfViewer("simulation", message);
  const picker = $("generation-buttons");
  if (picker) picker.innerHTML = "";
  Board.selectedSimGen = null;
  Board.selectedCandidateIndex = null;
  Board.lastSearchTreeGen = null;
  Board.lastSimulationPdfPath = "";
  Board.simulationWeightPreview = null;
  $("recalculation-popup")?.classList.add("hidden");
  clearSearchTreeFrame(message);
}

function renderSimulationSidebar(main, sim, gen, onRecalculate) {
  const tp = main?.trainingParameters || {};
  const settings = sim?.settings || {};
  const results = sim?.results || {};
  const savedWeights = sim?.scoreWeights || tp.scoreWeights || {};
  const preview = Board.simulationWeightPreview?.gen === gen ? Board.simulationWeightPreview : null;
  const accuracyWeight = preview?.weights?.accuracy ?? savedWeights.weight_acc ?? 1;
  const paramCountWeight = preview?.weights?.paramCount ?? savedWeights.weight_countW ?? 0;
  $("simulation-sidebar").innerHTML = `
    <button type="button" class="nav-link-btn" id="goto-training">← Go to the overview board</button>
    <h2>Simulation</h2>
    <div class="sidebar-section">
      <h3>Simulation settings</h3>
      <dl>${dlRows([
        ["Simulation max time", `${settings.simulationMaxTimeSec ?? tp.simulationTimeSec ?? "—"} s`],
        ["Simulation max depth", settings.simulationMaxDepth ?? tp.simulationEpochs ?? "—"],
        ["UCB1 settings", settings.ucb1Enabled != null ? (settings.ucb1Enabled ? "Enabled" : "Disabled") : "—"],
        ["Simulation algorithm", settings.algorithm ?? tp.simulationAlgorithm ?? "—"],
        ["Exploration constant (c)", settings.explorationConstant ?? "—"],
        ["Rollout policy", settings.rolloutPolicy ?? "Default"],
        ["Max branching factor", settings.maxBranchingFactor ?? "—"],
        ["Score weights", formatScoreWeights(sim.scoreWeights || tp.scoreWeights)],
      ])}</dl>
    </div>
    <div class="sidebar-section">
      <h3>Simulation results (current run)</h3>
      <dl>${dlRows([
        ["Mean time of simulation run", `${results.meanSimulationRunTimeSec ?? sim?.durationSec ?? "—"} s`],
        ["Average score of chosen action", results.averageChosenActionScore ?? sim?.scoreChosen ?? "—"],
        ["Time of chosen simulation", `${results.timeOfChosenSimulationSec ?? sim?.durationSec ?? "—"} s`],
        ["Depth of tree reached", results.depthReached ?? sim?.maxDepth ?? "—"],
        ["Action chosen", results.chosenAction ?? sim?.actionChosen ?? "—"],
        ["Score of action chosen", results.chosenActionScore ?? sim?.scoreChosen ?? "—"],
      ])}</dl>
    </div>
    <div class="sidebar-section settings-preview">
      <h3>Check other settings</h3>
      <p class="helper-text">Change only the multipliers. Saved accuracy and parameter results stay unchanged.</p>
      <label for="preview-accuracy-weight">Accuracy weight</label>
      <input id="preview-accuracy-weight" type="number" min="0" step="0.01" value="${accuracyWeight}">
      <label for="preview-param-weight">Parameter count weight</label>
      <input id="preview-param-weight" type="number" min="0" step="0.01" value="${paramCountWeight}">
      <button type="button" class="preview-recalculate-btn" id="preview-recalculate">Recalculate</button>
      <p class="preview-error hidden" id="preview-error"></p>
    </div>`;
  $("goto-training").onclick = () => navigateTo("training");
  $("preview-recalculate").onclick = onRecalculate;
}

function normalizeCandidates(sim) {
  const raw = sim.candidateActions || sim.candidates || [];
  return raw.map((c, i) => ({
    index: i,
    name: c.name || shortActionLabel(c.action) || `Action ${i + 1}`,
    action: c.action,
    score: c.ucbScore ?? c.score,
    compositeScore: c.compositeScore ?? c.scoreBreakdown?.composite,
    scoreMetric: c.scoreMetric || "UCB1",
    accuracyAfter: c.accuracyAfter,
    valLossAfter: c.valLossAfter,
    paramsAfter: c.paramsAfter ?? c.paramCount,
    visits: c.visits,
    isChosen: c.isChosen ?? c.chosen ?? false,
    graphPdf: c.graphPdf,
    scoreBreakdown: c.scoreBreakdown,
    structure: c.structure,
  }));
}

function candidatesWithPreview(candidates, preview) {
  if (!preview) return candidates;
  const rows = new Map(preview.result.candidates.map((row) => [row.index, row]));
  return candidates.map((candidate) => {
    const row = rows.get(candidate.index);
    if (!row) return candidate;
    return {
      ...candidate,
      score: row.score,
      compositeScore: row.score,
      scoreMetric: "recalculated composite",
      scoreBreakdown: row.scoreBreakdown,
      isProjectedChoice: row.action === preview.result.projectedAction,
    };
  });
}

function showRecalculationPopup(result) {
  const popup = $("recalculation-popup");
  if (!popup) return;
  $("recalculation-action").textContent = result.projectedName || "Unavailable";
  $("recalculation-same-action").textContent = result.sameAction ? "Yes" : "No";
  const unavailable = $("recalculation-unavailable");
  if (result.unavailableActions.length) {
    unavailable.textContent = `${result.unavailableActions.length} action(s) could not be recalculated because saved score terms are missing.`;
    unavailable.classList.remove("hidden");
  } else {
    unavailable.classList.add("hidden");
  }
  popup.classList.remove("hidden");
}

function renderCandidateActions(candidates, onSelect) {
  const box = $("candidates");
  box.innerHTML = "";
  if (!candidates.length) {
    box.innerHTML = `<p class="helper-text">No candidate actions recorded for this generation.</p>`;
    return;
  }
  candidates.forEach((c, idx) => {
    const card = document.createElement("button");
    card.type = "button";
    const selected = Board.selectedCandidateIndex === idx;
    card.className = "candidate-card"
      + (c.isChosen ? " chosen" : "")
      + (c.isProjectedChoice ? " projected" : "")
      + (selected ? " selected" : "");
    card.innerHTML = `
      <div class="candidate-header">
        <div class="candidate-icon">${c.isChosen ? "✓" : "⬡"}</div>
        <div class="action-name">${c.name}</div>
      </div>
      <div class="candidate-stat">UCB / score: <strong>${fmtNum(c.score)}</strong> (${c.scoreMetric})</div>
      <div class="candidate-stat">Val accuracy: <strong>${fmtNum(c.accuracyAfter)}</strong> · Loss: <strong>${fmtNum(c.valLossAfter)}</strong></div>
      <div class="candidate-stat">Params: <strong>${c.paramsAfter ?? "—"}</strong> · Visits: <strong>${c.visits ?? "—"}</strong></div>
      ${scoreBreakdownHtml(c.scoreBreakdown)}
      ${structureHtml(c.structure)}
      <div class="candidate-hint">${c.graphPdf ? "Click to preview this architecture" : "Graph not saved for this candidate"}</div>`;
    card.onclick = () => {
      Board.selectedCandidateIndex = idx;
      renderCandidateActions(candidates, onSelect);
      if (c.graphPdf && onSelect) onSelect(c);
    };
    box.appendChild(card);
  });
}

function renderGenerationPicker(gens, selected) {
  const box = $("generation-buttons");
  box.innerHTML = "";
  for (const gen of gens) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "generation-button" + (gen === selected ? " active" : "");
    btn.textContent = gen + 1;
    btn.onclick = () => loadSimulation(gen);
    box.appendChild(btn);
  }
}

export async function refreshSimulationBoard() {
  try {
    const gens = await listSimulationGenerations();
    if (!gens.length) {
      renderSimulationEmptyState();
      return;
    }
    if (Board.selectedSimGen == null || !gens.includes(Board.selectedSimGen)) {
      Board.selectedSimGen = gens.at(-1);
    }
    const selected = Board.selectedSimGen;
    if (snapshotChanged("simulation:gens", simulationGensSnapshot(gens, selected))) {
      renderGenerationPicker(gens, selected);
    }
    await loadSimulation(selected, { updatePicker: false, fromPoll: true });
  } catch (err) {
    console.error("Simulation board refresh failed:", err);
    renderSimulationEmptyState("Could not load simulation data. Try reloading the experiment.");
  }
}

export async function loadSimulation(gen, { updatePicker = true, fromPoll = false } = {}) {
  const prevGen = Board.selectedSimGen;
  Board.selectedSimGen = gen;
  if (!fromPoll) Board.selectedCandidateIndex = null;
  if (updatePicker) {
    document.querySelectorAll(".generation-button").forEach((btn) => {
      btn.classList.toggle("active", Number(btn.textContent) === gen + 1);
    });
  }
  if (prevGen !== gen) {
    Board.lastSearchTreeGen = null;
    Board.lastSimulationPdfPath = "";
    Board.simulationWeightPreview = null;
    delete Board.snapshots[`simulation:content:${prevGen}`];
    delete Board.snapshots[`simulation:sidebar:${prevGen}`];
  }
  let sim;
  let main;
  try {
    sim = await api(`/api/simulation/${gen}`);
    main = await api("/api/experiment/main");
  } catch {
    const msg = `No simulation data for generation ${gen + 1} yet.`;
    $("candidates").innerHTML = `<p class="helper-text">${escapeHtml(msg)}</p>`;
    if ($("sim-pdf-title")) $("sim-pdf-title").textContent = `Simulation — generation ${gen + 1}`;
    if ($("sim-pdf-path")) $("sim-pdf-path").textContent = "";
    clearPdfViewer("simulation", msg);
    clearSearchTreeFrame(msg);
    return;
  }
  const activePreview = Board.simulationWeightPreview?.gen === gen
    ? Board.simulationWeightPreview
    : null;
  let candidates = candidatesWithPreview(normalizeCandidates(sim), activePreview);
  const defaultPdf = sim.files?.simulationGraphPdf || `graphs/gen_${gen}_simulation_simplified.pdf`;
  const fallbackPdfs = [
    `graphs/gen_${gen}_simulation_full.pdf`,
    `graphs/gen_${gen}_full.pdf`,
    `graphs/gen_${gen}_simplified.pdf`,
  ];

  const showCandidatePdf = (c) => {
    const label = c.isChosen ? " (chosen)" : " (alternative)";
    $("sim-pdf-title").textContent = `Candidate: ${c.name}${label}`;
    $("sim-pdf-path").textContent = `File path: ${Board.experimentPath}/${c.graphPdf}`;
    Board.lastSimulationPdfPath = c.graphPdf;
    renderPdfViewer("simulation", c.graphPdf, [defaultPdf, ...fallbackPdfs]);
  };
  const recalculate = async () => {
    const button = $("preview-recalculate");
    const error = $("preview-error");
    const accuracyWeight = Number($("preview-accuracy-weight").value);
    const paramCountWeight = Number($("preview-param-weight").value);
    if (
      !Number.isFinite(accuracyWeight)
      || !Number.isFinite(paramCountWeight)
      || accuracyWeight < 0
      || paramCountWeight < 0
      || accuracyWeight + paramCountWeight <= 0
    ) {
      error.textContent = "Use non-negative weights with at least one value greater than zero.";
      error.classList.remove("hidden");
      return;
    }
    button.disabled = true;
    error.classList.add("hidden");
    try {
      const query = `accuracy_weight=${encodeURIComponent(accuracyWeight)}&param_count_weight=${encodeURIComponent(paramCountWeight)}`;
      const result = await api(`/api/simulation/${gen}/recalculate?${query}`);
      const preview = { gen, weights: result.weights, result };
      Board.simulationWeightPreview = preview;
      candidates = candidatesWithPreview(normalizeCandidates(sim), preview);
      renderCandidateActions(candidates, showCandidatePdf);
      Board.lastSearchTreeGen = null;
      renderSearchTree(gen, sim, result.weights);
      showRecalculationPopup(result);
    } catch (err) {
      error.textContent = err.message || "Could not recalculate the saved scores.";
      error.classList.remove("hidden");
    } finally {
      button.disabled = false;
    }
  };

  const contentKey = `simulation:content:${gen}`;
  const sidebarKey = `simulation:sidebar:${gen}`;
  const contentSnap = simulationContentSnapshot(sim, gen, candidates);
  const contentChanged = snapshotChanged(contentKey, contentSnap);
  const sidebarChanged = snapshotChanged(sidebarKey, simulationSidebarSnapshot(main, sim));

  if (sidebarChanged) renderSimulationSidebar(main, sim, gen, recalculate);
  if (contentChanged) renderCandidateActions(candidates, showCandidatePdf);
  renderSearchTree(gen, sim, activePreview?.weights);

  const chosen = candidates.find((c) => c.isChosen);
  const defaultTitle = chosen
    ? `Simulation Graph for Generation: ${gen} — ${chosen.name} (chosen)`
    : `Simulation Graph for Generation: ${gen} (PDF)`;

  if (!Board.selectedCandidateIndex && Board.lastSimulationPdfPath !== defaultPdf) {
    $("sim-pdf-title").textContent = defaultTitle;
    $("sim-pdf-path").textContent = `File path: ${Board.experimentPath}/${defaultPdf}`;
    Board.lastSimulationPdfPath = defaultPdf;
    renderPdfViewer("simulation", defaultPdf, fallbackPdfs);
  }
}

export function initSimulation() {
  Board.refreshHandlers.push(async () => {
    if ($("view-simulation")?.classList.contains("hidden")) return;
    await refreshSimulationBoard();
  });
  bindPdfToolbar("sim-pdf-toolbar", "simulation");
  $("recalculation-popup-close").onclick = () => $("recalculation-popup").classList.add("hidden");
}
