/** Simulation board: candidates, scores, architecture PDFs. */

import {
  Board,
  $,
  api,
  dlRows,
  fmtNum,
  formatScoreWeights,
  scoreBreakdownHtml,
  shortActionLabel,
  structureHtml,
  showView,
} from "../../shared/core.js";
import { bindPdfToolbar, renderPdfViewer } from "../../shared/pdf.js";

function renderSimulationSidebar(main, sim) {
  const tp = main?.trainingParameters || {};
  const settings = sim?.settings || {};
  const results = sim?.results || {};
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
    </div>`;
  $("goto-training").onclick = () => showView("training");
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

function renderStartingStructure(sim) {
  const start = sim.startingStructure || {};
  $("start-structure").innerHTML = `
    <h4>Starting structure</h4>
    <dl>${dlRows([
      ["Total params", start.totalParams ?? sim.paramCountBefore ?? "—"],
      ["Initial accuracy", start.accuracy ?? sim.valAccBefore ?? "—"],
    ])}</dl>`;
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
    card.className = "candidate-card" + (c.isChosen ? " chosen" : "") + (selected ? " selected" : "");
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
  let gens;
  try {
    gens = (await api("/api/generations")).generations || [];
  } catch {
    return;
  }
  if (!gens.length) return;
  if (Board.selectedSimGen == null || !gens.includes(Board.selectedSimGen)) {
    Board.selectedSimGen = gens.at(-1);
  }
  renderGenerationPicker(gens, Board.selectedSimGen);
  await loadSimulation(Board.selectedSimGen, false);
}

export async function loadSimulation(gen, updatePicker = true) {
  Board.selectedSimGen = gen;
  Board.selectedCandidateIndex = null;
  if (updatePicker) {
    document.querySelectorAll(".generation-button").forEach((btn) => {
      btn.classList.toggle("active", Number(btn.textContent) === gen + 1);
    });
  }
  let sim;
  let main;
  try {
    sim = await api(`/api/simulation/${gen}`);
    main = await api("/api/experiment/main");
  } catch {
    $("sim-pdf-title").textContent = `Simulation Graph for Generation: ${gen} (PDF)`;
    $("candidates").innerHTML = `<p class="helper-text">No simulation data for generation ${gen + 1} yet.</p>`;
    return;
  }
  renderSimulationSidebar(main, sim);
  renderStartingStructure(sim);
  const candidates = normalizeCandidates(sim);
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
    renderPdfViewer("simulation", c.graphPdf, [defaultPdf, ...fallbackPdfs]);
  };

  renderCandidateActions(candidates, showCandidatePdf);

  const chosen = candidates.find((c) => c.isChosen);
  $("sim-pdf-title").textContent = chosen
    ? `Simulation Graph for Generation: ${gen} — ${chosen.name} (chosen)`
    : `Simulation Graph for Generation: ${gen} (PDF)`;
  $("sim-pdf-path").textContent = `File path: ${Board.experimentPath}/${defaultPdf}`;
  renderPdfViewer("simulation", defaultPdf, fallbackPdfs);
}

export function initSimulation() {
  Board.refreshHandlers.push(async () => {
    await refreshSimulationBoard();
  });
  bindPdfToolbar("sim-pdf-toolbar", "simulation");
}
