/** Training board: charts, timeline, architecture PDF. */

import {
  Board,
  $,
  dlRows,
  fmtNum,
  formatElapsed,
  refreshAll,
  shortActionLabel,
  showView,
  stopPoll,
} from "../../shared/core.js";
import { bindPdfToolbar, renderPdfViewer } from "../../shared/pdf.js";
import { loadRecent } from "../home/home.js";

function renderTrainingSidebar(main) {
  const tp = main.trainingParameters || {};
  const status = main.status || "running";
  $("training-sidebar").innerHTML = `
    <button type="button" class="nav-link-btn" id="goto-home">← Back to folder picker</button>
    <h2>Experiment info</h2>
    <div class="sidebar-section">
      <dl>${dlRows([
        ["Experiment started on", main.experimentStartedOn],
        ["Training time elapsed", formatElapsed(main.trainingTimeElapsedSec)],
        ["Model", main.model?.name || main.experimentName],
        ["Dataset", main.dataset],
        ["Device", main.device],
        ["Status", `<span class="status-pill ${status}"><span class="status-dot status-active"></span>${status}</span>`],
      ])}</dl>
    </div>
    <div class="sidebar-section">
      <h3>Training parameters</h3>
      <dl>${dlRows([
        ["Total generations", tp.totalGenerations],
        ["Current generation", tp.currentGeneration],
        ["Current epoch", `${tp.currentEpoch ?? 0} / ${tp.epochsPerGeneration ?? 0}`],
        ["Total epochs", tp.totalEpochs],
        ["Batch size", tp.batchSize ?? "—"],
        ["Optimizer", tp.optimizer ?? "SGD"],
        ["Learning rate used", tp.learningRateUsed ?? tp.learningRateAlpha ?? "—"],
        ["Learning rate mode", tp.learningRateMode ?? "—"],
        ["Weight decay", tp.weightDecay ?? "—"],
        ["Gradient clip", tp.gradientClip ?? "—"],
        ["Random seed", tp.randomSeed ?? "—"],
      ])}</dl>
    </div>`;
  $("goto-home").onclick = () => {
    stopPoll();
    Board.experimentPath = "";
    showView("home");
    loadRecent();
  };
}

function renderLastSimulation(sim) {
  const box = $("last-sim");
  if (!sim) {
    box.innerHTML = "<dd>No simulation yet</dd>";
    return;
  }
  box.innerHTML = dlRows([
    ["Amount of actions analyzed", sim.actionsAnalyzed],
    ["Depth of simulation tree reached", sim.treeDepth],
    ["Time of execution", sim.executionTimeSec != null ? `${sim.executionTimeSec}s` : "—"],
    ["Action chosen", sim.actionShortLabel || shortActionLabel(sim.actionChosen)],
    ["Score of that action", sim.scoreChosen != null ? `${fmtNum(sim.scoreChosen)} (UCB1)` : "—"],
  ]);
}

function plotChart(canvasId, key, rows, xKey, yKey, color) {
  const canvas = $(canvasId);
  if (!canvas || !rows.length) return;
  if (Board.charts[key]) Board.charts[key].destroy();
  Board.charts[key] = new Chart(canvas, {
    type: "line",
    data: {
      labels: rows.map((r) => r[xKey]),
      datasets: [{ data: rows.map((r) => r[yKey]), borderColor: color, tension: 0.25, pointRadius: 2 }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { maxTicksLimit: 8 } } },
    },
  });
}

function renderTrainingCharts(training, tp) {
  if (!training?.epochs?.length) return;
  const epochs = training.epochs;
  const curGen = tp?.currentGeneration ?? epochs.at(-1)?.generation ?? 0;
  const genRows = epochs.filter((e) => e.generation === curGen);
  plotChart("chart-gen", "gen", genRows, "epochInGeneration", "valLoss", "#2563eb");
  plotChart("chart-global", "global", epochs, "globalEpoch", "valLoss", "#0ea5e9");
}

function buildTimelineFallback(tp, training) {
  const totalGen = tp.totalGenerations || 1;
  const epg = tp.epochsPerGeneration || 1;
  const curGen = tp.currentGeneration ?? 0;
  const epochs = training?.epochs || [];
  const rows = [];
  for (let g = 0; g < totalGen; g++) {
    const genEpochs = epochs.filter((e) => e.generation === g);
    rows.push({
      generation: g,
      startEpoch: g * epg,
      endEpoch: g * epg + epg,
      currentEpoch: g === curGen ? tp.currentEpoch : null,
      currentEpochIndex: g === curGen ? tp.currentEpoch : null,
      isCurrent: g === curGen,
      epochValues: genEpochs.map((e) => e.valAcc),
      actionExecuted: null,
    });
  }
  return rows;
}

function loadTrainingGeneration(gen, main) {
  const primary = Board.useSimplifiedGraph
    ? `graphs/gen_${gen}_simplified.pdf`
    : `graphs/gen_${gen}_full.pdf`;
  const fallbacks = [];
  for (let g = gen; g >= 0; g--) {
    if (Board.useSimplifiedGraph) {
      fallbacks.push(`graphs/gen_${g}_simulation_simplified.pdf`, `graphs/gen_${g}_simplified.pdf`);
    } else {
      fallbacks.push(`graphs/gen_${g}_simulation_full.pdf`, `graphs/gen_${g}_full.pdf`);
    }
  }
  fallbacks.push("graphs/start_simplified.pdf", "graphs/start_full.pdf");
  $("training-pdf-title").textContent = `Architecture Graph for Generation: ${gen} (PDF)`;
  renderPdfViewer("training", primary, fallbacks.filter((p) => p !== primary));
}

function renderGenerationTimeline(main, training) {
  const root = $("generation-timeline");
  if (!root) return;
  root.innerHTML = "";
  const tp = main.trainingParameters || {};
  const curGen = tp.currentGeneration ?? 0;
  if (Board.selectedTrainingGen == null) Board.selectedTrainingGen = curGen;
  const timeline = main.generationTimeline?.length
    ? main.generationTimeline
    : buildTimelineFallback(tp, training);

  for (const gen of timeline) {
    const g = gen.generation;
    const isCurrent = gen.isCurrent ?? g === curGen;
    const isSelected = g === Board.selectedTrainingGen;
    const values = gen.epochValues?.length
      ? gen.epochValues
      : Array.from({ length: tp.epochsPerGeneration || 5 }, (_, i) => 0.25 + (i % 3) * 0.08);
    const maxVal = Math.max(...values, 0.01);

    const card = document.createElement("button");
    card.type = "button";
    card.className = `generation-card${isCurrent ? " current" : ""}${isSelected ? " selected" : ""}`;

    const bars = values.map((v, idx) => {
      const height = Math.max(8, Math.round((v / maxVal) * 36));
      const active = isCurrent && gen.currentEpochIndex === idx;
      const marker = active ? '<span class="current-epoch-dot"></span>' : "";
      return `<span class="epoch-bar${active ? " active" : ""}" style="height:${height}px">${marker}</span>`;
    }).join("");

    const action = gen.actionExecuted;
    const actionBadge = action
      ? `<div class="action-badge" title="${action.action || ""}">⚡ ${action.shortLabel || shortActionLabel(action.action)}</div>`
      : "";

    card.innerHTML = `
      <div class="generation-title">Generation ${g + 1}</div>
      <div class="generation-range">(${gen.startEpoch} – ${gen.endEpoch})</div>
      <div class="generation-mini-chart">${bars || '<span class="epoch-bar placeholder"></span>'}</div>
      ${actionBadge}`;

    card.onclick = () => {
      Board.selectedTrainingGen = g;
      renderGenerationTimeline(main, training);
      loadTrainingGeneration(g, main);
    };
    root.appendChild(card);
  }
}

function architectureGraphCandidates(main) {
  const graphs = main.graphs || {};
  const gen = main.trainingParameters?.currentGeneration ?? 0;
  const primary = Board.useSimplifiedGraph
    ? (graphs.latestSimplified || `graphs/gen_${gen}_simplified.pdf`)
    : (graphs.latestFull || `graphs/gen_${gen}_full.pdf`);
  const fallbacks = [];
  for (let g = gen; g >= 0; g--) {
    if (Board.useSimplifiedGraph) {
      fallbacks.push(`graphs/gen_${g}_simulation_simplified.pdf`, `graphs/gen_${g}_simplified.pdf`);
    } else {
      fallbacks.push(`graphs/gen_${g}_simulation_full.pdf`, `graphs/gen_${g}_full.pdf`);
    }
  }
  fallbacks.push(
    Board.useSimplifiedGraph ? "graphs/start_simplified.pdf" : "graphs/start_full.pdf",
    Board.useSimplifiedGraph ? "graphs/start_full.pdf" : "graphs/start_simplified.pdf",
  );
  return [primary, ...fallbacks.filter((p) => p !== primary)];
}

export function renderTrainingBoard(main, training) {
  renderTrainingSidebar(main);
  renderLastSimulation(main.lastSimulation);
  renderTrainingCharts(training, main.trainingParameters);
  renderGenerationTimeline(main, training);

  const tp = main.trainingParameters || {};
  const curGen = tp.currentGeneration ?? 0;
  if (Board.selectedTrainingGen == null) Board.selectedTrainingGen = curGen;
  const viewGen = Board.selectedTrainingGen;
  const candidates = architectureGraphCandidates(main);
  const pdfPath = viewGen === curGen ? candidates[0] : (
    Board.useSimplifiedGraph ? `graphs/gen_${viewGen}_simplified.pdf` : `graphs/gen_${viewGen}_full.pdf`
  );
  $("training-pdf-title").textContent = `Architecture Graph for Generation: ${viewGen} (PDF)`;
  $("training-pdf-path").textContent = `File path: ${Board.experimentPath}/${pdfPath}`;
  renderPdfViewer("training", pdfPath, candidates.slice(1));
}

export function initTraining(onGotoSimulation) {
  Board.refreshHandlers.push(async (main, training) => {
    renderTrainingBoard(main, training);
  });

  const simplifiedToggle = $("simplified-toggle");
  if (simplifiedToggle) {
    simplifiedToggle.onchange = (e) => {
      Board.useSimplifiedGraph = e.target.checked;
      refreshAll();
    };
  }

  const gotoSim = $("goto-simulation");
  if (gotoSim) gotoSim.onclick = onGotoSimulation;
  bindPdfToolbar("training-pdf-toolbar", "training");
}
