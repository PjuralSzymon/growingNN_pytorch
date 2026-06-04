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

const TIMELINE_PX_PER_EPOCH = 6;
const ACTION_ICONS = { add: "+", delete: "−", remove: "−", other: "⚡" };

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function actionKind(actionStr) {
  const s = String(actionStr || "").toLowerCase();
  if (s.includes("delete")) return "delete";
  if (s.includes("add")) return "add";
  if (s.includes("remove")) return "remove";
  return "other";
}

function buildTimelineFallback(tp, training) {
  const totalGen = tp.totalGenerations || 1;
  const epg = tp.epochsPerGeneration || 1;
  const curGen = tp.currentGeneration ?? 0;
  const curEpoch = tp.currentEpoch ?? 0;
  const epochs = training?.epochs || [];
  const rows = [];
  for (let g = 0; g < totalGen; g++) {
    const genEpochs = epochs.filter((e) => e.generation === g);
    rows.push({
      generation: g,
      startEpoch: g * epg,
      endEpoch: g * epg + epg,
      currentEpoch: g === curGen ? curEpoch : null,
      isCurrent: g === curGen,
      epochValues: genEpochs.map((e) => e.valAcc),
      actionExecuted: null,
    });
  }
  return rows;
}

function getTimelineRows(main, training) {
  const tp = main.trainingParameters || {};
  const timeline = main.generationTimeline?.length
    ? main.generationTimeline
    : buildTimelineFallback(tp, training);
  const totalEpochs =
    timeline.at(-1)?.endEpoch ??
    tp.totalEpochs ??
    (tp.totalGenerations || 1) * (tp.epochsPerGeneration || 1);
  return { tp, timeline, totalEpochs };
}

/** Marker + badge share one position: global = generation start + epoch in generation. */
function resolveCurrentPosition(tp, timeline, training) {
  const curGen = tp.currentGeneration ?? 0;
  const epg = tp.epochsPerGeneration || 1;
  const row = timeline.find((g) => g.generation === curGen);
  const start = row?.startEpoch ?? curGen * epg;
  const end = row?.endEpoch ?? start + epg;

  let epochInGen = row?.currentEpoch ?? row?.currentEpochIndex ?? tp.currentEpoch;
  let globalEpoch = start + (epochInGen ?? 0);

  const lastMetric = training?.epochs?.at(-1);
  if (lastMetric?.generation === curGen) {
    if (lastMetric.globalEpoch != null) globalEpoch = lastMetric.globalEpoch;
    if (lastMetric.epochInGeneration != null) epochInGen = lastMetric.epochInGeneration;
  }

  if (epochInGen == null) epochInGen = Math.max(0, globalEpoch - start);
  globalEpoch = Math.min(end, Math.max(start, start + epochInGen));
  epochInGen = Math.max(0, Math.min(epg, globalEpoch - start));

  return { globalEpoch, epochInGen };
}

function epochLeftPct(globalEpoch, totalEpochs) {
  if (totalEpochs <= 0) return 0;
  return Math.min(100, Math.max(0, (globalEpoch / totalEpochs) * 100));
}

let timelineMain = null;
let timelinePopoverBound = false;

function hideTimelinePopover() {
  const pop = $("timeline-action-popover");
  if (pop) pop.classList.add("hidden");
}

function showTimelineActionPopover(action, anchor) {
  const pop = $("timeline-action-popover");
  if (!pop || !anchor) return;
  const label = action.shortLabel || shortActionLabel(action.action);
  const epochNote =
    action.atGlobalEpoch != null ? `Global epoch ${action.atGlobalEpoch}` : "";
  pop.innerHTML = `
    <h4>${escapeHtml(label)}</h4>
    <p>${escapeHtml(action.action || label)}</p>
    ${epochNote ? `<div class="popover-meta">${escapeHtml(epochNote)}</div>` : ""}`;
  pop.classList.remove("hidden");
  const rect = anchor.getBoundingClientRect();
  const popRect = pop.getBoundingClientRect();
  pop.style.left = `${Math.min(window.innerWidth - popRect.width - 12, Math.max(8, rect.left))}px`;
  let top = rect.top - popRect.height - 8;
  if (top < 8) top = rect.bottom + 8;
  pop.style.top = `${top}px`;
}

function bindTimelineEvents(root) {
  if (!root || root.dataset.bound === "1") return;
  root.dataset.bound = "1";
  root.addEventListener("click", (e) => {
    const marker = e.target.closest(".timeline-action-marker");
    if (marker) {
      e.stopPropagation();
      try {
        const action = JSON.parse(decodeURIComponent(marker.getAttribute("data-action") || "%7B%7D"));
        showTimelineActionPopover(action, marker);
      } catch (_) { /* */ }
      return;
    }
    const label = e.target.closest(".timeline-gen-label");
    if (!label || !timelineMain) return;
    const g = Number(label.dataset.generation);
    if (Number.isNaN(g)) return;
    Board.selectedTrainingGen = g;
    renderTrainingTimeline(timelineMain, timelineTraining);
    loadTrainingGeneration(g, timelineMain);
  });
  if (!timelinePopoverBound) {
    timelinePopoverBound = true;
    document.addEventListener("click", (e) => {
      if (e.target.closest(".timeline-action-marker") || e.target.closest("#timeline-action-popover")) return;
      hideTimelinePopover();
    });
  }
}

let timelineTraining = null;

function renderTrainingTimeline(main, training) {
  const root = $("training-timeline");
  if (!root) return;
  timelineMain = main;
  timelineTraining = training;
  hideTimelinePopover();

  const { tp, timeline, totalEpochs } = getTimelineRows(main, training);
  const curGen = tp.currentGeneration ?? 0;
  if (Board.selectedTrainingGen == null) Board.selectedTrainingGen = curGen;
  const selectedGen = Board.selectedTrainingGen;
  const { globalEpoch, epochInGen } = resolveCurrentPosition(tp, timeline, training);

  const minWidth = Math.max(totalEpochs * TIMELINE_PX_PER_EPOCH, 720);
  root.style.width = `${minWidth}px`;

  const labels = timeline
    .map((gen) => {
      const g = gen.generation;
      const span = Math.max(1, (gen.endEpoch ?? 0) - (gen.startEpoch ?? 0));
      const widthPct = (span / totalEpochs) * 100;
      const active = g === selectedGen;
      const displayGen = g + 1;
      return `<button type="button" class="timeline-gen-label${active ? " active" : ""}" data-generation="${g}" style="width:${widthPct}%">Generation ${displayGen}<span class="gen-range">(${gen.startEpoch} - ${gen.endEpoch})</span></button>`;
    })
    .join("");

  const segments = timeline
    .map((gen) => {
      const span = Math.max(1, gen.endEpoch - gen.startEpoch);
      const widthPct = (span / totalEpochs) * 100;
      const lineClass = gen.generation % 2 === 0 ? "line-epoch" : "line-generation";
      return `<div class="timeline-segment ${lineClass}" style="width:${widthPct}%"></div>`;
    })
    .join("");

  const ticks = Array.from({ length: totalEpochs + 1 }, (_, e) => {
    const major = e % 10 === 0;
    return `<span class="epoch-tick${major ? " major" : ""}" style="left:${epochLeftPct(e, totalEpochs)}%"></span>`;
  }).join("");

  const actions = timeline
    .filter((gen) => gen.actionExecuted)
    .map((gen) => {
      const action = gen.actionExecuted;
      const at =
        action.atGlobalEpoch ??
        gen.endEpoch - 1;
      const kind = actionKind(action.action);
      const label = action.shortLabel || shortActionLabel(action.action);
      const payload = encodeURIComponent(JSON.stringify({ ...action, shortLabel: label }));
      return `<button type="button" class="timeline-action-marker action-${kind}" style="left:${epochLeftPct(at, totalEpochs)}%" title="${escapeHtml(label)}" data-action="${payload}">${ACTION_ICONS[kind]}</button>`;
    })
    .join("");

  const curLeft = epochLeftPct(globalEpoch, totalEpochs);
  const scale = Array.from({ length: Math.floor(totalEpochs / 10) + 1 }, (_, i) => {
    const e = i * 10;
    return `<span class="scale-label" style="left:${epochLeftPct(e, totalEpochs)}%">${e}</span>`;
  }).join("");

  root.innerHTML = `
    <div class="timeline-labels">${labels}</div>
    <div class="timeline-track-area">
      <div class="timeline-track">${segments}</div>
      <div class="timeline-epoch-ticks">${ticks}</div>
      ${actions}
      <div class="timeline-current" style="left:${curLeft}%">
        <span class="current-dot"></span>
        <span class="current-stem"></span>
        <span class="current-badge">Epoch ${epochInGen}</span>
      </div>
    </div>
    <div class="timeline-scale">${scale}</div>`;

  bindTimelineEvents(root);
  const viewport = root.closest(".timeline-viewport");
  if (viewport) {
    requestAnimationFrame(() => {
      const marker = root.querySelector(".timeline-current");
      if (marker) {
        const left = (curLeft / 100) * root.offsetWidth;
        viewport.scrollLeft = Math.max(0, left - viewport.clientWidth * 0.4);
      }
    });
  }
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
  renderTrainingTimeline(main, training);

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
