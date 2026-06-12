/** Application bootstrap: mount pages and wire navigation. */

import { api, mountPageHtml, resetSnapshots, startPoll } from "./shared/lib.js?v=5";
import {
  initNavigation,
  navigateTo,
  showView,
  viewFromLocation,
} from "./shared/navigation.js?v=5";
import { initHome, loadRecent } from "./pages/home/home.js?v=5";
import { initTraining, renderTrainingBoard } from "./pages/training/training.js?v=5";
import { initSimulation, refreshSimulationBoard } from "./pages/simulation/simulation.js?v=5";

async function restoreTrainingView() {
  try {
    const main = await api("/api/experiment/main");
    let training = null;
    try {
      training = await api("/api/experiment/training");
    } catch (_) { /* */ }
    renderTrainingBoard(main, training);
  } catch (_) { /* keep last UI */ }
}

async function boot() {
  await Promise.all([
    mountPageHtml("home"),
    mountPageHtml("training"),
    mountPageHtml("simulation"),
  ]);

  initHome();
  initTraining(async () => {
    navigateTo("simulation");
    try {
      await refreshSimulationBoard();
    } catch (err) {
      console.error("Failed to open simulation board:", err);
    }
  });
  initSimulation();

  initNavigation({
    onHome: loadRecent,
    onTraining: restoreTrainingView,
    onSimulation: refreshSimulationBoard,
  });

  const initialView = viewFromLocation();
  if (initialView === "home") {
    showView("home");
    history.replaceState({ view: "home" }, "", "/");
    await loadRecent();
    return;
  }

  showView(initialView);
  history.replaceState({ view: initialView }, "", initialView === "home" ? "/" : `/#/${initialView}`);
  await loadRecent();
}

boot().catch((err) => {
  console.error("GrowingNN Board failed to start:", err);
  const status = document.getElementById("load-status");
  const message = `Failed to load UI: ${err.message}`;
  if (status) {
    status.textContent = message;
    status.classList.add("warn");
    return;
  }
  const home = document.getElementById("view-home");
  if (home) {
    home.innerHTML = `<div class="start-card"><p class="load-status warn">${message}</p></div>`;
  }
});
