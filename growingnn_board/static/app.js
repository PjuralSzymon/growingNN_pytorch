/** Application bootstrap: mount pages and wire navigation. */

import { mountPageHtml, showView } from "./shared/core.js";
import { initHome, loadRecent } from "./pages/home/home.js";
import { initTraining } from "./pages/training/training.js";
import { initSimulation, refreshSimulationBoard } from "./pages/simulation/simulation.js";

async function boot() {
  await Promise.all([
    mountPageHtml("home"),
    mountPageHtml("training"),
    mountPageHtml("simulation"),
  ]);

  initHome();
  initTraining(async () => {
    showView("simulation");
    try {
      await refreshSimulationBoard();
    } catch (err) {
      console.error("Failed to open simulation board:", err);
    }
  });
  initSimulation();

  showView("home");
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
