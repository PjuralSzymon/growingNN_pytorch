/** View routing and browser history (hash-based SPA navigation). */

const VALID_VIEWS = new Set(["home", "training", "simulation"]);

export function showView(name) {
  document.getElementById("view-home")?.classList.toggle("hidden", name !== "home");
  document.getElementById("view-training")?.classList.toggle("hidden", name !== "training");
  document.getElementById("view-simulation")?.classList.toggle("hidden", name !== "simulation");
}

export function viewFromLocation() {
  const hash = location.hash.replace(/^#\/?/, "");
  return VALID_VIEWS.has(hash) ? hash : "home";
}

export function navigateTo(view, { replace = false } = {}) {
  if (!VALID_VIEWS.has(view)) view = "home";
  showView(view);
  const url = view === "home" ? "/" : `/#/${view}`;
  const state = { view };
  if (replace) history.replaceState(state, "", url);
  else history.pushState(state, "", url);
}

export function initNavigation({ onTraining, onSimulation, onHome } = {}) {
  window.addEventListener("popstate", async () => {
    const { Board, startPoll, stopPoll } = await import("./lib.js?v=5");
    const view = history.state?.view || viewFromLocation();
    showView(view);
    if (view === "home") {
      stopPoll();
      if (onHome) await onHome();
      return;
    }
    if (!Board.experimentPath) {
      navigateTo("home", { replace: true });
      if (onHome) await onHome();
      return;
    }
    if (view === "training") {
      startPoll();
      if (onTraining) await onTraining();
      return;
    }
    if (view === "simulation" && onSimulation) await onSimulation();
  });
}
