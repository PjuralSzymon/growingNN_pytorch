/** PDF.js viewer shared by training and simulation pages. */

import { API, Board, $ } from "./lib.js?v=5";

export const pdfViewers = {
  training: {
    doc: null,
    page: 1,
    scale: 1.0,
    path: "",
    canvasId: "training-pdf-canvas",
    viewportId: "training-pdf-viewport",
    pageInfoId: "training-pdf-page-info",
    scaleInfoId: "training-pdf-scale",
  },
  simulation: {
    doc: null,
    page: 1,
    scale: 1.0,
    path: "",
    canvasId: "sim-pdf-canvas",
    viewportId: "sim-pdf-viewport",
    pageInfoId: "sim-pdf-page-info",
    scaleInfoId: "sim-pdf-scale",
  },
};

export function initPdfJs() {
  if (window.pdfjsLib) {
    pdfjsLib.GlobalWorkerOptions.workerSrc =
      "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
  }
}

function clearPdfError(viewportId) {
  const vp = $(viewportId);
  vp?.querySelector(".pdf-error")?.remove();
}

function showPdfError(viewportId, path) {
  const vp = $(viewportId);
  if (!vp) return;
  let err = vp.querySelector(".pdf-error");
  if (!err) {
    err = document.createElement("p");
    err.className = "pdf-error helper-text";
    err.style.padding = "16px";
    vp.appendChild(err);
  }
  err.textContent = `Graph PDF not available yet: ${path}`;
}

export async function renderPdfViewer(name, relativePath, fallbacks = []) {
  initPdfJs();
  const v = pdfViewers[name];
  if (!window.pdfjsLib || !relativePath) return;

  const paths = [relativePath, ...fallbacks];
  for (const path of paths) {
    if (v.path === path && v.doc) {
      return;
    }
    const url = `${API}/api/files/pdf?path=${encodeURIComponent(path)}`;
    try {
      v.doc = await pdfjsLib.getDocument(url).promise;
      v.path = path;
      v.page = 1;
      clearPdfError(v.viewportId);
      if (name === "training") {
        $("training-pdf-path").textContent = `File path: ${Board.experimentPath}/${path}`;
      }
      if (name === "simulation") {
        $("sim-pdf-path").textContent = `File path: ${Board.experimentPath}/${path}`;
      }
      await drawPdfPage(name);
      return;
    } catch (_) {
      continue;
    }
  }
  showPdfError(v.viewportId, relativePath);
}

export function clearPdfViewer(name, message = "Not available") {
  const v = pdfViewers[name];
  if (!v) return;
  v.doc = null;
  v.path = "";
  v.page = 1;
  const canvas = $(v.canvasId);
  if (canvas) {
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  const pageInfo = $(v.pageInfoId);
  if (pageInfo) pageInfo.textContent = "—";
  const scaleInfo = $(v.scaleInfoId);
  if (scaleInfo) scaleInfo.textContent = "—";
  showPdfError(v.viewportId, message);
}

async function drawPdfPage(name) {
  const v = pdfViewers[name];
  if (!v.doc) return;
  const page = await v.doc.getPage(v.page);
  const viewport = page.getViewport({ scale: v.scale });
  const canvas = $(v.canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  canvas.height = viewport.height;
  canvas.width = viewport.width;
  await page.render({ canvasContext: ctx, viewport }).promise;
  $(v.pageInfoId).textContent = `Page ${v.page} / ${v.doc.numPages}`;
  $(v.scaleInfoId).textContent = `${Math.round(v.scale * 100)}%`;
}

export function bindPdfToolbar(toolbarId, name) {
  const toolbar = $(toolbarId);
  if (!toolbar) return;
  toolbar.querySelectorAll("button").forEach((btn) => {
    btn.onclick = async () => {
      const v = pdfViewers[name];
      const action = btn.dataset.action;
      if (action === "zoom-in") {
        v.scale = Math.min(3, v.scale + 0.2);
        await drawPdfPage(name);
      }
      if (action === "zoom-out") {
        v.scale = Math.max(0.4, v.scale - 0.2);
        await drawPdfPage(name);
      }
      if (action === "prev" && v.page > 1) {
        v.page--;
        await drawPdfPage(name);
      }
      if (action === "next" && v.doc && v.page < v.doc.numPages) {
        v.page++;
        await drawPdfPage(name);
      }
      if (action === "download" && v.path) {
        window.open(`${API}/api/files/pdf?path=${encodeURIComponent(v.path)}`, "_blank");
      }
      if (action === "print" && v.doc) {
        const canvas = $(v.canvasId);
        const w = window.open("");
        w.document.write(`<img src="${canvas.toDataURL()}" onload="window.print();window.close()" />`);
      }
      if (action === "fullscreen") {
        $(v.viewportId).requestFullscreen?.();
      }
    };
  });
}
