const root = document.documentElement;
const themeButton = document.querySelector(".theme-toggle");
const savedTheme = localStorage.getItem("growingnn-theme");
const preferredDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

root.dataset.theme = savedTheme || (preferredDark ? "dark" : "light");

themeButton?.addEventListener("click", () => {
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem("growingnn-theme", root.dataset.theme);
});

document.querySelector(".menu-toggle")?.addEventListener("click", () => {
  document.querySelector(".sidebar")?.classList.toggle("open");
});

document.querySelectorAll(".nav-group button").forEach((button) => {
  button.addEventListener("click", () => {
    button.setAttribute("aria-expanded", button.getAttribute("aria-expanded") !== "true");
  });
});

const dialog = document.querySelector(".search-dialog");
const searchInput = dialog?.querySelector("input");
const results = dialog?.querySelector(".search-results");
let searchIndex = [];

async function openSearch() {
  dialog?.classList.add("open");
  searchInput?.focus();
  if (!searchIndex.length) {
    try {
      searchIndex = await fetch("/search-index.json").then((response) => response.json());
    } catch {
      results.innerHTML = "<p>Search index could not be loaded.</p>";
    }
  }
}

function closeSearch() {
  dialog?.classList.remove("open");
  if (searchInput) searchInput.value = "";
}

document.querySelector(".search-trigger")?.addEventListener("click", openSearch);
dialog?.querySelector(".search-input button")?.addEventListener("click", closeSearch);
dialog?.addEventListener("click", (event) => {
  if (event.target === dialog) closeSearch();
});

document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    openSearch();
  }
  if (event.key === "Escape") closeSearch();
});

searchInput?.addEventListener("input", () => {
  const query = searchInput.value.trim().toLowerCase();
  if (!query) {
    results.innerHTML = "<p>Start typing to search every page.</p>";
    return;
  }
  const terms = query.split(/\s+/);
  const matches = searchIndex
    .map((page) => {
      const value = `${page.title} ${page.section} ${page.text}`.toLowerCase();
      const score = terms.reduce((total, term) => total + (value.includes(term) ? 1 : 0), 0);
      return { page, score };
    })
    .filter(({ score }) => score === terms.length)
    .slice(0, 12);

  results.innerHTML = matches.length
    ? matches
        .map(
          ({ page }) =>
            `<a class="search-result" href="${page.url}"><small>${page.section}</small><strong>${page.title}</strong><span>${page.text}</span></a>`,
        )
        .join("")
    : "<p>No pages match this search.</p>";
});

document.querySelectorAll("pre").forEach((block) => {
  const button = document.createElement("button");
  button.className = "copy-code";
  button.textContent = "Copy";
  button.addEventListener("click", async () => {
    await navigator.clipboard.writeText(block.innerText.replace(/^Copy/, ""));
    button.textContent = "Copied";
    setTimeout(() => (button.textContent = "Copy"), 1200);
  });
  block.style.position = "relative";
  button.style.cssText =
    "position:absolute;right:9px;top:9px;border:1px solid #3a3c49;border-radius:5px;background:#242631;color:#aaa;padding:3px 7px;font-size:9px;cursor:pointer";
  block.appendChild(button);
});

