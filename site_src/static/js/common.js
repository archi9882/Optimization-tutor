// Theme toggle + KaTeX rendering, shared by every page.

const root = document.documentElement;

function currentTheme() {
  if (root.dataset.theme) return root.dataset.theme;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

document.querySelector(".theme-toggle")?.addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  root.dataset.theme = next;
  try { localStorage.setItem("ot-theme", next); } catch (e) { /* no storage */ }
  window.dispatchEvent(new CustomEvent("ot-theme"));
});

matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", () => window.dispatchEvent(new CustomEvent("ot-theme")));

/** Typeset $...$ math inside `el` (default: <main>); false if KaTeX is not loaded yet. */
export function renderMath(el = document.querySelector("main")) {
  if (!window.renderMathInElement) return false;
  window.renderMathInElement(el, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
    ],
    throwOnError: false,
  });
  return true;
}

if (!renderMath()) window.addEventListener("load", () => renderMath(), { once: true });
