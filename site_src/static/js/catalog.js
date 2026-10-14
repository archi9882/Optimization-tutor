// Home page: filter the algorithm cards by search text and chips.
// Within one chip group the selections are OR-ed; across groups AND-ed.

const cards = JSON.parse(document.getElementById("cards-data").textContent);
const byId = Object.fromEntries(cards.map((c) => [c.id, c]));
const q = document.getElementById("q");
const onlyReady = document.getElementById("only-ready");
const count = document.getElementById("count");
const empty = document.getElementById("empty");
const selected = { problem: new Set(), info: new Set(), domains: new Set() };

function matches(c) {
  const text = q.value.trim().toLowerCase();
  if (text && !`${c.name} ${c.en} ${c.summary}`.toLowerCase().includes(text)) return false;
  if (onlyReady.checked && !c.status.page) return false;
  for (const [group, values] of Object.entries(selected)) {
    if (!values.size) continue;
    const field = c[group];
    const have = Array.isArray(field) ? field : [field];
    if (!have.some((v) => values.has(v))) return false;
  }
  return true;
}

function apply() {
  let shown = 0;
  for (const el of document.querySelectorAll(".card")) {
    const ok = matches(byId[el.dataset.id]);
    el.hidden = !ok;
    shown += ok;
  }
  for (const fam of document.querySelectorAll(".family")) {
    fam.hidden = !fam.querySelector(".card:not([hidden])");
  }
  count.textContent = `顯示 ${shown} / ${cards.length}`;
  empty.hidden = shown > 0;
}

for (const row of document.querySelectorAll(".chip-row")) {
  const group = row.dataset.group;
  row.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    const on = chip.getAttribute("aria-pressed") !== "true";
    chip.setAttribute("aria-pressed", String(on));
    on ? selected[group].add(chip.dataset.value) : selected[group].delete(chip.dataset.value);
    apply();
  });
}
q.addEventListener("input", apply);
onlyReady.addEventListener("change", apply);
apply();
