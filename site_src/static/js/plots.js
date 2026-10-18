// Shared canvas renderers.
//   ContourPlot     – renderer ①: 2D level sets + optimizer trajectories
//                     (grid.constraints: shaded infeasible area + g = 0 curves);
//                     with setRegion() also renderer ②: feasible polygon,
//                     constraint lines and iso-objective lines for LP/QP
//   ConvergencePlot – f(x_k) − f* on a log axis, one line per optimizer
// Both read colors from CSS custom properties so they follow the theme.

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const seriesColor = (i) => css(`--s${(i % 8) + 1}`);

function fmt(v) {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return v.toFixed(a < 1 ? 4 : a < 100 ? 3 : 1);
}
export { fmt };

/** True on devices without hover (phones, tablets): hints say "點一下" instead of "滑鼠移到". */
export const touchUI = () => typeof matchMedia === "function" && matchMedia("(hover: none)").matches;

function setupCanvas(canvas, cssW, cssH) {
  const dpr = window.devicePixelRatio || 1;
  canvas.style.height = `${cssH}px`;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

// Marching squares: returns line segments [[x1,y1,x2,y2], ...] in grid index space.
function isoSegments(t, level) {
  const segs = [];
  const ny = t.length, nx = t[0].length;
  const lerp = (a, b) => (level - a) / (b - a);
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = t[j][i], b = t[j][i + 1], c = t[j + 1][i + 1], d = t[j + 1][i];
      const idx = (a > level) | ((b > level) << 1) | ((c > level) << 2) | ((d > level) << 3);
      if (idx === 0 || idx === 15) continue;
      const top = [i + lerp(a, b), j], right = [i + 1, j + lerp(b, c)];
      const bottom = [i + lerp(d, c), j + 1], left = [i, j + lerp(a, d)];
      const add = (p, q) => segs.push([p[0], p[1], q[0], q[1]]);
      switch (idx) {
        case 1: case 14: add(left, top); break;
        case 2: case 13: add(top, right); break;
        case 3: case 12: add(left, right); break;
        case 4: case 11: add(right, bottom); break;
        case 6: case 9: add(top, bottom); break;
        case 7: case 8: add(left, bottom); break;
        case 5: add(left, top); add(right, bottom); break;
        case 10: add(top, right); add(left, bottom); break;
      }
    }
  }
  return segs;
}

export class ContourPlot {
  constructor(container, { maxHeight = 520 } = {}) {
    this.wrap = document.createElement("div");
    this.wrap.className = "plot";
    this.canvas = document.createElement("canvas");
    this.canvas.setAttribute("role", "img");
    this.tip = document.createElement("div");
    this.tip.className = "tooltip";
    this.tip.hidden = true;
    this.wrap.append(this.canvas, this.tip);
    container.append(this.wrap);
    this.maxHeight = maxHeight;
    this.series = [];
    this.frame = Infinity;
    this.pickCb = null;
    this.resizeCb = null;
    this.pointer = "mouse";
    this.canvas.addEventListener("pointerdown", (e) => { this.pointer = e.pointerType || "mouse"; });
    this.canvas.addEventListener("pointermove", (e) => {
      if (e.pointerType !== "mouse") return;   // touch drags scroll the page
      this.pointer = "mouse";
      this.hover(e);
    });
    // Browsers send a compatibility mouseleave after a tap: keep the tapped tooltip.
    this.canvas.addEventListener("mouseleave", () => { if (this.pointer === "mouse") this.tip.hidden = true; });
    this.canvas.addEventListener("click", (e) => {
      if (!this.grid) return;
      // Touch has no hover: a tap on (or next to) a path point shows its tooltip.
      if (this.pointer !== "mouse") {
        if (this.showPoint(e.offsetX, e.offsetY, 18)) return;
        if (!this.pickCb) { this.hover(e); return; }
        this.tip.hidden = true;
      }
      if (!this.pickCb) return;
      const [x, y] = this.toData(e.offsetX, e.offsetY);
      this.pickCb([x, y]);
    });
    new ResizeObserver(() => this.resize()).observe(this.wrap);
    window.addEventListener("ot-theme", () => { this.cache = null; this.draw(); });
  }

  onPick(cb) { this.pickCb = cb; this.canvas.style.cursor = "crosshair"; }
  /** cb(heightPx) after every resize (the lab matches the convergence plot to it on phones). */
  onResize(cb) { this.resizeCb = cb; if (this.h) cb(this.h); }

  setGrid(grid) {
    this.grid = grid;
    const z = grid.z.flat();
    const zmin = Math.min(...z), zmax = Math.max(...z);
    const eps = (zmax - zmin) * 1e-5 + 1e-12;
    // Linear objectives: evenly spaced (parallel, equidistant) iso-lines.
    this.t = grid.linear ? grid.z : grid.z.map((row) => row.map((v) => Math.log10(v - zmin + eps)));
    const sorted = this.t.flat().sort((a, b) => a - b);
    const lo = sorted[Math.floor(sorted.length * 0.01)], hi = sorted[Math.floor(sorted.length * 0.99)];
    const nLev = grid.linear ? 12 : 16;
    this.levels = Array.from({ length: nLev }, (_, k) => lo + ((hi - lo) * (k + 0.5)) / nLev);
    this.segs = this.levels.map((lv) => isoSegments(this.t, lv));
    this.canvas.setAttribute("aria-label", `${grid.name} 的等高線圖與最佳化路徑`);
    this.resize();
  }

  setSeries(series) { this.series = series; this.draw(); }

  // Renderer ②: {polygon, constraints: [{a, b, label}], target, lattice}.
  setRegion(region) { this.region = region; this.cache = null; this.draw(); }

  // Per-step layer on top of the region: {polygon, cuts: [{a, b, label}],
  // marks: [{x, kind: "lp" | "best", label}]}. Redrawn every frame.
  setOverlay(ov) { this.overlay = ov; this.draw(); }

  inPolygon(poly, x, y) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }

  onBoundary(poly, x, y) {
    return poly.some(([xi, yi], i) => {
      const [xj, yj] = poly[(i + 1) % poly.length];
      const cross = (xj - xi) * (y - yi) - (yj - yi) * (x - xi);
      const dot = (x - xi) * (xj - xi) + (y - yi) * (yj - yi);
      return Math.abs(cross) < 1e-7 && dot >= -1e-9 && dot <= (xj - xi) ** 2 + (yj - yi) ** 2 + 1e-9;
    });
  }

  feasible(poly, x, y) { return poly.length > 0 && (this.inPolygon(poly, x, y) || this.onBoundary(poly, x, y)); }

  drawOverlay(c) {
    const ov = this.overlay;
    if (ov.polygon && ov.polygon.length) {
      c.save();
      c.beginPath();
      ov.polygon.forEach(([x, y], i) => { const [px, py] = this.toPx(x, y); i ? c.lineTo(px, py) : c.moveTo(px, py); });
      c.closePath();
      c.globalAlpha = 0.22; c.fillStyle = css("--accent"); c.fill();
      c.globalAlpha = 1; c.lineWidth = 2; c.strokeStyle = css("--accent"); c.stroke();
      c.restore();
      // Integer points the current sub-problem still contains.
      if (this.region?.lattice) {
        const [x0, x1, y0, y1] = this.grid.bounds;
        c.fillStyle = css("--accent-ink");
        for (let i = Math.ceil(x0); i <= x1; i++) for (let j = Math.ceil(y0); j <= y1; j++) {
          if (!this.feasible(ov.polygon, i, j)) continue;
          const [px, py] = this.toPx(i, j);
          c.beginPath(); c.arc(px, py, 3.5, 0, 7); c.fill();
        }
      }
    }
    for (const cut of ov.cuts || []) {
      const seg = this.clipLine(cut.a, cut.b);
      if (!seg) continue;
      const [p, q] = seg.map(([x, y]) => this.toPx(x, y));
      c.save();
      c.strokeStyle = css("--s2"); c.lineWidth = 2;
      c.beginPath(); c.moveTo(...p); c.lineTo(...q); c.stroke();
      // Label partway along the cut, staggered so successive cuts do not collide.
      const k = (ov.cuts || []).indexOf(cut), u = 0.3 + 0.12 * k;
      const lx = Math.min(this.w - 34, Math.max(4, p[0] + (q[0] - p[0]) * u + 6));
      const ly = Math.min(this.h - 16, Math.max(4, p[1] + (q[1] - p[1]) * u));
      c.font = `12px ${css("--font")}`;
      c.fillStyle = css("--surface"); c.globalAlpha = 0.85;
      c.fillRect(lx - 2, ly - 1, c.measureText(cut.label).width + 4, 15);
      c.globalAlpha = 1; c.fillStyle = css("--s2");
      c.textAlign = "left"; c.textBaseline = "top";
      c.fillText(cut.label, lx, ly);
      c.restore();
    }
    for (const m of ov.marks || []) {
      if (!m.x) continue;
      const [px, py] = this.toPx(...m.x);
      c.save();
      if (m.kind === "best") {
        c.strokeStyle = css("--good"); c.lineWidth = 2.5;
        c.beginPath(); c.arc(px, py, 8, 0, 7); c.stroke();
      } else {
        c.fillStyle = css("--s1"); c.strokeStyle = css("--surface"); c.lineWidth = 2;
        c.beginPath(); c.arc(px, py, 5.5, 0, 7); c.fill(); c.stroke();
      }
      if (m.label) {
        c.font = `12px ${css("--font")}`; c.fillStyle = m.kind === "best" ? css("--good") : css("--ink");
        c.textAlign = "left"; c.textBaseline = "bottom";
        c.fillText(m.label, Math.min(px + 10, this.w - 60), py - 6);
      }
      c.restore();
    }
  }

  // Segment of the line a·x = b inside the view box, or null.
  clipLine(a, b) {
    const [x0, x1, y0, y1] = this.grid.bounds;
    const pts = [];
    if (Math.abs(a[1]) > 1e-12) for (const x of [x0, x1]) pts.push([x, (b - a[0] * x) / a[1]]);
    if (Math.abs(a[0]) > 1e-12) for (const y of [y0, y1]) pts.push([(b - a[1] * y) / a[0], y]);
    const inside = pts.filter(([x, y]) => x >= x0 - 1e-9 && x <= x1 + 1e-9 && y >= y0 - 1e-9 && y <= y1 + 1e-9);
    if (inside.length < 2) return null;
    inside.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
    return [inside[0], inside[inside.length - 1]];
  }

  drawRegion(c) {
    const r = this.region;
    // Dim everything outside the feasible polygon.
    if (r.polygon.length) {
      c.save();
      c.beginPath();
      c.rect(0, 0, this.w, this.h);
      r.polygon.forEach(([x, y], i) => { const [px, py] = this.toPx(x, y); i ? c.lineTo(px, py) : c.moveTo(px, py); });
      c.closePath();
      c.fillStyle = css("--surface");
      c.globalAlpha = 0.72;
      c.fill("evenodd");
      c.globalAlpha = 0.10;
      c.fillStyle = css("--accent");
      c.beginPath();
      r.polygon.forEach(([x, y], i) => { const [px, py] = this.toPx(x, y); i ? c.lineTo(px, py) : c.moveTo(px, py); });
      c.closePath();
      c.fill();
      c.restore();
    }
    // Integer lattice: feasible points dark, infeasible ones faint.
    if (r.lattice) {
      const [x0, x1, y0, y1] = this.grid.bounds;
      c.save();
      for (let i = Math.ceil(x0); i <= x1; i++) for (let j = Math.ceil(y0); j <= y1; j++) {
        const ok = this.feasible(r.polygon, i, j);
        const [px, py] = this.toPx(i, j);
        c.fillStyle = ok ? css("--ink-2") : css("--axis");
        c.beginPath(); c.arc(px, py, ok ? 3 : 2, 0, 7); c.fill();
      }
      c.restore();
    }
    // Axes x1 = 0 and x2 = 0.
    c.save();
    c.strokeStyle = css("--axis");
    c.lineWidth = 1;
    const [ax, ay] = this.toPx(0, 0);
    c.beginPath(); c.moveTo(0, ay); c.lineTo(this.w, ay); c.moveTo(ax, 0); c.lineTo(ax, this.h); c.stroke();
    c.font = `11px ${css("--font")}`;
    c.fillStyle = css("--muted");
    if (r.var_labels) {
      c.textAlign = "right"; c.textBaseline = "bottom";
      c.fillText(r.var_labels[0], this.w - 4, ay - 3);
      c.textAlign = "left"; c.textBaseline = "top";
      c.fillText(r.var_labels[1], ax + 4, 3);
    }
    c.restore();
    // Constraint lines with their resource name.
    c.save();
    c.font = `12px ${css("--font")}`;
    r.constraints.forEach((con, k) => {
      const seg = this.clipLine(con.a, con.b);
      if (!seg) return;
      const [p, q] = seg.map(([x, y]) => this.toPx(x, y));
      c.strokeStyle = css("--ink-2");
      c.lineWidth = 1.25;
      c.setLineDash([6, 4]);
      c.beginPath(); c.moveTo(...p); c.lineTo(...q); c.stroke();
      c.setLineDash([]);
      // Label near the upper end of the line, nudged inside the canvas.
      const top = p[1] < q[1] ? p : q, other = top === p ? q : p;
      const u = 0.12 + 0.06 * k;
      let lx = top[0] + (other[0] - top[0]) * u, ly = top[1] + (other[1] - top[1]) * u;
      const name = con.label.split("：")[0];
      const tw = c.measureText(name).width;
      lx = Math.max(4, Math.min(this.w - tw - 8, lx + 6));
      ly = Math.max(14, Math.min(this.h - 6, ly));
      c.fillStyle = css("--surface");
      c.globalAlpha = 0.85;
      c.fillRect(lx - 2, ly - 12, tw + 4, 15);
      c.globalAlpha = 1;
      c.fillStyle = css("--ink-2");
      c.textAlign = "left"; c.textBaseline = "alphabetic";
      c.fillText(name, lx, ly);
    });
    c.restore();
    if (r.target) {
      const [tx, ty] = this.toPx(...r.target);
      c.save();
      c.strokeStyle = css("--ink-2");
      c.lineWidth = 1.5;
      c.beginPath(); c.arc(tx, ty, 6, 0, 7); c.moveTo(tx - 9, ty); c.lineTo(tx + 9, ty); c.moveTo(tx, ty - 9); c.lineTo(tx, ty + 9); c.stroke();
      c.font = `12px ${css("--font")}`;
      c.fillStyle = css("--ink-2");
      c.textAlign = "right"; c.textBaseline = "middle";
      c.fillText("目標", tx - 10, ty);
      c.restore();
    }
  }
  // Constrained problems: grid.constraints = [{kind, label, z}] with g sampled
  // on the grid (g <= 0 feasible, or g = 0). Wash out (per pixel, bilinear g)
  // where any inequality is violated, then draw each g = 0 curve with
  // marching squares.
  drawConstraints(c) {
    const g = this.grid, cons = g.constraints, nx = g.x.length, ny = g.y.length;
    const ineq = cons.filter((k) => k.kind !== "eq");
    // Bilinear value of a sampled constraint at fractional grid indices.
    const at = (z, fi, fj) => {
      fi = Math.min(nx - 1.000001, Math.max(0, fi)); fj = Math.min(ny - 1.000001, Math.max(0, fj));
      const i = Math.floor(fi), j = Math.floor(fj), u = fi - i, v = fj - j;
      return (1 - u) * (1 - v) * z[j][i] + u * (1 - v) * z[j][i + 1] + u * v * z[j + 1][i + 1] + (1 - u) * v * z[j + 1][i];
    };
    const rgbOf = (ctx, name) => {
      ctx.fillStyle = css(name);
      const hex = ctx.fillStyle; // normalised "#rrggbb"
      return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
    };
    if (ineq.length) {
      // Infeasible area: a wash that fades the contours plus diagonal hatching.
      const W = this.canvas.width, H = this.canvas.height, dpr = W / this.w;
      const off = document.createElement("canvas");
      off.width = W; off.height = H;
      const octx = off.getContext("2d");
      const wash = rgbOf(octx, "--surface"), hatch = rgbOf(octx, "--muted");
      const period = Math.round(9 * dpr), thick = Math.max(1, Math.round(1.2 * dpr));
      const img = octx.createImageData(W, H);
      for (let py = 0; py < H; py++) {
        const fj = (1 - py / H) * (ny - 1);
        for (let px = 0; px < W; px++) {
          const fi = (px / W) * (nx - 1);
          if (!ineq.some(({ z }) => at(z, fi, fj) > 0)) continue;
          const o = 4 * (py * W + px), line = (px + py) % period < thick;
          const col = line ? hatch : wash;
          img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2];
          img.data[o + 3] = line ? 170 : 200;
        }
      }
      octx.putImageData(img, 0, 0);
      c.save();
      c.setTransform(1, 0, 0, 1, 0, 0);
      c.drawImage(off, 0, 0);
      c.restore();
    }
    const sx = this.w / (nx - 1), sy = this.h / (ny - 1);
    c.save();
    c.font = `12px ${css("--font")}`;
    const box = (text, x, y, color) => {
      // Text centred at (x, y) on a translucent background, kept inside the canvas.
      const tw = c.measureText(text).width;
      x = Math.max(4 + tw / 2, Math.min(this.w - 4 - tw / 2, x));
      y = Math.max(10, Math.min(this.h - 10, y));
      c.fillStyle = css("--surface"); c.globalAlpha = 0.9;
      c.fillRect(x - tw / 2 - 3, y - 8, tw + 6, 16);
      c.globalAlpha = 1; c.fillStyle = color;
      c.textAlign = "center"; c.textBaseline = "middle";
      c.fillText(text, x, y);
    };
    cons.forEach((k, n) => {
      const segs = isoSegments(k.z, 0);
      c.strokeStyle = k.kind === "eq" ? css("--accent") : css("--ink-2");
      c.lineWidth = k.kind === "eq" ? 2 : 1.5;
      c.beginPath();
      for (const [a, b, p, q] of segs) { c.moveTo(a * sx, this.h - b * sy); c.lineTo(p * sx, this.h - q * sy); }
      c.stroke();
      if (!segs.length || !k.label) return;
      // Label the part of the curve that really bounds the feasible set (all
      // other inequalities hold there), on the infeasible side of the curve.
      const others = ineq.filter((o) => o !== k);
      const mids = segs.map(([a, b, p, q]) => [(a + p) / 2, (b + q) / 2])
        .filter(([fi, fj]) => others.every((o) => at(o.z, fi, fj) <= 1e-9));
      const pool = mids.length ? mids : segs.map(([a, b, p, q]) => [(a + p) / 2, (b + q) / 2]);
      const [fi, fj] = pool[Math.floor(pool.length * (0.35 + 0.1 * (n % 3)))];
      let gx = (at(k.z, fi + 0.5, fj) - at(k.z, fi - 0.5, fj)) / sx;   // outward normal in px
      let gy = -(at(k.z, fi, fj + 0.5) - at(k.z, fi, fj - 0.5)) / sy;
      const gn = Math.hypot(gx, gy) || 1;
      gx /= gn; gy /= gn;
      const tw = c.measureText(k.label).width;
      const d = 10 + Math.abs(gx) * tw / 2;
      box(k.label, fi * sx + gx * d, this.h - fj * sy + gy * d, css("--ink-2"));
    });
    if (g.unconstrained) {
      // Where f alone would go: the constraint is what stops it.
      const [px, py] = this.toPx(...g.unconstrained);
      c.strokeStyle = css("--muted"); c.lineWidth = 1.5;
      c.beginPath(); c.arc(px, py, 5, 0, 7); c.stroke();
      const text = "無限制最小值", tw = c.measureText(text).width;
      const right = px + 12 + tw + 4 <= this.w;
      box(text, right ? px + 12 + tw / 2 : px - 12 - tw / 2, py, css("--muted"));
    }
    c.restore();
  }

  setFrame(k) { this.frame = k; this.draw(); }

  resize() {
    if (!this.grid) return;
    const [x0, x1, y0, y1] = this.grid.bounds;
    const w = this.wrap.clientWidth;
    if (!w) return;
    // Equal axis scaling: gradients must look perpendicular to level sets.
    let h = (w * (y1 - y0)) / (x1 - x0);
    this.w = w;
    this.h = Math.min(h, this.maxHeight);
    if (h > this.maxHeight) this.w = (this.maxHeight * (x1 - x0)) / (y1 - y0);
    this.canvas.style.width = `${this.w}px`;
    this.ctx = setupCanvas(this.canvas, this.w, this.h);
    this.cache = null;
    this.draw();
    if (this.resizeCb) this.resizeCb(this.h);
  }

  toPx(x, y) {
    const [x0, x1, y0, y1] = this.grid.bounds;
    return [((x - x0) / (x1 - x0)) * this.w, this.h - ((y - y0) / (y1 - y0)) * this.h];
  }

  toData(px, py) {
    const [x0, x1, y0, y1] = this.grid.bounds;
    return [x0 + (px / this.w) * (x1 - x0), y0 + ((this.h - py) / this.h) * (y1 - y0)];
  }

  drawContours() {
    const off = document.createElement("canvas");
    const dpr = window.devicePixelRatio || 1;
    off.width = this.canvas.width; off.height = this.canvas.height;
    const c = off.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.fillStyle = css("--surface");
    c.fillRect(0, 0, this.w, this.h);
    const nx = this.grid.x.length, ny = this.grid.y.length;
    const sx = this.w / (nx - 1), sy = this.h / (ny - 1);
    c.lineWidth = 1;
    this.segs.forEach((segs, k) => {
      c.strokeStyle = css("--contour");
      c.globalAlpha = 0.45 + (0.55 * (this.segs.length - k)) / this.segs.length;
      c.beginPath();
      for (const [a, b, p, q] of segs) {
        c.moveTo(a * sx, this.h - b * sy);
        c.lineTo(p * sx, this.h - q * sy);
      }
      c.stroke();
    });
    c.globalAlpha = 1;
    if (this.region) this.drawRegion(c);
    if (this.grid.constraints) this.drawConstraints(c);
    c.strokeStyle = css("--ink-2");
    c.lineWidth = 1.5;
    for (const m of this.grid.minima) {
      const [px, py] = this.toPx(m[0], m[1]);
      c.beginPath();
      c.moveTo(px - 5, py); c.lineTo(px + 5, py);
      c.moveTo(px, py - 5); c.lineTo(px, py + 5);
      c.stroke();
    }
    this.cache = off;
  }

  draw() {
    if (!this.ctx || !this.grid) return;
    if (!this.cache) this.drawContours();
    const c = this.ctx;
    c.save();
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.drawImage(this.cache, 0, 0);
    c.restore();
    if (this.overlay) this.drawOverlay(c);
    const surface = css("--surface");
    for (const s of this.series) {
      const n = Math.min(s.xs.length, this.frame + 1);
      if (n < 1) continue;
      c.save();
      c.strokeStyle = s.color;
      c.lineWidth = s.dash ? 1.5 : 2;
      c.lineJoin = "round";
      if (s.dash) c.setLineDash([5, 4]);
      c.globalAlpha = s.dash ? 0.8 : 1;
      c.beginPath();
      s.xs.slice(0, n).forEach(([x, y], i) => {
        const [px, py] = this.toPx(x, y);
        i ? c.lineTo(px, py) : c.moveTo(px, py);
      });
      c.stroke();
      c.setLineDash([]);
      if (s.dots) {
        // Every visited point (simplex vertices, barrier iterates).
        c.fillStyle = s.color;
        s.xs.slice(0, n).forEach(([x, y], i) => {
          const [px, py] = this.toPx(x, y);
          c.beginPath(); c.arc(px, py, s.big && s.big[i] ? 4 : 2.5, 0, 7); c.fill();
        });
      }
      // start point
      const [sx0, sy0] = this.toPx(...s.xs[0]);
      c.fillStyle = surface; c.beginPath(); c.arc(sx0, sy0, 4, 0, 7); c.fill();
      c.lineWidth = 2; c.stroke();
      if (!s.dash && s.poly) {
        // Per-iterate geometry: NM simplex / poll stencil (filled) or a fan of directions (open).
        const open = s.polyKind === "open";
        const shape = s.poly[Math.min(n, s.poly.length) - 1];
        if (shape && shape.length) {
          c.save();
          c.beginPath();
          shape.forEach(([x, y], i) => { const [px, py] = this.toPx(x, y); i ? c.lineTo(px, py) : c.moveTo(px, py); });
          if (!open) { c.closePath(); c.globalAlpha = 0.25; c.fillStyle = s.color; c.fill(); }
          c.globalAlpha = 0.9; c.lineWidth = 1.5; c.stroke();
          c.fillStyle = s.color;
          (open ? [shape[0], shape[shape.length - 1]] : shape).forEach(([x, y]) => {
            const [px, py] = this.toPx(x, y);
            c.beginPath(); c.arc(px, py, 2.5, 0, 7); c.fill();
          });
          c.restore();
        }
      }
      if (!s.dash && s.radius && n - 1 < s.radius.length) {
        // Trust region used for the next step, drawn around the current point.
        const [cx, cy] = this.toPx(...s.xs[n - 1]);
        const [x0, x1] = this.grid.bounds;
        c.save();
        c.setLineDash([4, 4]);
        c.lineWidth = 1.5;
        c.globalAlpha = 0.9;
        c.beginPath();
        c.arc(cx, cy, (s.radius[n - 1] * this.w) / (x1 - x0), 0, 2 * Math.PI);
        c.stroke();
        c.restore();
      }
      if (!s.dash) {
        const [px, py] = this.toPx(...s.xs[n - 1]);
        const ended = n === s.xs.length;
        if (ended && s.status === "diverged") {
          c.lineWidth = 2.5;
          c.beginPath();
          c.moveTo(px - 6, py - 6); c.lineTo(px + 6, py + 6);
          c.moveTo(px + 6, py - 6); c.lineTo(px - 6, py + 6);
          c.stroke();
        } else {
          c.beginPath(); c.arc(px, py, 5, 0, 7);
          c.fillStyle = s.color; c.fill();
          c.lineWidth = 2; c.strokeStyle = surface; c.stroke();
        }
      }
      c.restore();
    }
  }

  // Bilinear interpolation of f on the grid (for the hover read-out).
  fAt(x, y) {
    const g = this.grid, nx = g.x.length, ny = g.y.length;
    const fi = ((x - g.x[0]) / (g.x[nx - 1] - g.x[0])) * (nx - 1);
    const fj = ((y - g.y[0]) / (g.y[ny - 1] - g.y[0])) * (ny - 1);
    const i = Math.max(0, Math.min(nx - 2, Math.floor(fi))), j = Math.max(0, Math.min(ny - 2, Math.floor(fj)));
    const u = fi - i, v = fj - j;
    return (1 - u) * (1 - v) * g.z[j][i] + u * (1 - v) * g.z[j][i + 1] + u * v * g.z[j + 1][i + 1] + (1 - u) * v * g.z[j + 1][i];
  }

  // Closest drawn path point within r px of (px, py), or null.
  nearestPoint(px, py, r) {
    let best = null;
    for (const s of this.series) {
      const n = Math.min(s.xs.length, this.frame + 1);
      for (let k = 0; k < n; k++) {
        const [qx, qy] = this.toPx(...s.xs[k]);
        const d = Math.hypot(qx - px, qy - py);
        if (d < r && (!best || d < best.d)) best = { d, s, k, qx, qy };
      }
    }
    return best;
  }

  // Tooltip of the path point near (px, py); false if there is none.
  showPoint(px, py, r = 10) {
    const best = this.nearestPoint(px, py, r);
    if (!best) return false;
    const [x, y] = best.s.xs[best.k];
    const f = best.s.fs ? best.s.fs[best.k] : this.fAt(x, y);
    const fl = this.grid.flabel || "f";
    this.tip.textContent = `${best.s.label} · 第 ${best.k} 步 · (${fmt(x)}, ${fmt(y)}) · ${fl} = ${fmt(f)}`;
    this.placeTip(best.qx, best.qy);
    return true;
  }

  // Show the tooltip above (x, y), shifted sideways so it stays inside the plot.
  placeTip(x, y) {
    this.tip.hidden = false;
    const half = this.tip.offsetWidth / 2;
    const left = half * 2 >= this.w ? this.w / 2 : Math.min(Math.max(x, half), this.w - half);
    this.tip.style.left = `${left}px`; this.tip.style.top = `${y}px`;
  }

  hover(e) {
    if (!this.grid) return;
    const px = e.offsetX, py = e.offsetY;
    if (this.showPoint(px, py)) return;
    const [x, y] = this.toData(px, py);
    const fl = this.grid.flabel || "f", sign = this.grid.sign || 1;
    this.tip.textContent = `(${x.toFixed(2)}, ${y.toFixed(2)})  ${fl} ≈ ${fmt(sign * this.fAt(x, y))}${this.pickCb && this.pointer === "mouse" ? "  · 點擊設為起點" : ""}`;
    this.placeTip(px, py);
  }
}

export class ConvergencePlot {
  constructor(container, { height = 300 } = {}) {
    this.wrap = document.createElement("div");
    this.wrap.className = "plot";
    this.canvas = document.createElement("canvas");
    this.canvas.setAttribute("role", "img");
    this.canvas.setAttribute("aria-label", "f(x_k) − f* 隨迭代次數變化（對數座標）");
    this.tip = document.createElement("div");
    this.tip.className = "tooltip";
    this.tip.hidden = true;
    this.wrap.append(this.canvas, this.tip);
    container.append(this.wrap);
    this.height = height;
    this.series = [];
    this.frame = Infinity;
    this.fstar = 0;
    this.hoverK = null;
    this.canvas.addEventListener("mousemove", (e) => this.hover(e));
    this.pointer = "mouse";
    this.canvas.addEventListener("pointerdown", (e) => { this.pointer = e.pointerType || "mouse"; });
    this.canvas.addEventListener("pointermove", (e) => { if (e.pointerType === "mouse") this.pointer = "mouse"; });
    this.canvas.addEventListener("click", (e) => this.hover(e));   // touch: tap = hover
    this.canvas.addEventListener("mouseleave", () => {
      if (this.pointer !== "mouse") return;   // compatibility event after a tap
      this.tip.hidden = true; this.hoverK = null; this.draw();
    });
    new ResizeObserver(() => this.resize()).observe(this.wrap);
    window.addEventListener("ot-theme", () => this.draw());
  }

  setSeries(series, fstar, nMax) {
    this.series = series; this.fstar = fstar; this.nMax = nMax;
    const all = series.flatMap((s) => s.fs.map((f) => this.y(f)));
    this.ymin = Math.max(-16, Math.floor(Math.min(...all, 0)));
    this.ymax = Math.min(12, Math.ceil(Math.max(...all, 1)));
    if (this.ymax - this.ymin < 2) this.ymin = this.ymax - 2;
    this.draw();
  }

  setFrame(k) { this.frame = k; this.draw(); }
  setHeight(height) { if (height !== this.height) { this.height = height; this.resize(); } }
  y(f) { return Math.log10(Math.max(f - this.fstar, 1e-16)); }

  resize() {
    this.w = this.wrap.clientWidth;
    if (!this.w) return;
    this.canvas.style.width = `${this.w}px`;
    this.ctx = setupCanvas(this.canvas, this.w, this.height);
    this.draw();
  }

  get pad() { return { l: 46, r: 12, t: 12, b: 40 }; }
  px(k) { const p = this.pad; return p.l + (k / this.nMax) * (this.w - p.l - p.r); }
  py(v) {
    const p = this.pad;
    const c = Math.max(this.ymin, Math.min(this.ymax, v));
    return p.t + ((this.ymax - c) / (this.ymax - this.ymin)) * (this.height - p.t - p.b);
  }

  draw() {
    const c = this.ctx;
    if (!c || !this.nMax) return;
    const p = this.pad;
    c.fillStyle = css("--surface");
    c.fillRect(0, 0, this.w, this.height);
    c.font = `11px ${css("--font")}`;
    c.fillStyle = css("--muted");
    c.strokeStyle = css("--grid");
    c.lineWidth = 1;
    const span = this.ymax - this.ymin;
    const stepY = span > 12 ? 4 : span > 6 ? 2 : 1;
    c.textAlign = "right"; c.textBaseline = "middle";
    for (let v = Math.ceil(this.ymin / stepY) * stepY; v <= this.ymax; v += stepY) {
      const y = this.py(v);
      c.beginPath(); c.moveTo(p.l, y); c.lineTo(this.w - p.r, y); c.stroke();
      c.fillText(`1e${v}`, p.l - 6, y);
    }
    const stepX = [10, 20, 25, 50, 100, 200, 500, 1000].find((s) => this.nMax / s <= 6) || 1000;
    c.textAlign = "center"; c.textBaseline = "top";
    for (let k = 0; k <= this.nMax; k += stepX) c.fillText(String(k), this.px(k), this.height - p.b + 6);
    c.strokeStyle = css("--axis");
    c.beginPath(); c.moveTo(p.l, this.height - p.b); c.lineTo(this.w - p.r, this.height - p.b); c.stroke();
    c.textAlign = "center"; c.textBaseline = "bottom";
    c.fillText("迭代次數 k", p.l + (this.w - p.l - p.r) / 2, this.height - 2);

    for (const s of this.series) {
      const n = Math.min(s.fs.length, this.frame + 1);
      c.save();
      c.strokeStyle = s.color; c.lineWidth = s.dash ? 1.5 : 2; c.lineJoin = "round";
      if (s.dash) c.setLineDash([5, 4]);
      c.beginPath();
      for (let k = 0; k < n; k++) {
        const x = this.px(k), y = this.py(this.y(s.fs[k]));
        k ? c.lineTo(x, y) : c.moveTo(x, y);
      }
      c.stroke();
      c.restore();
    }
    const k = this.hoverK ?? (Number.isFinite(this.frame) ? this.frame : null);
    if (k !== null && k <= this.nMax) {
      c.strokeStyle = css("--axis");
      c.setLineDash([3, 3]);
      c.beginPath(); c.moveTo(this.px(k), p.t); c.lineTo(this.px(k), this.height - p.b); c.stroke();
      c.setLineDash([]);
    }
  }

  hover(e) {
    if (!this.nMax) return;
    const k = Math.round(((e.offsetX - this.pad.l) / (this.w - this.pad.l - this.pad.r)) * this.nMax);
    if (k < 0 || k > this.nMax) { this.tip.hidden = true; return; }
    this.hoverK = k;
    const lines = this.series.map((s) => {
      const f = s.fs[Math.min(k, s.fs.length - 1)];
      const done = k >= s.fs.length ? (s.status === "diverged" ? "（已發散）" : "（已停止）") : "";
      return `${s.label}: ${fmt(f - this.fstar)}${done}`;
    });
    this.tip.textContent = `k = ${k}   ` + lines.join("   ");
    this.tip.style.left = `${Math.min(Math.max(e.offsetX, 120), this.w - 120)}px`;
    this.tip.style.top = `${this.pad.t + 10}px`;
    this.tip.style.transform = "translate(-50%, 0)";
    this.tip.hidden = false;
    this.draw();
  }
}
