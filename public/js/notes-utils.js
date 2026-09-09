// Shared helpers for standalone interactive notes pages (public/notes/*.html).
// Vanilla JS, no dependencies. Exposed as the global `NotesUtils`.
(function (global) {
  const SVG_NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const k in attrs || {}) {
      if (k === "text") node.textContent = attrs[k];
      else if (k === "html") node.innerHTML = attrs[k];
      else if (k.startsWith("on") && typeof attrs[k] === "function") node.addEventListener(k.slice(2), attrs[k]);
      else node.setAttribute(k, attrs[k]);
    }
    for (const child of children || []) {
      if (child != null) node.appendChild(child);
    }
    return node;
  }

  function svg(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const k in attrs || {}) node.setAttribute(k, attrs[k]);
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  // -- Random variables -------------------------------------------------

  function bernoulli(p) {
    return Math.random() < p ? 1 : 0;
  }

  // Number of Bernoulli(p) trials until (and including) the first success.
  function flipUntilSuccess(p, maxTrials) {
    let n = 0;
    const cap = maxTrials || 500;
    do {
      n++;
    } while (Math.random() >= p && n < cap);
    return n;
  }

  function binomialCoeff(n, k) {
    if (k < 0 || k > n) return 0;
    k = Math.min(k, n - k);
    let result = 1;
    for (let i = 0; i < k; i++) {
      result = (result * (n - i)) / (i + 1);
    }
    return result;
  }

  function binomialPmf(n, q, k) {
    return binomialCoeff(n, k) * Math.pow(q, k) * Math.pow(1 - q, n - k);
  }

  function geometricPmf(q, k) {
    return Math.pow(1 - q, k - 1) * q;
  }

  function gaussianRandom(mu, sigma) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    return z * sigma + mu;
  }

  function normalPdf(x, mu, sigma) {
    return Math.exp(-((x - mu) ** 2) / (2 * sigma * sigma)) / (sigma * Math.sqrt(2 * Math.PI));
  }

  // Abramowitz & Stegun 7.1.26, max error ~1.5e-7
  function erf(x) {
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    const t = 1 / (1 + p * x);
    const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
  }

  function normalCdf(x, mu, sigma) {
    return 0.5 * (1 + erf((x - mu) / (sigma * Math.sqrt(2))));
  }

  function mean(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  function variance(arr) {
    const m = mean(arr);
    return mean(arr.map((x) => (x - m) ** 2));
  }

  // -- Grouped bar chart --------------------------------------------------
  // series: [{ name, color, values: [{k, v}, ...] }], all sharing the same k's.
  function barChart(container, series, opts) {
    opts = opts || {};
    const width = opts.width || 480;
    const height = opts.height || 240;
    const margin = Object.assign({ top: 12, right: 12, bottom: 30, left: 38 }, opts.margin);
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    clear(container);
    container.setAttribute("viewBox", `0 0 ${width} ${height}`);
    container.setAttribute("width", "100%");

    const keys = series[0].values.map((d) => d.k);
    const yMax = opts.yMax != null ? opts.yMax : Math.max(
      1e-9,
      ...series.flatMap((s) => s.values.map((d) => d.v))
    ) * 1.15;

    const g = svg("g", { transform: `translate(${margin.left},${margin.top})` });
    container.appendChild(g);

    // axes
    g.appendChild(svg("line", { x1: 0, y1: innerH, x2: innerW, y2: innerH, stroke: "#9b9e91" }));
    g.appendChild(svg("line", { x1: 0, y1: 0, x2: 0, y2: innerH, stroke: "#9b9e91" }));

    const nTicks = 4;
    for (let i = 0; i <= nTicks; i++) {
      const v = (yMax * i) / nTicks;
      const y = innerH - (v / yMax) * innerH;
      g.appendChild(svg("line", { x1: -4, y1: y, x2: innerW, y2: y, stroke: "#efebe2" }));
      const label = svg("text", { x: -8, y: y + 3, "text-anchor": "end", "font-size": 10, fill: "#7c7f72" });
      label.textContent = v.toFixed(v < 0.1 ? 3 : 2);
      g.appendChild(label);
    }

    const groupW = innerW / keys.length;
    const barW = (groupW * 0.7) / series.length;

    keys.forEach((k, i) => {
      const groupX = i * groupW + groupW * 0.15;
      series.forEach((s, si) => {
        const d = s.values[i];
        const barH = (d.v / yMax) * innerH;
        g.appendChild(
          svg("rect", {
            x: groupX + si * barW,
            y: innerH - barH,
            width: Math.max(0, barW - 1.5),
            height: Math.max(0, barH),
            fill: s.color,
            opacity: s.opacity != null ? s.opacity : 1,
          })
        );
      });
      const label = svg("text", {
        x: i * groupW + groupW / 2,
        y: innerH + 16,
        "text-anchor": "middle",
        "font-size": 10,
        fill: "#6b6e62",
      });
      label.textContent = String(k);
      g.appendChild(label);
    });

    if (opts.legend) {
      const ly = -2;
      let lx = 0;
      series.forEach((s) => {
        g.appendChild(svg("rect", { x: lx, y: ly - 8, width: 9, height: 9, fill: s.color }));
        const t = svg("text", { x: lx + 13, y: ly, "font-size": 10, fill: "#565a4e" });
        t.textContent = s.name;
        g.appendChild(t);
        lx += 13 + s.name.length * 5.6 + 14;
      });
    }
  }

  // -- Simple line chart (e.g. running estimate vs. N) ---------------------
  function lineChart(container, points, opts) {
    opts = opts || {};
    const width = opts.width || 480;
    const height = opts.height || 200;
    const margin = Object.assign({ top: 12, right: 12, bottom: 26, left: 38 }, opts.margin);
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    clear(container);
    container.setAttribute("viewBox", `0 0 ${width} ${height}`);
    container.setAttribute("width", "100%");

    const yDomain = opts.yDomain || (() => {
      const lo = Math.min(...points, opts.refValue != null ? opts.refValue : Infinity);
      const hi = Math.max(...points, opts.refValue != null ? opts.refValue : -Infinity);
      const pad = (hi - lo) * 0.15 || 1;
      return [lo - pad, hi + pad];
    })();
    const xMax = Math.max(1, points.length - 1);

    const g = svg("g", { transform: `translate(${margin.left},${margin.top})` });
    container.appendChild(g);

    g.appendChild(svg("line", { x1: 0, y1: innerH, x2: innerW, y2: innerH, stroke: "#9b9e91" }));
    g.appendChild(svg("line", { x1: 0, y1: 0, x2: 0, y2: innerH, stroke: "#9b9e91" }));

    const yToPx = (y) => innerH - ((y - yDomain[0]) / (yDomain[1] - yDomain[0])) * innerH;

    if (opts.refValue != null) {
      const ry = yToPx(opts.refValue);
      g.appendChild(svg("line", { x1: 0, y1: ry, x2: innerW, y2: ry, stroke: "#b08a2a", "stroke-dasharray": "4,3" }));
      const t = svg("text", { x: innerW, y: ry - 4, "text-anchor": "end", "font-size": 10, fill: "#8a6a1c" });
      t.textContent = (opts.refLabel || "true p") + " = " + opts.refValue.toFixed(opts.refPrecision != null ? opts.refPrecision : 2);
      g.appendChild(t);
    }

    for (let i = 0; i <= 4; i++) {
      const v = yDomain[0] + ((yDomain[1] - yDomain[0]) * i) / 4;
      const y = yToPx(v);
      const label = svg("text", { x: -8, y: y + 3, "text-anchor": "end", "font-size": 10, fill: "#7c7f72" });
      label.textContent = v.toFixed(2);
      g.appendChild(label);
    }

    if (points.length > 1) {
      const d = points
        .map((p, i) => `${i === 0 ? "M" : "L"} ${(i / xMax) * innerW} ${yToPx(p)}`)
        .join(" ");
      g.appendChild(svg("path", { d, fill: "none", stroke: "#1f5a6b", "stroke-width": 1.8 }));
    }

    const xLabel = svg("text", { x: innerW, y: innerH + 18, "text-anchor": "end", "font-size": 10, fill: "#7c7f72" });
    xLabel.textContent = "n = " + (points.length - 1);
    g.appendChild(xLabel);
  }

  // -- Density curve with a shaded region (uniform / normal probability demos) --
  // opts: { curve(x)->y, xDomain:[x0,x1], yDomain?:[0,ymax], shadeX0?, shadeX1?,
  //         shadeRegions?:[[x0,x1],...] (e.g. two-tailed p-value areas), shadeColor?,
  //         width?, height?, margin?, nPoints?, xTicks?:[{x,label}] }
  function densityArea(container, opts) {
    const width = opts.width || 480;
    const height = opts.height || 220;
    const margin = Object.assign({ top: 12, right: 12, bottom: 30, left: 30 }, opts.margin);
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const [x0, x1] = opts.xDomain;
    const n = opts.nPoints || 200;

    clear(container);
    container.setAttribute("viewBox", `0 0 ${width} ${height}`);
    container.setAttribute("width", "100%");

    const pts = [];
    for (let i = 0; i <= n; i++) {
      const x = x0 + ((x1 - x0) * i) / n;
      pts.push([x, opts.curve(x)]);
    }
    const yMax = opts.yDomain ? opts.yDomain[1] : Math.max(...pts.map((p) => p[1])) * 1.15;

    const g = svg("g", { transform: `translate(${margin.left},${margin.top})` });
    container.appendChild(g);

    const xToPx = (x) => ((x - x0) / (x1 - x0)) * innerW;
    const yToPx = (y) => innerH - (y / yMax) * innerH;

    g.appendChild(svg("line", { x1: 0, y1: innerH, x2: innerW, y2: innerH, stroke: "#9b9e91" }));

    const regions = opts.shadeRegions || (opts.shadeX0 != null && opts.shadeX1 != null ? [[opts.shadeX0, opts.shadeX1]] : []);
    regions.forEach(([r0, r1]) => {
      const sx0 = Math.max(x0, r0);
      const sx1 = Math.min(x1, r1);
      if (sx1 <= sx0) return;
      const shadePts = [];
      for (let i = 0; i <= n; i++) {
        const x = sx0 + ((sx1 - sx0) * i) / n;
        shadePts.push(`${xToPx(x)},${yToPx(opts.curve(x))}`);
      }
      const d = `M ${xToPx(sx0)},${innerH} L ${shadePts.join(" L ")} L ${xToPx(sx1)},${innerH} Z`;
      g.appendChild(svg("path", { d, fill: opts.shadeColor || "#1f5a6b", opacity: 0.35 }));
    });

    const linePath = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${xToPx(p[0])} ${yToPx(p[1])}`).join(" ");
    g.appendChild(svg("path", { d: linePath, fill: "none", stroke: opts.curveColor || "#3a3a34", "stroke-width": 2 }));

    (opts.xTicks || []).forEach((t) => {
      const px = xToPx(t.x);
      g.appendChild(svg("line", { x1: px, y1: innerH, x2: px, y2: innerH + 4, stroke: "#9b9e91" }));
      const label = svg("text", { x: px, y: innerH + 16, "text-anchor": "middle", "font-size": 10, fill: "#6b6e62" });
      label.textContent = t.label;
      g.appendChild(label);
    });
  }

  // -- Histogram with an overlaid density curve (CLT-style demos) --
  // opts: { bins:[{x0,x1,density}], curve(x)->y, xDomain, yDomain?, width?, height?,
  //         margin?, barColor?, curveColor?, legend?:[{name,color}] }
  // For more than one histogram sharing the same axes (e.g. one replicate's
  // raw samples alongside the accumulated sample means), pass
  // series:[{bins,color,opacity}, ...] instead of bins/barColor; opts.bins
  // is still supported as shorthand for a single series.
  function histWithCurve(container, opts) {
    const width = opts.width || 480;
    const height = opts.height || 240;
    const margin = Object.assign({ top: 16, right: 12, bottom: 30, left: 34 }, opts.margin);
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const [x0, x1] = opts.xDomain;

    clear(container);
    container.setAttribute("viewBox", `0 0 ${width} ${height}`);
    container.setAttribute("width", "100%");

    const seriesList = opts.series || [{
      bins: opts.bins,
      color: opts.barColor || "#1f5a6b",
      opacity: opts.barOpacity != null ? opts.barOpacity : 0.75,
    }];

    const barMax = Math.max(0, ...seriesList.flatMap((s) => s.bins.map((b) => b.density)));
    const curveMax = opts.curve ? Math.max(...Array.from({ length: 40 }, (_, i) => opts.curve(x0 + ((x1 - x0) * i) / 39))) : 0;
    const yMax = opts.yDomain ? opts.yDomain[1] : Math.max(barMax, curveMax) * 1.2 || 1;

    const g = svg("g", { transform: `translate(${margin.left},${margin.top})` });
    container.appendChild(g);

    const xToPx = (x) => ((x - x0) / (x1 - x0)) * innerW;
    const yToPx = (y) => innerH - (y / yMax) * innerH;

    g.appendChild(svg("line", { x1: 0, y1: innerH, x2: innerW, y2: innerH, stroke: "#9b9e91" }));
    g.appendChild(svg("line", { x1: 0, y1: 0, x2: 0, y2: innerH, stroke: "#9b9e91" }));

    for (let i = 0; i <= 4; i++) {
      const v = (yMax * i) / 4;
      const y = yToPx(v);
      g.appendChild(svg("line", { x1: -4, y1: y, x2: innerW, y2: y, stroke: "#efebe2" }));
    }

    seriesList.forEach((s) => {
      s.bins.forEach((b) => {
        const bx0 = Math.max(x0, b.x0), bx1 = Math.min(x1, b.x1);
        if (bx1 <= bx0) return;
        g.appendChild(
          svg("rect", {
            x: xToPx(bx0),
            y: yToPx(b.density),
            width: Math.max(0, xToPx(bx1) - xToPx(bx0) - 1),
            height: Math.max(0, innerH - yToPx(b.density)),
            fill: s.color,
            opacity: s.opacity != null ? s.opacity : 0.75,
          })
        );
      });
    });

    if (opts.curve) {
      const n = 80;
      const d = Array.from({ length: n + 1 }, (_, i) => {
        const x = x0 + ((x1 - x0) * i) / n;
        return `${i === 0 ? "M" : "L"} ${xToPx(x)} ${yToPx(opts.curve(x))}`;
      }).join(" ");
      g.appendChild(svg("path", { d, fill: "none", stroke: opts.curveColor || "#b08a2a", "stroke-width": 2 }));
    }

    (opts.xTicks || []).forEach((t) => {
      const px = xToPx(t.x);
      const label = svg("text", { x: px, y: innerH + 16, "text-anchor": "middle", "font-size": 10, fill: "#6b6e62" });
      label.textContent = t.label;
      g.appendChild(label);
    });

    if (opts.legend) {
      let lx = 0;
      opts.legend.forEach((s) => {
        g.appendChild(svg("rect", { x: lx, y: -10, width: 9, height: 9, fill: s.color }));
        const t = svg("text", { x: lx + 13, y: -2, "font-size": 10, fill: "#565a4e" });
        t.textContent = s.name;
        g.appendChild(t);
        lx += 13 + s.name.length * 5.6 + 14;
      });
    }
  }

  // -- Scatter plot with a fitted line and optional residual segments (RSS / OLS demos) --
  // opts: { points:[{x,y}], xDomain, yDomain, slope, intercept, showResiduals?,
  //         refSlope?, refIntercept? (dashed comparison line, e.g. the true OLS fit),
  //         width?, height?, margin?, pointColor?, lineColor?, residualColor?, refColor? }
  function scatterFit(container, opts) {
    const width = opts.width || 480;
    const height = opts.height || 280;
    const margin = Object.assign({ top: 12, right: 12, bottom: 30, left: 34 }, opts.margin);
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const [x0, x1] = opts.xDomain;
    const [y0, y1] = opts.yDomain;

    clear(container);
    container.setAttribute("viewBox", `0 0 ${width} ${height}`);
    container.setAttribute("width", "100%");

    const g = svg("g", { transform: `translate(${margin.left},${margin.top})` });
    container.appendChild(g);

    const xToPx = (x) => ((x - x0) / (x1 - x0)) * innerW;
    const yToPx = (y) => innerH - ((y - y0) / (y1 - y0)) * innerH;

    g.appendChild(svg("line", { x1: 0, y1: innerH, x2: innerW, y2: innerH, stroke: "#9b9e91" }));
    g.appendChild(svg("line", { x1: 0, y1: 0, x2: 0, y2: innerH, stroke: "#9b9e91" }));

    for (let i = 0; i <= 4; i++) {
      const v = y0 + ((y1 - y0) * i) / 4;
      const y = yToPx(v);
      g.appendChild(svg("line", { x1: -4, y1: y, x2: innerW, y2: y, stroke: "#efebe2" }));
      const label = svg("text", { x: -8, y: y + 3, "text-anchor": "end", "font-size": 10, fill: "#7c7f72" });
      label.textContent = v.toFixed(1);
      g.appendChild(label);
    }
    for (let i = 0; i <= 4; i++) {
      const v = x0 + ((x1 - x0) * i) / 4;
      const x = xToPx(v);
      const label = svg("text", { x, y: innerH + 16, "text-anchor": "middle", "font-size": 10, fill: "#7c7f72" });
      label.textContent = v.toFixed(1);
      g.appendChild(label);
    }

    function linePts(slope, intercept) {
      const yAt0 = slope * x0 + intercept, yAt1 = slope * x1 + intercept;
      return { x1: xToPx(x0), y1: yToPx(yAt0), x2: xToPx(x1), y2: yToPx(yAt1) };
    }

    if (opts.refSlope != null && opts.refIntercept != null) {
      const p = linePts(opts.refSlope, opts.refIntercept);
      g.appendChild(svg("line", Object.assign(p, { stroke: opts.refColor || "#7c7f72", "stroke-width": 1.5, "stroke-dasharray": "5,4" })));
    }

    if (opts.showResiduals) {
      opts.points.forEach((p) => {
        const yFit = opts.slope * p.x + opts.intercept;
        g.appendChild(
          svg("line", {
            x1: xToPx(p.x), y1: yToPx(p.y), x2: xToPx(p.x), y2: yToPx(yFit),
            stroke: opts.residualColor || "#a2542c", "stroke-width": 1.3, opacity: 0.7,
          })
        );
      });
    }

    const fitP = linePts(opts.slope, opts.intercept);
    g.appendChild(svg("line", Object.assign(fitP, { stroke: opts.lineColor || "#1f5a6b", "stroke-width": 2 })));

    opts.points.forEach((p) => {
      g.appendChild(svg("circle", { cx: xToPx(p.x), cy: yToPx(p.y), r: 3.5, fill: opts.pointColor || "#3a3a34", opacity: 0.85 }));
    });
  }

  // -- Least-squares polynomial regression (feature-map / bias-variance demos) --
  // Solves the (degree+1)x(degree+1) normal-equations system via Gaussian
  // elimination with partial pivoting. Fine for the small degrees (<=12) and
  // x ranges used in these demos; not meant for numerically serious use.
  function solveLinear(A, b) {
    const n = b.length;
    const M = A.map((row, i) => row.concat([b[i]]));
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      const tmp = M[col]; M[col] = M[piv]; M[piv] = tmp;
      if (Math.abs(M[col][col]) < 1e-12) continue;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const factor = M[r][col] / M[col][col];
        for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
      }
    }
    return M.map((row, i) => (Math.abs(row[i]) < 1e-12 ? 0 : row[n] / row[i]));
  }

  // Returns coefficients [c0, c1, ..., c_degree] with y ≈ c0 + c1 x + ... + c_degree x^degree.
  function polyFit(xs, ys, degree) {
    const n = degree + 1;
    const powSums = new Array(2 * n - 1).fill(0);
    const b = new Array(n).fill(0);
    for (let k = 0; k < xs.length; k++) {
      let p = 1;
      const pows = [1];
      for (let i = 1; i < 2 * n - 1; i++) { p *= xs[k]; pows.push(p); }
      for (let i = 0; i < 2 * n - 1; i++) powSums[i] += pows[i];
      let py = 1;
      for (let i = 0; i < n; i++) { b[i] += py * ys[k]; py *= xs[k]; }
    }
    const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => powSums[i + j]));
    return solveLinear(A, b);
  }

  function polyEval(coeffs, x) {
    let result = 0, p = 1;
    for (let i = 0; i < coeffs.length; i++) { result += coeffs[i] * p; p *= x; }
    return result;
  }

  global.NotesUtils = {
    el,
    svg,
    clear,
    bernoulli,
    flipUntilSuccess,
    binomialCoeff,
    binomialPmf,
    geometricPmf,
    gaussianRandom,
    normalPdf,
    normalCdf,
    mean,
    variance,
    barChart,
    lineChart,
    densityArea,
    histWithCurve,
    scatterFit,
    polyFit,
    polyEval,
  };
})(window);

// Printing/exporting a notes page (including tools/build_pdfs_html.sh's
// headless-Chrome render) should show every Drill/Proof/aside, and every
// demo populated with an example, not whatever happens to be expanded/drawn
// on screen. Two fixes, both scoped to print media only:
(function () {
  let openedByUs = [];

  // 1. CSS alone can't force a closed <details> open in Chromium, so flip
  //    the `open` attribute here and put it back afterward.
  function openAllDetails() {
    openedByUs = Array.from(document.querySelectorAll("details:not([open])"));
    openedByUs.forEach((d) => { d.open = true; });
  }

  function restoreDetails() {
    openedByUs.forEach((d) => { d.open = false; });
    openedByUs = [];
  }

  // 2. Most demos start with an empty chart until a "draw/simulate/new
  //    dataset" button is clicked - the button itself is hidden in print
  //    (see notes.css), so without this the printed figure is blank. Click
  //    every demo button once, in document order, skipping ones that only
  //    clear/reset state. Left in place afterward (unlike the <details>
  //    toggle above, there's no generic "go back to blank" to restore, and
  //    a populated demo is a harmless thing to leave behind after a manual
  //    print/cancel).
  function populateDemos() {
    document.querySelectorAll(".demo button").forEach((btn) => {
      if (/reset|clear/i.test(btn.textContent)) return;
      btn.click();
    });
  }

  function forPrint() { openAllDetails(); populateDemos(); }

  if (window.matchMedia) {
    const mql = window.matchMedia("print");
    if (mql.matches) forPrint();
    if (mql.addEventListener) {
      mql.addEventListener("change", (e) => { if (e.matches) forPrint(); else restoreDetails(); });
    }
  }
  window.addEventListener("beforeprint", forPrint);
  window.addEventListener("afterprint", restoreDetails);
})();
