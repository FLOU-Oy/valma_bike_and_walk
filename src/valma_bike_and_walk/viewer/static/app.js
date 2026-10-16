// valma view: tiles carry geometry and a link index; colours come from
// per-link arrays fetched once per measure and kept here.
"use strict";

const NETWORK_MEASURES = new Set(["speed", "speed_forward", "speed_reverse", "under_construction"]);
const $ = (id) => document.getElementById(id);
const state = {
  meta: null, run: "", measure: "volume", compare: "", minValue: 0,
  showCounters: true, values: null, scaleMax: 1, kind: "volume",
  version: 0, selected: -1, counters: null,
  flowIndex: null, flowName: "", pollToken: 0,
  map: null, routeMode: false, origin: null, destination: null, route: null,
  vias: [], alternative: null, calibration: null, calibToken: 0,
  munis: null, muniCompare: null, muniSort: { key: "bike_km", desc: true },
  muniSelected: -1, muniOutline: null, muniToken: 0,
  tl: null, tlCompare: null, tlGroup: "country", tlName: "", tlToken: 0,
};
const arrays = new Map();

function status(text) {
  const el = $("status");
  el.textContent = text || "";
  el.style.display = text ? "block" : "none";
}

async function fetchArray(run, measure) {
  const key = `${run}|${measure}`;
  if (!arrays.has(key)) {
    status(`Loading ${measure}…`);
    const res = await fetch(`/api/attr/${encodeURI(run)}/${measure}.bin`);
    if (!res.ok) throw new Error(`${measure}: ${res.status}`);
    arrays.set(key, new Float32Array(await res.arrayBuffer()));
    status("");
  }
  return arrays.get(key);
}

function quantile(values, q, filter) {
  const sample = [];
  const step = Math.max(1, Math.floor(values.length / 200000));
  for (let i = 0; i < values.length; i += step) {
    const v = values[i];
    if (filter(v)) sample.push(Math.abs(v));
  }
  if (!sample.length) return 1;
  sample.sort((a, b) => a - b);
  return sample[Math.min(sample.length - 1, Math.floor(q * sample.length))] || 1;
}

async function refresh() {
  const m = state.measure;
  if (state.flowIndex != null) {
    const measure = NETWORK_MEASURES.has(m) ? "volume" : m;
    state.values = await fetchArray(`counter/${state.flowIndex}`, measure);
    state.kind = "flow";
    state.scaleMax = quantile(state.values, 0.995, (v) => v > 0);
  } else if (NETWORK_MEASURES.has(m)) {
    state.values = await fetchArray("network", m);
    state.kind = m === "under_construction" ? "flag" : "speed";
    state.scaleMax = 25;
  } else if (state.run) {
    const a = await fetchArray(state.run, m);
    if (state.compare && state.compare !== state.run) {
      const b = await fetchArray(state.compare, m);
      const d = new Float32Array(a.length);
      for (let i = 0; i < a.length; i++) d[i] = a[i] - b[i];
      state.values = d;
      state.kind = "diff";
      state.scaleMax = quantile(d, 0.99, (v) => v !== 0);
    } else {
      state.values = a;
      state.kind = "volume";
      state.scaleMax = quantile(a, 0.995, (v) => v > 0);
    }
  } else {
    state.values = null;
  }
  state.version++;
  render();
  legend();
}

// -- colours -------------------------------------------------------------

const SEQ = [[255, 237, 160], [254, 178, 76], [240, 59, 32], [189, 0, 38], [84, 0, 70]];
const DIV_NEG = [33, 102, 172], DIV_POS = [178, 24, 43], DIV_MID = [235, 235, 235];
const SPEED = [[165, 0, 38], [244, 109, 67], [254, 224, 139], [102, 189, 99], [0, 104, 55]];
const FLOW = [[199, 233, 180], [65, 182, 196], [34, 94, 168], [37, 52, 148], [8, 29, 88]];

function ramp(stops, t) {
  t = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t)), f = t - i;
  return stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * f));
}
function mix(a, b, t) { return a.map((c, k) => Math.round(c + (b[k] - c) * t)); }

function value(i) { return state.values ? state.values[i] : 0; }

function lineColor(f) {
  const v = value(f.properties.i);
  switch (state.kind) {
    case "volume":
      if (!(v > 0) || v < state.minValue) return [0, 0, 0, 0];
      return [...ramp(SEQ, Math.log1p(v) / Math.log1p(state.scaleMax)), 230];
    case "flow":
      if (!(v > 0) || v < state.minValue) return [0, 0, 0, 0];
      return [...ramp(FLOW, Math.log1p(v) / Math.log1p(state.scaleMax)), 235];
    case "diff": {
      if (v === 0) return [160, 160, 160, 60];
      const t = Math.sqrt(Math.min(1, Math.abs(v) / state.scaleMax));
      return [...mix(DIV_MID, v < 0 ? DIV_NEG : DIV_POS, t), 230];
    }
    case "speed":
      return [...ramp(SPEED, v / state.scaleMax), 220];
    case "flag":
      return v > 0 ? [214, 51, 132, 255] : [150, 150, 150, 70];
    default:
      return [120, 120, 120, 160];
  }
}

function lineWidth(f) {
  const v = value(f.properties.i);
  if (state.kind === "volume" || state.kind === "flow") {
    if (!(v > 0) || v < state.minValue) return 0;
    return 0.6 + 7 * Math.sqrt(Math.min(1, v / state.scaleMax));
  }
  if (state.kind === "diff") return 0.6 + 6 * Math.sqrt(Math.min(1, Math.abs(v) / state.scaleMax));
  if (state.kind === "flag") return v > 0 ? 3 : 0.8;
  return 1.6;
}

function ratioColor(r) {
  if (r == null) return [130, 130, 130, 200];
  const t = Math.max(-1, Math.min(1, Math.log2(r) / 2));  // ±2 doublings saturate
  return [...mix(DIV_MID, t < 0 ? DIV_NEG : DIV_POS, Math.abs(t)), 240];
}

// -- layers --------------------------------------------------------------

const overlay = new deck.MapboxOverlay({ interleaved: false, layers: [], getTooltip: tooltip });

function render() {
  const layers = [];
  if (state.muniOutline) {
    layers.push(new deck.GeoJsonLayer({
      id: "municipality", data: state.muniOutline, stroked: true, filled: true,
      getFillColor: [11, 114, 133, 18], getLineColor: [11, 114, 133, 255],
      lineWidthUnits: "pixels", getLineWidth: 2.5,
    }));
  }
  layers.push(
    new deck.MVTLayer({
      id: "links",
      data: `${location.origin}/tiles/links/{z}/{x}/{y}.mvt`,
      binary: false,
      minZoom: 0,
      maxZoom: 14,
      lineWidthUnits: "pixels",
      lineCapRounded: true,
      getLineColor: lineColor,
      getLineWidth: lineWidth,
      pickable: true,
      autoHighlight: true,
      highlightColor: [11, 114, 133, 255],
      uniqueIdProperty: "i",
      highlightedFeatureId: state.selected >= 0 ? state.selected : null,
      updateTriggers: { getLineColor: [state.version, state.minValue], getLineWidth: [state.version, state.minValue] },
      onClick: (info) => !state.routeMode && info.object && inspectLink(info.object.properties.i),
    }),
  );
  if (state.alternative) {
    layers.push(new deck.PathLayer({
      id: "alt-casing", data: [state.alternative], getPath: (f) => f.geometry.coordinates,
      getColor: [255, 255, 255, 230], widthUnits: "pixels", getWidth: 9, capRounded: true, jointRounded: true,
    }), new deck.PathLayer({
      id: "alt-line", data: [state.alternative], getPath: (f) => f.geometry.coordinates,
      getColor: [232, 89, 12, 255], widthUnits: "pixels", getWidth: 5, capRounded: true, jointRounded: true,
    }));
  }
  if (state.route) {
    layers.push(new deck.PathLayer({
      id: "route", data: [state.route], getPath: (f) => f.geometry.coordinates,
      getColor: [255, 255, 255, 230], widthUnits: "pixels", getWidth: 9, capRounded: true, jointRounded: true,
    }), new deck.PathLayer({
      id: "route-line", data: [state.route], getPath: (f) => f.geometry.coordinates,
      getColor: [11, 114, 133, 255], widthUnits: "pixels", getWidth: 5, capRounded: true, jointRounded: true,
      pickable: true,
      onClick: (info) => { if (info.coordinate) addVia(info.coordinate); return true; },
    }));
  }
  if (state.showCounters && state.counters) {
    const feats = state.counters.features;
    layers.push(
      new deck.PathLayer({
        id: "count-lines", data: feats, getPath: (f) => f.properties.line,
        getColor: [20, 20, 20, 200], widthUnits: "pixels", getWidth: 2,
      }),
      new deck.ScatterplotLayer({
        id: "counters", data: feats, pickable: true, stroked: true,
        radiusUnits: "pixels", lineWidthUnits: "pixels", getLineWidth: 1,
        getPosition: (f) => f.geometry.coordinates,
        getRadius: (f) => 4 + Math.sqrt(f.properties.observed) / 4,
        getFillColor: (f) => ratioColor(f.properties.ratio),
        getLineColor: [40, 40, 40, 255],
        updateTriggers: { getFillColor: [state.run] },
        onClick: (info) => !state.routeMode && info.object && showCounter(info.object.properties),
      }),
    );
  }
  overlay.setProps({ layers });
}

function fmt(v, d = 0) {
  return v == null || Number.isNaN(v) ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d });
}

function tooltip({ object, layer }) {
  if (!object || !layer) return null;
  if (layer.id === "route-line") return "Click to add a via point, then drag it to your route";
  if (layer.id === "counters") {
    const p = object.properties;
    return `${p.name}\nobserved ${fmt(p.observed)} · model ${fmt(p.modeled)} · ×${fmt(p.ratio, 2)}`;
  }
  if (layer.id.startsWith("links")) {
    const i = object.properties.i;
    const unit = state.kind === "speed" ? " km/h" : "";
    return `link row ${i}: ${fmt(value(i), state.kind === "speed" ? 1 : 0)}${unit}`;
  }
  return null;
}

// -- legend and details --------------------------------------------------

function gradient(stops) {
  return `linear-gradient(to right, ${stops.map((c) => `rgb(${c.join(",")})`).join(", ")})`;
}

function legend() {
  const el = $("legend");
  const label = $("measure").selectedOptions[0].textContent;
  if (state.kind === "flow") {
    el.innerHTML = `<h2>Trips through ${state.flowName}</h2><div class="bar" style="background:${gradient(FLOW)}"></div>
      <div class="scale"><span>1</span><span>${fmt(Math.sqrt(state.scaleMax))}</span><span>${fmt(state.scaleMax)}+</span></div>
      <button id="clear-flow">Back to all trips</button>`;
  } else if (state.kind === "volume") {
    el.innerHTML = `<h2>${label}</h2><div class="bar" style="background:${gradient(SEQ)}"></div>
      <div class="scale"><span>1</span><span>${fmt(Math.sqrt(state.scaleMax))}</span><span>${fmt(state.scaleMax)}+</span></div>`;
  } else if (state.kind === "diff") {
    el.innerHTML = `<h2>${label}: ${state.run} − ${state.compare}</h2>
      <div class="bar" style="background:${gradient([DIV_NEG, DIV_MID, DIV_POS])}"></div>
      <div class="scale"><span>−${fmt(state.scaleMax)}</span><span>0</span><span>+${fmt(state.scaleMax)}</span></div>`;
  } else if (state.kind === "speed") {
    el.innerHTML = `<h2>${label}</h2><div class="bar" style="background:${gradient(SPEED)}"></div>
      <div class="scale"><span>0</span><span>12.5</span><span>25 km/h</span></div>`;
  } else {
    el.innerHTML = `<h2>${label}</h2><p class="hint">Pink: read from highway=construction.</p>`;
  }
  if (state.showCounters && state.counters) {
    el.innerHTML += `<h2>Counters: model ÷ observed</h2>
      <div class="bar" style="background:${gradient([DIV_NEG, DIV_MID, DIV_POS])}"></div>
      <div class="scale"><span>×0.25</span><span>×1</span><span>×4</span></div>`;
  }
  // After the last innerHTML change, which would drop a listener added earlier.
  const clear = $("clear-flow");
  if (clear) clear.addEventListener("click", clearFlow);
}

function row(k, v, cls = "") { return `<tr><td>${k}</td><td class="${cls}">${v}</td></tr>`; }

async function inspectLink(i) {
  state.selected = i;
  render();
  const res = await fetch(`/api/link/${i}`);
  if (!res.ok) return;
  const d = await res.json();
  const s = d.speed, dirs = s.directions || {};
  const tagOrder = ["link_id", "highway", "name", "surface", "segregated", "cycleway", "cycleway_left",
    "cycleway_right", "cycleway_both", "oneway", "oneway_bicycle", "car_volume", "crossing",
    "traffic_signal_at_start", "traffic_signal_at_end", "crossing_node_at_start",
    "crossing_node_at_end", "crossing_way_share", "ascent_m", "descent_m", "under_construction",
    "speed_override_kmh", "bridge", "osm_way_id"];
  const tags = tagOrder.filter((k) => k in d.tags).map((k) => row(k, d.tags[k])).join("");
  const dir = (k) => dirs[k] || {};
  const both = (label, key, digits = 2, suffix = "") =>
    `<tr><td>${label}</td><td class="num">${fmt(dir("forward")[key], digits)}${suffix}</td><td class="num">${fmt(dir("reverse")[key], digits)}${suffix}</td></tr>`;
  const facility = `<tr><td>Facility</td><td class="num">${dir("forward").facility ?? "–"}</td><td class="num">${dir("reverse").facility ?? "–"}</td></tr>`;
  const vols = Object.entries(d.volumes).map(([name, v]) =>
    `<tr><td>${name}</td><td class="num">${fmt(v.forward)}</td><td class="num">${fmt(v.reverse)}</td></tr>`).join("");
  $("details").innerHTML = `
    <h2>Link row ${i}${d.tags.name ? ` · ${d.tags.name}` : ""}</h2>
    <table>
      ${row("Nominal speed", `${fmt(s.nominal_kmh, 1)} km/h (${s.highway})`, "num")}
      ${row("Surface factor", fmt(s.surface_factor, 2), "num")}
      ${s.segregated_factor != null ? row("Segregation factor", fmt(s.segregated_factor, 2), "num") : ""}
      ${s.traffic_factor != null ? row("Traffic factor", `${fmt(s.traffic_factor, 3)} (car_volume ${fmt(s.car_volume)})`, "num") : ""}
      ${row("Length", `${fmt(s.length_m, 1)} m`, "num")}
    </table>
    <table>
      <tr><th></th><th class="num">digitised</th><th class="num">against</th></tr>
      ${facility}
      ${both("Carriageway divisor", "carriageway_divisor", 3)}
      ${both("Speed", "speed_kmh", 1, " km/h")}
      ${both("Moving time", "moving_time_s", 1, " s")}
      ${both("Penalties", "penalties_s", 1, " s")}
      ${both("Travel time", "travel_time_s", 1, " s")}
    </table>
    ${vols ? `<h2>Volumes</h2><table><tr><th></th><th class="num">digitised</th><th class="num">against</th></tr>${vols}</table>` : ""}
    <h2>Tags</h2><table>${tags}</table>`;
}

// -- routing -------------------------------------------------------------

function marker(color, lngLat) {
  const m = new maplibregl.Marker({ color, draggable: true }).setLngLat(lngLat).addTo(state.map);
  m.on("dragend", () => {
    if (!(state.origin && state.destination)) return;
    if (state.vias.length) runCalibration(); else planRoute();
  });
  return m;
}

function routeHint(text) {
  const el = $("route-hint");
  el.hidden = !text;
  el.textContent = text || "";
}

function startRouting() {
  clearRoute();
  state.routeMode = true;
  state.map.getCanvas().style.cursor = "crosshair";
  $("route-toggle").textContent = "Cancel";
  routeHint("Click the origin on the map.");
}

function stopPlacing() {
  state.routeMode = false;
  state.map.getCanvas().style.cursor = "";
  $("route-toggle").textContent = "Plan a route";
}

function clearRoute() {
  stopPlacing();
  for (const m of [state.origin, state.destination, ...state.vias]) if (m) m.remove();
  state.origin = state.destination = state.route = null;
  state.vias = [];
  state.alternative = state.calibration = null;
  $("route-clear").hidden = true;
  $("route-result").innerHTML = "";
  routeHint("");
  render();
}

function onMapClick(e) {
  if (!state.routeMode) return;
  if (!state.origin) {
    state.origin = marker("#2b8a3e", e.lngLat);
    routeHint("Click the destination.");
  } else {
    state.destination = marker("#c92a2a", e.lngLat);
    stopPlacing();
    $("route-clear").hidden = false;
    routeHint("Drag the end markers to re-route. Click the route to add a via point and drag it onto the route you think is right.");
    planRoute();
  }
}

async function planRoute() {
  const a = state.origin.getLngLat(), b = state.destination.getLngLat();
  $("route-result").innerHTML = `<p class="hint">Routing…</p>`;
  const q = `origin=${a.lng},${a.lat}&destination=${b.lng},${b.lat}`;
  const res = await fetch(`/api/route?${q}`);
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))).detail || res.status;
    state.route = null;
    render();
    $("route-result").innerHTML = `<p class="hint">No route: ${detail}</p>`;
    return;
  }
  state.route = await res.json();
  render();
  const p = state.route.properties;
  const types = Object.entries(p.by_highway).slice(0, 6)
    .map(([k, v]) => row(k || "–", `${fmt(v / 1000, 2)} km (${fmt(100 * v / p.length_m)} %)`, "num")).join("");
  const steps = p.steps.map((s) =>
    `<tr><td>${s.name || `<span class="hint">${s.highway}</span>`}</td>
      <td class="num">${fmt(s.length_m)} m</td><td class="num">${fmt(s.speed_kmh, 1)}</td></tr>`).join("");
  $("route-result").innerHTML = `
    <table>
      ${row("Length", `${fmt(p.length_m / 1000, 2)} km`, "num")}
      ${row("Travel time (model cost)", `${fmt(p.time_s / 60, 1)} min`, "num")}
      ${row("Average model speed", `${fmt(p.speed_kmh, 1)} km/h`, "num")}
      ${row("Links", fmt(p.links), "num")}
    </table>
    <h2>By road type</h2><table>${types}</table>
    <h2>Route</h2><table><tr><th>Street</th><th class="num">length</th><th class="num">km/h</th></tr>${steps}</table>`;
}

// -- calibration: dragging the route --------------------------------------

function addVia(coord) {
  const m = new maplibregl.Marker({ color: "#e8590c", draggable: true, scale: 0.8 })
    .setLngLat(coord).addTo(state.map);
  m.on("dragend", runCalibration);
  m.getElement().title = "Drag onto the preferred route; double-click to remove";
  m.getElement().addEventListener("dblclick", (e) => {
    e.stopPropagation();
    m.remove();
    state.vias = state.vias.filter((v) => v !== m);
    if (state.vias.length) runCalibration();
    else { state.alternative = state.calibration = null; planRoute(); }
  });
  // Keep vias in route order: insert where the click falls along the route.
  state.vias.push(m);
  state.vias.sort((a, b) => along(a.getLngLat()) - along(b.getLngLat()));
  runCalibration();
}

function along(lngLat) {
  // Index of the nearest model-route vertex: good enough to order vias.
  const coords = state.route ? state.route.geometry.coordinates : [];
  let best = 0, bestD = Infinity;
  coords.forEach(([x, y], i) => {
    const d = (x - lngLat.lng) ** 2 + (y - lngLat.lat) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  });
  return best;
}

function points() {
  const ll = (m) => { const p = m.getLngLat(); return [p.lng, p.lat]; };
  return { origin: ll(state.origin), destination: ll(state.destination), vias: state.vias.map(ll) };
}

async function runCalibration() {
  const token = ++state.calibToken;
  $("route-result").innerHTML = `<p class="hint">Comparing routes…</p>`;
  const res = await fetch("/api/calibrate", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(points()),
  });
  if (token !== state.calibToken) return;
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))).detail || res.status;
    $("route-result").innerHTML = `<p class="hint">Could not compare: ${detail}</p>`;
    return;
  }
  const c = await res.json();
  state.calibration = c;
  state.route = c.model;
  state.alternative = c.alternative;
  render();
  showCalibration(c);
}

function showCalibration(c) {
  const m = c.model_parts, a = c.alternative_parts;
  const diff = (x, y, d = 1) => `${y - x >= 0 ? "+" : ""}${fmt(y - x, d)}`;
  const causes = Object.keys(m.parts_s)
    .filter((k) => Math.abs(m.parts_s[k]) >= 0.5 || Math.abs(a.parts_s[k] || 0) >= 0.5)
    .map((k) => `<tr><td>${k}</td><td class="num">${fmt(m.parts_s[k])}</td><td class="num">${fmt(a.parts_s[k] || 0)}</td>
      <td class="num">${diff(m.parts_s[k], a.parts_s[k] || 0, 0)}</td></tr>`).join("");
  const be = c.break_even.slice(0, 10).map((r) => `<tr><td>${r.parameter}</td>
      <td class="num">${fmt(r.current, 3)}</td>
      <td class="num">${r.break_even == null ? "–" : fmt(r.break_even, 3)}</td>
      <td class="num">${r.change_pct == null ? "out of range" : `${r.change_pct >= 0 ? "+" : ""}${fmt(r.change_pct)} %`}</td></tr>`).join("");
  $("route-result").innerHTML = `
    ${c.warning ? `<p style="color:#c92a2a"><b>Warning:</b> ${c.warning}</p>` : ""}
    <p><span style="color:#0b7285">■</span> model route &nbsp; <span style="color:#e8590c">■</span> your route</p>
    <table>
      <tr><th></th><th class="num">model</th><th class="num">yours</th><th class="num">diff</th></tr>
      <tr><td>Length (km)</td><td class="num">${fmt(m.length_m / 1000, 2)}</td><td class="num">${fmt(a.length_m / 1000, 2)}</td><td class="num">${diff(m.length_m / 1000, a.length_m / 1000, 2)}</td></tr>
      <tr><td>Model cost (min)</td><td class="num">${fmt(m.cost_s / 60, 1)}</td><td class="num">${fmt(a.cost_s / 60, 1)}</td><td class="num">${diff(m.cost_s / 60, a.cost_s / 60, 1)}</td></tr>
      <tr><td>Signalised links</td><td class="num">${m.signal_links}</td><td class="num">${a.signal_links}</td><td class="num">${diff(m.signal_links, a.signal_links, 0)}</td></tr>
    </table>
    <h2>Cost by cause (s)</h2>
    <table><tr><th></th><th class="num">model</th><th class="num">yours</th><th class="num">diff</th></tr>${causes}</table>
    <h2>What would make your route win</h2>
    <p class="hint">Each parameter alone, others unchanged: the value at which your route costs no more than the model's. Smallest change first.</p>
    <table><tr><th>parameter</th><th class="num">now</th><th class="num">break-even</th><th class="num">change</th></tr>${be}</table>
    <input id="case-note" placeholder="Note for this case (optional)" style="width:100%;margin-top:8px;box-sizing:border-box">
    <button id="copy-report">Copy report</button>
    <button id="save-case">Save case</button>
    <span id="case-status" class="hint"></span>`;
  $("copy-report").addEventListener("click", async () => {
    await navigator.clipboard.writeText(c.report);
    $("case-status").textContent = " Copied.";
  });
  $("save-case").addEventListener("click", async () => {
    const res = await fetch("/api/calibrate/save", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...points(), note: $("case-note").value }),
    });
    const r = await res.json();
    $("case-status").textContent = res.ok ? ` Saved (${r.observations} cases in ${r.saved}).` : ` Failed: ${r.detail}`;
  });
}

// -- municipalities ------------------------------------------------------

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ESCAPES[c]); }

async function fetchMunicipalities(run) {
  const res = await fetch(`/api/municipalities?run=${encodeURIComponent(run)}`);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || res.status);
  return body;
}

async function loadMunicipalities() {
  if (!state.meta.municipalities || !state.run) { $("muni-box").hidden = true; return; }
  $("muni-box").hidden = false;
  const token = ++state.muniToken;
  const compare = state.compare && state.compare !== state.run ? state.compare : "";
  try {
    const [a, b] = await Promise.all([
      fetchMunicipalities(state.run), compare ? fetchMunicipalities(compare) : null,
    ]);
    if (token !== state.muniToken) return;
    state.munis = a;
    state.muniCompare = b;
  } catch (err) {
    if (token !== state.muniToken) return;
    state.munis = state.muniCompare = null;
    $("muni-total").textContent = "";
    $("muni-table").innerHTML = `<tr><td class="hint">${esc(err.message)}</td></tr>`;
    return;
  }
  municipalityTable();
  $("plot-open").hidden = !state.munis.fit;
  if (!state.munis.fit) $("plot").hidden = true;
  else if (!$("plot").hidden) drawPlot();
  if (state.muniSelected >= 0) showMunicipality(state.muniSelected);
}

// Modal share against speed, one dot per municipality, sized by its trips.
function drawPlot() {
  const { rows, fit, run } = state.munis;
  const mode = state.meta.mode;
  const pts = rows.filter((r) => r.speed_kmh != null && r.mode_share != null);
  const W = 600, H = 380, L = 48, R = 14, T = 12, B = 40;
  const xs = pts.map((r) => r.speed_kmh), ys = pts.map((r) => 100 * r.mode_share);
  const x0 = Math.floor(Math.min(...xs)), x1 = Math.ceil(Math.max(...xs));
  const yStep = Math.max(...ys) > 25 ? 5 : 2;
  const y1 = Math.ceil(Math.max(...ys) / yStep) * yStep;
  const sx = (v) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
  const sy = (v) => H - B - (v / y1) * (H - T - B);
  const maxTrips = Math.max(...pts.map((r) => r.all_trips));
  const radius = (r) => 2.5 + 11 * Math.sqrt(r.all_trips / maxTrips);

  let svg = "";
  for (let v = x0; v <= x1; v++) {
    svg += `<line class="grid" x1="${sx(v)}" x2="${sx(v)}" y1="${T}" y2="${H - B}"/>
      <text x="${sx(v)}" y="${H - B + 15}" text-anchor="middle">${v}</text>`;
  }
  for (let v = 0; v <= y1; v += yStep) {
    svg += `<line class="grid" x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}"/>
      <text x="${L - 6}" y="${sy(v) + 4}" text-anchor="end">${v} %</text>`;
  }
  svg += `<line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>
    <line class="axis" x1="${L}" x2="${L}" y1="${T}" y2="${H - B}"/>
    <text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">Average ${esc(mode)} speed, km/h (bike-km ÷ bike-hours)</text>
    <text transform="translate(12 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${esc(mode)} share of trips</text>`;
  const fy = (v) => 100 * (fit.intercept + fit.slope * v);
  svg += `<line class="fit" x1="${sx(x0)}" y1="${sy(fy(x0))}" x2="${sx(x1)}" y2="${sy(fy(x1))}"/>`;
  // Big ones first so the small dots stay clickable on top.
  const order = [...pts].sort((a, b) => b.all_trips - a.all_trips);
  for (const r of order) {
    svg += `<circle data-i="${r.index}" class="${r.index === state.muniSelected ? "sel" : ""}"
      cx="${sx(r.speed_kmh)}" cy="${sy(100 * r.mode_share)}" r="${radius(r)}">
      <title>${esc(r.name)}: ${fmt(r.speed_kmh, 2)} km/h, ${pct(r.mode_share)} of ${fmt(r.all_trips)} trips</title></circle>`;
  }
  const labelled = new Set(order.slice(0, 6).map((r) => r.index));
  if (state.muniSelected >= 0) labelled.add(state.muniSelected);
  for (const r of pts.filter((p) => labelled.has(p.index))) {
    svg += `<text class="name" x="${sx(r.speed_kmh) + radius(r) + 3}" y="${sy(100 * r.mode_share) + 4}">${esc(r.name)}</text>`;
  }
  $("plot-svg").innerHTML = svg;
  $("plot-svg").querySelectorAll("circle").forEach((c) =>
    c.addEventListener("click", () => selectMunicipality(Number(c.dataset.i))));
  $("plot-title").textContent = `${mode} share vs speed · ${run}`;
  $("plot-stats").innerHTML = `<b>R² = ${fmt(fit.r2, 3)}</b> (each municipality once) ·
    R² = ${fmt(fit.r2_weighted, 3)} weighted by trips · n = ${fit.n} ·
    slope ${signed(100 * fit.slope, 2)} percentage points per km/h. Dot size: trips, all modes.`;
}

function speedDelta(i) {
  const a = state.munis.rows[i].speed_kmh, b = state.muniCompare?.rows[i].speed_kmh;
  return a != null && b != null ? a - b : null;
}

function pct(v, d = 1) { return v == null ? "–" : `${fmt(100 * v, d)} %`; }

function signed(v, d) { return v == null ? "–" : `${v >= 0 ? "+" : ""}${fmt(v, d)}`; }

function municipalityTable() {
  const { rows, total } = state.munis;
  const cmp = !!state.muniCompare;
  const shares = total.mode_share != null;
  $("muni-total").textContent = `· Finland ${fmt(total.speed_kmh, 2)} km/h`;
  const q = $("muni-search").value.trim().toLocaleLowerCase();
  const { key, desc } = state.muniSort;
  const get = (r) => (key === "delta" ? speedDelta(r.index) : r[key]);
  const shown = rows
    .filter((r) => !q || r.name.toLocaleLowerCase().includes(q))
    .sort((a, b) => {
      if (key === "name") return (desc ? -1 : 1) * a.name.localeCompare(b.name, "fi");
      const x = get(a), y = get(b);
      if (x == null || y == null) return (x == null) - (y == null);  // blanks last
      return desc ? y - x : x - y;
    });
  const head = (k, label, num = true) =>
    `<th class="${num ? "num" : ""}" data-key="${k}">${label}${key === k ? (desc ? " ▾" : " ▴") : ""}</th>`;
  const body = shown.map((r) => `<tr data-i="${r.index}" class="${r.index === state.muniSelected ? "sel" : ""}">
      <td>${esc(r.name)}</td><td class="num">${fmt(r.speed_kmh, 2)}</td>
      ${cmp ? `<td class="num">${signed(speedDelta(r.index), 2)}</td>` : ""}
      ${shares ? `<td class="num">${pct(r.mode_share)}</td>` : ""}
      <td class="num">${fmt(r.bike_km)}</td></tr>`).join("");
  $("muni-table").innerHTML = `<thead><tr>${head("name", "Municipality", false)}${head("speed_kmh", "km/h")}
      ${cmp ? head("delta", "Δ vs B") : ""}${shares ? head("mode_share", "share") : ""}${head("bike_km", "bike-km")}</tr></thead>
    <tbody>${body || `<tr><td class="hint" colspan="5">No match.</td></tr>`}</tbody>`;
  $("muni-table").querySelectorAll("th").forEach((th) => th.addEventListener("click", () => {
    const k = th.dataset.key;
    state.muniSort = { key: k, desc: state.muniSort.key === k ? !state.muniSort.desc : k !== "name" };
    municipalityTable();
  }));
  $("muni-table").querySelectorAll("tbody tr[data-i]").forEach((tr) =>
    tr.addEventListener("click", () => selectMunicipality(Number(tr.dataset.i))));
}

async function selectMunicipality(i) {
  state.muniSelected = i;
  municipalityTable();
  if (state.tlGroup === "municipality" && state.munis) {
    state.tlName = state.munis.rows[i].name;
    tripLengthOptions();
    drawTripLengths();
  }
  if (!$("plot").hidden) drawPlot();
  showMunicipality(i);
  const res = await fetch(`/api/municipality/${i}`);
  if (!res.ok || state.muniSelected !== i) return;
  state.muniOutline = await res.json();
  const [w, s, e, n] = state.muniOutline.properties.bounds;
  state.map.fitBounds([[w, s], [e, n]], { padding: 40, duration: 600 });
  render();
}

function clearMunicipality() {
  state.muniSelected = -1;
  state.muniOutline = null;
  $("details").innerHTML = `<p class="hint">Click a link or a counter for details.</p>`;
  if (state.munis) municipalityTable();
  if (!$("plot").hidden) drawPlot();
  render();
}

function showMunicipality(i) {
  if (!state.munis) return;
  const r = state.munis.rows[i], total = state.munis.total;
  const ranked = state.munis.rows.filter((x) => x.speed_kmh != null)
    .sort((a, b) => b.speed_kmh - a.speed_kmh);
  const rank = ranked.findIndex((x) => x.index === i) + 1;
  const b = state.muniCompare?.rows[i];
  const colB = (v, d) => (b ? `<td class="num">${fmt(v, d)}</td>` : "");
  const line = (label, key, d, unit = "") =>
    `<tr><td>${label}</td><td class="num">${fmt(r[key], d)}${unit}</td>${colB(b?.[key], d)}</tr>`;
  $("details").innerHTML = `
    <h2>${esc(r.name)}</h2>
    <table>
      ${b ? `<tr><th></th><th class="num">${esc(state.run)}</th><th class="num">${esc(state.compare)}</th></tr>` : ""}
      ${line("Average speed", "speed_kmh", 2, " km/h")}
      ${line("Bike-km / day", "bike_km", 0)}
      ${line("Bike-hours / day", "bike_hours", 1)}
      ${line("Links ridden", "links", 0)}
      ${r.mode_share != null ? `<tr><td>${esc(state.meta.mode)} share of trips</td><td class="num">${pct(r.mode_share, 2)}</td>${b ? "<td></td>" : ""}</tr>
        <tr><td>Trips / day, all modes</td><td class="num">${fmt(r.all_trips)}</td>${b ? "<td></td>" : ""}</tr>` : ""}
      <tr><td>Rank by speed</td><td class="num">${rank ? `${rank} / ${ranked.length}` : "–"}</td>${b ? "<td></td>" : ""}</tr>
      <tr><td>Finland average</td><td class="num">${fmt(total.speed_kmh, 2)} km/h</td>${colB(state.muniCompare?.total.speed_kmh, 2)}</tr>
    </table>
    <button id="muni-clear">Clear selection</button>`;
  $("muni-clear").addEventListener("click", clearMunicipality);
}

// -- trip lengths --------------------------------------------------------

async function fetchTripLengths(run) {
  const res = await fetch(`/api/trip-lengths?run=${encodeURIComponent(run)}`);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || res.status);
  return body;
}

async function loadTripLengths() {
  if (!state.meta.trip_lengths || !state.run) { $("tl-box").hidden = true; return; }
  $("tl-box").hidden = false;
  const token = ++state.tlToken;
  const compare = state.compare && state.compare !== state.run ? state.compare : "";
  try {
    const [a, b] = await Promise.all([
      fetchTripLengths(state.run), compare ? fetchTripLengths(compare).catch(() => null) : null,
    ]);
    if (token !== state.tlToken) return;
    state.tl = a;
    state.tlCompare = b;
  } catch (err) {
    if (token !== state.tlToken) return;
    state.tl = state.tlCompare = null;
    $("tl-svg").innerHTML = "";
    $("tl-stats").textContent = err.message;
    return;
  }
  $("tl-group").querySelector('option[value="region"]').disabled = !Object.keys(state.tl.regions).length;
  tripLengthOptions();
  drawTripLengths();
}

function tripLengthOptions() {
  const group = state.tlGroup;
  $("tl-name-label").hidden = group === "country";
  if (group === "country" || !state.tl) return;
  const names = group === "region"
    ? Object.keys(state.tl.regions)
    : Object.keys(state.tl.municipalities).sort((a, b) => a.localeCompare(b, "fi"));
  if (!names.includes(state.tlName)) state.tlName = names[0] || "";
  $("tl-name-title").textContent = group === "region" ? "Region" : "Municipality";
  $("tl-name").innerHTML = names.map((n) =>
    `<option${n === state.tlName ? " selected" : ""}>${esc(n)}</option>`).join("");
}

function tripLengthGroup(d) {
  if (!d) return null;
  if (state.tlGroup === "region") return d.regions[state.tlName] || null;
  if (state.tlGroup === "municipality") return d.municipalities[state.tlName] || null;
  return d.country;
}

function drawTripLengths() {
  if (!state.tl) return;
  const a = tripLengthGroup(state.tl), b = tripLengthGroup(state.tlCompare);
  const labels = state.tl.labels;
  if (!a || !a.total) {
    $("tl-svg").innerHTML = "";
    $("tl-stats").textContent = "No trips start here in this run.";
    return;
  }
  const W = 312, H = 200, L = 6, R = 6, T = 16, B = 30;
  const top = Math.max(...a.shares, ...(b ? b.shares : [0])) * 1.12 || 1;
  const slot = (W - L - R) / labels.length;
  const y = (v) => H - B - (v / top) * (H - T - B);
  let svg = `<line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>`;
  labels.forEach((label, i) => {
    const x = L + i * slot, v = a.shares[i];
    svg += `<rect class="bar" x="${x + slot * 0.14}" y="${y(v)}" width="${slot * 0.72}" height="${H - B - y(v)}">
        <title>${label} km: ${pct(v)} (${fmt(a.trips[i])} trips)</title></rect>
      <text class="val" x="${x + slot / 2}" y="${y(v) - 3}" text-anchor="middle">${fmt(100 * v, 0)}</text>
      <text x="${x + slot / 2}" y="${H - B + 13}" text-anchor="middle">${label}</text>`;
    if (b) {
      const w = b.shares[i];
      svg += `<rect class="bar-b" x="${x + slot * 0.14}" y="${y(w)}" width="${slot * 0.72}" height="${H - B - y(w)}">
        <title>${esc(state.compare)}: ${pct(w)}</title></rect>`;
    }
  });
  svg += `<text x="${W / 2}" y="${H - 3}" text-anchor="middle">km on the network · bars: % of trips</text>`;
  $("tl-svg").innerHTML = svg;
  const where = state.tlGroup === "country" ? "Whole country" : state.tlName;
  const all = a.total + a.unreached;
  const members = a.municipalities ? ` (${a.municipalities.length} municipalities)` : "";
  $("tl-stats").innerHTML = `<b>${esc(where)}</b>${esc(members)}: ${fmt(a.total)} trips/day,
    mean ${fmt(a.mean_km, 2)} km${b && b.mean_km != null ? ` (B: ${fmt(b.mean_km, 2)} km)` : ""}.
    Intrazonal (estimated) ${pct(a.intrazonal / all)} · beyond the cutoff ${pct(a.unreached / all)}.
    ${b ? `<br><span style="color:#e8590c">▭</span> outline: ${esc(state.compare)}` : ""}`;
}

function clearFlow() {
  state.flowIndex = null;
  state.flowName = "";
  state.pollToken++;
  refresh();
}

async function counterFlow(p) {
  const token = ++state.pollToken;
  const box = $("flow-status");
  const started = Date.now();
  let res = await fetch(`/api/counter-flow/${p.index}`, { method: "POST" });
  while (token === state.pollToken) {
    if (!res.ok) { box.textContent = `Failed: ${res.status} ${await res.text()}`; return; }
    const s = await res.json();
    if (s.state === "done") {
      const sm = s.summary || {};
      const tops = (rows) => (rows || []).slice(0, 5).map(([z, t]) => `${z} (${fmt(t, 1)})`).join(", ") || "–";
      box.innerHTML = `<table>
          ${row("Trips through the counter", fmt(sm.trips, 1), "num")}
          ${row("OD pairs", fmt(sm.od_pairs), "num")}
          ${row("Origins searched", `${fmt(sm.origins_searched)} of ${fmt(sm.origins_total)}`, "num")}
          ${sm.seconds != null ? row("Computed in", `${fmt(sm.seconds, 0)} s`, "num") : ""}
        </table>
        <p><b>Largest origins:</b> ${tops(sm.top_origins)}<br><b>Largest destinations:</b> ${tops(sm.top_destinations)}</p>`;
      state.flowIndex = p.index;
      state.flowName = p.name;
      await refresh();
      return;
    }
    if (s.state === "error") { box.textContent = `Failed: ${s.error}`; return; }
    const secs = Math.round((Date.now() - started) / 1000);
    box.textContent = `${s.state === "queued" ? "Queued" : "Computing"}${s.step ? `: ${s.step}` : ""} (${secs} s)…`;
    await new Promise((r) => setTimeout(r, 2000));
    if (token !== state.pollToken) return;
    res = await fetch(`/api/counter-flow/${p.index}`);
  }
}

function showCounter(p) {
  const links = p.links.map((i) => `<a class="link" data-i="${i}">${i}</a> (${fmt(value(i))})`).join(", ");
  $("details").innerHTML = `
    <h2>${p.name}</h2>
    <table>
      ${row("City", p.city ?? "–")}
      ${row("Observed (annual avg/day)", fmt(p.observed), "num")}
      ${row("Model", fmt(p.modeled), "num")}
      ${row("Model ÷ observed", fmt(p.ratio, 2), "num")}
    </table>
    <h2>Links on the count line</h2><p>${links || "none"}</p>
    <h2>Trips through this counter</h2>
    <div id="flow-status">${state.meta.select ? "Starting…" : "Start valma view with --demand and --centroids to map them."}</div>`;
  $("details").querySelectorAll("a.link").forEach((a) =>
    a.addEventListener("click", () => inspectLink(Number(a.dataset.i))));
  if (state.meta.select) counterFlow(p);
}

async function loadCounters() {
  if (!state.meta.counters) { state.counters = null; return; }
  const res = await fetch(`/api/counters?run=${encodeURIComponent(state.run)}`);
  state.counters = await res.json();
}

// -- start ---------------------------------------------------------------

(async function main() {
  state.meta = await (await fetch("/api/meta")).json();
  const m = state.meta;
  $("summary").textContent = `${m.mode} · ${m.links.toLocaleString()} links`;
  for (const id of ["run", "compare"]) {
    for (const name of m.runs) $(id).insertAdjacentHTML("beforeend", `<option>${name}</option>`);
  }
  state.run = m.runs[0] || "";
  if (!m.runs.length) { state.measure = "speed"; $("measure").value = "speed"; }

  const map = new maplibregl.Map({
    container: "map",
    style: "https://tiles.openfreemap.org/styles/positron",
    bounds: [[m.bounds[0], m.bounds[1]], [m.bounds[2], m.bounds[3]]],
    hash: true,
  });
  map.addControl(new maplibregl.NavigationControl(), "top-left");
  map.addControl(new maplibregl.ScaleControl(), "bottom-right");
  map.addControl(overlay);
  state.map = map;
  map.on("click", onMapClick);
  if (m.route) {
    $("route-toggle").addEventListener("click", () => (state.routeMode ? clearRoute() : startRouting()));
    $("route-clear").addEventListener("click", clearRoute);
  } else {
    $("route-box").hidden = true;
  }

  $("run").addEventListener("change", async (e) => {
    state.run = e.target.value; await loadCounters(); refresh(); loadMunicipalities(); loadTripLengths();
  });
  $("measure").addEventListener("change", (e) => { state.measure = e.target.value; refresh(); });
  $("compare").addEventListener("change", (e) => {
    state.compare = e.target.value; refresh(); loadMunicipalities(); loadTripLengths();
  });
  $("tl-group").addEventListener("change", (e) => {
    state.tlGroup = e.target.value;
    if (state.tlGroup === "municipality" && state.muniSelected >= 0 && state.munis) {
      state.tlName = state.munis.rows[state.muniSelected].name;
    }
    tripLengthOptions();
    drawTripLengths();
  });
  $("tl-name").addEventListener("change", (e) => { state.tlName = e.target.value; drawTripLengths(); });
  $("muni-search").addEventListener("input", () => state.munis && municipalityTable());
  $("plot-open").addEventListener("click", () => { $("plot").hidden = false; drawPlot(); });
  $("plot-close").addEventListener("click", () => { $("plot").hidden = true; });
  $("minvalue").addEventListener("input", (e) => {
    state.minValue = Number(e.target.value);
    $("minvalue-out").textContent = e.target.value;
    render();
  });
  $("counters").addEventListener("change", (e) => { state.showCounters = e.target.checked; render(); legend(); });

  await loadCounters();
  await refresh();
  loadMunicipalities();
  loadTripLengths();
})().catch((err) => status(`Error: ${err.message}`));
