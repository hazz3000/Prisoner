// Mirrors prisoners/circles.py. Runs as a Web Worker for index.html and diagnose.html.
const STRATS = ["blast", "rotate", "hot", "hot_retry", "oracle"];
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r) { return Math.sqrt(-2 * Math.log(r() || 1e-12)) * Math.cos(2 * Math.PI * r()); }
function gamma(a, r) {
  if (a < 1) return gamma(a + 1, r) * Math.pow(r() || 1e-12, 1 / a);
  const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do { x = gauss(r); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = r() || 1e-12;
    if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v;
  }
}
function beta(a, b, r) { const x = gamma(a, r), y = gamma(b, r); return x / (x + y); }
function shuffle(a, r) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

function makeWorld(s, r) {
  const z = [gauss(r)];
  for (let g = 1; g < s.circles; g++) z.push(s.similarity * z[g - 1] + Math.sqrt(1 - s.similarity ** 2) * gauss(r));
  const sig = s.spread;
  const circleRate = z.map(v => s.base * Math.exp(sig * v - sig * sig / 2));
  const circleOf = new Int32Array(s.ips), rate = new Float64Array(s.ips);
  const members = Array.from({ length: s.circles }, () => []);
  for (let i = 0; i < s.ips; i++) {
    const c = Math.floor(i * s.circles / s.ips);
    circleOf[i] = c;
    rate[i] = Math.min(0.5, circleRate[c] * Math.exp(0.5 * gauss(r) - 0.125));
    members[c].push(i);
  }
  return { circleOf, rate, members };
}

function runOnce(strategy, s, w, r) {
  const n = s.ips, G = s.circles;
  const converted = new Uint8Array(n);
  const shows = Array.from({ length: n }, () => []);
  const imps = new Float64Array(G), convs = new Float64Array(G);
  const loop = [];
  for (let c = 0; c < G; c++) loop.push(...shuffle(w.members[c].slice(), r));
  let pointer = 0, totalConv = 0;
  const curve = [];
  const recent = (i, day) => { let k = 0; for (const d of shows[i]) if (day - d <= s.cooldown) k++; return k; };
  const effective = (i, day) => w.rate[i] * Math.pow(s.fatigue, recent(i, day));
  const priorA = 1, priorB = 1 / Math.max(s.base, 1e-4);
  const bestCircle = open => {
    let best = -1, pick = open[0];
    for (const c of open) {
      const l = (c - 1 + G) % G, rr = (c + 1) % G;
      const a = convs[c] + 0.5 * (convs[l] + convs[rr]);
      const b = imps[c] - convs[c] + 0.5 * (imps[l] - convs[l] + imps[rr] - convs[rr]);
      const d = beta(priorA + a, priorB + b, r);
      if (d > best) { best = d; pick = c; }
    }
    return pick;
  };
  for (let day = 0; day < s.days; day++) {
    const want = Math.min(s.perDay, n - totalConv);
    let chosen = [];
    if (strategy === "blast") {
      const pool = []; for (let i = 0; i < n; i++) if (!converted[i]) pool.push(i);
      for (let k = 0; k < want; k++) { const j = k + Math.floor(r() * (pool.length - k)); [pool[k], pool[j]] = [pool[j], pool[k]]; }
      chosen = pool.slice(0, want);
    } else if (strategy === "rotate") {
      const picked = new Set(); let steps = 0;
      while (chosen.length < want && steps < n) {
        const i = loop[pointer]; pointer = (pointer + 1) % n; steps++;
        if (!converted[i] && !picked.has(i)) { chosen.push(i); picked.add(i); }
      }
    } else if (strategy === "hot" || strategy === "hot_retry") {
      const avail = new Map();
      for (let c = 0; c < G; c++) {
        const pool = w.members[c].filter(i => !converted[i] && (strategy === "hot" || recent(i, day) === 0));
        if (pool.length) avail.set(c, shuffle(pool, r));
      }
      const chunk = Math.max(1, Math.floor(want / 25));
      while (chosen.length < want && avail.size) {
        const c = bestCircle([...avail.keys()]);
        const pool = avail.get(c);
        const take = pool.splice(0, Math.min(chunk, want - chosen.length));
        chosen.push(...take);
        if (!pool.length) avail.delete(c);
      }
      if (strategy === "hot_retry" && chosen.length < want) {
        const picked = new Set(chosen), rest = [];
        for (let i = 0; i < n; i++) if (!converted[i] && !picked.has(i)) rest.push(i);
        rest.sort((a, b) => (shows[a].length ? shows[a][shows[a].length - 1] : -1) - (shows[b].length ? shows[b][shows[b].length - 1] : -1));
        chosen.push(...rest.slice(0, want - chosen.length));
      }
    } else if (strategy === "oracle") {
      const pool = [];
      for (let i = 0; i < n; i++) if (!converted[i]) pool.push([recent(i, day) > 0 ? 1 : 0, effective(i, day), i]);
      pool.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
      chosen = pool.slice(0, want).map(x => x[2]);
    }
    for (const i of chosen) {
      const p = effective(i, day);
      shows[i].push(day);
      const c = w.circleOf[i];
      imps[c]++;
      if (r() < p) { converted[i] = 1; convs[c]++; totalConv++; }
    }
    curve.push(totalConv);
  }
  return { total: totalConv, curve, imps: Array.from(imps) };
}

function sampleLog(s) {
  // a random-blast campaign log with known settings, like diagnose.sample_log
  const r = mulberry32(s.seed), w = makeWorld(s, r);
  const converted = new Uint8Array(s.ips), shows = Array.from({ length: s.ips }, () => []);
  const host = new Int32Array(s.ips);
  w.members.forEach(m => m.forEach((i, k) => host[i] = k + 1));
  const start = Date.UTC(2026, 8, 1), lines = ["date,ip,converted"];
  for (let day = 0; day < s.days; day++) {
    const pool = []; for (let i = 0; i < s.ips; i++) if (!converted[i]) pool.push(i);
    const want = Math.min(s.perDay, pool.length);
    for (let k = 0; k < want; k++) { const j = k + Math.floor(r() * (pool.length - k)); [pool[k], pool[j]] = [pool[j], pool[k]]; }
    const ds = new Date(start + day * 86400000).toISOString().slice(0, 10);
    for (const i of pool.slice(0, want)) {
      let recent = 0; for (const d of shows[i]) if (day - d <= s.cooldown) recent++;
      const conv = r() < w.rate[i] * Math.pow(s.fatigue, recent) ? 1 : 0;
      shows[i].push(day); if (conv) converted[i] = 1;
      const c = w.circleOf[i];
      lines.push(`${ds},10.${c >> 8}.${c & 255}.${host[i]},${conv}`);
    }
  }
  return lines.join(String.fromCharCode(10));
}

onmessage = e => {
  const { s, job } = e.data;
  if (e.data.type === "sample") { postMessage({ job, csv: sampleLog(s), final: true }); return; }
  const STRATS_RUN = s.strategies || STRATS;
  const r = mulberry32(s.seed);
  const out = {}; STRATS_RUN.forEach(k => out[k] = { runs: [], curve: new Array(s.days).fill(0) });
  let ring = null;
  for (let run = 0; run < s.runs; run++) {
    const w = makeWorld(s, r);
    const runSeed = Math.floor(r() * 2 ** 31);
    STRATS_RUN.forEach(k => {
      const res = runOnce(k, s, w, mulberry32(runSeed + STRATS.indexOf(k) * 7919));
      out[k].runs.push(res.total);
      res.curve.forEach((v, d) => out[k].curve[d] += v);
      if (run === 0 && k === "hot_retry") {
        const circleRate = w.members.map(m => m.reduce((a, i) => a + w.rate[i], 0) / m.length);
        ring = { circleRate, imps: res.imps };
      }
    });
    postMessage({ job, done: run + 1, total: s.runs, out, ring, final: run + 1 === s.runs });
  }
};
