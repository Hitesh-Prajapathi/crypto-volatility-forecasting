"use strict";
(function () {
  var state = { fc: null, bt: null, asset: null, horizon: "1" };

  function $(id) { return document.getElementById(id); }
  function el(tag, text, cls) {
    var e = document.createElement(tag);
    if (text !== undefined && text !== null) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }
  function fmt(x, d) { return x === null || x === undefined ? "-" : Number(x).toFixed(d); }
  function pct(x, d) { return x === null || x === undefined ? "-" : (100 * x).toFixed(d) + "%"; }
  function sci(x) { return x === null || x === undefined ? "-" : Number(x).toExponential(2); }
  function theme() {
    var s = getComputedStyle(document.documentElement);
    return {
      fg: s.getPropertyValue("--fg").trim(), grid: s.getPropertyValue("--grid").trim(),
      a1: s.getPropertyValue("--accent").trim(), a2: s.getPropertyValue("--accent2").trim(),
      muted: s.getPropertyValue("--muted").trim()
    };
  }
  function layout(t, extra) {
    var base = {
      margin: { l: 52, r: 12, t: 8, b: 40 },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: t.fg, size: 12 },
      xaxis: { gridcolor: t.grid, zerolinecolor: t.grid },
      yaxis: { gridcolor: t.grid, zerolinecolor: t.grid },
      legend: { orientation: "h", y: -0.2 }, hovermode: "x unified"
    };
    for (var k in extra) base[k] = extra[k];
    return base;
  }
  function getJSON(url) {
    return fetch(url, { cache: "no-cache" }).then(function (r) {
      if (!r.ok) throw new Error(url + ": " + r.status);
      return r.json();
    });
  }

  function horizonsPresent() {
    var hs = {};
    Object.keys(state.fc.assets).forEach(function (a) {
      Object.keys(state.fc.assets[a].horizons).forEach(function (h) { hs[h] = true; });
    });
    return Object.keys(hs).sort(function (a, b) { return a - b; });
  }

  function renderTable() {
    var body = document.querySelector("#forecast-table tbody");
    body.textContent = "";
    Object.keys(state.fc.assets).forEach(function (a) {
      var blk = state.fc.assets[a].horizons[state.horizon];
      var tr = el("tr");
      if (!blk) {
        tr.appendChild(el("td", a));
        var td = el("td", "no forecast yet"); td.colSpan = 6; tr.appendChild(td);
      } else {
        var l = blk.latest;
        tr.appendChild(el("td", a));
        tr.appendChild(el("td", l.origin_date));
        tr.appendChild(el("td", l.target_date));
        tr.appendChild(el("td", pct(l.forecast_vol, 1), "num"));
        tr.appendChild(el("td", sci(l.forecast_var), "num"));
        tr.appendChild(el("td", state.horizon === "1" ? pct(l.var99, 2) : "-", "num"));
        tr.appendChild(el("td", blk.model_name + " (v" + blk.model_version + ", " + blk.role + ")"));
      }
      body.appendChild(tr);
    });
  }

  function renderCharts() {
    var t = theme();
    var blk = state.fc.assets[state.asset] && state.fc.assets[state.asset].horizons[state.horizon];
    var vol = $("chart-vol"), ql = $("chart-qlike");
    if (!blk) {
      vol.textContent = ""; ql.textContent = "";
      $("qlike-msg").textContent = "No forecasts for this asset and horizon yet.";
      return;
    }
    var hist = blk.history;
    var x = hist.map(function (r) { return r.target_date; });
    var traces = [
      { x: x, y: hist.map(function (r) { return r.forecast_vol === null ? null : 100 * r.forecast_vol; }),
        name: "Forecast", mode: "lines+markers", line: { color: t.a1, width: 2 }, marker: { size: 4 } },
      { x: x, y: hist.map(function (r) { return r.realized_vol === null ? null : 100 * r.realized_vol; }),
        name: "Realized", mode: "lines+markers", line: { color: t.a2, width: 2 }, marker: { size: 4 },
        connectgaps: false }
    ];
    Plotly.react(vol, traces, layout(t, { yaxis: { title: "annualized vol (%)", gridcolor: t.grid } }),
      { responsive: true, displayModeBar: false });

    var c = blk.cum_qlike;
    if (c.n < c.min_days) {
      $("qlike-msg").textContent = "Cumulative QLIKE appears once " + c.min_days +
        " days are scored (" + c.n + " so far). Realized variance of day t is known the next UTC day.";
      Plotly.purge(ql);
      ql.textContent = "";
    } else {
      $("qlike-msg").textContent = c.n + " scored days; QLIKE ratio vs HAR so far: " + fmt(c.ratio_vs_har, 3) + ".";
      Plotly.react(ql, [
        { x: c.dates, y: c.champion, name: blk.model_name, mode: "lines", line: { color: t.a1, width: 2 } },
        { x: c.dates, y: c.har, name: "HAR", mode: "lines", line: { color: t.muted, width: 2, dash: "dot" } }
      ], layout(t, { yaxis: { title: "cumulative QLIKE", gridcolor: t.grid } }),
        { responsive: true, displayModeBar: false });
    }
  }

  function renderTabs() {
    var box = $("asset-tabs");
    box.textContent = "";
    Object.keys(state.fc.assets).forEach(function (a) {
      var b = el("button", a.replace("USDT", ""));
      b.type = "button"; b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", a === state.asset ? "true" : "false");
      b.addEventListener("click", function () { state.asset = a; renderTabs(); renderCharts(); });
      box.appendChild(b);
    });
  }

  function renderBacktest() {
    var bt = state.bt;
    if (!bt) return;
    var body = document.querySelector("#backtest-table tbody");
    body.textContent = "";
    Object.keys(bt.tables).forEach(function (a) {
      Object.keys(bt.tables[a]).sort().forEach(function (h) {
        var blk = bt.tables[a][h], by = {};
        blk.rows.forEach(function (r) { by[r.model] = r; });
        var x = by.xgb, d = by.log_har_dow;
        var tr = el("tr");
        tr.appendChild(el("td", a));
        tr.appendChild(el("td", h));
        tr.appendChild(el("td", String(blk.n), "num"));
        tr.appendChild(el("td", x ? fmt(x.ratio_vs_har, 3) : "-", "num"));
        tr.appendChild(el("td", d ? fmt(d.ratio_vs_har, 3) : "-", "num"));
        tr.appendChild(el("td", x ? fmt(x.ratio_vs_log_har_dow, 3) : "-", "num"));
        tr.appendChild(el("td", x ? fmt(x.dm_p_vs_log_har_dow, 3) : "-", "num"));
        body.appendChild(tr);
      });
    });
    $("backtest-note").textContent = "Honest reading: " + bt.note;
    $("backtest-src").textContent = "From " + bt.source + ".";
    $("backtest").hidden = false;
  }

  function init() {
    var fc = state.fc;
    $("load-msg").hidden = true;
    $("disclaimer").textContent = fc.disclaimer;
    $("demo-banner").hidden = !fc.demo;
    $("meta").textContent = "Generated " + fc.generated_at + " from commit " + fc.git_sha +
      ". Forecasts are out of sample: each is stored before its target day happens.";
    var hs = horizonsPresent();
    if (!hs.length) hs = fc.horizons.map(String);
    state.horizon = hs[0];
    var sel = $("horizon-select");
    hs.forEach(function (h) { sel.appendChild(el("option", h + "-day", null)).value = h; });
    sel.addEventListener("change", function () { state.horizon = sel.value; renderTable(); renderCharts(); });
    state.asset = Object.keys(fc.assets)[0];
    if (fc.empty) {
      var m = $("empty-msg");
      m.textContent = "No live forecasts yet: the track record starts with the first daily run. " +
        "Nothing is shown here until real, out-of-sample forecasts exist.";
      m.hidden = false;
      $("track-start").textContent = "Track record starts: first scheduled run (no forecasts stored yet).";
    } else {
      $("track-start").textContent = "Track record starts " + fc.track_record_start + ".";
    }
    renderTable(); renderTabs(); renderCharts();
    if (window.matchMedia) {
      var mq = window.matchMedia("(prefers-color-scheme: dark)");
      var rerender = function () { renderCharts(); };
      if (mq.addEventListener) mq.addEventListener("change", rerender);
    }
  }

  getJSON("data/forecasts.json").then(function (fc) {
    state.fc = fc;
    init();
    return getJSON("data/backtest_summary.json").then(function (bt) { state.bt = bt; renderBacktest(); },
      function () { /* backtest section stays hidden */ });
  }).catch(function (err) {
    $("load-msg").textContent = "Could not load forecasts (" + err.message + ").";
  });
})();
