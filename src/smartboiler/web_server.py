# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Flask web dashboard for SmartBoiler add-on.
# Exposed via HA ingress on port 8099.
# Security: HA Supervisor handles auth before forwarding; we add rate limiting + input validation.

import logging
import threading
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from flask import Flask, Response, abort, jsonify, render_template_string, request

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Injected state provider — set via set_state_provider() before starting
_state_provider: Optional[Callable[[], Dict]] = None


def set_state_provider(provider: Callable[[], Dict]) -> None:
    """Inject a callable that returns current system state for the dashboard."""
    global _state_provider
    _state_provider = provider


def _get_state() -> Dict:
    if _state_provider is None:
        return {}
    try:
        return _state_provider()
    except Exception as e:
        logger.error("State provider error: %s", e)
        return {}


# Injected extra data provider (history / accuracy / predictor)
_extra_provider: Optional[Callable[[str, Dict], Dict]] = None


def set_extra_provider(provider: Callable[[str, Dict], Dict]) -> None:
    """Inject a callable for extended data queries (history, accuracy, predictor)."""
    global _extra_provider
    _extra_provider = provider


def _get_extra(endpoint: str, params: Dict) -> Dict:
    if _extra_provider is None:
        return {}
    try:
        return _extra_provider(endpoint, params) or {}
    except Exception as e:
        logger.error("Extra provider error (%s): %s", endpoint, e)
        return {}



# Injected CalendarManager — set via set_calendar_manager() before starting
_calendar_manager: Optional[Any] = None


def set_calendar_manager(cm: Any) -> None:
    """Inject the CalendarManager so calendar routes can use it."""
    global _calendar_manager
    _calendar_manager = cm

# ── Rate limiting (simple token bucket, no external dependencies) ─────────

_request_counts: Dict[str, int] = {}
_request_lock = threading.Lock()
RATE_LIMIT = 120  # max requests per IP per sliding window


def _check_rate_limit(ip: str, limit: int = RATE_LIMIT) -> bool:
    with _request_lock:
        count = _request_counts.get(ip, 0)
        if count >= limit:
            return False
        _request_counts[ip] = count + 1
    return True


def _reset_rate_limits() -> None:
    with _request_lock:
        _request_counts.clear()


def _rate_limited() -> bool:
    ip = request.remote_addr or "unknown"
    if not _check_rate_limit(ip):
        logger.warning("Rate limit exceeded for %s", ip)
        return True
    return False


# ── Dashboard HTML ────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>SmartBoiler</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"
          crossorigin="anonymous"></script>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    :root{
      --blue:#2b6cb0;--blue-l:#ebf8ff;--green:#276749;--green-l:#c6f6d5;
      --orange:#c05621;--orange-l:#feebc8;--red:#9b2c2c;--red-l:#fed7d7;
      --gray:#718096;--bg:#f0f4f8;--card:#fff;--border:#e2e8f0;
    }
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background:var(--bg);color:#2d3748}
    header{background:var(--blue);color:#fff;padding:.75rem 1.5rem;
           display:flex;align-items:center;gap:.75rem}
    header h1{font-size:1.2rem;font-weight:700}
    .dot{width:12px;height:12px;border-radius:50%;background:#a0aec0;flex-shrink:0}
    .dot.off{background:#fc8181}.dot.on_idle{background:#f6e05e}.dot.on_heating{background:#68d391}
    #ts{margin-left:auto;font-size:.78rem;opacity:.75}

    /* Tabs */
    .tabs{background:#1a4a8a;display:flex;padding:0 1rem}
    .tab-btn{color:rgba(255,255,255,.65);background:none;border:none;
             border-bottom:2px solid transparent;padding:.6rem 1.1rem;
             font-size:.88rem;cursor:pointer;transition:all .15s}
    .tab-btn:hover{color:#fff}
    .tab-btn.active{color:#fff;border-bottom-color:#90cdf4}

    /* Tab panes */
    .tab-pane{display:none}.tab-pane.active{display:block}

    /* Main grid */
    main{max-width:1400px;margin:1.25rem auto;padding:0 1rem;
         display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1.25rem}

    /* Cards */
    .card{background:var(--card);border-radius:12px;padding:1.25rem;
          box-shadow:0 1px 4px rgba(0,0,0,.08)}
    .card h2{font-size:.78rem;color:var(--gray);text-transform:uppercase;
             letter-spacing:.06em;margin-bottom:.9rem}
    .big{font-size:2.4rem;font-weight:700;color:var(--blue);line-height:1}
    .unit{font-size:1rem;color:#a0aec0}
    .badge{display:inline-block;padding:.2rem .7rem;border-radius:999px;
           font-size:.78rem;font-weight:600;margin-top:.6rem}
    .badge.unavailable{background:#e2e8f0;color:#4a5568}
    .badge.off{background:var(--red-l);color:var(--red)}
    .badge.on_idle{background:#fefcbf;color:#744210}
    .badge.on_heating{background:var(--green-l);color:var(--green)}
    .row{display:flex;justify-content:space-between;padding:.3rem 0;
         border-bottom:1px solid #f7fafc;font-size:.88rem}
    .row:last-child{border-bottom:none}

    /* 24h grid */
    .hgrid{display:grid;grid-template-columns:repeat(24,1fr);gap:1px;margin-top:.5rem}
    .hcell{height:32px;border-radius:3px;display:flex;align-items:center;
           justify-content:center;font-size:.6rem;font-weight:600;color:#fff;cursor:default}
    .hcell.heat{background:#f6ad55}.hcell.pv{background:#68d391}
    .hcell.hdo{background:#fc8181}.hcell.idle{background:#e2e8f0;color:var(--gray)}
    .hcell.cal-vacation{background:#bee3f8;color:#2a69ac}
    .hcell.cal-off{background:#b2b2b2;color:#fff}
    .hcell.cal-boost{background:#d6bcfa;color:#44337a}

    /* Calendar events */
    .evt-list{margin-top:.5rem}
    .evt-row{display:flex;align-items:center;gap:.5rem;padding:.35rem 0;border-bottom:1px solid #f0f4f8;font-size:.84rem}
    .evt-row:last-child{border-bottom:none}
    .evt-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
    .evt-info{flex:1;min-width:0}
    .evt-title{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .evt-time{font-size:.75rem;color:#718096}
    .evt-del{background:none;border:none;cursor:pointer;color:#fc8181;font-size:1rem;padding:0 .25rem}
    .btn-add{display:inline-block;margin-top:.75rem;padding:.4rem .9rem;background:#2b6cb0;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:.85rem}
    .btn-add:hover{background:#2c5282}
    /* Modal */
    .overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:100;align-items:center;justify-content:center}
    .overlay.open{display:flex}
    .modal{background:#fff;border-radius:14px;padding:1.5rem;width:min(420px,92vw);box-shadow:0 8px 32px rgba(0,0,0,.15)}
    .modal h3{margin-bottom:1rem;font-size:1rem}
    .form-row{margin-bottom:.75rem}
    .form-row label{display:block;font-size:.8rem;color:#718096;margin-bottom:.25rem}
    .form-row input,.form-row select{width:100%;padding:.4rem .6rem;border:1px solid #e2e8f0;border-radius:6px;font-size:.9rem}
    .modal-btns{display:flex;gap:.5rem;justify-content:flex-end;margin-top:1rem}
    .btn-cancel{padding:.4rem .9rem;border:1px solid #e2e8f0;border-radius:8px;background:#fff;cursor:pointer;font-size:.85rem}
    .btn-save{padding:.4rem .9rem;background:#2b6cb0;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:.85rem}

    /* Legend */
    .legend{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.6rem}
    .lbadge{font-size:.75rem;padding:.15rem .6rem;border-radius:6px;font-weight:600}

    /* Span helpers */
    .span2{grid-column:span 2}.span3{grid-column:span 3}
    .full{grid-column:1/-1}
    @media(max-width:700px){.span2,.span3,.full{grid-column:span 1}}

    /* Charts */
    canvas{max-height:220px}.chart-lg canvas{max-height:300px}

    /* Period/group selector */
    .period-bar{display:flex;align-items:center;gap:.5rem;
                padding:1rem 1.5rem .25rem;max-width:1400px;margin:0 auto;flex-wrap:wrap}
    .period-bar span{font-size:.82rem;color:var(--gray)}
    .pbtn{padding:.3rem .85rem;border-radius:6px;border:1px solid var(--border);
          background:#fff;font-size:.82rem;cursor:pointer;color:var(--gray);transition:all .15s}
    .pbtn.active{background:var(--blue);color:#fff;border-color:var(--blue)}

    /* Heatmap */
    .hm-wrap{overflow-x:auto;margin-top:.5rem}
    .hm{display:grid;grid-template-columns:36px repeat(24,1fr);gap:1px;min-width:560px}
    .hm-row-label{font-size:.7rem;color:var(--gray);display:flex;align-items:center;
                  justify-content:flex-end;padding-right:4px}
    .hm-cell{height:22px;border-radius:2px;cursor:default}
    .hm-cell:hover{outline:1px solid #4a5568}
    .hm-col-hdr{height:18px;display:flex;align-items:center;justify-content:center;
                font-size:.65rem;color:var(--gray)}

    /* Stat summary cards */
    .stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.75rem;margin-bottom:1rem}
    .stat-card{background:var(--blue-l);border-radius:8px;padding:.75rem 1rem;text-align:center}
    .stat-card .val{font-size:1.6rem;font-weight:700;color:var(--blue)}
    .stat-card .lbl{font-size:.75rem;color:var(--gray);margin-top:.2rem}

    /* Generic */
    .no-data{text-align:center;color:var(--gray);padding:2rem;font-size:.9rem}
    footer{text-align:center;color:#a0aec0;font-size:.78rem;padding:1.5rem 0}
    table{border-collapse:collapse;width:100%;font-size:.85rem}
    th,td{padding:.35rem .5rem;text-align:left;border-bottom:1px solid var(--border)}
    th{font-weight:600;font-size:.78rem;color:var(--gray);text-transform:uppercase;letter-spacing:.04em}
    td.num{text-align:right}
  </style>
</head>
<body>
<header>
  <div class="dot" id="dot"></div>
  <h1>SmartBoiler Dashboard</h1>
  <span id="ts"></span>
</header>
<nav class="tabs">
  <button class="tab-btn active" data-tab="overview"   onclick="switchTab('overview')">Overview</button>
  <button class="tab-btn"        data-tab="history"    onclick="switchTab('history')">History</button>
  <button class="tab-btn"        data-tab="prediction" onclick="switchTab('prediction')">Prediction</button>
  <button class="tab-btn"        data-tab="statistics" onclick="switchTab('statistics')">Statistics</button>
</nav>

<!-- ═══════════════════ OVERVIEW ════════════════════════════════════════ -->
<div id="tab-overview" class="tab-pane active">
<main>
  <div class="card">
    <h2>Boiler Status</h2>
    <div class="big" id="temp">—<span class="unit"> °C</span></div>
    <div id="relay-badge" class="badge off">💤 OFF</div>
    <div style="margin-top:.9rem">
      <div class="row"><span>Set temp</span><span id="set-tmp">—</span></div>
      <div class="row"><span>Min temp</span><span id="min-tmp">—</span></div>
      <div class="row"><span>Heating until</span><span id="heat-until">—</span></div>
      <div class="row"><span>Last legionella</span><span id="leg">—</span></div>
      <div class="row"><span>Predictor data</span><span id="pred-data">—</span></div>
    </div>
  </div>

  <div class="card span2">
    <h2>24-Hour Heating Plan</h2>
    <div class="hgrid" id="hgrid"></div>
    <div class="legend">
      <span class="lbadge" style="background:#f6ad55;color:#fff">🔥 Heating</span>
      <span class="lbadge" style="background:#68d391;color:#fff">☀️ PV free</span>
      <span class="lbadge" style="background:#fc8181;color:#fff">⛔ HDO</span>
      <span class="lbadge" style="background:#e2e8f0;color:var(--gray)">💤 Idle</span>
      <span class="lbadge" style="background:#bee3f8;color:#2a69ac">🏖 Vacation</span>
      <span class="lbadge" style="background:#b2b2b2;color:#fff">🚫 Off</span>
      <span class="lbadge" style="background:#d6bcfa;color:#44337a">⚡ Boost</span>
    </div>
  </div>

  <div class="card span2">
    <h2>Consumption Forecast — next 24 h</h2>
    <canvas id="cChart"></canvas>
  </div>

  <div class="card">
    <h2>Spot Electricity Prices (EUR/MWh)</h2>
    <canvas id="pChart"></canvas>
  </div>

  <div class="card">
    <h2>HDO Schedule (learned)</h2>
    <div id="hdo"></div>
  </div>

  <div class="card">
    <h2>System Info</h2>
    <div id="sysinfo"></div>
  </div>
</main>
</div>

<!-- ═══════════════════ HISTORY ══════════════════════════════════════════ -->
<div id="tab-history" class="tab-pane">
<div class="period-bar" id="hist-pbar">
  <span>Period:</span>
  <button class="pbtn"        data-p="1d"  onclick="setHistPeriod('1d')">1 day</button>
  <button class="pbtn active" data-p="7d"  onclick="setHistPeriod('7d')">7 days</button>
  <button class="pbtn"        data-p="30d" onclick="setHistPeriod('30d')">30 days</button>
  <button class="pbtn"        data-p="90d" onclick="setHistPeriod('90d')">90 days</button>
</div>
<main>
  <div class="card full chart-lg">
    <h2>Water Consumption &amp; Heating Activity</h2>
    <div id="hist-nodata" class="no-data" style="display:none">No history data yet.</div>
    <canvas id="histChart"></canvas>
  </div>
  <div class="card span2 chart-lg">
    <h2>Daily Heating Hours vs Consumption</h2>
    <canvas id="heatHoursChart"></canvas>
  </div>
  <div class="card">
    <h2>Period Summary</h2>
    <div id="hist-summary"><p class="no-data">Loading…</p></div>
  </div>
</main>
</div>

<!-- ═══════════════════ PREDICTION ═══════════════════════════════════════ -->
<div id="tab-prediction" class="tab-pane">
<main>
  <div class="card full">
    <h2>Consumption Heatmap — Predicted kWh (Weekday × Hour of Day)</h2>
    <div class="hm-wrap"><div class="hm" id="heatmap"></div></div>
    <div class="legend" style="margin-top:.75rem">
      <span style="font-size:.78rem;color:var(--gray)">
        Color: white = no consumption · dark blue = high consumption
      </span>
    </div>
  </div>
  <div class="card span2 chart-lg">
    <h2>Predicted vs Actual Mean — by Hour of Day</h2>
    <div id="acc-nodata" class="no-data" style="display:none">Not enough data yet.</div>
    <canvas id="accChart"></canvas>
  </div>
  <div class="card">
    <h2>Prediction Accuracy</h2>
    <div id="acc-stats"><p class="no-data">Loading…</p></div>
  </div>
</main>
</div>

<!-- ═══════════════════ STATISTICS ═══════════════════════════════════════ -->
<div id="tab-statistics" class="tab-pane">
<div class="period-bar">
  <span>Group by:</span>
  <button class="pbtn active" data-g="daily"   onclick="setStatGroup('daily')">Daily</button>
  <button class="pbtn"        data-g="weekly"  onclick="setStatGroup('weekly')">Weekly</button>
  <button class="pbtn"        data-g="monthly" onclick="setStatGroup('monthly')">Monthly</button>
</div>
<main>
  <div class="card full">
    <h2>Overall Summary (last 90 days)</h2>
    <div class="stat-grid" id="stat-cards"><p class="no-data">Loading…</p></div>
  </div>
  <div class="card full chart-lg">
    <h2>Consumption Over Time</h2>
    <canvas id="statChart"></canvas>
  </div>
  <div class="card full">
    <h2>Detailed Table (most recent 30 entries)</h2>
    <div id="stat-table" style="overflow-x:auto"></div>
  </div>
  <!-- Calendar Events -->
  <div class="card span2" id="cal-card">
    <h2>Calendar Events</h2>
    <div class="evt-list" id="cal-list"><p style="color:#a0aec0;font-size:.85rem">No calendar configured</p></div>
    <button class="btn-add" id="btn-add-evt" style="display:none" onclick="openModal()">+ Add Event</button>
  </div>
</main>
</div>

<!-- Add Event Modal -->
<div class="overlay" id="modal-overlay" onclick="closeModalIfBg(event)">
  <div class="modal">
    <h3>Add Boiler Calendar Event</h3>
    <div class="form-row">
      <label>Event Type</label>
      <select id="evt-type" onchange="onTypeChange()">
        <option value="vacation_min">Vacation (min temp)</option>
        <option value="vacation_off">Off (legionella only)</option>
        <option value="boost_max">Boost (max temp)</option>
        <option value="boost_temp">Boost (specific temp)</option>
      </select>
    </div>
    <div class="form-row">
      <label>Start (YYYY-MM-DD HH:MM)</label>
      <input type="text" id="evt-start" placeholder="2026-03-20 08:00">
    </div>
    <div class="form-row">
      <label>End (YYYY-MM-DD HH:MM)</label>
      <input type="text" id="evt-end" placeholder="2026-03-27 20:00">
    </div>
    <div class="form-row" id="row-target" style="display:none">
      <label>Target Temperature (°C)</label>
      <input type="number" id="evt-target" min="30" max="95" value="65">
    </div>
    <div class="modal-btns">
      <button class="btn-cancel" onclick="closeModal()">Cancel</button>
      <button class="btn-save" onclick="saveEvent()">Save</button>
    </div>
  </div>
</div>

<footer>Auto-refreshes every 60 s &middot; SmartBoiler</footer>

<script>
// ── App state ──────────────────────────────────────────────────────────────
const S = {
  activeTab: 'overview',
  histPeriod: '7d',
  statGroup: 'daily',
  cache: {},       // cacheKey → data
  charts: {},      // canvasId → Chart instance
};

// ── Chart helpers ──────────────────────────────────────────────────────────
function mkChart(id, cfg) {
  if (S.charts[id]) { S.charts[id].destroy(); }
  const el = document.getElementById(id);
  if (!el) return null;
  S.charts[id] = new Chart(el, cfg);
  return S.charts[id];
}

// ── Tab switching ──────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  const btn = document.querySelector('.tab-btn[data-tab="' + name + '"]');
  if (btn) btn.classList.add('active');
  S.activeTab = name;
  if (name === 'history')    loadHistory(S.histPeriod);
  if (name === 'prediction') loadPrediction();
  if (name === 'statistics') loadStatistics();
}

// ═══════════════════════════ OVERVIEW ════════════════════════════════════
async function loadOverview() {
  try {
    const r = await fetch('api/status');
    if (!r.ok) return;
    const d = await r.json();
    renderOverview(d);
    document.getElementById('ts').textContent = 'Updated ' + new Date().toLocaleTimeString();
  } catch(e) { console.error(e); }
}

function renderOverview(d) {
  const tmp = d.boiler_temp;
  document.getElementById('temp').innerHTML =
    (tmp !== null ? tmp.toFixed(1) : '—') + '<span class="unit"> °C</span>';

  const status = d.boiler_status || 'unavailable';
  const meta = {
    unavailable: {dot:'',          label:'⚠️ Unavailable'},
    off:         {dot:'off',       label:'💤 OFF'},
    on_idle:     {dot:'on_idle',   label:'✓ On — standby'},
    on_heating:  {dot:'on_heating',label:'🔥 Heating'},
  }[status] || {dot:'', label:'⚠️ Unavailable'};

  document.getElementById('dot').className = 'dot ' + meta.dot;
  const rb = document.getElementById('relay-badge');
  rb.className = 'badge ' + status;
  rb.textContent = meta.label;

  document.getElementById('set-tmp').textContent   = (d.set_tmp  ?? '—') + ' °C';
  document.getElementById('min-tmp').textContent   = (d.min_tmp  ?? '—') + ' °C';
  document.getElementById('heat-until').textContent = d.heating_until || '—';
  document.getElementById('leg').textContent       = d.last_legionella || '—';
  document.getElementById('pred-data').textContent = d.predictor_has_data ? 'Ready' : 'Collecting…';

  // Hour grid
  const grid = document.getElementById('hgrid');
  grid.innerHTML = '';
  (d.plan_slots || []).forEach(s => {
    const c = document.createElement('div');
    const calMode = s.calendar_mode;
    let cellClass = 'idle';
    if (s.hdo_blocked)                           cellClass = 'hdo';
    else if (calMode === 'vacation_off')          cellClass = 'cal-off';
    else if (calMode === 'vacation_min')          cellClass = 'cal-vacation';
    else if (calMode === 'boost_max' || calMode === 'boost_temp') cellClass = 'cal-boost';
    else if (s.pv_free)                          cellClass = 'pv';
    else if (s.heating)                          cellClass = 'heat';
    c.className = 'hcell ' + cellClass;
    c.title = `${s.label}: ${s.heating?'heating':'idle'}${s.hdo_blocked?' (HDO)':''}${s.pv_free?' (PV)':''}`;
    c.textContent = s.label.split(':')[0];
    grid.appendChild(c);
  });

  // Consumption forecast
  mkChart('cChart', {
    type: 'bar',
    data: {
      labels: (d.forecast_24h || []).map((_,i) => i + 'h'),
      datasets: [{label:'kWh', data:(d.forecast_24h||[]).map(v=>v.toFixed(4)),
                  backgroundColor:'#90cdf4', borderRadius:4}],
    },
    options: {plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true}}, animation:false},
  });

  // Spot prices
  const pd2 = d.spot_prices_today || {};
  const pKeys = Object.keys(pd2).sort((a,b) => Number(a)-Number(b));
  if (pKeys.length > 0) {
    mkChart('pChart', {
      type: 'line',
      data: {
        labels: pKeys.map(h => h + ':00'),
        datasets: [{label:'EUR/MWh', data:pKeys.map(k=>pd2[k]),
                    borderColor:'#f6ad55', backgroundColor:'rgba(246,173,85,.1)',
                    fill:true, tension:.3}],
      },
      options: {plugins:{legend:{display:false}}, scales:{y:{beginAtZero:false}}, animation:false},
    });
  }

  // HDO
  document.getElementById('hdo').innerHTML = Object.entries(d.hdo_schedule || {}).map(([day,hrs]) =>
    `<div class="row"><span>${day}</span><span>${
      hrs.length ? hrs.map(h=>String(h).padStart(2,'0')+':00').join(', ') : '—'
    }</span></div>`
  ).join('');

  // Sys info
  document.getElementById('sysinfo').innerHTML = (d.sys_info || []).map(r =>
    `<div class="row"><span>${r[0]}</span><span>${r[1]}</span></div>`
  ).join('');

  // Show/hide Add Event button based on whether calendar is configured
  const hasCal = !!(d.calendar_entity_id);
  document.getElementById('btn-add-evt').style.display = hasCal ? 'inline-block' : 'none';
}

// ── Calendar events ────────────────────────────────────────────────────────────────────────────
const _CAL_COLORS = {
  vacation_min: '#bee3f8',
  vacation_off: '#b2b2b2',
  boost_max:    '#d6bcfa',
  boost_temp:   '#d6bcfa',
};

async function loadCalendar() {
  const list = document.getElementById('cal-list');
  try {
    const r = await fetch('api/calendar/events?days=7');
    if (!r.ok) return;
    const events = await r.json();
    if (!events.length) {
      list.innerHTML = '<p style="color:#a0aec0;font-size:.85rem">No upcoming boiler events</p>';
      return;
    }
    list.innerHTML = events.map(e => `
      <div class="evt-row">
        <span class="evt-dot" style="background:${_CAL_COLORS[e.type]||'#a0aec0'}"></span>
        <div class="evt-info">
          <div class="evt-title">${e.summary}${e.active?' <span style="color:#276749;font-size:.75rem">● now</span>':''}</div>
          <div class="evt-time">${e.start} – ${e.end} &bull; ${e.type_label}</div>
        </div>
        <button class="evt-del" title="Delete" onclick="delEvent(${JSON.stringify(e.id)})">&#x2715;</button>
      </div>`).join('');
  } catch(err) { console.error('calendar load error', err); }
}

async function delEvent(id) {
  if (!confirm('Delete this event?')) return;
  await fetch('api/calendar/events/' + encodeURIComponent(id), {method:'DELETE'});
  loadCalendar();
}

function openModal() {
  const t = new Date(); t.setDate(t.getDate()+1); t.setHours(8,0,0,0);
  const fmt = d => d.toISOString().slice(0,10) + ' ' + d.toTimeString().slice(0,5);
  document.getElementById('evt-start').value = fmt(t);
  const e = new Date(t); e.setDate(e.getDate()+7); e.setHours(20,0,0,0);
  document.getElementById('evt-end').value = fmt(e);
  document.getElementById('modal-overlay').classList.add('open');
}

function closeModal() {
  document.getElementById('modal-overlay').classList.remove('open');
}

function closeModalIfBg(ev) {
  if (ev.target.id === 'modal-overlay') closeModal();
}

function onTypeChange() {
  const t = document.getElementById('evt-type').value;
  document.getElementById('row-target').style.display = t === 'boost_temp' ? 'block' : 'none';
}

async function saveEvent() {
  const body = {
    event_type:  document.getElementById('evt-type').value,
    start:       document.getElementById('evt-start').value,
    end:         document.getElementById('evt-end').value,
  };
  if (body.event_type === 'boost_temp')
    body.target_temp = parseFloat(document.getElementById('evt-target').value);
  const r = await fetch('api/calendar/events', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  const j = await r.json();
  if (j.ok) { closeModal(); loadCalendar(); }
  else      { alert('Error: ' + (j.error || 'unknown')); }
}

// ═══════════════════════════ HISTORY ═════════════════════════════════════
function setHistPeriod(p) {
  S.histPeriod = p;
  document.querySelectorAll('#hist-pbar .pbtn').forEach(b =>
    b.classList.toggle('active', b.dataset.p === p));
  delete S.cache['hist_' + p];   // force refresh
  loadHistory(p);
}

async function loadHistory(period) {
  const key = 'hist_' + period;
  if (!S.cache[key]) {
    try {
      const r = await fetch('api/history?period=' + period);
      if (!r.ok) return;
      S.cache[key] = await r.json();
    } catch(e) { console.error(e); return; }
  }
  renderHistory(S.cache[key]);
}

function renderHistory(d) {
  const nodata = document.getElementById('hist-nodata');
  const canvas = document.getElementById('histChart');
  if (!d.labels || d.labels.length === 0) {
    nodata.style.display = '';
    canvas.style.display = 'none';
    document.getElementById('hist-summary').innerHTML = '<p class="no-data">No history data yet.</p>';
    return;
  }
  nodata.style.display = 'none';
  canvas.style.display = '';

  const hasPower = d.power_w && d.power_w.some(v => v > 0);
  const datasets = [
    {
      label: 'Water consumption (kWh)', type:'bar',
      data: d.consumption,
      backgroundColor: 'rgba(144,205,244,.75)', borderRadius: 2,
      yAxisID: 'yKwh', order: 2,
    },
    {
      label: 'Heating active (%)', type:'line',
      data: d.relay_on.map(v => +(v * 100).toFixed(1)),
      borderColor: '#f6ad55', backgroundColor: 'rgba(246,173,85,.08)',
      fill: true, tension: .3, pointRadius: 0,
      yAxisID: 'yHeat', order: 1,
    },
  ];
  if (hasPower) {
    datasets.push({
      label: 'Power (W)', type:'line',
      data: d.power_w,
      borderColor: '#fc8181', borderDash:[3,2], pointRadius:0, tension:.2,
      yAxisID: 'yPow', order: 0,
    });
  }
  mkChart('histChart', {
    data: {labels: d.labels, datasets},
    options: {
      responsive:true,
      interaction:{mode:'index', intersect:false},
      plugins:{legend:{display:true, position:'top'}},
      scales:{
        x:    {ticks:{maxTicksLimit:14, maxRotation:45}, grid:{display:false}},
        yKwh: {beginAtZero:true, title:{display:true,text:'kWh'}, position:'left'},
        yHeat:{beginAtZero:true, max:100, title:{display:true,text:'% heating'},
               position:'right', grid:{drawOnChartArea:false}},
        ...(hasPower ? {yPow:{display:false}} : {}),
      },
      animation:false,
    },
  });

  const ds = d.daily_stats || [];
  if (ds.length > 0) {
    mkChart('heatHoursChart', {
      type:'bar',
      data:{
        labels: ds.map(r => r.date),
        datasets:[
          {label:'Heating hours', data:ds.map(r=>r.heating_hours),
           backgroundColor:'#f6ad55', borderRadius:3, yAxisID:'yH'},
          {label:'kWh consumed',  data:ds.map(r=>r.consumption_kwh),
           backgroundColor:'#90cdf4', borderRadius:3, yAxisID:'yK'},
        ],
      },
      options:{
        plugins:{legend:{display:true,position:'top'}},
        scales:{
          yH:{beginAtZero:true, title:{display:true,text:'h'}, position:'left'},
          yK:{beginAtZero:true, title:{display:true,text:'kWh'}, position:'right',
              grid:{drawOnChartArea:false}},
        },
        animation:false,
      },
    });

    const totKwh  = ds.reduce((s,r) => s + r.consumption_kwh, 0);
    const totHeat = ds.reduce((s,r) => s + r.heating_hours, 0);
    const n = ds.length || 1;
    document.getElementById('hist-summary').innerHTML = [
      ['Days covered',          ds.length],
      ['Total consumption',     totKwh.toFixed(2) + ' kWh'],
      ['Total heating hours',   totHeat + ' h'],
      ['Avg daily consumption', (totKwh/n).toFixed(2) + ' kWh'],
      ['Avg daily heating',     (totHeat/n).toFixed(1) + ' h'],
    ].map(([k,v]) => `<div class="row"><span>${k}</span><span><b>${v}</b></span></div>`).join('');
  }
}

// ═══════════════════════════ PREDICTION ══════════════════════════════════
async function loadPrediction() {
  if (!S.cache.pred) {
    try {
      const [pr, ac] = await Promise.all([
        fetch('api/predictor').then(r => r.json()),
        fetch('api/accuracy').then(r => r.json()),
      ]);
      S.cache.pred = {predictor:pr, accuracy:ac};
    } catch(e) { console.error(e); return; }
  }
  renderPrediction(S.cache.pred);
}

function renderPrediction({predictor, accuracy}) {
  const days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  const hm   = predictor.heatmap || {};

  // find max for colour scale
  let maxVal = 0;
  days.forEach(d => Object.values(hm[d]||{}).forEach(c => { if(c.predicted>maxVal) maxVal=c.predicted; }));
  if (maxVal < 0.001) maxVal = 0.001;

  const hmDiv = document.getElementById('heatmap');
  hmDiv.innerHTML = '';

  // header row: empty corner + hour labels
  const corner = document.createElement('div');
  corner.className = 'hm-row-label';
  hmDiv.appendChild(corner);
  for (let h = 0; h < 24; h++) {
    const el = document.createElement('div');
    el.className = 'hm-col-hdr';
    el.textContent = h;
    hmDiv.appendChild(el);
  }

  // data rows
  days.forEach(day => {
    const label = document.createElement('div');
    label.className = 'hm-row-label';
    label.textContent = day;
    hmDiv.appendChild(label);
    for (let h = 0; h < 24; h++) {
      const cell = (hm[day]||{})[String(h)] || {predicted:0, count:0};
      const t = cell.predicted / maxVal;        // 0-1
      const L = Math.round(96 - t * 66);        // hsl lightness 96%→30%
      const el = document.createElement('div');
      el.className = 'hm-cell';
      el.style.background = `hsl(210,75%,${L}%)`;
      el.title = `${day} ${h}:00 — predicted: ${cell.predicted.toFixed(4)} kWh (${cell.count} samples)`;
      hmDiv.appendChild(el);
    }
  });

  // Predicted vs Actual chart
  const acc = accuracy.by_hour || [];
  const accCanvas = document.getElementById('accChart');
  const accNodata = document.getElementById('acc-nodata');
  if (acc.length === 0) {
    accNodata.style.display = '';
    accCanvas.style.display = 'none';
  } else {
    accNodata.style.display = 'none';
    accCanvas.style.display = '';
    mkChart('accChart', {
      type:'bar',
      data:{
        labels: acc.map(r => r.hour + ':00'),
        datasets:[
          {label:'Predicted', data:acc.map(r=>r.predicted),
           backgroundColor:'rgba(246,173,85,.85)', borderRadius:3},
          {label:'Actual mean', data:acc.map(r=>r.actual_mean),
           backgroundColor:'rgba(144,205,244,.85)', borderRadius:3},
        ],
      },
      options:{
        plugins:{legend:{display:true,position:'top'}},
        scales:{y:{beginAtZero:true, title:{display:true,text:'kWh'}}},
        animation:false,
      },
    });
  }

  // Accuracy stats panel
  const q = predictor.quantile != null ? 'p' + Math.round(predictor.quantile * 100) : '—';
  const biasVal = accuracy.bias;
  const biasStr = biasVal != null
    ? (biasVal >= 0 ? '+' : '') + biasVal.toFixed(4) + ' kWh (' + (biasVal>0?'over':'under') + '-predicts)'
    : '—';
  document.getElementById('acc-stats').innerHTML = [
    ['Quantile used',          q],
    ['Total history samples',  predictor.total_samples ?? '—'],
    ['Global fallback value',  predictor.global_fallback != null
                               ? predictor.global_fallback.toFixed(4) + ' kWh' : '—'],
    ['Mean absolute error',    accuracy.mae != null ? accuracy.mae.toFixed(4) + ' kWh' : '—'],
    ['Bias',                   biasStr],
    ['Hours with data',        acc.length + ' / 24'],
  ].map(([k,v]) => `<div class="row"><span>${k}</span><span><b>${v}</b></span></div>`).join('');
}

// ═══════════════════════════ STATISTICS ══════════════════════════════════
function setStatGroup(g) {
  S.statGroup = g;
  document.querySelectorAll('.pbtn[data-g]').forEach(b =>
    b.classList.toggle('active', b.dataset.g === g));
  if (S.cache['hist_90d']) renderStatistics(S.cache['hist_90d'], g);
}

async function loadStatistics() {
  if (!S.cache['hist_90d']) {
    try {
      const r = await fetch('api/history?period=90d');
      if (!r.ok) return;
      S.cache['hist_90d'] = await r.json();
    } catch(e) { console.error(e); return; }
  }
  renderStatistics(S.cache['hist_90d'], S.statGroup);
}

function renderStatistics(d, group) {
  const daily = d.daily_stats || [];
  if (daily.length === 0) {
    document.getElementById('stat-cards').innerHTML = '<p class="no-data">No history data yet.</p>';
    return;
  }

  // Summary cards (always over full 90d)
  const totKwh  = daily.reduce((s,r) => s + r.consumption_kwh, 0);
  const totHeat = daily.reduce((s,r) => s + r.heating_hours,   0);
  const n = daily.length || 1;
  document.getElementById('stat-cards').innerHTML = `
    <div class="stat-card"><div class="val">${totKwh.toFixed(1)}</div><div class="lbl">Total kWh (90d)</div></div>
    <div class="stat-card"><div class="val">${(totKwh/n).toFixed(2)}</div><div class="lbl">Avg kWh/day</div></div>
    <div class="stat-card"><div class="val">${totHeat}</div><div class="lbl">Heating hours (90d)</div></div>
    <div class="stat-card"><div class="val">${(totHeat/n).toFixed(1)}</div><div class="lbl">Avg h heating/day</div></div>
  `;

  // Aggregate
  let rows = daily;
  if (group === 'weekly')  rows = aggregateBy(daily, r => isoWeekMonday(r.date), 'W ');
  if (group === 'monthly') rows = aggregateBy(daily, r => r.date.slice(0,7), '');

  mkChart('statChart', {
    type:'bar',
    data:{
      labels: rows.map(r => r.label || r.date),
      datasets:[
        {label:'kWh consumed',  data:rows.map(r=>r.consumption_kwh.toFixed(3)),
         backgroundColor:'#90cdf4', borderRadius:3, yAxisID:'yK'},
        {label:'Heating hours', data:rows.map(r=>r.heating_hours),
         backgroundColor:'#f6ad55', borderRadius:3, yAxisID:'yH'},
      ],
    },
    options:{
      plugins:{legend:{display:true,position:'top'}},
      scales:{
        yK:{beginAtZero:true, title:{display:true,text:'kWh'}, position:'left'},
        yH:{beginAtZero:true, title:{display:true,text:'h'},   position:'right', grid:{drawOnChartArea:false}},
      },
      animation:false,
    },
  });

  const tableRows = rows.slice(-30).reverse().map(r => `
    <tr>
      <td>${r.label||r.date}</td>
      <td class="num">${r.consumption_kwh.toFixed(3)}</td>
      <td class="num">${r.heating_hours}</td>
    </tr>`).join('');
  document.getElementById('stat-table').innerHTML = `
    <table>
      <thead><tr><th>Period</th><th style="text-align:right">kWh consumed</th><th style="text-align:right">Heating hours</th></tr></thead>
      <tbody>${tableRows}</tbody>
    </table>`;
}

function aggregateBy(daily, keyFn, prefix) {
  const map = {};
  daily.forEach(r => {
    const key = keyFn(r);
    if (!map[key]) map[key] = {label: prefix + key, consumption_kwh:0, heating_hours:0};
    map[key].consumption_kwh += r.consumption_kwh;
    map[key].heating_hours   += r.heating_hours;
  });
  return Object.values(map).map(r => ({...r, consumption_kwh: parseFloat(r.consumption_kwh.toFixed(3))}));
}

function isoWeekMonday(dateStr) {
  const d = new Date(dateStr);
  const day = (d.getDay() + 6) % 7;
  const mon = new Date(d);
  mon.setDate(d.getDate() - day);
  return mon.toISOString().slice(0,10);
}

// ── Boot ──────────────────────────────────────────────────────────────────
loadOverview();
loadCalendar();
setInterval(loadOverview, 60000);
setInterval(loadCalendar, 120000);  // refresh calendar every 2 min
setInterval(() => fetch('api/ping').catch(()=>{}), 55000);
</script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────


@app.after_request
def add_security_headers(response: Response) -> Response:
    """Add security headers to all responses."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'"
    )
    return response


@app.route("/")
def index() -> str:
    if _rate_limited():
        abort(429)
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/status")
def api_status() -> Response:
    if _rate_limited():
        abort(429)
    try:
        state = _get_state()
        safe_keys = {
            "boiler_temp", "relay_on", "boiler_status", "set_tmp", "min_tmp",
            "heating_until", "last_legionella", "predictor_has_data",
            "forecast_24h", "plan_slots", "spot_prices_today",
            "hdo_schedule", "sys_info", "calendar_events", "calendar_entity_id",
        }
        return jsonify({k: v for k, v in state.items() if k in safe_keys})
    except Exception as e:
        logger.error("api_status error: %s", e)
        abort(500)


@app.route("/api/ping")
def api_ping() -> Response:
    return jsonify({"ok": True})


@app.route("/api/history")
def api_history() -> Response:
    if _rate_limited():
        abort(429)
    period = request.args.get("period", "7d")
    if period not in ("1d", "7d", "30d", "90d"):
        period = "7d"
    try:
        return jsonify(_get_extra("history", {"period": period}))
    except Exception as e:
        logger.error("api_history error: %s", e)
        abort(500)


@app.route("/api/accuracy")
def api_accuracy() -> Response:
    if _rate_limited():
        abort(429)
    try:
        return jsonify(_get_extra("accuracy", {}))
    except Exception as e:
        logger.error("api_accuracy error: %s", e)
        abort(500)


@app.route("/api/predictor")
def api_predictor() -> Response:
    if _rate_limited():
        abort(429)
    try:
        return jsonify(_get_extra("predictor", {}))
    except Exception as e:
        logger.error("api_predictor error: %s", e)
        abort(500)


@app.route("/api/entities")
def api_entities() -> Response:
    """Return entity list for UI-based entity selection (name + id only)."""
    if _rate_limited():
        abort(429)
    state = _get_state()
    entities = state.get("available_entities", [])
    safe = [
        {
            "entity_id": e.get("entity_id", ""),
            "name": e.get("friendly_name") or e.get("entity_id", ""),
        }
        for e in entities
        if e.get("entity_id")
    ]
    return jsonify(safe)



# ── Calendar routes ────────────────────────────────────────────────────────


@app.route("/api/calendar/events", methods=["GET"])
def api_calendar_events() -> Response:
    """Return upcoming calendar events (next 7 days)."""
    if _rate_limited():
        abort(429)
    if _calendar_manager is None:
        return jsonify([])
    try:
        days = min(int(request.args.get("days", 7)), 30)
        return jsonify(_calendar_manager.upcoming_events_json(days=days))
    except Exception as e:
        logger.error("api_calendar_events error: %s", e)
        abort(500)


@app.route("/api/calendar/events", methods=["POST"])
def api_calendar_create() -> Response:
    """Create a boiler calendar event.

    Body JSON:
      { "event_type": "vacation_min"|"vacation_off"|"boost_max"|"boost_temp",
        "start": "2026-03-15 08:00",
        "end":   "2026-03-22 20:00",
        "target_temp": 65  // only for boost_temp
      }
    """
    if _rate_limited():
        abort(429)
    if _calendar_manager is None:
        return jsonify({"ok": False, "error": "Calendar not configured"}), 503
    try:
        body = request.get_json(force=True) or {}
        event_type = str(body.get("event_type", ""))
        if event_type not in ("vacation_min", "vacation_off", "boost_max", "boost_temp"):
            return jsonify({"ok": False, "error": "Invalid event_type"}), 400

        start_raw = str(body.get("start", ""))
        end_raw   = str(body.get("end",   ""))
        try:
            start = datetime.strptime(start_raw, "%Y-%m-%d %H:%M")
            end   = datetime.strptime(end_raw,   "%Y-%m-%d %H:%M")
        except ValueError:
            return jsonify({"ok": False, "error": "Invalid date format; use YYYY-MM-DD HH:MM"}), 400

        if end <= start:
            return jsonify({"ok": False, "error": "end must be after start"}), 400

        target_temp = None
        if event_type == "boost_temp":
            try:
                target_temp = float(body.get("target_temp", 0))
                if not (30 <= target_temp <= 95):
                    raise ValueError
            except (ValueError, TypeError):
                return jsonify({"ok": False, "error": "target_temp must be 30–95"}), 400

        ok = _calendar_manager.create_event(event_type, start, end, target_temp)
        return jsonify({"ok": ok})
    except Exception as e:
        logger.error("api_calendar_create error: %s", e)
        abort(500)


@app.route("/api/calendar/events/<path:event_id>", methods=["DELETE"])
def api_calendar_delete(event_id: str) -> Response:
    """Delete a calendar event by its uid."""
    if _rate_limited():
        abort(429)
    if _calendar_manager is None:
        return jsonify({"ok": False, "error": "Calendar not configured"}), 503
    try:
        ok = _calendar_manager.delete_event(event_id)
        return jsonify({"ok": ok})
    except Exception as e:
        logger.error("api_calendar_delete error: %s", e)
        abort(500)


# ── Server start ──────────────────────────────────────────────────────────


def run_dashboard(
    host: str = "0.0.0.0",
    port: int = 8099,
    debug: bool = False,
) -> None:
    """Start the Flask dashboard. Call this in a daemon thread."""
    import threading
    # Reset rate limit buckets every minute
    def _rate_reset_loop():
        import time
        while True:
            time.sleep(60)
            _reset_rate_limits()

    t = threading.Thread(target=_rate_reset_loop, daemon=True)
    t.start()

    app.run(host=host, port=port, debug=debug, use_reloader=False)
