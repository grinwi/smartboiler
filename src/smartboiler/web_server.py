# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Flask web dashboard for SmartBoiler add-on.
# Exposed via HA ingress on port 8099.
# Security: HA Supervisor handles auth before forwarding; we add rate limiting + input validation.

import logging
import threading
from typing import Callable, Dict, Optional

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
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background:#f0f4f8;color:#2d3748}
    header{background:#2b6cb0;color:#fff;padding:1rem 1.5rem;
           display:flex;align-items:center;gap:.75rem}
    header h1{font-size:1.25rem;font-weight:700}
    .dot{width:13px;height:13px;border-radius:50%;background:#a0aec0;flex-shrink:0}
    .dot.off{background:#fc8181}
    .dot.on_idle{background:#f6e05e}
    .dot.on_heating{background:#68d391}
    #ts{margin-left:auto;font-size:.8rem;opacity:.75}
    main{max-width:1280px;margin:1.25rem auto;padding:0 1rem;
         display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1.25rem}
    .card{background:#fff;border-radius:12px;padding:1.25rem;
          box-shadow:0 1px 4px rgba(0,0,0,.08)}
    .card h2{font-size:.8rem;color:#718096;text-transform:uppercase;
             letter-spacing:.06em;margin-bottom:.9rem}
    .big{font-size:2.6rem;font-weight:700;color:#2b6cb0;line-height:1}
    .unit{font-size:1rem;color:#a0aec0}
    .badge{display:inline-block;padding:.2rem .7rem;border-radius:999px;
           font-size:.78rem;font-weight:600;margin-top:.6rem}
    .badge.unavailable{background:#e2e8f0;color:#4a5568}
    .badge.off{background:#fed7d7;color:#9b2c2c}
    .badge.on_idle{background:#fefcbf;color:#744210}
    .badge.on_heating{background:#c6f6d5;color:#276749}
    .row{display:flex;justify-content:space-between;padding:.3rem 0;
         border-bottom:1px solid #f7fafc;font-size:.88rem}
    .row:last-child{border-bottom:none}
    /* 24h hour grid */
    .hgrid{display:grid;grid-template-columns:repeat(24,1fr);gap:1px;margin-top:.5rem}
    .hcell{height:32px;border-radius:3px;display:flex;align-items:center;
           justify-content:center;font-size:.6rem;font-weight:600;color:#fff;cursor:default}
    .hcell.heat{background:#f6ad55}
    .hcell.pv{background:#68d391}
    .hcell.hdo{background:#fc8181}
    .hcell.idle{background:#e2e8f0;color:#718096}
    .legend{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.6rem}
    .lbadge{font-size:.75rem;padding:.15rem .6rem;border-radius:6px;font-weight:600}
    canvas{max-height:200px}
    .span2{grid-column:span 2}
    @media(max-width:700px){.span2{grid-column:span 1}}
    footer{text-align:center;color:#a0aec0;font-size:.78rem;padding:1.5rem 0}
  </style>
</head>
<body>
<header>
  <div class="dot" id="dot"></div>
  <h1>SmartBoiler Dashboard</h1>
  <span id="ts"></span>
</header>
<main>
  <!-- Boiler status -->
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

  <!-- 24h heating plan grid -->
  <div class="card span2">
    <h2>24-Hour Heating Plan</h2>
    <div class="hgrid" id="hgrid"></div>
    <div class="legend">
      <span class="lbadge" style="background:#f6ad55;color:#fff">🔥 Heating</span>
      <span class="lbadge" style="background:#68d391;color:#fff">☀️ PV free</span>
      <span class="lbadge" style="background:#fc8181;color:#fff">⛔ HDO</span>
      <span class="lbadge" style="background:#e2e8f0;color:#718096">💤 Idle</span>
    </div>
  </div>

  <!-- Consumption forecast -->
  <div class="card span2">
    <h2>Consumption Forecast — next 24 h</h2>
    <canvas id="cChart"></canvas>
  </div>

  <!-- Spot prices -->
  <div class="card">
    <h2>Spot Electricity Prices (EUR/MWh)</h2>
    <canvas id="pChart"></canvas>
  </div>

  <!-- HDO schedule -->
  <div class="card">
    <h2>HDO Schedule (learned)</h2>
    <div id="hdo"></div>
  </div>

  <!-- System info -->
  <div class="card">
    <h2>System Info</h2>
    <div id="sysinfo"></div>
  </div>
</main>
<footer>Auto-refreshes every 60 s &middot; SmartBoiler</footer>

<script>
let cChart=null,pChart=null;

async function load(){
  try{
    const r=await fetch('api/status');
    if(!r.ok)return;
    const d=await r.json();
    render(d);
    document.getElementById('ts').textContent='Updated '+new Date().toLocaleTimeString();
  }catch(e){console.error(e);}
}

function render(d){
  // Status
  const tmp=d.boiler_temp;
  document.getElementById('temp').innerHTML=
    (tmp!==null?tmp.toFixed(1):'—')+'<span class="unit"> °C</span>';
  const status=d.boiler_status||'unavailable';
  const statusMeta={
    unavailable:{dot:'',   label:'⚠️ Unavailable'},
    off:         {dot:'off',      label:'💤 OFF'},
    on_idle:     {dot:'on_idle',  label:'✓ On — standby'},
    on_heating:  {dot:'on_heating',label:'🔥 Heating'},
  };
  const meta=statusMeta[status]||statusMeta.unavailable;
  document.getElementById('dot').className='dot '+meta.dot;
  const rb=document.getElementById('relay-badge');
  rb.className='badge '+status;
  rb.textContent=meta.label;
  document.getElementById('set-tmp').textContent=(d.set_tmp??'—')+' °C';
  document.getElementById('min-tmp').textContent=(d.min_tmp??'—')+' °C';
  document.getElementById('heat-until').textContent=d.heating_until||'—';
  document.getElementById('leg').textContent=d.last_legionella||'—';
  document.getElementById('pred-data').textContent=d.predictor_has_data?'Ready':'Collecting…';

  // Hour grid
  const grid=document.getElementById('hgrid');
  grid.innerHTML='';
  (d.plan_slots||[]).forEach(s=>{
    const c=document.createElement('div');
    c.className='hcell '+(s.hdo_blocked?'hdo':s.pv_free?'pv':s.heating?'heat':'idle');
    c.title=`${s.label}: ${s.heating?'heating':'idle'}${s.hdo_blocked?' (HDO)':''}${s.pv_free?' (PV)':''}`;
    c.textContent=s.label.split(':')[0];
    grid.appendChild(c);
  });

  // Consumption chart
  const cLabels=(d.forecast_24h||[]).map((_,i)=>i+'h');
  const cVals=(d.forecast_24h||[]).map(v=>v.toFixed(3));
  if(cChart)cChart.destroy();
  cChart=new Chart(document.getElementById('cChart'),{
    type:'bar',
    data:{labels:cLabels,datasets:[{
      label:'kWh',data:cVals,backgroundColor:'#90cdf4',borderRadius:4
    }]},
    options:{plugins:{legend:{display:false}},
             scales:{y:{beginAtZero:true}},animation:false}
  });

  // Price chart
  const pd2=d.spot_prices_today||{};
  const pKeys=Object.keys(pd2).sort((a,b)=>Number(a)-Number(b));
  const pVals=pKeys.map(k=>pd2[k]);
  if(pChart)pChart.destroy();
  if(pKeys.length>0){
    pChart=new Chart(document.getElementById('pChart'),{
      type:'line',
      data:{labels:pKeys.map(h=>h+':00'),datasets:[{
        label:'EUR/MWh',data:pVals,
        borderColor:'#f6ad55',backgroundColor:'rgba(246,173,85,.1)',
        fill:true,tension:.3
      }]},
      options:{plugins:{legend:{display:false}},
               scales:{y:{beginAtZero:false}},animation:false}
    });
  } else {
    document.getElementById('pChart').parentElement.querySelector('h2').nextElementSibling
      && (document.getElementById('pChart').style.display='none');
    document.getElementById('pChart').insertAdjacentHTML('afterend',
      '<p style="color:#a0aec0;margin-top:.5rem">No price data yet</p>');
  }

  // HDO
  const hdoDiv=document.getElementById('hdo');
  hdoDiv.innerHTML=Object.entries(d.hdo_schedule||{}).map(([day,hrs])=>
    `<div class="row"><span>${day}</span><span>${
      hrs.length?hrs.map(h=>String(h).padStart(2,'0')+':00').join(', '):'—'
    }</span></div>`
  ).join('');

  // Sys info
  document.getElementById('sysinfo').innerHTML=(d.sys_info||[]).map(r=>
    `<div class="row"><span>${r[0]}</span><span>${r[1]}</span></div>`
  ).join('');
}

load();
setInterval(load,60000);
// Reset minute counter each minute (keeps page fresh)
setInterval(()=>fetch('api/ping').catch(()=>{}),55000);
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
        # Strip any sensitive fields before returning
        safe_keys = {
            "boiler_temp", "relay_on", "boiler_status", "set_tmp", "min_tmp",
            "heating_until", "last_legionella", "predictor_has_data",
            "forecast_24h", "plan_slots", "spot_prices_today",
            "hdo_schedule", "sys_info",
        }
        safe = {k: v for k, v in state.items() if k in safe_keys}
        return jsonify(safe)
    except Exception as e:
        logger.error("api_status error: %s", e)
        abort(500)


@app.route("/api/ping")
def api_ping() -> Response:
    return jsonify({"ok": True})


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
