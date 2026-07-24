"""Embedded dependency-free assets for the localhost dashboard."""

from __future__ import annotations

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Train Guard</title>
  <link rel="stylesheet" href="/assets/app.css">
</head>
<body>
  <header>
    <div><span class="eyebrow">LOCAL CONTROL PLANE</span><h1>Train Guard</h1></div>
    <div class="toolbar">
      <select id="run-filter" aria-label="Training run"></select>
      <span id="connection" class="badge">connecting</span>
    </div>
  </header>
  <main>
    <section id="summary" class="summary"></section>
    <section class="grid">
      <article class="panel span-2">
        <div class="panel-head"><h2>Training signals</h2><span id="sample-time"></span></div>
        <div id="charts" class="charts"></div>
      </article>
      <article class="panel">
        <div class="panel-head"><h2>GPU fleet</h2></div>
        <div id="gpus" class="stack empty">No GPU samples</div>
      </article>
      <article class="panel">
        <div class="panel-head"><h2>Managed process</h2><span id="control-mode"></span></div>
        <div id="process" class="stack empty">No supervised process</div>
        <div id="controls" class="controls"></div>
      </article>
      <article class="panel span-2">
        <div class="panel-head"><h2>Alert timeline</h2><span id="alert-count"></span></div>
        <div id="alerts" class="timeline empty">No active alerts</div>
      </article>
      <article class="panel">
        <div class="panel-head"><h2>Checkpoints</h2></div>
        <div id="checkpoints" class="stack empty">No checkpoints</div>
      </article>
      <article class="panel">
        <div class="panel-head"><h2>Recovery history</h2></div>
        <div id="recoveries" class="stack empty">No recovery attempts</div>
      </article>
    </section>
  </main>
  <div id="toast" role="status"></div>
  <script src="/assets/app.js" defer></script>
</body>
</html>
"""

CSS = """
:root{color-scheme:dark;--bg:#090d12;--surface:#101721;--line:#243142;--muted:#8ea0b5;
--text:#edf4fb;--cyan:#51d7e8;--green:#55d68b;--amber:#f1b84b;--red:#ff6b75}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0,#122334 0,var(--bg) 38%);
color:var(--text);font:14px Inter,ui-sans-serif,system-ui,sans-serif;min-height:100vh}
header{height:92px;display:flex;align-items:center;justify-content:space-between;padding:0 3vw;border-bottom:1px solid var(--line);
background:rgba(9,13,18,.8);backdrop-filter:blur(18px);position:sticky;top:0;z-index:2}
h1{font-size:24px;margin:2px 0 0;letter-spacing:-.04em}h2{font-size:14px;margin:0}.eyebrow{font:10px ui-monospace,monospace;
letter-spacing:.2em;color:var(--cyan)}.toolbar{display:flex;gap:12px;align-items:center}select,button{font:inherit}
select{background:var(--surface);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px 12px}
main{padding:26px 3vw 60px;max-width:1500px;margin:auto}.summary{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:12px}
.metric,.panel{background:linear-gradient(160deg,rgba(19,29,41,.96),rgba(12,18,26,.96));border:1px solid var(--line);
border-radius:12px;box-shadow:0 16px 50px rgba(0,0,0,.18)}.metric{padding:16px}.metric label{display:block;color:var(--muted);
font-size:11px;text-transform:uppercase;letter-spacing:.08em}.metric strong{display:block;font:24px ui-monospace,monospace;margin-top:8px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.panel{padding:16px;min-height:220px}.span-2{grid-column:span 2}
.panel-head{display:flex;justify-content:space-between;align-items:center;color:var(--muted);margin-bottom:16px}.charts{display:grid;
grid-template-columns:repeat(3,1fr);gap:10px}.chart{border:1px solid var(--line);border-radius:9px;padding:10px;background:#0b1119}
.chart svg{width:100%;height:110px}.chart polyline{fill:none;stroke:var(--cyan);stroke-width:2;vector-effect:non-scaling-stroke}
.chart label{color:var(--muted);font-size:11px}.chart b{float:right;font:13px ui-monospace,monospace}.stack{display:grid;gap:8px}
.row{padding:10px;border:1px solid var(--line);border-radius:8px;background:#0c131c}.row-head{display:flex;justify-content:space-between;
gap:8px}.muted,.empty{color:var(--muted)}.bar{height:4px;background:#202b39;border-radius:4px;margin-top:8px;overflow:hidden}
.bar i{height:100%;display:block;background:var(--cyan)}.timeline{display:grid;gap:8px;max-height:390px;overflow:auto}
.alert{border-left:3px solid var(--amber);padding:10px 12px;background:#0c131c;border-radius:0 8px 8px 0}.alert.critical,
.alert.error{border-color:var(--red)}.alert.warning{border-color:var(--amber)}.badge{font:11px ui-monospace,monospace;
padding:5px 8px;border-radius:99px;border:1px solid var(--line);color:var(--muted)}.badge.live{color:var(--green);border-color:#235c42}
.controls{display:grid;grid-template-columns:repeat(2,1fr);gap:7px;margin-top:12px}.controls button{background:#172333;color:var(--text);
border:1px solid var(--line);padding:8px;border-radius:7px;cursor:pointer}.controls button:hover{border-color:var(--cyan)}
.controls button.danger{color:#ff9ca4}.controls button:disabled{opacity:.35;cursor:not-allowed}#toast{position:fixed;right:20px;bottom:20px;
padding:12px 16px;background:#152131;border:1px solid var(--line);border-radius:8px;display:none}
@media(max-width:1000px){.summary{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr 1fr}.charts{grid-template-columns:1fr}}
@media(max-width:650px){header{padding:0 16px}.grid{display:block}.panel{margin-bottom:12px}.summary{grid-template-columns:1fr 1fr}
.toolbar select{max-width:150px}}
"""

JS = r"""
const $=s=>document.querySelector(s);let currentRun="",token=sessionStorage.getItem("tg-token")||"";
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=v=>Number.isFinite(Number(v))?Number(v).toFixed(3):"--";
function card(label,value){return `<div class="metric"><label>${esc(label)}</label><strong>${esc(value)}</strong></div>`}
function rows(items,render,empty){return items?.length?items.map(render).join(""):`<div class="empty">${esc(empty)}</div>`}
function chart(name,values){const nums=(values||[]).map(Number).filter(Number.isFinite);let points="";
if(nums.length>1){const lo=Math.min(...nums),hi=Math.max(...nums),span=hi-lo||1;points=nums.map((v,i)=>
`${i*100/(nums.length-1)},${95-(v-lo)*85/span}`).join(" ")}
return `<div class="chart"><label>${esc(name)}</label><b>${nums.length?fmt(nums.at(-1)):"--"}</b>
<svg viewBox="0 0 100 100" preserveAspectRatio="none"><polyline points="${points}"/></svg></div>`}
function render(s){const latest=s.latest_sample||{},m=latest.metrics||{},proc=s.managed_process||{};
const gpuPayload=latest.gpus;const gpus=Array.isArray(gpuPayload)?gpuPayload:(Array.isArray(gpuPayload?.gpus)?gpuPayload.gpus:[]);
$("#summary").innerHTML=card("Run",s.run_id||"--")+card("Phase",s.phase||"unknown")+
card("Step",m.step??latest.global_step??"--")+card("Loss",fmt(m.loss))+card("Active alerts",s.active_alerts?.length||0);
$("#sample-time").textContent=latest.timestamp||"no samples";
$("#charts").innerHTML=chart("Loss",s.series?.loss)+chart("Grad norm",s.series?.grad_norm)+chart("Throughput",s.series?.throughput);
$("#gpus").innerHTML=rows(gpus,g=>`<div class="row"><div class="row-head"><b>GPU ${esc(g.index)}</b>
<span>${fmt(g.temperature_c)} C</span></div><div class="muted">${fmt(g.memory_used_mb)} / ${fmt(g.memory_total_mb)} MiB</div>
<div class="bar"><i style="width:${Math.max(0,Math.min(100,Number(g.utilization_gpu)||0))}%"></i></div></div>`,"No GPU samples");
$("#process").innerHTML=proc.pid?`<div class="row"><div class="row-head"><b>PID ${proc.pid}</b><span>${esc(proc.status)}</span></div>
<div class="muted">${esc(proc.capabilities?.join(", ")||"observe only")}</div></div>`:`<div class="empty">No supervised process</div>`;
$("#control-mode").textContent=s.control_enabled?"CONTROL ENABLED":"READ ONLY";
const actions=["pause","resume","graceful_stop","terminate","validated_restart"];
$("#controls").innerHTML=actions.map(a=>`<button data-action="${a}" class="${a==="terminate"?"danger":""}"
${!s.control_enabled||!proc.capabilities?.includes(a)?"disabled":""}>${a.replace("_"," ")}</button>`).join("");
$("#controls").querySelectorAll("button").forEach(b=>b.onclick=()=>command(b.dataset.action));
$("#alerts").innerHTML=rows(s.active_alerts,a=>`<div class="alert ${esc(a.event?.severity)}"><div class="row-head">
<b>${esc(a.event?.kind)}</b><span>x${esc(a.occurrence_count)}</span></div><div>${esc(a.event?.message)}</div>
<small class="muted">${esc(a.updated_at)}</small></div>`,"No active alerts");
$("#alert-count").textContent=`${s.active_alerts?.length||0} active`;
$("#checkpoints").innerHTML=rows(s.checkpoints,c=>`<div class="row"><b>${esc(c.name||c)}</b>
<div class="muted">${esc(c.status||"observed")}</div></div>`,"No checkpoints");
$("#recoveries").innerHTML=rows(s.recoveries,r=>`<div class="row"><div class="row-head"><b>${esc(r.action||r.type)}</b>
<span>${esc(r.status||r.outcome)}</span></div><small class="muted">${esc(r.created_at||r.timestamp)}</small></div>`,"No recovery attempts");
const runs=s.runs||[];const select=$("#run-filter");const before=select.value;
select.innerHTML=runs.map(r=>`<option value="${esc(r)}">${esc(r)}</option>`).join("");
select.value=currentRun||before||s.run_id||""}
async function refresh(){try{const q=currentRun?`?run_id=${encodeURIComponent(currentRun)}`:"";const r=await fetch(`/api/status${q}`);
if(!r.ok)throw Error(await r.text());render(await r.json());$("#connection").textContent="LIVE";$("#connection").className="badge live"}
catch(e){$("#connection").textContent="OFFLINE";$("#connection").className="badge"}}
async function command(action){if(!token){token=prompt("Local control token")||"";sessionStorage.setItem("tg-token",token)}
const body={run_id:currentRun||$("#run-filter").value,action};const r=await fetch("/api/commands",{method:"POST",
headers:{"Content-Type":"application/json","Authorization":`Bearer ${token}`},body:JSON.stringify(body)});
const msg=r.ok?"Command queued":`Rejected: ${await r.text()}`;$("#toast").textContent=msg;$("#toast").style.display="block";
setTimeout(()=>$("#toast").style.display="none",3000);if(r.status===401){token="";sessionStorage.removeItem("tg-token")}}
$("#run-filter").onchange=e=>{currentRun=e.target.value;refresh()};refresh();setInterval(refresh,2000);
"""

__all__ = ["CSS", "HTML", "JS"]
