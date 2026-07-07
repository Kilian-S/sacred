"""Build the interactive multi-convoy interdiction visualiser on the REAL Kaliningrad graph:
reads multiconvoy_kali_data.json and writes the self-contained HTML artifact (data embedded)."""
import json

SCRATCH = "/private/tmp/claude-501/-Users-kilian-Kilian-ICL-Thesis-code-sacred/3614da47-f0d2-4cbf-8bff-4bff1d7b87b2/scratchpad"
DATA = open(f"{SCRATCH}/multiconvoy_kali_data.json").read()
d = json.loads(DATA)
LD = round(d["loss_det"] * 100, 1)
LM = round(d["loss_mixed"] * 100, 1)

HTML = r"""<style>
  :root{
    --ground:#080d14; --grid:#0c141f; --panel:#0f1826; --panel-2:#0b121c;
    --line:#182437; --line-bright:#243651;
    --ink:#c6d3e2; --ink-dim:#7a8ca4; --ink-faint:#4d6076;
    --sacred:#33e0c2; --sacred-dim:#1c8f7e; --danger:#ff5a4f; --danger-dim:#8f2f2a; --amber:#f0a63a; --safe:#5ec98f;
    --font-mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
    --font-sans:system-ui,-apple-system,"Segoe UI",sans-serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--font-sans)}
  body{min-height:100vh;padding:clamp(16px,3vw,40px)}
  .wrap{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:20px}
  header{display:flex;flex-direction:column;gap:8px;border-bottom:1px solid var(--line);padding-bottom:18px}
  .eyebrow{font-family:var(--font-mono);font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:var(--sacred)}
  h1{font-size:clamp(23px,4vw,36px);font-weight:650;line-height:1.05;margin:0;text-wrap:balance;letter-spacing:-.01em}
  .lede{color:var(--ink-dim);max-width:70ch;font-size:15px;line-height:1.5}
  .lede b{color:var(--ink);font-weight:600}
  .spec{font-family:var(--font-mono);font-size:12px;color:var(--ink-faint);display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:2px}
  .spec b{color:var(--ink);font-weight:600}
  .console{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(280px,1fr);gap:18px}
  @media (max-width:840px){.console{grid-template-columns:1fr}}
  .map{position:relative;background:radial-gradient(130% 120% at 50% 0%,#0c1521 0%,#060a10 100%);
       border:1px solid var(--line-bright);border-radius:12px;overflow:hidden}
  .map svg{display:block;width:100%;height:auto}
  .maptag{position:absolute;top:10px;left:13px;font-family:var(--font-mono);font-size:10.5px;letter-spacing:.2em;text-transform:uppercase;color:var(--ink-faint)}
  .banner{position:absolute;left:50%;bottom:12px;transform:translateX(-50%);font-family:var(--font-mono);font-size:12.5px;letter-spacing:.13em;text-transform:uppercase;padding:6px 15px;border-radius:6px;border:1px solid;white-space:nowrap;opacity:0;transition:opacity .25s}
  .banner.show{opacity:1}
  .banner.fail{color:var(--danger);border-color:var(--danger-dim);background:rgba(255,90,79,.10)}
  .banner.ok{color:var(--sacred);border-color:var(--sacred-dim);background:rgba(51,224,194,.10)}
  .rail{display:flex;flex-direction:column;gap:14px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 15px}
  .card h2{font-family:var(--font-mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--ink-faint);margin:0 0 11px;font-weight:600}
  .toggle{display:grid;grid-template-columns:1fr 1fr;gap:6px;background:var(--panel-2);border:1px solid var(--line);border-radius:8px;padding:4px}
  .toggle button{appearance:none;border:0;background:transparent;color:var(--ink-dim);font-family:var(--font-mono);font-size:12px;padding:9px 6px;border-radius:5px;cursor:pointer;display:flex;flex-direction:column;gap:2px;align-items:center;transition:background .15s,color .15s}
  .toggle button small{font-size:10px;color:var(--ink-faint)}
  .toggle button[aria-pressed="true"]{background:rgba(51,224,194,.12);color:var(--sacred)}
  .toggle button[aria-pressed="true"].classical{background:rgba(240,166,58,.13);color:var(--amber)}
  .toggle button[aria-pressed="true"] small{color:inherit;opacity:.75}
  .toggle button:focus-visible{outline:2px solid var(--sacred);outline-offset:2px}
  .readout{font-family:var(--font-mono);font-size:12.5px;display:flex;flex-direction:column;gap:9px;margin-top:13px}
  .row{display:flex;justify-content:space-between;align-items:baseline;gap:10px}
  .row .k{color:var(--ink-faint);text-transform:uppercase;letter-spacing:.08em;font-size:10.5px}
  .row .v{color:var(--ink);text-align:right;font-variant-numeric:tabular-nums}
  .v.fail{color:var(--danger)} .v.ok{color:var(--sacred)}
  .meter{display:flex;flex-direction:column;gap:6px} .meter+.meter{margin-top:13px}
  .meter .top{display:flex;justify-content:space-between;align-items:baseline;font-family:var(--font-mono)}
  .meter .name{font-size:12px} .meter .name.sacred{color:var(--sacred)} .meter .name.classical{color:var(--amber)}
  .meter .pct{font-size:15px;font-weight:600;font-variant-numeric:tabular-nums}
  .track{position:relative;height:12px;border-radius:6px;background:var(--panel-2);border:1px solid var(--line);overflow:hidden}
  .fill{position:absolute;inset:0 auto 0 0;width:0%;border-radius:6px 0 0 6px;transition:width .18s linear}
  .fill.sacred{background:linear-gradient(90deg,var(--sacred-dim),var(--sacred))}
  .fill.classical{background:linear-gradient(90deg,#7a5312,var(--amber))}
  .oracle-tick{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink);opacity:.85}
  .meter .sub{font-family:var(--font-mono);font-size:10.5px;color:var(--ink-faint);display:flex;justify-content:space-between}
  .controls{display:flex;gap:8px;align-items:center}
  .controls button{appearance:none;font-family:var(--font-mono);font-size:11px;color:var(--ink-dim);background:var(--panel-2);border:1px solid var(--line);border-radius:6px;padding:8px 12px;cursor:pointer;transition:border-color .15s,color .15s}
  .controls button:hover{border-color:var(--line-bright);color:var(--ink)}
  .controls button:focus-visible{outline:2px solid var(--sacred);outline-offset:2px}
  .controls .speed{margin-left:auto;color:var(--ink-faint);font-family:var(--font-mono);font-size:11px}
  .legend{display:flex;flex-wrap:wrap;gap:8px 20px;font-family:var(--font-mono);font-size:11.5px;color:var(--ink-dim)}
  .legend .it{display:flex;align-items:center;gap:7px}
  .dot{width:10px;height:10px;border-radius:50%} .sw{width:16px;height:3px;border-radius:2px}
  footer{color:var(--ink-faint);font-size:12.5px;line-height:1.55;border-top:1px solid var(--line);padding-top:16px;max-width:82ch}
  footer b{color:var(--ink-dim);font-weight:600}
  @media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">SACRED &middot; Kaliningrad &middot; contested resupply</div>
    <h1>Multi-convoy interdiction on the Kaliningrad road network</h1>
    <p class="lede">A fleet must reach the forward base (FOB __FOB__) from base __BASE__ across the real
      street network while a hidden interdictor commits one ambush in advance. The <b>classical
      planner</b> fixes the convoys on a coordinated route assignment, predictable, so it gets
      ambushed. <b>SACRED</b> routes them by a randomised mixed strategy. Watch a sortie on the map;
      the meters run thousands in the background and settle at the game's true mission-failure rates.</p>
    <div class="spec"><span>routes <b>__NROUTES__</b></span><span>convoys <b>__N__</b></span><span>interdictor assets <b>__K__</b></span><span>interception <b>soft</b></span><span>objective <b>mission-failure = lose any convoy</b></span></div>
  </header>
  <div class="console">
    <div class="map">
      <svg id="tac" viewBox="0 0 760 500" role="img" aria-label="Kaliningrad road network with candidate convoy routes, animated convoys, and an interdiction point"></svg>
      <div class="maptag">Kaliningrad &middot; 290 intersections</div>
      <div class="banner" id="banner"></div>
    </div>
    <div class="rail">
      <div class="card">
        <h2>Routing doctrine</h2>
        <div class="toggle" role="group" aria-label="Routing doctrine">
          <button id="btn-sacred" aria-pressed="true">SACRED<small>randomised</small></button>
          <button id="btn-classical" class="classical" aria-pressed="false">Classical<small>fixed plan</small></button>
        </div>
        <div class="readout">
          <div class="row"><span class="k">Interdiction target</span><span class="v" id="ro-amb">&mdash;</span></div>
          <div class="row"><span class="k">Convoy routing</span><span class="v" id="ro-route">&mdash;</span></div>
          <div class="row"><span class="k">This sortie</span><span class="v" id="ro-out">&mdash;</span></div>
        </div>
      </div>
      <div class="card">
        <h2>Mission-failure rate &nbsp;&middot;&nbsp; live over <span id="ns" style="color:var(--ink-dim)">0</span> sorties</h2>
        <div class="meter">
          <div class="top"><span class="name sacred">SACRED (adversarial)</span><span class="pct" id="m-s" style="color:var(--sacred)">0.0%</span></div>
          <div class="track"><div class="fill sacred" id="f-s"></div><div class="oracle-tick" id="t-s"></div></div>
          <div class="sub"><span>&darr; lower is better</span><span>equilibrium __LM__%</span></div>
        </div>
        <div class="meter">
          <div class="top"><span class="name classical">Classical planner (ALNS)</span><span class="pct" id="m-c" style="color:var(--amber)">0.0%</span></div>
          <div class="track"><div class="fill classical" id="f-c"></div><div class="oracle-tick" id="t-c"></div></div>
          <div class="sub"><span>predictable &rarr; ambushed</span><span>equilibrium __LD__%</span></div>
        </div>
      </div>
      <div class="card"><h2>Control</h2>
        <div class="controls"><button id="pause">Pause</button><button id="step">Step sortie</button><span class="speed" id="spd"></span></div>
      </div>
    </div>
  </div>
  <div class="legend">
    <div class="it"><span class="dot" style="background:var(--sacred)"></span>convoy</div>
    <div class="it"><span class="dot" style="background:var(--danger)"></span>interdiction point</div>
    <div class="it"><span class="sw" style="background:var(--safe)"></span>low &rarr;</div>
    <div class="it"><span class="sw" style="background:var(--danger)"></span>high corridor exposure</div>
  </div>
  <footer>Real Kaliningrad graph (290 intersections), OD __BASE__ &rarr; __FOB__. Both doctrines play the
    <b>computed game-theoretic optimum</b> (best deterministic fleet plan vs the minimax mixed strategy),
    so the meters converge to the oracle's exact values, this shows the environment and the ground-truth
    result, not a trained network yet. The multi-convoy game gives the classical planner a genuine
    coordination problem to solve, and it still loses; that is the direction now in build.</footer>
</div>

<script>
const DATA=__DATA__;const nodes=DATA.nodes;const SVG="http://www.w3.org/2000/svg";
const tac=document.getElementById("tac"),W=760,H=500,PAD=40;
const ids=Object.keys(nodes);
const lats=ids.map(n=>nodes[n][1]);const lat0=(Math.min(...lats)+Math.max(...lats))/2*Math.PI/180;const cosL=Math.cos(lat0);
const PXs=ids.map(n=>[nodes[n][0]*cosL,nodes[n][1]]);
const minx=Math.min(...PXs.map(p=>p[0])),maxx=Math.max(...PXs.map(p=>p[0])),miny=Math.min(...PXs.map(p=>p[1])),maxy=Math.max(...PXs.map(p=>p[1]));
const scale=Math.min((W-2*PAD)/(maxx-minx),(H-2*PAD)/(maxy-miny));
const offx=(W-(maxx-minx)*scale)/2,offy=(H-(maxy-miny)*scale)/2;
function P(n){const q=[nodes[n][0]*cosL,nodes[n][1]];return[offx+(q[0]-minx)*scale,H-offy-(q[1]-miny)*scale];}
function el(t,a){const e=document.createElementNS(SVG,t);for(const k in a)e.setAttribute(k,a[k]);return e;}
function edgeVuln(e){const k=e.slice().sort().join("|");for(const it of DATA.isets)if(it.edge.slice().sort().join("|")===k)return it.vuln;return 0;}
const routeVuln=DATA.route_edges.map(re=>Math.max(...re.map(edgeVuln)));
function heat(v){const s=[[0.25,[94,201,143]],[0.55,[240,166,58]],[0.9,[255,90,79]]];let a=s[0],b=s[s.length-1];for(let i=0;i<s.length-1;i++)if(v>=s[i][0]&&v<=s[i+1][0]){a=s[i];b=s[i+1];break;}const t=Math.max(0,Math.min(1,(v-a[0])/((b[0]-a[0])||1)));const c=a[1].map((x,i)=>Math.round(x+(b[1][i]-x)*t));return`rgb(${c[0]},${c[1]},${c[2]})`;}

const gNet=el("g",{}),gRoutes=el("g",{}),gNodes=el("g",{}),gAmbush=el("g",{}),gConvoy=el("g",{});
DATA.edges.forEach(([u,v])=>{if(nodes[u]&&nodes[v]){const a=P(u),b=P(v);gNet.appendChild(el("line",{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:"var(--line-bright)","stroke-width":1,"stroke-opacity":.42}));}});
function routeGeom(r){const pts=r.map(P),seg=[];let total=0;for(let i=0;i<pts.length-1;i++){const l=Math.hypot(pts[i+1][0]-pts[i][0],pts[i+1][1]-pts[i][1]);seg.push({l,cum:total});total+=l;}return{pts,seg,total};}
const RG=DATA.routes.map(routeGeom);
function polyAt(g,t){t=Math.max(0,Math.min(1,t));const d=t*g.total;let i=0;while(i<g.seg.length-1&&g.seg[i].cum+g.seg[i].l<d)i++;const s=g.seg[i],lt=s.l>0?(d-s.cum)/s.l:0;return[g.pts[i][0]+(g.pts[i+1][0]-g.pts[i][0])*lt,g.pts[i][1]+(g.pts[i+1][1]-g.pts[i][1])*lt];}
function ambushParam(ri,edge){const r=DATA.routes[ri],g=RG[ri],k=edge.slice().sort().join("|");for(let i=0;i<r.length-1;i++)if([r[i],r[i+1]].slice().sort().join("|")===k)return(g.seg[i].cum+g.seg[i].l/2)/g.total;return null;}
DATA.routes.forEach((r,i)=>{const dd="M"+RG[i].pts.map(p=>p[0].toFixed(1)+","+p[1].toFixed(1)).join(" L");const col=heat(routeVuln[i]);gRoutes.appendChild(el("path",{d:dd,fill:"none",stroke:col,"stroke-width":9,"stroke-opacity":.07,"stroke-linecap":"round","stroke-linejoin":"round"}));gRoutes.appendChild(el("path",{d:dd,fill:"none",stroke:col,"stroke-width":2.5,"stroke-opacity":.6,"stroke-linecap":"round","stroke-linejoin":"round"}));const mp=RG[i].pts[Math.floor(RG[i].pts.length/2)];const lb=el("text",{x:mp[0],y:mp[1]-9,"text-anchor":"middle",fill:col,"font-family":"ui-monospace,monospace","font-size":10.5,"fill-opacity":.9});lb.textContent="route "+(i+1)+" · "+Math.round(routeVuln[i]*100)+"%";gRoutes.appendChild(lb);});
[[DATA.base,"var(--sacred)","BASE"],[DATA.fob,"#9fb6d4","FOB"]].forEach(([n,c,lab])=>{const[px,py]=P(n);gNodes.appendChild(el("rect",{x:px-11,y:py-11,width:22,height:22,rx:5,fill:"#0b1420",stroke:c,"stroke-width":1.7}));gNodes.appendChild(el("rect",{x:px-5,y:py-5,width:10,height:10,rx:2,fill:c,"fill-opacity":.9}));const t=el("text",{x:px,y:py-16,"text-anchor":"middle",fill:c,"font-family":"ui-monospace,monospace","font-size":11,"letter-spacing":".08em"});t.textContent=lab+" "+n;gNodes.appendChild(t);});
tac.append(gNet,gRoutes,gNodes,gAmbush,gConvoy);

function sampleDist(p){let r=Math.random(),c=0;for(let i=0;i<p.length;i++){c+=p[i];if(r<=c)return i;}return p.length-1;}
function occToRoutes(o){const out=[];o.forEach((c,r)=>{for(let i=0;i<c;i++)out.push(r);});return out;}
function drawSortie(mode){const occ=mode==="sacred"?DATA.occupancies[sampleDist(DATA.defender_mixed)]:DATA.det_occupancy;const routes=occToRoutes(occ);const iset=mode==="sacred"?sampleDist(DATA.attacker_eq):DATA.attacker_br_det;const edge=DATA.isets[iset].edge,vuln=DATA.isets[iset].vuln;const caught=routes.map(r=>{const a=ambushParam(r,edge);return a!==null&&Math.random()<vuln;});return{occ,routes,iset,edge,vuln,caught,fail:caught.some(x=>x)};}
function isetRoute(edge){const k=edge.slice().sort().join("|");return DATA.route_edges.findIndex(re=>re.some(e=>e.slice().sort().join("|")===k));}
function describe(routes){const c={};routes.forEach(r=>c[r]=(c[r]||0)+1);const keys=Object.keys(c);if(keys.length===1)return"all "+routes.length+" → route "+(+keys[0]+1);return keys.map(r=>"route "+(+r+1)+" ×"+c[r]).join(", ");}

let mode="sacred",paused=false,SPEED=1,sortie=null,t0=0,DUR=2200;
const banner=document.getElementById("banner");
function beginSortie(){sortie=drawSortie(mode);gConvoy.textContent="";gAmbush.textContent="";banner.className="banner";
  const ri=isetRoute(sortie.edge);let ambPt;const hitR=sortie.routes.find(r=>ambushParam(r,sortie.edge)!==null);
  if(hitR!==undefined)ambPt=polyAt(RG[hitR],ambushParam(hitR,sortie.edge));else ambPt=polyAt(RG[ri],ambushParam(ri,sortie.edge)??0.5);
  const amb=el("g",{});amb.appendChild(el("circle",{cx:ambPt[0],cy:ambPt[1],r:15,fill:"none",stroke:"var(--danger)","stroke-width":1.5,"stroke-opacity":.5,"stroke-dasharray":"3 4"}));
  amb.appendChild(el("line",{x1:ambPt[0]-8,y1:ambPt[1],x2:ambPt[0]+8,y2:ambPt[1],stroke:"var(--danger)","stroke-width":1.5}));
  amb.appendChild(el("line",{x1:ambPt[0],y1:ambPt[1]-8,x2:ambPt[0],y2:ambPt[1]+8,stroke:"var(--danger)","stroke-width":1.5}));
  amb.appendChild(el("circle",{cx:ambPt[0],cy:ambPt[1],r:5,fill:"var(--danger)"}));gAmbush.appendChild(amb);
  sortie._marks=sortie.routes.map(r=>{const m=el("g",{});const halo=el("circle",{cx:0,cy:0,r:8,fill:"var(--sacred)","fill-opacity":.16});const core=el("circle",{cx:0,cy:0,r:4.5,fill:"var(--sacred)",stroke:"#0b1420","stroke-width":1.3});m.append(halo,core);gConvoy.appendChild(m);return{g:m,core,halo,route:r,done:false,burst:false};});
  document.getElementById("ro-amb").textContent="route "+(ri+1)+" · "+Math.round(sortie.vuln*100)+"%";
  document.getElementById("ro-route").textContent=describe(sortie.routes);
  const o=document.getElementById("ro-out");o.textContent="in transit…";o.className="v";t0=performance.now();sortie._ended=false;}
function endSortie(){const o=document.getElementById("ro-out");o.textContent=sortie.fail?"MISSION FAILED":"delivered";o.className="v "+(sortie.fail?"fail":"ok");banner.textContent=sortie.fail?"◆ mission failed — convoy intercepted":"✓ mission accomplished";banner.className="banner show "+(sortie.fail?"fail":"ok");}
function burst(h){const s=performance.now();(function g(now){const k=(now-s)/440;if(k>=1){h.setAttribute("fill-opacity",0);return;}h.setAttribute("r",4+26*k);h.setAttribute("fill-opacity",.4*(1-k));requestAnimationFrame(g);})(s);}
function frame(now){requestAnimationFrame(frame);document.getElementById("spd").textContent=SPEED.toFixed(1)+"× · "+(paused?"paused":"running");bg();if(!sortie||paused)return;
  const t=(now-t0)/(DUR/SPEED);if(t>=1.2){beginSortie();return;}
  const pulse=gAmbush.querySelector("circle:last-child");if(pulse){const s=1+.5*Math.sin(now/140);pulse.setAttribute("r",5*s);}
  sortie._marks.forEach((m,i)=>{if(m.burst)return;const ap=ambushParam(m.route,sortie.edge);const die=sortie.caught[i]&&ap!==null;const stop=die?ap:1;const tt=Math.min(t,stop);if(tt<=1){const[x,y]=polyAt(RG[m.route],tt);m.g.setAttribute("transform","translate("+x+","+y+")");}if(die&&t>=ap&&!m.done){m.done=true;m.burst=true;m.core.setAttribute("fill","var(--danger)");m.halo.setAttribute("fill","var(--danger)");burst(m.halo);}if(!die&&t>=1&&!m.done){m.done=true;m.core.setAttribute("fill","var(--safe)");m.halo.setAttribute("fill","var(--safe)");}});
  if(t>=1&&!sortie._ended){sortie._ended=true;endSortie();}}
let nS=0,fS=0,fC=0;
function bg(){if(paused)return;for(let i=0;i<80;i++){nS++;if(drawSortie("sacred").fail)fS++;if(drawSortie("classical").fail)fC++;}const rs=fS/nS,rc=fC/nS;document.getElementById("ns").textContent=nS.toLocaleString();document.getElementById("m-s").textContent=(rs*100).toFixed(1)+"%";document.getElementById("m-c").textContent=(rc*100).toFixed(1)+"%";document.getElementById("f-s").style.width=(rs*100)+"%";document.getElementById("f-c").style.width=(rc*100)+"%";}
document.getElementById("t-s").style.left=(DATA.loss_mixed*100)+"%";document.getElementById("t-c").style.left=(DATA.loss_det*100)+"%";
function setMode(m){mode=m;document.getElementById("btn-sacred").setAttribute("aria-pressed",m==="sacred");document.getElementById("btn-classical").setAttribute("aria-pressed",m==="classical");beginSortie();}
document.getElementById("btn-sacred").onclick=()=>setMode("sacred");
document.getElementById("btn-classical").onclick=()=>setMode("classical");
document.getElementById("pause").onclick=e=>{paused=!paused;e.target.textContent=paused?"Resume":"Pause";if(!paused)t0=performance.now();};
document.getElementById("step").onclick=()=>{paused=false;beginSortie();paused=true;document.getElementById("pause").textContent="Resume";};
if(window.matchMedia("(prefers-reduced-motion:reduce)").matches)DUR=1;
beginSortie();requestAnimationFrame(frame);
</script>"""

HTML = (HTML.replace("__DATA__", DATA).replace("__BASE__", d["base"]).replace("__FOB__", d["fob"])
        .replace("__N__", str(d["N"])).replace("__K__", str(d["K"])).replace("__NROUTES__", str(len(d["routes"])))
        .replace("__LM__", str(LM)).replace("__LD__", str(LD)))
open(f"{SCRATCH}/multiconvoy_view.html", "w").write(HTML)
print(f"[written] multiconvoy_view.html  ({len(HTML)} bytes)  loss_det={LD}% loss_mixed={LM}%")
