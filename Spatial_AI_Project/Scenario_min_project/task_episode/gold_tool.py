# -*- coding: utf-8 -*-
"""Gold 라벨링 도구 v3 — 에피소드 앵커 (Task: episode / gold).

egomotion arc로 클립을 에피소드로 분할.
- **stop 만 GT 자동 확정**(AUTO_GT, 순수 kinematic·평가 제외).
- **회전·차선변경(turn/u_turn/lane_change)은 arc가 '트리거 힌트'일 뿐** — 교차로/신호/분기/커브
  맥락을 보고 **사람이 확정**(HUMAN_KEYS, 평가 대상). arc 자동확정 아님(CLAUDE.md §2 두 층위).
사람은 각 에피소드에 회전세부·상호작용·맥락을 태그 + '기동 밖(배경)' 카드 + ODD 클립당 1회.
기존 트랜스코드 영상(gold_label/vids)·sample_clips.json 재사용.

실행: ./run.sh task_episode/gold_tool.py  (PYTHONPATH로 common 로드)
"""
import os, sys, json
# common(taxonomy)·task_episode 경로 확보 — run.sh PYTHONPATH 밖에서 직접 실행도 대비
_ROOT = "/home/daejun/vla-tagging"
for _p in (f"{_ROOT}/common", f"{_ROOT}/task_episode", _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import taxonomy

OUTDIR = "/home/daejun/vla-tagging/gold_label"
sample = json.load(open(f"{OUTDIR}/sample_clips.json"))
episodes = json.load(open(f"{OUTDIR}/episodes.json"))

# 사람이 태그할 이벤트만 (AUTO_GT 제외), 축별.
# 이제 turn/u_turn/lane_change 가 HUMAN_KEYS 이므로 'ego 기동' 축에 노출됨(사람이 맥락으로 확정).
AXIS_RENAME = {"ego 기동": "ego 기동 (회전·차선변경 = 맥락 보고 확정)"}
human_axes = []
for ax, items in taxonomy.BY_AXIS.items():
    its = [{"k": k, "l": l, "src": s, "hint": h} for (k, l, s, h) in items if k in taxonomy.HUMAN_KEYS]
    if its:
        human_axes.append({"axis": AXIS_RENAME.get(ax, ax), "items": its})

odd_js = json.dumps([
    {"dim": d, "label": lbl, "opts": [{"k": ok_, "l": ol} for (ok_, ol) in opts]}
    for (d, lbl, opts) in taxonomy.ODD], ensure_ascii=False)
events_js = json.dumps(human_axes, ensure_ascii=False)
auto_labels = {"turn_left": "좌회전", "turn_right": "우회전", "stop": "정지", "u_turn": "U턴",
               "lane_change_left": "좌차선변경", "lane_change_right": "우차선변경"}
autogt_js = json.dumps(sorted(taxonomy.AUTO_GT))   # GT 확정 arc 태그(=stop). 나머지 회전은 힌트.
clips_js = json.dumps([
    {"id": c, "vid": f"vids/{c[:8]}.mp4", "dur": episodes[c]["dur"],
     "eps": episodes[c]["episodes"]} for c in sample], ensure_ascii=False)

HTML = r"""<!doctype html><html lang=ko><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>VLA long-tail gold 라벨링 (에피소드)</title>
<style>
:root{--paper:#f5f7f8;--panel:#fff;--ink:#151d21;--muted:#65757c;--line:#dde4e6;--accent:#0d8fa6;--ok:#2c9c68;--warn:#c98a2a;--chip:#eef2f3;--gt:#2c9c68}
@media(prefers-color-scheme:dark){:root{--paper:#0c1214;--panel:#141d21;--ink:#e6eef1;--muted:#8598a0;--line:#233036;--accent:#2ab6cc;--ok:#3fb27f;--warn:#d8a14a;--chip:#1b262b;--gt:#3fb27f}}
:root[data-theme=dark]{--paper:#0c1214;--panel:#141d21;--ink:#e6eef1;--muted:#8598a0;--line:#233036;--accent:#2ab6cc;--ok:#3fb27f;--warn:#d8a14a;--chip:#1b262b;--gt:#3fb27f}
:root[data-theme=light]{--paper:#f5f7f8;--panel:#fff;--ink:#151d21;--muted:#65757c;--line:#dde4e6;--accent:#0d8fa6;--ok:#2c9c68;--warn:#c98a2a;--chip:#eef2f3;--gt:#2c9c68}
*{box-sizing:border-box}html,body{margin:0}body{background:var(--paper);color:var(--ink);font-family:system-ui,"Noto Sans KR",sans-serif;line-height:1.45}
header{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--paper) 92%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:10px 18px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header h1{font-size:15px;margin:0;font-weight:700}
.prog{font-family:ui-monospace,monospace;font-size:12.5px;color:var(--muted)}
.bar{flex:1;min-width:100px;height:6px;background:var(--chip);border-radius:4px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--accent);width:0}
button{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:7px;padding:6px 12px;cursor:pointer}
button.p{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
button:hover{filter:brightness(1.05)}
.wrap{max-width:1120px;margin:0 auto;padding:16px 18px 60px}
.grid{display:grid;grid-template-columns:minmax(0,420px) minmax(0,1fr);gap:20px;align-items:start}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.vcol{position:sticky;top:60px}
video{width:100%;border-radius:10px;border:1px solid var(--line);background:#000}
.cid{font-family:ui-monospace,monospace;font-size:11.5px;color:var(--muted);margin-top:6px;word-break:break-all}
.odd{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:12px}
.dim{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:8px 10px}
.dim label{font-size:10.5px;color:var(--muted);display:block;margin-bottom:4px}
.dim select{width:100%;font:inherit;font-size:12px;padding:5px;border-radius:6px;border:1px solid var(--line);background:var(--paper);color:var(--ink)}
.ep{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:11px 13px;margin-bottom:12px}
.ep.bg{border-style:dashed}
.eph{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:9px}
.win{font-family:ui-monospace,monospace;font-size:12px;background:var(--chip);border-radius:6px;padding:3px 8px;cursor:pointer;border:1px solid var(--line)}
.win:hover{border-color:var(--accent)}
.gt{font-family:ui-monospace,monospace;font-size:11px;color:var(--gt);border:1px solid var(--gt);border-radius:12px;padding:2px 9px}
.hint{font-family:ui-monospace,monospace;font-size:11px;color:var(--warn);border:1px dashed var(--warn);border-radius:12px;padding:2px 9px}
.arc{font-family:ui-monospace,monospace;font-size:10.5px;color:var(--muted)}
.axis{margin:8px 0}
.axh{font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:var(--accent);font-weight:700;margin:0 0 5px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{display:inline-flex;align-items:center;gap:5px;border:1px solid var(--line);background:var(--paper);border-radius:18px;padding:4px 10px;font-size:12px;cursor:pointer;user-select:none}
.chip .src{font-family:ui-monospace,monospace;font-size:8.5px;color:var(--muted)}
.chip:hover{border-color:var(--accent)}.chip.on{background:var(--accent);border-color:var(--accent);color:#fff}.chip.on .src{color:#dff}
.note{width:100%;font:inherit;font-size:12px;padding:6px;border-radius:7px;border:1px solid var(--line);background:var(--paper);color:var(--ink);margin-top:7px}
.tnum{width:56px;font:inherit;font-size:11.5px;padding:2px 4px;border-radius:5px;border:1px solid var(--line);background:var(--paper);color:var(--ink)}
.tip{font-size:11.5px;color:var(--muted);margin:2px 0 12px}
kbd{font-family:ui-monospace,monospace;font-size:11px;background:var(--chip);border:1px solid var(--line);border-radius:4px;padding:0 5px}
h3{font-size:12px;margin:14px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
</style></head><body>
<header>
  <h1>long-tail gold · 에피소드</h1>
  <span class=prog id=prog>0/0</span>
  <div class=bar><i id=barfill></i></div>
  <button id=btnPrev>◀</button><button id=btnNext class=p>다음 ▶</button>
  <button id=btnDl>gold.json 내려받기</button>
  <button id=btnImp>이어하기</button>
  <input type=file id=fileImp accept=.json style=display:none>
</header>
<div class=wrap>
<p class=tip><b>정지(stop)만 GT 자동</b>(<span class=gt>초록</span> = egomotion). <b>회전·차선변경은 arc 힌트</b>(<span style="color:var(--warn)">주황</span>)일 뿐 — 교차로/신호/분기/커브 <b>맥락을 보고 회전 칩을 직접 확정</b>(arc가 우회전을 제안해도 실제로는 커브·분기일 수 있음). 각 에피소드 시간창(▶) 클릭 → 그 구간 재생 후 <b>회전세부·상호작용·맥락</b>을 체크(그 시간창에 자동 동기화). 기동 밖 특이상황은 <b>+ 구간 추가</b>. 자동저장. <kbd>←</kbd><kbd>→</kbd> 이동 <kbd>Space</kbd> 재생.</p>
<div class=grid>
  <div class=vcol>
    <video id=vid controls preload=metadata playsinline></video>
    <div class=cid id=cid></div>
  </div>
  <div id=eps></div>
</div>
</div>
<script>
const CLIPS=__CLIPS__, EAXES=__EVENTS__, ODD=__ODD__, AUTOL=__AUTOL__, AUTOGT=__AUTOGT__;
const HINTK=["turn_left","turn_right","u_turn","lane_change_left","lane_change_right"]; // arc 힌트(사람 확정)
const KEY="vla_gold_ep_v2";
let store=JSON.parse(localStorage.getItem(KEY)||"{}");
let idx=0, seekEnd=null;
const $=s=>document.querySelector(s);
const vid=()=>$("#vid");

function rec(){const id=CLIPS[idx].id;
  if(!store[id]){const c=CLIPS[idx];
    store[id]={odd:{},episodes:c.eps.map(e=>({win:[e.t0,e.t1],auto:e.auto,arc:e.arc,human:[],note:""})),
               manual:[]};}
  const r=store[id];
  if(!r.episodes)r.episodes=[]; if(!r.manual)r.manual=[]; if(!r.odd)r.odd={};
  // 구버전 bg → manual 이관
  if(r.bg){if(r.bg.human&&r.bg.human.length)r.manual.push({win:[0,CLIPS[idx].dur],arc:["manual"],human:r.bg.human,note:r.bg.note||"",manual:true});delete r.bg;}
  return r;
}
function save(){localStorage.setItem(KEY,JSON.stringify(store));}

function chipGroup(slot,onchange){
  // slot = {human:[...]}; returns DOM
  const box=document.createElement("div");
  for(const ax of EAXES){
    const a=document.createElement("div");a.className="axis";
    a.innerHTML=`<p class=axh>${ax.axis}</p>`;
    const cw=document.createElement("div");cw.className="chips";
    for(const it of ax.items){
      const c=document.createElement("div");c.className="chip";c.title=it.hint;
      c.innerHTML=`${it.l} <span class=src>${it.src}</span>`;
      if(slot.human.includes(it.k))c.classList.add("on");
      c.onclick=()=>{const i=slot.human.indexOf(it.k);
        if(i<0)slot.human.push(it.k);else slot.human.splice(i,1);
        c.classList.toggle("on");save();onchange&&onchange();};
      cw.appendChild(c);
    }
    a.appendChild(cw);box.appendChild(a);
  }
  return box;
}

function seek(t0,t1){const v=vid();v.currentTime=t0;seekEnd=t1;v.play();}
vid().ontimeupdate=()=>{if(seekEnd!=null&&vid().currentTime>=seekEnd){vid().pause();seekEnd=null;}};

function epCard(ep,manual,onRemove){
  const card=document.createElement("div");card.className="ep"+(manual?" bg":"");
  const h=document.createElement("div");h.className="eph";
  if(manual){
    const play=document.createElement("span");play.className="win";play.textContent="▶";
    play.onclick=()=>seek(ep.win[0],ep.win[1]);
    const s=document.createElement("input");s.type="number";s.step="0.5";s.value=ep.win[0];s.className="tnum";
    const e=document.createElement("input");e.type="number";e.step="0.5";e.value=ep.win[1];e.className="tnum";
    s.onchange=()=>{ep.win[0]=+s.value;save();};e.onchange=()=>{ep.win[1]=+e.value;save();};
    const lab=document.createElement("span");lab.className="arc";lab.textContent="수동 구간";
    h.append(play,lab,s,document.createTextNode("–"),e,document.createTextNode("s"));
    const rm=document.createElement("button");rm.textContent="삭제";rm.style.marginLeft="auto";rm.style.padding="2px 8px";
    rm.onclick=onRemove;h.appendChild(rm);
  }else{
    // stop 등 AUTO_GT 만 GT-확정(초록). 회전·차선변경은 arc 힌트(주황) — 사람이 칩으로 확정.
    const gtk=ep.arc.filter(a=>AUTOGT.includes(a));
    const hintk=ep.arc.filter(a=>HINTK.includes(a));
    const gtb=gtk.map(a=>`<span class=gt>GT ${AUTOL[a]||a}</span>`).join(" ");
    const hintb=hintk.map(a=>`<span class=hint title="arc 제안 — 교차로/분기/커브 맥락 보고 회전 칩으로 확정">arc? ${AUTOL[a]||a}</span>`).join(" ");
    const none=(!gtk.length&&!hintk.length)?`<span class=gt style="opacity:.6">GT 기동 무(주행)</span>`:"";
    h.innerHTML=`<span class=win>▶ ${ep.win[0]}–${ep.win[1]}s</span> ${gtb} ${hintb} ${none} <span class=arc>arc: ${ep.arc.join('+')}</span>`;
    h.querySelector(".win").onclick=()=>seek(ep.win[0],ep.win[1]);
  }
  card.appendChild(h);
  card.appendChild(chipGroup(ep,updateProg));
  const nt=document.createElement("textarea");nt.className="note";nt.rows=1;nt.placeholder="메모";
  nt.value=ep.note||"";nt.oninput=()=>{ep.note=nt.value;save();};
  card.appendChild(nt);
  return card;
}
function renderEps(){
  const r=rec(),c=CLIPS[idx],w=$("#eps");w.innerHTML="";
  r.episodes.forEach(ep=>w.appendChild(epCard(ep,false)));
  r.manual.forEach((ep,i)=>w.appendChild(epCard(ep,true,()=>{r.manual.splice(i,1);save();renderEps();})));
  if(r.episodes.length===0&&r.manual.length===0){
    const hint=document.createElement("div");hint.className="tip";
    hint.textContent="GT 기동 에피소드 없음(순수 주행). 특이 상황이 있으면 '구간 추가'로 시간대를 지정해 태그하세요.";
    w.appendChild(hint);
  }
  const add=document.createElement("button");add.textContent="+ 구간 추가 (기동 밖 상황)";add.style.marginTop="4px";
  add.onclick=()=>{const cur=Math.floor(vid().currentTime||0);
    r.manual.push({win:[cur,Math.min(cur+4,c.dur)],arc:["manual"],human:[],note:"",manual:true});save();renderEps();};
  w.appendChild(add);
}
function labeled(id){const r=store[id];if(!r)return false;
  const eps=(r.episodes||[]).concat(r.manual||[]);
  return eps.some(e=>e.human&&e.human.length);}
function updateProg(){const done=CLIPS.filter(c=>labeled(c.id)).length;
  $("#prog").textContent=`${idx+1}/${CLIPS.length} · 라벨 ${done}`;
  $("#barfill").style.width=(100*(idx+1)/CLIPS.length)+"%";}
function load(){const c=CLIPS[idx];vid().src=c.vid;seekEnd=null;
  $("#cid").textContent=`${idx+1}. ${c.id}`;rec();renderEps();updateProg();}
function go(d){idx=Math.max(0,Math.min(CLIPS.length-1,idx+d));load();window.scrollTo(0,0);}
$("#btnNext").onclick=()=>go(1);$("#btnPrev").onclick=()=>go(-1);
$("#btnDl").onclick=()=>{const b=new Blob([JSON.stringify(store,null,1)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="gold.json";a.click();};
$("#btnImp").onclick=()=>$("#fileImp").click();
$("#fileImp").onchange=e=>{const f=e.target.files[0];if(!f)return;const rd=new FileReader();
  rd.onload=()=>{try{store=JSON.parse(rd.result);save();load();alert("불러오기 완료");}catch(x){alert("파싱 실패");}};rd.readAsText(f);};
document.onkeydown=e=>{if(["TEXTAREA","SELECT"].includes(e.target.tagName))return;
  if(e.key=="ArrowRight")go(1);else if(e.key=="ArrowLeft")go(-1);
  else if(e.key==" "){e.preventDefault();const v=vid();v.paused?v.play():v.pause();}};
load();
</script></body></html>"""

HTML = (HTML.replace("__CLIPS__", clips_js)
            .replace("__EVENTS__", events_js)
            .replace("__ODD__", odd_js)
            .replace("__AUTOL__", json.dumps(auto_labels, ensure_ascii=False))
            .replace("__AUTOGT__", autogt_js))
open(f"{OUTDIR}/index.html", "w").write(HTML)
print(f"wrote {OUTDIR}/index.html ({len(HTML)} bytes) · {len(sample)} clips"
      f" · AUTO_GT={sorted(taxonomy.AUTO_GT)} · human events={len(taxonomy.HUMAN_KEYS)}"
      f" · turn in human={'turn_left' in taxonomy.HUMAN_KEYS}")
