import json
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # cari baca_pdf.py di folder file ini sendiri

import gradio as gr
import uvicorn
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

try:
    import pymupdf as fitz
except ImportError:
    import fitz

import importlib

import baca_pdf_v2 as bp

bp = importlib.reload(bp)

if not hasattr(bp, "analisis_pdf") or not hasattr(bp, "PALET_KOTAK"):
    raise ImportError(f"baca_pdf_v2.py yang terbaca ({bp.__file__}) bukan versi terbaru. "
                      "Ganti dengan baca_pdf_v2.py terbaru (yang ada warna per kotak), simpan, lalu restart kernel.")

JUDUL = "Sistem Deteksi Segiempat dan Ekstraksi Teks PDF"
DPI_PRATINJAU = 110
SESI = {}

#  Editor untuk di web (HTML + JavaScript)
HTML_EDITOR = r"""<!doctype html>
<html lang="id"><head><meta charset="utf-8"><title>Editor kotak</title>
<style>
*{box-sizing:border-box} body{margin:0;font:14px system-ui,Segoe UI,Arial,sans-serif;color:#222;background:#f4f4f6;height:100vh;display:flex;flex-direction:column}
#bar{display:flex;gap:6px;align-items:center;flex-wrap:wrap;padding:6px 10px;background:#fff;border-bottom:1px solid #ccc}
#bar button,#bar select{font:inherit;padding:4px 9px;border:1px solid #aaa;border-radius:6px;background:#fafafa;cursor:pointer}
#bar button:hover{background:#eef}
#bar .utama{background:#1a7f37;color:#fff;border-color:#1a7f37}
#bar .gap{flex:1}
#status{color:#555;font-size:13px}
#main{flex:1;display:flex;min-height:0}
#kanvas{flex:1;overflow:auto;background:#d9d9de;padding:10px}
#svg{background:#fff;display:block;touch-action:none;box-shadow:0 1px 6px #0004;user-select:none}
#panel{width:330px;overflow:auto;background:#fff;border-left:1px solid #ccc;padding:8px}
.kartu{border:1px solid #ccc;border-radius:8px;padding:6px;margin-bottom:8px;background:#fafafa}
.kartu.pilih{border-color:#1a7f37;box-shadow:0 0 0 2px #1a7f3744}
.kartu h4{margin:0 0 4px;display:flex;align-items:center;gap:6px;font-size:13px}
.kartu h4 span{flex:1;cursor:pointer}
.kartu input{width:100%;margin:2px 0;padding:4px 6px;border:1px solid #bbb;border-radius:5px;border-left-width:6px;font:inherit}
.kartu button{font:inherit;font-size:12px;padding:2px 7px;border:1px solid #aaa;border-radius:5px;background:#fff;cursor:pointer}
.bantuan{font-size:12px;color:#555;margin-bottom:8px;line-height:1.45}
.tambahBesar{width:100%;padding:9px;margin:2px 0 14px;font:inherit;font-weight:600;color:#1a7f37;background:#f3fbf5;border:2px dashed #1a7f37;border-radius:8px;cursor:pointer}
.tambahBesar:hover{background:#e3f5e8}
body.menggambar #svg{cursor:crosshair}
</style></head><body>
<div id="bar">
  <button id="prev">&#9664;</button><span>Halaman</span><select id="pilihHal"></select><button id="next">&#9654;</button>
  <span style="width:10px"></span>
  <button id="zoomOut">Zoom &minus;</button><button id="zoomIn">Zoom +</button>
  <span style="width:10px"></span>
  <button id="tambah">+ Tambah kotak</button><button id="urung">&#8630; Urungkan</button>
  <span class="gap"></span><span id="status">Memuat ...</span>
  <button id="simpan" class="utama">Buat PDF &amp; unduh</button>
</div>
<div id="main"><div id="kanvas"><svg id="svg" xmlns="http://www.w3.org/2000/svg"></svg></div><div id="panel"></div></div>
<script>
const SID="__SID__";
const NS="http://www.w3.org/2000/svg";
const PALET=__PALET__;     // warna kotak (sama dengan yang dipakai di PDF); tiap kotak punya satu warna
function hsv2hex(h,s,v){const f=(n,k=(n+h*6)%6)=>v-v*s*Math.max(Math.min(k,4-k,1),0);const x=n=>Math.round(255*f(n)).toString(16).padStart(2,"0");return "#"+x(5)+x(3)+x(1);}
function warnaKe(n){return n<PALET.length?PALET[n]:hsv2hex((n*0.61803398875)%1,0.75,0.72);}
function warnaBaru(){const pakai=new Set(page().kotak.map(k=>k.warna));for(const c of PALET){if(!pakai.has(c))return c;}return warnaKe(page().kotak.length);}
const MIN=6;
let HAL=[], hi=0, sel=null, zoom=1, undoStack=[], drag=null;
const $=id=>document.getElementById(id);
const rad=d=>d*Math.PI/180;
function el(n,a,p){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
function pts(o){const c=Math.cos(rad(o.a)),s=Math.sin(rad(o.a));return [[-1,-1],[1,-1],[1,1],[-1,1]].map(([i,j])=>{const x=i*o.w/2,y=j*o.h/2;return [o.cx+x*c-y*s,o.cy+x*s+y*c];});}
function toLocal(o,p){const c=Math.cos(rad(o.a)),s=Math.sin(rad(o.a)),dx=p[0]-o.cx,dy=p[1]-o.cy;return [dx*c+dy*s,-dx*s+dy*c];}
function toWorld(o,x,y){const c=Math.cos(rad(o.a)),s=Math.sin(rad(o.a));return [o.cx+x*c-y*s,o.cy+x*s+y*c];}
function norm(a){while(a>180)a-=360;while(a<=-180)a+=360;return a;}
const clone=o=>JSON.parse(JSON.stringify(o));
function page(){return HAL[hi];}
function selObj(){ if(!sel)return null; const k=page().kotak[sel.k]; if(!k)return null; return sel.t<0?k:k.item[sel.t]; }
function scaleK(){ const r=$("svg").getBoundingClientRect(); return r.width/page().W; }
function pushUndo(){ undoStack.push({hi:hi,kotak:JSON.stringify(page().kotak)}); if(undoStack.length>60)undoStack.shift(); }
function setStatus(t){$("status").textContent=t;}

async function muat(){
  const r=await fetch(`/editor/${SID}/data`); HAL=await r.json();
  const sel_=$("pilihHal"); sel_.innerHTML="";
  HAL.forEach((h,i)=>{const o=document.createElement("option");o.value=i;o.textContent=h.halaman;sel_.appendChild(o);});
  bukaHalaman(0);
}
function bukaHalaman(i){
  hi=Math.max(0,Math.min(HAL.length-1,i)); sel=null; $("pilihHal").value=hi;
  const h=page(), svg=$("svg");
  const lebarArea=Math.max(300,$("kanvas").clientWidth-24);
  svg.setAttribute("viewBox",`0 0 ${h.W} ${h.H}`);
  svg.style.width=(lebarArea*zoom)+"px"; svg.style.height=(lebarArea*zoom*h.H/h.W)+"px";
  gambar(); panel();
  setStatus(`Halaman ${h.halaman}: ${h.kotak.length} kotak`);
}
function gambar(){
  const h=page(), svg=$("svg"), k=scaleK();
  svg.innerHTML="";
  const im=el("image",{x:0,y:0,width:h.W,height:h.H,href:`/editor/${SID}/img/${hi}.png`,"data-bg":1},svg);
  h.kotak.forEach((kt,ki)=>{
    const g=el("polygon",{points:pts(kt.obj).map(p=>p.join(",")).join(" "),fill:kt.warna,"fill-opacity":0.06,stroke:kt.warna,"stroke-width":2.5,"vector-effect":"non-scaling-stroke","pointer-events":"all","data-k":ki,"data-t":-1,"data-role":"body",style:"cursor:move"},svg);
  });
  h.kotak.forEach((kt,ki)=>{
    kt.item.forEach((it,ti)=>{
      el("polygon",{points:pts(it.obj).map(p=>p.join(",")).join(" "),fill:"none",stroke:kt.warna,"stroke-width":1.8,"vector-effect":"non-scaling-stroke","pointer-events":"all","data-k":ki,"data-t":ti,"data-role":"body",style:"cursor:move"},svg);
    });
  });
  h.kotak.forEach((kt,ki)=>{      // label nomor + tulisan di sisi kiri-atas kotak
    const P=pts(kt.obj); let a=P[0]; for(const p of P){ if(p[0]+p[1]<a[0]+a[1]) a=p; }
    const t=el("text",{x:a[0],y:a[1]-3/k,"font-size":13/k,"font-weight":"bold",fill:kt.warna,"paint-order":"stroke",stroke:"#fff","stroke-width":3/k,"pointer-events":"none"},svg);
    t.appendChild(document.createTextNode(String(ki+1)));
    kt.item.forEach(it=>{const s=el("tspan",{fill:kt.warna,"font-weight":"normal","dx":6/k},t);s.textContent=it.teks;});
  });
  const so=selObj(), o=so?so.obj:null;
  if(o){        // pegangan: 8 titik ubah ukuran + 1 titik putar
    const hs=6/k;
    el("polygon",{points:pts(o).map(p=>p.join(",")).join(" "),fill:"none",stroke:"#1565c0","stroke-width":1.5,"stroke-dasharray":"5 3","vector-effect":"non-scaling-stroke","pointer-events":"none"},svg);
    for(const hy of [-1,0,1]) for(const hx of [-1,0,1]){
      if(hx===0&&hy===0) continue;
      const w=toWorld(o,hx*o.w/2,hy*o.h/2);
      const cur=(hx===0)?"ns-resize":(hy===0?"ew-resize":((hx*hy>0)?"nwse-resize":"nesw-resize"));
      el("rect",{x:w[0]-hs,y:w[1]-hs,width:2*hs,height:2*hs,fill:"#fff",stroke:"#1565c0","stroke-width":1.5,"vector-effect":"non-scaling-stroke","data-role":"h","data-hx":hx,"data-hy":hy,style:"cursor:"+cur,transform:`rotate(${o.a} ${w[0]} ${w[1]})`},svg);
    }
    const top=toWorld(o,0,-o.h/2), rot=toWorld(o,0,-o.h/2-24/k);
    el("line",{x1:top[0],y1:top[1],x2:rot[0],y2:rot[1],stroke:"#1565c0","stroke-width":1.5,"vector-effect":"non-scaling-stroke","pointer-events":"none"},svg);
    el("circle",{cx:rot[0],cy:rot[1],r:hs*1.2,fill:"#1565c0","data-role":"rot",style:"cursor:grab"},svg);
  }
}
function panel(){
  const h=page(), P=$("panel"); P.innerHTML="";
  const b=document.createElement("div"); b.className="bantuan";
  b.innerHTML="<b>Cara pakai:</b> Klik kotak untuk mulai mengedit. Tarik untuk menggeser posisi, seret kotak kecil di tepi untuk mengubah ukuran, dan tarik bulatan biru di atas untuk memutar. Klik area berwarna di dalam kotak untuk mengubah teksnya.";
  P.appendChild(b);
  h.kotak.forEach((kt,ki)=>{
    const c=document.createElement("div"); c.className="kartu"+(sel&&sel.k===ki?" pilih":""); c.dataset.k=ki;
    c.style.borderLeft="8px solid "+kt.warna;
    const hd=document.createElement("h4");
    const sp=document.createElement("span"); sp.textContent="Kotak "+(ki+1); sp.style.color=kt.warna; sp.style.fontWeight="700"; sp.onclick=()=>{sel={k:ki,t:-1};gambar();panel();};
    const del=document.createElement("button"); del.textContent="Hapus"; del.onclick=()=>{hapusKotak(ki);};
    hd.appendChild(sp); hd.appendChild(del); c.appendChild(hd);
    const buatInput=(it,ti)=>{
      const inp=document.createElement("input"); inp.value=it?it.teks:""; inp.placeholder="(ketik tulisan)";
      inp.style.borderLeftColor=kt.warna;
      inp.onfocus=()=>{ if(!(sel&&sel.k===ki)){sel={k:ki,t:ti===null?-1:ti};gambar();[...P.children].forEach(x=>{if(x.classList.contains("kartu"))x.classList.toggle("pilih",+x.dataset.k===ki);});}};
      inp.oninput=()=>{
        if(it){ it.teks=inp.value; gambar(); }
        else if(inp.value.trim()){
          pushUndo(); tambahItem(ki,inp.value); const baru=kt.item[kt.item.length-1];
          inp.style.borderLeftColor=baru.warna;
          inp.oninput=()=>{ baru.teks=inp.value; gambar(); };      // huruf berikutnya ikut tersimpan
          inp.onchange=()=>{ if(!inp.value.trim()){ pushUndo(); kt.item.splice(kt.item.indexOf(baru),1); sel=null; gambar(); panel(); } };
          gambar();
        }
      };
      inp.onchange=()=>{ if(it&&!inp.value.trim()){ pushUndo(); kt.item.splice(ti,1); sel=null; gambar(); panel(); } };
      return inp;
    };
    if(kt.item.length===0) c.appendChild(buatInput(null,null));
    kt.item.forEach((it,ti)=>c.appendChild(buatInput(it,ti)));
    if(kt.item.length>0){ const ad=document.createElement("button"); ad.textContent="+ tulisan"; ad.onclick=()=>{pushUndo();tambahItem(ki,"");gambar();panel();const ins=c.querySelectorAll("input");ins[ins.length-1].focus();}; c.appendChild(ad); }
    P.appendChild(c);
  });
  const tb=document.createElement("button"); tb.className="tambahBesar"; tb.textContent="+ Tambahkan kotak"; tb.onclick=mulaiGambar;
  P.appendChild(tb);
}
function tambahItem(ki,teks){
  const kt=page().kotak[ki], o=kt.obj;
  kt.item.push({id:null,d:true,teks:teks,warna:kt.warna,obj:{cx:o.cx,cy:o.cy,w:o.w*0.6,h:o.h*0.4,a:o.a}});
}
function hapusKotak(ki){ pushUndo(); page().kotak.splice(ki,1); sel=null; gambar(); panel(); setStatus(`Halaman ${page().halaman}: ${page().kotak.length} kotak`); }

function titik(e){ const svg=$("svg"), m=svg.getScreenCTM().inverse(); const p=svg.createSVGPoint(); p.x=e.clientX;p.y=e.clientY; const q=p.matrixTransform(m); return [q.x,q.y]; }
$("svg").addEventListener("pointerdown",e=>{
  const t=e.target, role=t.getAttribute("data-role"), p=titik(e);
  if(modeGambar){        // sedang menggambar kotak baru: mulai dari titik ini
    const r=el("rect",{x:p[0],y:p[1],width:0,height:0,fill:"rgba(26,160,60,0.12)",stroke:"#12a03c","stroke-width":2,"stroke-dasharray":"6 4","vector-effect":"non-scaling-stroke","pointer-events":"none",id:"rGambar"},$("svg"));
    drag={role:"gambar",p0:p,rect:r}; $("svg").setPointerCapture(e.pointerId); e.preventDefault(); return;
  }
  if(!role){ sel=null; gambar(); panel(); return; }
  if(role==="body"){ sel={k:+t.getAttribute("data-k"),t:+t.getAttribute("data-t")}; gambar(); panel(); }
  const o=selObj(); if(!o) return;
  pushUndo();
  drag={role:role==="body"?"move":role,hx:+t.getAttribute("data-hx"),hy:+t.getAttribute("data-hy"),p0:p,sel:{...sel},o0:clone(o),kids0:sel.t<0?clone(page().kotak[sel.k].item):null};
  $("svg").setPointerCapture(e.pointerId); e.preventDefault();
});
$("svg").addEventListener("pointermove",e=>{
  if(!drag) return;
  if(drag.role==="gambar"){
    const q=titik(e), x=Math.min(drag.p0[0],q[0]), y=Math.min(drag.p0[1],q[1]);
    drag.rect.setAttribute("x",x); drag.rect.setAttribute("y",y);
    drag.rect.setAttribute("width",Math.abs(q[0]-drag.p0[0])); drag.rect.setAttribute("height",Math.abs(q[1]-drag.p0[1])); return;
  }
  const p=titik(e), kt=page().kotak[drag.sel.k]; const o=drag.sel.t<0?kt:kt.item[drag.sel.t]; const o0=drag.o0.obj;
  if(drag.role==="move"){
    const dx=p[0]-drag.p0[0], dy=p[1]-drag.p0[1];
    o.obj.cx=o0.cx+dx; o.obj.cy=o0.cy+dy; o.d=true;
    if(drag.kids0) kt.item.forEach((it,i)=>{it.obj.cx=drag.kids0[i].obj.cx+dx; it.obj.cy=drag.kids0[i].obj.cy+dy; it.d=true;});
  } else if(drag.role==="rot"){
    let a=Math.atan2(p[1]-o0.cy,p[0]-o0.cx)*180/Math.PI+90; if(e.shiftKey) a=Math.round(a/15)*15; a=norm(a);
    const da=a-o0.a; o.obj.a=a; o.d=true;
    if(drag.kids0) kt.item.forEach((it,i)=>{       // tulisan di dalam kotak ikut berputar mengelilingi pusat kotak
      const c0=drag.kids0[i].obj, r=rad(da), dx=c0.cx-o0.cx, dy=c0.cy-o0.cy;
      it.obj.cx=o0.cx+dx*Math.cos(r)-dy*Math.sin(r); it.obj.cy=o0.cy+dx*Math.sin(r)+dy*Math.cos(r); it.obj.a=norm(c0.a+da); it.d=true;});
  } else if(drag.role==="h"){
    const pl=toLocal(o0,p); let nw=o0.w,nh=o0.h,cxl=0,cyl=0;
    if(drag.hx!==0){const ax=-drag.hx*o0.w/2; nw=Math.max(MIN,drag.hx*(pl[0]-ax)); cxl=(ax+(ax+drag.hx*nw))/2;}
    if(drag.hy!==0){const ay=-drag.hy*o0.h/2; nh=Math.max(MIN,drag.hy*(pl[1]-ay)); cyl=(ay+(ay+drag.hy*nh))/2;}
    const w=toWorld(o0,cxl,cyl); o.obj.w=nw;o.obj.h=nh;o.obj.cx=w[0];o.obj.cy=w[1]; o.d=true;
  }
  gambar();
});
function selesai(e){
  if(drag&&drag.role==="gambar"){
    const q=titik(e), h=page(), d=drag; drag=null; d.rect.remove();
    let w=Math.abs(q[0]-d.p0[0]), hh=Math.abs(q[1]-d.p0[1]), cx=(q[0]+d.p0[0])/2, cy=(q[1]+d.p0[1])/2;
    if(w<MIN||hh<MIN){ w=h.W*0.16; hh=h.W*0.1; cx=d.p0[0]; cy=d.p0[1]; }     // cuma klik: kotak ukuran standar di titik itu
    pushUndo(); h.kotak.push({id:null,d:true,warna:warnaBaru(),obj:{cx:cx,cy:cy,w:w,h:hh,a:0},item:[]});
    selesaiGambar(); sel={k:h.kotak.length-1,t:-1}; gambar(); panel();
    setStatus(`Halaman ${h.halaman}: ${h.kotak.length} kotak. Ketik tulisannya di panel kanan.`);
    const kartu=document.querySelector(`.kartu[data-k="${h.kotak.length-1}"]`);
    if(kartu){ kartu.scrollIntoView({block:"nearest"}); const inp=kartu.querySelector("input"); if(inp) inp.focus(); }
    return;
  }
  if(drag){drag=null; gambar();}
}
$("svg").addEventListener("pointerup",selesai); $("svg").addEventListener("pointercancel",selesai);

document.addEventListener("keydown",e=>{
  if(e.key==="Escape"&&modeGambar){ selesaiGambar(); setStatus("Menggambar kotak dibatalkan."); return; }
  const ket=(document.activeElement&&document.activeElement.tagName==="INPUT");
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="z"&&!ket){ e.preventDefault(); urung(); return; }
  if(ket||!sel) return;
  const o=selObj(); if(!o) return;
  if(e.key==="Delete"||e.key==="Backspace"){ e.preventDefault(); pushUndo();
    if(sel.t<0) hapusKotak(sel.k); else { page().kotak[sel.k].item.splice(sel.t,1); sel=null; gambar(); panel(); } return; }
  const d=e.shiftKey?5:1, mv={ArrowLeft:[-d,0],ArrowRight:[d,0],ArrowUp:[0,-d],ArrowDown:[0,d]}[e.key];
  if(mv){ e.preventDefault(); pushUndo(); o.obj.cx+=mv[0]; o.obj.cy+=mv[1]; o.d=true;
    if(sel.t<0) page().kotak[sel.k].item.forEach(it=>{it.obj.cx+=mv[0];it.obj.cy+=mv[1];it.d=true;}); gambar(); }
});
function urung(){ const s=undoStack.pop(); if(!s) return; if(s.hi!==hi) bukaHalaman(s.hi); page().kotak=JSON.parse(s.kotak); sel=null; gambar(); panel(); }
$("urung").onclick=urung;
$("prev").onclick=()=>bukaHalaman(hi-1); $("next").onclick=()=>bukaHalaman(hi+1);
$("pilihHal").onchange=()=>bukaHalaman(+$("pilihHal").value);
$("zoomIn").onclick=()=>{zoom=Math.min(4,zoom*1.25);bukaHalamanTanpaReset();}; $("zoomOut").onclick=()=>{zoom=Math.max(0.4,zoom/1.25);bukaHalamanTanpaReset();};
function bukaHalamanTanpaReset(){ const s=sel; bukaHalaman(hi); sel=s; gambar(); panel(); }
let modeGambar=false;
function mulaiGambar(){
  modeGambar=true; sel=null; gambar(); document.body.classList.add("menggambar");
  setStatus("Seret di halaman untuk menggambar kotak baru (klik sekali = ukuran standar). Esc = batal.");
}
function selesaiGambar(){ modeGambar=false; document.body.classList.remove("menggambar"); }
$("tambah").onclick=mulaiGambar;
$("simpan").onclick=async()=>{
  setStatus("Membuat PDF ...");
  const edit=HAL.map(h=>({halaman:h.halaman,kotak:h.kotak.map(k=>({id:k.id,d:!!k.d,warna:k.warna,obj:k.obj,item:k.item.map(i=>({id:i.id,d:!!i.d,obj:i.obj,teks:i.teks}))}))}));
  try{
    const r=await fetch(`/editor/${SID}/simpan`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(edit)});
    const j=await r.json(); if(!j.url) throw new Error(j.pesan||"gagal");
    const a=document.createElement("a"); a.href=j.url; a.download=""; document.body.appendChild(a); a.click(); a.remove();
    setStatus("PDF dibuat: "+j.total+" kotak.");
  }catch(err){ setStatus("Gagal membuat PDF: "+err.message); }
};
window.addEventListener("resize",()=>{ if(HAL.length) bukaHalamanTanpaReset(); });
muat();
</script></body></html>
"""

#  Server lokal
app = APIRouter()

def _sesi(sid):
    s = SESI.get(sid)
    if s is None:
        raise KeyError(sid)
    return s

@app.get("/editor/{sid}", response_class=HTMLResponse)
def editor_halaman(sid: str):
    if sid not in SESI:
        return HTMLResponse("<p style='font-family:sans-serif;padding:24px'>Sesi editor ini sudah tidak ada "
                            "(program dijalankan ulang). Refresh halaman utama, lalu upload PDF dan klik Proses lagi.</p>", status_code=404)
    palet = json.dumps([bp._hex(c) for c in bp.PALET_KOTAK])
    return HTML_EDITOR.replace("__SID__", sid).replace("__PALET__", palet)

@app.get("/editor/{sid}/data")
def editor_data(sid: str):
    s = _sesi(sid)
    with fitz.open(s["masuk"]) as d:
        ukuran = [(p.rect.width, p.rect.height) for p in d]
    hasil = bp.data_ke_editor(s["data"])
    for h in hasil:
        h["W"], h["H"] = ukuran[h["halaman"] - 1]
    return JSONResponse(hasil)

@app.get("/editor/{sid}/img/{i}.png")
def editor_gambar(sid: str, i: int):
    s = _sesi(sid)
    f = Path(s["folder"]) / f"asli_{i}.png"
    if not f.exists():
        with fitz.open(s["masuk"]) as d:
            d[s["data"][i]["halaman"] - 1].get_pixmap(dpi=DPI_PRATINJAU).save(str(f))
    return FileResponse(str(f), media_type="image/png")

@app.post("/editor/{sid}/simpan")
async def editor_simpan(sid: str, request: Request):
    s = _sesi(sid)
    try:
        edit = await request.json()
        baru = bp.editor_ke_data(s["data"], edit)
        keluar = Path(s["folder"]) / f"{s['nama']}_hasil_disunting.pdf"
        ringkasan = bp.tulis_pdf(s["masuk"], baru, keluar)
    except Exception as e:
        return JSONResponse({"pesan": f"{type(e).__name__}: {e}"}, status_code=500)
    s["pdf"] = str(keluar)
    return {"url": f"/editor/{sid}/pdf?t={int(time.time())}", "total": sum(len(h["kotak"]) for h in ringkasan)}

@app.get("/editor/{sid}/pdf")
def editor_pdf(sid: str):
    s = _sesi(sid)
    return FileResponse(s["pdf"], media_type="application/pdf", filename=Path(s["pdf"]).name)

# Website Gradio
def jalankan(berkas, progres=gr.Progress()):
    if berkas is None:
        raise gr.Error("Pilih file PDF dulu.")
    folder = Path(tempfile.mkdtemp(prefix="kotak_pdf_"))
    masuk = folder / Path(berkas).name
    shutil.copy(berkas, masuk)
    try:
        data = bp.analisis_pdf(masuk, laporan=lambda t: progres(0.5, desc=t))
    except FileNotFoundError as e:
        raise gr.Error(f"File model YOLO tidak ditemukan. Cek MODEL di baca_pdf.py. ({e})")
    hasil_awal = folder / (masuk.stem + "_hasil.pdf")
    ringkasan = bp.tulis_pdf(masuk, data, hasil_awal)
    sid = secrets.token_urlsafe(8)
    SESI[sid] = {"masuk": str(masuk), "folder": str(folder), "nama": masuk.stem, "data": data, "pdf": str(hasil_awal)}
    total = sum(len(h["kotak"]) for h in ringkasan)
    status = f"Selesai! {total} kotak ditemukan di {len(ringkasan)} halaman. Edit di editor di bawah ini, lalu klik \"Buat PDF & Download\"."
    iframe = (f'<p><a href="/editor/{sid}" target="_blank">Klik disini untuk membuka editor di tab baru </a></p>'
              f'<iframe src="/editor/{sid}" style="width:100%;height:82vh;border:1px solid #bbb;border-radius:8px"></iframe>')
    return status, iframe

with gr.Blocks(title=JUDUL) as web:
    gr.Markdown(f"# {JUDUL}\nUnggah file PDF kamu, dan biarkan sistem mendeteksi jumlah kotak serta membaca teks di dalamnya secara otomatis. Kamu juga dapat mengedit hasil ekstraksi terlebih dahulu sebelum download dokumen PDF finalnya.")
    with gr.Row():
        berkas = gr.File(label="Upload PDF disini", file_types=[".pdf"], type="filepath", scale=2)
        with gr.Column(scale=1):
            tombol = gr.Button("Proses", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
    editor = gr.HTML()
    tombol.click(jalankan, [berkas], [status, editor])


#  Menjalankan server
BAGIKAN_ONLINE = True
SANDI = None


def mulai(buka_browser=True):
    gr.close_all()
    web.launch(server_name="127.0.0.1", share=BAGIKAN_ONLINE, auth=SANDI,
               inbrowser=buka_browser, prevent_thread_lock=True, show_error=True)
    web.app.include_router(app)
    return web


if __name__ == "__main__":
    mulai()
    try:
        get_ipython()
    except NameError:
        web.block_thread()