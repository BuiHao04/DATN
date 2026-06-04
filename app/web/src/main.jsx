import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

const css = `
:root{--bg:#f3f1f8;--panel:#fff;--line:#d6d3e3;--muted:#6b7280;--primary:#3049b9;--text:#16161f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);font-family:Inter,Segoe UI,Arial,sans-serif;color:var(--text)}
.app{display:grid;grid-template-columns:260px 1fr;min-height:100vh}
.sidebar{background:#f6f5fb;border-right:1px solid #cfcbdc;display:flex;flex-direction:column}
.brand{padding:22px 20px 12px}.brand h1{margin:0;color:#223ca8;font-size:28px;font-weight:800}.brand p{margin:6px 0 0;color:#6b7280;font-size:12px}
.nav{padding:8px 12px;display:flex;flex-direction:column;gap:6px}
.nav a{display:flex;align-items:center;padding:10px 12px;border-radius:10px;text-decoration:none;color:#2f3442;font-size:20px;position:relative}
.nav a.active{background:#e9e7f7;color:#233fae;font-weight:700}.nav a.active:before{content:"";position:absolute;left:-12px;top:0;bottom:0;width:4px;background:#233fae}
.bottom{margin-top:auto;padding:12px;border-top:1px solid #d6d3e3;display:flex;flex-direction:column;gap:4px}
.bottom a{padding:8px 10px;text-decoration:none;color:#3b4252;font-size:14px}
.main{display:flex;flex-direction:column;min-width:0}
.topbar{height:58px;background:#f6f5fb;border-bottom:1px solid #cbc8d8;display:flex;align-items:center;justify-content:space-between;padding:0 16px}
.search{width:48%;max-width:680px;border:1px solid #c8c5d6;border-radius:9px;background:#efedf5;padding:10px 12px;font-size:13px}
.content{padding:12px}
.title{font-size:26px;font-weight:800;margin:0}.subtitle{margin:6px 0 12px;color:#33394a;font-size:14px;line-height:1.45}
.step{background:#fff;border:1px solid #c8c4d8;border-radius:12px;margin-bottom:10px;overflow:hidden}
.step-head{display:flex;justify-content:space-between;align-items:center;padding:10px 12px;background:#faf9ff;border-bottom:1px solid #d9d5e4}
.step-head h3{margin:0;font-size:16px}
.step-body{padding:10px 12px}
.row{display:grid;grid-template-columns:1fr auto auto;gap:8px;align-items:center}
.input{width:100%;padding:9px 10px;border:1px solid #c8c5d6;border-radius:8px;background:#f8f7fd;font-size:13px}
.btn{border:1px solid #b7b4c6;background:#eef0f6;color:#202736;border-radius:8px;padding:8px 12px;font-size:13px;font-weight:600;cursor:pointer}
.btn.primary{background:#3049b9;border-color:#3049b9;color:#fff}.btn.dark{background:#161b27;border-color:#161b27;color:#fff}
.drop{border:2px dashed #cbc8d8;border-radius:10px;height:72px;display:flex;align-items:center;justify-content:center;text-align:center;background:#fcfcff;margin:6px 0;cursor:pointer}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700}.queued{background:#fef3c7;color:#92400e}.running{background:#dbeafe;color:#1e40af}.success{background:#dcfce7;color:#166534}.failed{background:#fee2e2;color:#991b1b}.idle{background:#e5e7eb;color:#374151}
.p{font-size:13px;color:#2f3442;line-height:1.45;margin:0}
.bar{height:8px;border-radius:999px;background:#e8e6f2;overflow:hidden}.bar>div{height:100%;background:#3049b9}
.term{background:#121827;color:#dbeafe;border-radius:8px;border:1px solid #22283a;overflow:hidden}.term h4{margin:0;padding:8px 10px;border-bottom:1px solid #2a2f45;font-size:11px;color:#b8c2e2}.term pre{margin:0;padding:10px;max-height:170px;overflow:auto;font-family:Consolas,monospace;font-size:12px;white-space:pre-wrap}
.tiny{font-size:12px;color:#6b7280}
.table-wrap{margin-top:6px;border:1px solid #d4d1e0;border-radius:8px;overflow:hidden;background:#fff;max-height:180px;overflow:auto}
.tbl{width:100%;border-collapse:collapse;font-size:12px}
.tbl th,.tbl td{padding:5px 7px;border-bottom:1px solid #eceaf4;text-align:left}
.tbl th{background:#f7f6fc;font-weight:700}
.tbl tr:last-child td{border-bottom:none}
.path{max-width:560px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.step-upload .step-body{padding:8px 10px}
.upload-compact{display:grid;grid-template-columns:1fr 210px auto auto;gap:8px;align-items:center}
.upload-compact .input{height:34px}
.thumb{width:56px;height:56px;object-fit:cover;border:1px solid #d4d1e0;border-radius:6px;cursor:pointer;background:#f3f3f8}
.modal{position:fixed;inset:0;background:rgba(10,14,24,.62);display:flex;align-items:center;justify-content:center;z-index:9999}
.modal-box{background:#fff;border-radius:10px;padding:10px;max-width:92vw;max-height:92vh;position:relative}
.modal-box img{max-width:88vw;max-height:84vh;display:block}
.modal-close{position:absolute;top:6px;right:6px;border:0;background:#111827;color:#fff;border-radius:6px;padding:4px 8px;cursor:pointer}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:10px}
.card{background:#fff;border:1px solid #d4d0e1;border-radius:12px;padding:12px}
.card .k{font-size:12px;color:#6b7280;margin-bottom:6px}.card .v{font-size:22px;font-weight:800}
.split{display:grid;grid-template-columns:1.15fr .85fr;gap:10px}
.stack{display:flex;flex-direction:column;gap:10px}
.actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
@media (max-width:1200px){.app{grid-template-columns:220px 1fr}.path{max-width:260px}}
@media (max-width:1100px){.cards,.split{grid-template-columns:1fr}}
`;

const NAV = [["prep", "Chuẩn bị dữ liệu"],["train", "Huấn luyện"],["results", "Kết quả & Đánh giá"]];
const LABELS = [
  "MERCHANT_NAME",
  "MERCHANT_ADDRESS",
  "MERCHANT_PHONE",
  "TAX_CODE",
  "INVOICE_ID",
  "DATE",
  "TIME",
  "CASHIER",
  "ITEM_NAME",
  "ITEM_QTY",
  "ITEM_UNIT_PRICE",
  "ITEM_AMOUNT",
  "SUBTOTAL",
  "SERVICE_FEE",
  "DISCOUNT",
  "TAX_AMOUNT",
  "TOTAL_AMOUNT",
  "PAYMENT_METHOD",
  "OTHER",
];

function shortName(s){s=String(s||""); return s.length>26?`${s.slice(0,24)}...`:s;}

function App(){
  const [page,setPage]=useState("prep");
  const [jobs,setJobs]=useState([]);
  const [out,setOut]=useState("");
  const [files,setFiles]=useState([]);
  const [pickMode,setPickMode]=useState("files");
  const [pickerKey,setPickerKey]=useState(0);
  const [subdir,setSubdir]=useState("");
  const [recentRaw,setRecentRaw]=useState([]);
  const [ocr,setOcr]=useState({input_dir:"data/stage_b_raw_images",output_dir:"data/labeling_stage_b",lang:"vi",save_debug_images:1,copy_images:1});
  const [dataset,setDataset]=useState({input_csv:"data/labeling_stage_b/nodes_to_label.csv",output_json:"data/stage_b_vi_dataset.json"});
  const [labelRows,setLabelRows]=useState([]);
  const [allowedLabels,setAllowedLabels]=useState(LABELS);
  const [labelHint,setLabelHint]=useState("");
  const [labelLimit,setLabelLimit]=useState(100);
  const [labelPage,setLabelPage]=useState(1);
  const [totalPages,setTotalPages]=useState(1);
  const [docView,setDocView]=useState([]);
  const [previewImage,setPreviewImage]=useState("");
  const [missingOnly,setMissingOnly]=useState(false);
  const [trainA,setTrainA]=useState({dataset_json:"data/stage_a_dataset.json",val_dataset_json:"",checkpoint:"outputs/checkpoints/gcn_stage_a.pt",init_checkpoint:"",epochs:30,lr:0.001,early_stop_patience:0});
  const [trainB,setTrainB]=useState({dataset_json:"data/stage_b_vi_dataset.json",val_dataset_json:"",base_checkpoint:"outputs/checkpoints/gcn_stage_a.pt",checkpoint:"outputs/checkpoints/gcn_stage_b.pt",epochs:20,lr:0.0005,early_stop_patience:0});
  const [trainFull,setTrainFull]=useState({stage_a_json:"data/stage_a_dataset.json",stage_b_json:"data/stage_b_vi_dataset.json",stage_a_ckpt:"outputs/checkpoints/gcn_stage_a.pt",stage_b_ckpt:"outputs/checkpoints/gcn_stage_b.pt",stage_a_epochs:30,stage_b_epochs:20,stage_a_lr:0.001,stage_b_lr:0.0005,init_checkpoint:"",eval_json:"",output_eval:"outputs/gcn_eval_report.json"});
  const pollRef=useRef(null);

  const loadJobs=async()=>{try{const r=await fetch("/api/jobs");const d=await r.json();setJobs(Array.isArray(d)?d:[]);}catch{setJobs([])}};
  const loadRecentRaw=async()=>{try{const r=await fetch("/api/files/stage-b-raw-images");const d=await r.json();setRecentRaw((d.files||[]).slice(0,10)); if(d.input_dir) setOcr(s=>({...s,input_dir:d.input_dir}));}catch{setRecentRaw([])}};

  useEffect(()=>{loadJobs(); loadRecentRaw();},[]);
  useEffect(()=>{const run=jobs.some(j=>j.status==="queued"||j.status==="running"); pollRef.current=setTimeout(loadJobs, run?2500:8000); return ()=>clearTimeout(pollRef.current);},[jobs]);

  const latestByMode=useMemo(()=>{const m={}; for(const j of jobs) if(!m[j.mode]) m[j.mode]=j; return m;},[jobs]);
  const ocrJob=latestByMode["prepare_ocr_labeling"]; const dsJob=latestByMode["preprocess_gcn_dataset"]; const hfJob=latestByMode["convert_hf_cord_to_csv"];
  const aiLabelJob=latestByMode["labeling_auto_suggest"];
  const trainAJob=latestByMode["train_gcn_stage_a"];
  const trainBJob=latestByMode["train_gcn_stage_b"];
  const trainFullJob=latestByMode["train_gcn_full"];
  const statusClass=(s)=>s==="success"?"success":s==="failed"?"failed":s==="running"?"running":s==="queued"?"queued":"idle";
  const lastLog=(j)=>{if(!j) return "Chưa có log"; const t=(j.stderr||j.stdout||"").trim(); if(!t) return "Job đã tạo, đang chờ log..."; return t.split(/\r?\n/).slice(-25).join("\n");};

  const apiPost=async(url,body)=>{const r=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});const d=await r.json();setOut(JSON.stringify(d,null,2));await loadJobs();return d;};
  const upload=async()=>{if(!files.length) return; const fd=new FormData(); files.forEach(f=>fd.append("files",f)); if(subdir.trim()) fd.append("subdir",subdir.trim()); const r=await fetch("/api/files/upload-images",{method:"POST",body:fd}); const d=await r.json(); setOut(JSON.stringify(d,null,2)); await loadRecentRaw(); await loadJobs();};
  const clearOldFiles=async()=>{const d=await apiPost("/api/files/clear-stage-b-raw-images",{}); setFiles([]); setSubdir(""); setPickMode("files"); setPickerKey(v=>v+1); await loadRecentRaw(); return d;};
  const runOcrBatch=async()=>{await apiPost("/api/pipeline/prepare-ocr-labeling",ocr);};
  const loadLabelPreview=async(pageOverride=null, onlyMissingOverride=null)=>{
    try{
      const targetPage = pageOverride ?? Number(labelPage||1);
      const onlyMissing = onlyMissingOverride ?? missingOnly;
      const r=await fetch("/api/pipeline/labeling-sample",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({
          input_csv: dataset.input_csv,
          label_col: "label",
          text_col: "text",
          limit: Number(labelLimit||100),
          page: targetPage,
        })
      });
      const d=await r.json();
      if(!r.ok){
        setLabelRows([]);
        setLabelHint(`Lỗi tải dữ liệu gán nhãn: ${d?.detail || "unknown"}`);
        setOut(JSON.stringify(d,null,2));
        return;
      }
      setAllowedLabels(d.allowed_labels||LABELS);
      let rows=(d.rows||[]).map(x=>({...x,picked_label:x.label||""}));
      if (onlyMissing) rows = rows.filter((x)=>String(x.picked_label||"").trim()==="");
      setLabelRows(rows);
      setLabelPage(Number(d.page||targetPage));
      setTotalPages(Number(d.total_pages||1));
      setLabelHint(`Đã tải ${rows.length} dòng để rà soát. Còn ${d.empty_label_count||0} dòng thiếu nhãn trên tổng ${d.total_rows||0} dòng.`);
      setOut(JSON.stringify(d,null,2));
    }catch(e){
      setLabelRows([]);
      setLabelHint("Không gọi được API tải dòng thiếu nhãn.");
    }
  };
  const saveLabelUpdates=async()=>{const updates=labelRows.filter(r=>String(r.picked_label||"").trim()!=="").map(r=>({row_number:r.row_number,label:r.picked_label})); const d=await apiPost("/api/pipeline/labeling-apply",{input_csv:dataset.input_csv,label_col:"label",updates}); if(d?.updated>=0){setLabelHint(`Đã lưu ${d.updated} dòng.`); await loadLabelPreview();}};
  const autoSuggestLabels=async()=>{
    const d=await apiPost("/api/pipeline/labeling-auto-suggest-start",{
      input_csv:dataset.input_csv,
      label_col:"label",
      text_col:"text",
      doc_id_col:"doc_id",
      only_empty:1,
      llm_model:"gpt-4.1-mini",
      batch_docs:10,
      llm_text_batch_size:10,
      require_llm:1
    });
    if(d?.id){
      setLabelHint(`Đã tạo job AI gợi ý nhãn cho các dòng còn trống: ${d.id}. Job sẽ lưu sau từng batch và bấm chạy tiếp sẽ không gán lại phần đã xong.`);
    }
  };
  const runDatasetCompile=async()=>{const v=await apiPost("/api/pipeline/validate-label-csv",{input_csv:dataset.input_csv,label_col:"label"}); if(!v?.ok){setOut(JSON.stringify({status:"blocked",reason:"Còn label rỗng, cần gán nhãn trước",validate:v},null,2)); return;} await apiPost("/api/pipeline/preprocess-gcn-dataset",dataset);};
  const runCordImport=async()=>{await apiPost("/api/pipeline/convert-hf-cord-to-csv",{dataset_id:"naver-clova-ix/cord-v2",split:"train",output_csv:"data/cord_train_nodes.csv",streaming:1});};
  const runTrainA=async()=>{await apiPost("/api/pipeline/train-gcn-stage-a", {...trainA, val_dataset_json: trainA.val_dataset_json || null, init_checkpoint: trainA.init_checkpoint || null});};
  const runTrainB=async()=>{await apiPost("/api/pipeline/train-gcn-stage-b", {...trainB, val_dataset_json: trainB.val_dataset_json || null});};
  const runTrainFull=async()=>{await apiPost("/api/pipeline/train-gcn-full", {...trainFull, init_checkpoint: trainFull.init_checkpoint || null, eval_json: trainFull.eval_json || null});};
  const loadByDoc=async()=>{
    const d=await apiPost("/api/pipeline/labeling-by-doc",{
      input_csv:dataset.input_csv,
      label_col:"label",
      text_col:"text",
      doc_id_col:"doc_id",
      limit_docs:50
    });
    setDocView(d.docs||[]);
  };

  const ocrProgress = ocrJob?.progress || null;
  const ocrCurrent = ocrProgress?.current ?? 0;
  const ocrTotal = ocrProgress?.total ?? 0;
  const ocrPct = ocrProgress?.percent ?? 0;
  const ocrCurrentFile = ocrProgress?.current_file || "";
  const reviewByDoc = useMemo(()=>{
    const previewMap = new Map((docView||[]).map(d=>[String(d.doc_id||""), d.preview_path || ""]));
    const grouped = new Map();
    for(const r of labelRows){
      const key = String(r.doc_id||"UNKNOWN_DOC");
      if(!grouped.has(key)){
        grouped.set(key, { doc_id:key, preview_path: previewMap.get(key) || "", rows:[] });
      }
      grouped.get(key).rows.push(r);
    }
    return Array.from(grouped.values());
  }, [labelRows, docView]);
  const trainRunning = [trainAJob, trainBJob, trainFullJob].some(j=>j?.status==="running" || j?.status==="queued");
  const trainDoneCount = [trainAJob, trainBJob, trainFullJob].filter(j=>j?.status==="success").length;
  const trainFailCount = [trainAJob, trainBJob, trainFullJob].filter(j=>j?.status==="failed").length;

  return <>
    <style>{css}</style>
    <div className="app">
      <aside className="sidebar">
        <div className="brand"><h1>Nexus ML</h1><p>v2.4.0-stable</p></div>
        <nav className="nav">{NAV.map(([k,l])=><a key={k} href="#" className={page===k?"active":""} onClick={e=>{e.preventDefault();setPage(k);}}>{l}</a>)}</nav>
        <div className="bottom"><a href="#">Cài đặt</a><a href="#">Tài liệu</a></div>
      </aside>
      <main className="main">
        <div className="topbar"><input className="search" placeholder="Tìm kiếm bước xử lý..."/></div>
        <div className="content">
          {page==="train" ? <>
            <h2 className="title">Huấn luyện GCN</h2>
            <div className="subtitle">Màn này dùng để chạy train Stage A, fine-tune Stage B và full flow A {"->"} B. Backend đã có sẵn endpoint, giao diện này chỉ bọc lại để bạn cấu hình và theo dõi job.</div>

            <div className="cards">
              <div className="card"><div className="k">Job train đã chạy</div><div className="v">{[trainAJob,trainBJob,trainFullJob].filter(Boolean).length}</div></div>
              <div className="card"><div className="k">Đang chạy</div><div className="v">{trainRunning?1:0}</div></div>
              <div className="card"><div className="k">Thành công</div><div className="v">{trainDoneCount}</div></div>
              <div className="card"><div className="k">Thất bại</div><div className="v">{trainFailCount}</div></div>
            </div>

            <div className="split">
              <div className="stack">
                <div className="step">
                  <div className="step-head"><h3>Stage A: pretrain / train nền</h3><span className={`badge ${statusClass(trainAJob?.status)}`}>{(trainAJob?.status||"idle").toUpperCase()}</span></div>
                  <div className="step-body">
                    <div className="grid2">
                      <input className="input" value={trainA.dataset_json} onChange={e=>setTrainA({...trainA,dataset_json:e.target.value})} placeholder="Dataset JSON train"/>
                      <input className="input" value={trainA.val_dataset_json} onChange={e=>setTrainA({...trainA,val_dataset_json:e.target.value})} placeholder="Dataset JSON validation (tùy chọn)"/>
                    </div>
                    <div className="grid2" style={{marginTop:8}}>
                      <input className="input" value={trainA.checkpoint} onChange={e=>setTrainA({...trainA,checkpoint:e.target.value})} placeholder="Checkpoint output"/>
                      <input className="input" value={trainA.init_checkpoint} onChange={e=>setTrainA({...trainA,init_checkpoint:e.target.value})} placeholder="Init checkpoint (tùy chọn)"/>
                    </div>
                    <div className="grid3" style={{marginTop:8}}>
                      <input className="input" type="number" value={trainA.epochs} onChange={e=>setTrainA({...trainA,epochs:Number(e.target.value)})} placeholder="Epochs"/>
                      <input className="input" type="number" step="0.0001" value={trainA.lr} onChange={e=>setTrainA({...trainA,lr:Number(e.target.value)})} placeholder="Learning rate"/>
                      <input className="input" type="number" value={trainA.early_stop_patience} onChange={e=>setTrainA({...trainA,early_stop_patience:Number(e.target.value)})} placeholder="Early stop"/>
                    </div>
                    <div className="actions" style={{marginTop:10}}><div className="tiny">POST /api/pipeline/train-gcn-stage-a</div><button className="btn dark" onClick={runTrainA}>Train Stage A</button></div>
                  </div>
                </div>

                <div className="step">
                  <div className="step-head"><h3>Stage B: fine-tune hóa đơn tiếng Việt</h3><span className={`badge ${statusClass(trainBJob?.status)}`}>{(trainBJob?.status||"idle").toUpperCase()}</span></div>
                  <div className="step-body">
                    <div className="grid2">
                      <input className="input" value={trainB.dataset_json} onChange={e=>setTrainB({...trainB,dataset_json:e.target.value})} placeholder="Dataset JSON train"/>
                      <input className="input" value={trainB.val_dataset_json} onChange={e=>setTrainB({...trainB,val_dataset_json:e.target.value})} placeholder="Dataset JSON validation (tùy chọn)"/>
                    </div>
                    <div className="grid2" style={{marginTop:8}}>
                      <input className="input" value={trainB.base_checkpoint} onChange={e=>setTrainB({...trainB,base_checkpoint:e.target.value})} placeholder="Checkpoint Stage A"/>
                      <input className="input" value={trainB.checkpoint} onChange={e=>setTrainB({...trainB,checkpoint:e.target.value})} placeholder="Checkpoint Stage B output"/>
                    </div>
                    <div className="grid3" style={{marginTop:8}}>
                      <input className="input" type="number" value={trainB.epochs} onChange={e=>setTrainB({...trainB,epochs:Number(e.target.value)})} placeholder="Epochs"/>
                      <input className="input" type="number" step="0.0001" value={trainB.lr} onChange={e=>setTrainB({...trainB,lr:Number(e.target.value)})} placeholder="Learning rate"/>
                      <input className="input" type="number" value={trainB.early_stop_patience} onChange={e=>setTrainB({...trainB,early_stop_patience:Number(e.target.value)})} placeholder="Early stop"/>
                    </div>
                    <div className="actions" style={{marginTop:10}}><div className="tiny">POST /api/pipeline/train-gcn-stage-b</div><button className="btn dark" onClick={runTrainB}>Train Stage B</button></div>
                  </div>
                </div>

                <div className="step">
                  <div className="step-head"><h3>Full Flow: A {"->"} B {"->"} Eval</h3><span className={`badge ${statusClass(trainFullJob?.status)}`}>{(trainFullJob?.status||"idle").toUpperCase()}</span></div>
                  <div className="step-body">
                    <div className="grid2">
                      <input className="input" value={trainFull.stage_a_json} onChange={e=>setTrainFull({...trainFull,stage_a_json:e.target.value})} placeholder="Stage A dataset JSON"/>
                      <input className="input" value={trainFull.stage_b_json} onChange={e=>setTrainFull({...trainFull,stage_b_json:e.target.value})} placeholder="Stage B dataset JSON"/>
                    </div>
                    <div className="grid2" style={{marginTop:8}}>
                      <input className="input" value={trainFull.stage_a_ckpt} onChange={e=>setTrainFull({...trainFull,stage_a_ckpt:e.target.value})} placeholder="Checkpoint Stage A"/>
                      <input className="input" value={trainFull.stage_b_ckpt} onChange={e=>setTrainFull({...trainFull,stage_b_ckpt:e.target.value})} placeholder="Checkpoint Stage B"/>
                    </div>
                    <div className="grid2" style={{marginTop:8}}>
                      <input className="input" value={trainFull.eval_json} onChange={e=>setTrainFull({...trainFull,eval_json:e.target.value})} placeholder="Dataset eval (tùy chọn)"/>
                      <input className="input" value={trainFull.output_eval} onChange={e=>setTrainFull({...trainFull,output_eval:e.target.value})} placeholder="Output eval JSON"/>
                    </div>
                    <div className="grid2" style={{marginTop:8}}>
                      <input className="input" value={trainFull.init_checkpoint} onChange={e=>setTrainFull({...trainFull,init_checkpoint:e.target.value})} placeholder="Init checkpoint cho Stage A (tùy chọn)"/>
                      <div className="tiny" style={{display:"flex",alignItems:"center"}}>Dùng khi bạn muốn tiếp tục từ checkpoint có sẵn thay vì train Stage A từ đầu.</div>
                    </div>
                    <div className="grid3" style={{marginTop:8}}>
                      <input className="input" type="number" value={trainFull.stage_a_epochs} onChange={e=>setTrainFull({...trainFull,stage_a_epochs:Number(e.target.value)})} placeholder="Epochs A"/>
                      <input className="input" type="number" value={trainFull.stage_b_epochs} onChange={e=>setTrainFull({...trainFull,stage_b_epochs:Number(e.target.value)})} placeholder="Epochs B"/>
                      <div />
                    </div>
                    <div className="grid2" style={{marginTop:8}}>
                      <input className="input" type="number" step="0.0001" value={trainFull.stage_a_lr} onChange={e=>setTrainFull({...trainFull,stage_a_lr:Number(e.target.value)})} placeholder="LR A"/>
                      <input className="input" type="number" step="0.0001" value={trainFull.stage_b_lr} onChange={e=>setTrainFull({...trainFull,stage_b_lr:Number(e.target.value)})} placeholder="LR B"/>
                    </div>
                    <div className="actions" style={{marginTop:10}}><div className="tiny">POST /api/pipeline/train-gcn-full</div><button className="btn primary" onClick={runTrainFull}>Chạy full flow</button></div>
                  </div>
                </div>
              </div>

              <div className="stack">
                <div className="step">
                  <div className="step-head"><h3>Job train gần nhất</h3><span className={`badge ${statusClass(trainFullJob?.status || trainBJob?.status || trainAJob?.status)}`}>{((trainFullJob?.status || trainBJob?.status || trainAJob?.status || "idle")).toUpperCase()}</span></div>
                  <div className="step-body">
                    <div className="tiny">Stage A</div>
                    <div className="term" style={{marginTop:6}}><h4>TRAIN STAGE A</h4><pre>{lastLog(trainAJob)}</pre></div>
                    <div className="tiny" style={{marginTop:10}}>Stage B</div>
                    <div className="term" style={{marginTop:6}}><h4>TRAIN STAGE B</h4><pre>{lastLog(trainBJob)}</pre></div>
                    <div className="tiny" style={{marginTop:10}}>Full flow</div>
                    <div className="term" style={{marginTop:6}}><h4>TRAIN FULL FLOW</h4><pre>{lastLog(trainFullJob)}</pre></div>
                  </div>
                </div>

                <div className="step">
                  <div className="step-head"><h3>Gợi ý sử dụng</h3><span className="badge idle">READY</span></div>
                  <div className="step-body">
                    <p className="p">Nếu bạn mới bắt đầu, nên train Stage A trước để tạo checkpoint nền, sau đó mới fine-tune Stage B bằng checkpoint của Stage A.</p>
                    <p className="p" style={{marginTop:8}}>Nếu đã có sẵn `data/stage_b_vi_dataset.json`, bạn có thể bỏ qua full flow và chạy thẳng Stage B để tiết kiệm thời gian.</p>
                    <p className="p" style={{marginTop:8}}>Khi có validation dataset, hãy bật `early_stop_patience` lớn hơn 0 để tránh train quá tay.</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="step"><div className="step-head"><h3>Output API mới nhất</h3><span className="badge idle">READY</span></div><div className="step-body"><div className="term"><h4>APPLICATION/JSON</h4><pre>{out||"Chưa có phản hồi API"}</pre></div></div></div>
          </> : page!=="prep" ? <div className="step"><div className="step-body"><h3>{page}</h3><p className="p">Màn này sẽ làm tiếp sau.</p></div></div> : <>
            <h2 className="title">Chuẩn bị dữ liệu</h2>
            <div className="subtitle">4 bước quan trọng: Upload dữ liệu {"->"} OCR tạo nhãn {"->"} Hoàn tất nhãn và compile dataset {"->"} Import bộ dữ liệu mẫu nếu cần.</div>

            <div className="step">
              <div className="step-head"><h3>Bước 1: Upload ảnh đầu vào</h3><span className="tiny">POST /api/files/upload-images</span></div>
              <div className="step-body">
                <div className="row" style={{gridTemplateColumns:"auto auto 1fr", marginBottom:6}}>
                  <button className={`btn ${pickMode==="files"?"primary":""}`} onClick={()=>setPickMode("files")}>Chọn từng ảnh</button>
                  <button className={`btn ${pickMode==="folder"?"primary":""}`} onClick={()=>setPickMode("folder")}>Chọn thư mục</button>
                  <div className="tiny">Đã chọn: {files.length} file</div>
                </div>
                <label className="drop">
                  {pickMode==="files" ? (
                    <input key={`files-${pickerKey}`} type="file" multiple accept=".png,.jpg,.jpeg,.tif,.tiff" style={{display:"none"}} onChange={e=>setFiles(Array.from(e.target.files||[]))}/>
                  ) : (
                    <input key={`folder-${pickerKey}`} type="file" multiple webkitdirectory="true" directory="" style={{display:"none"}} onChange={e=>setFiles(Array.from(e.target.files||[]))}/>
                  )}
                  <div><div style={{fontWeight:700}}>{pickMode==="files"?"Chọn file ảnh":"Chọn thư mục ảnh"}</div><div className="tiny">PNG, JPG, TIFF</div></div>
                </label>
                <div className="row"><input className="input" placeholder="Thư mục con (tùy chọn)" value={subdir} onChange={e=>setSubdir(e.target.value)}/><button className="btn primary" onClick={upload}>Upload</button><button className="btn" onClick={clearOldFiles}>Xóa file cũ</button></div>
                <div className="table-wrap">
                  <table className="tbl">
                    <thead>
                      <tr>
                        <th style={{width:60}}>STT</th>
                        <th style={{width:220}}>Tên file</th>
                        <th>Đường dẫn</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recentRaw.slice(0,20).map((f,i)=>(
                        <tr key={i}>
                          <td>{i+1}</td>
                          <td>{shortName(String(f).split(/[\\/]/).pop())}</td>
                          <td className="path">{f}</td>
                        </tr>
                      ))}
                      {recentRaw.length===0 && (
                        <tr><td colSpan={3} className="tiny">Chưa có ảnh trong thư mục đầu vào.</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="step">
              <div className="step-head"><h3>Bước 2: OCR batch và tạo nodes_to_label.csv</h3><span className={`badge ${statusClass(ocrJob?.status)}`}>{(ocrJob?.status||"idle").toUpperCase()}</span></div>
              <div className="step-body">
                <div className="grid3"><input className="input" value={ocr.lang} onChange={e=>setOcr({...ocr,lang:e.target.value})}/><input className="input" value={ocr.input_dir} onChange={e=>setOcr({...ocr,input_dir:e.target.value})}/><input className="input" value={ocr.output_dir} onChange={e=>setOcr({...ocr,output_dir:e.target.value})}/></div>
                <div style={{marginTop:8}} className="p">Tiến độ xử lý file: <b>{ocrCurrent}</b> / <b>{ocrTotal}</b> ({ocrPct}%)</div>
                {ocrCurrentFile && <div className="tiny" style={{marginTop:4}}>Đang xử lý: <b>{ocrCurrentFile}</b></div>}
                <div className="bar" style={{marginTop:6}}><div style={{width:`${ocrPct}%`}}/></div>
                <div className="row" style={{gridTemplateColumns:"1fr auto",marginTop:10}}><div className="tiny">POST /api/pipeline/prepare-ocr-labeling</div><button className="btn dark" onClick={runOcrBatch}>Chạy bước 2</button></div>
                <div className="term" style={{marginTop:8}}><h4>LOG BƯỚC 2 (THỰC TẾ)</h4><pre>{lastLog(ocrJob)}</pre></div>
              </div>
            </div>

            <div className="step">
              <div className="step-head"><h3>Bước 3: Hoàn tất nhãn và compile dataset</h3><span className={`badge ${statusClass(dsJob?.status)}`}>{(dsJob?.status||"idle").toUpperCase()}</span></div>
              <div className="step-body">
                <div className="grid2"><input className="input" value={dataset.input_csv} onChange={e=>setDataset({...dataset,input_csv:e.target.value})}/><input className="input" value={dataset.output_json} onChange={e=>setDataset({...dataset,output_json:e.target.value})}/></div>
                <div className="row" style={{gridTemplateColumns:"100px 90px auto auto auto auto auto 1fr",marginTop:8}}>
                  <input className="input" type="number" min="10" max="500" value={labelLimit} onChange={e=>setLabelLimit(e.target.value)} />
                  <input className="input" type="number" min="1" value={labelPage} onChange={e=>setLabelPage(e.target.value)} />
                  <button className="btn" onClick={()=>{setMissingOnly(false); loadLabelPreview();}}>Lấy dữ liệu</button>
                  <button className="btn" onClick={()=>{setMissingOnly(true); loadLabelPreview(1, true);}}>Xem dữ liệu thiếu nhãn</button>
                  <button className="btn" onClick={()=>{const p=Math.max(1, Number(labelPage)-1); setLabelPage(p); loadLabelPreview(p, missingOnly);}}>Trang trước</button>
                  <button className="btn" onClick={()=>{const p=Math.min(totalPages, Number(labelPage)+1); setLabelPage(p); loadLabelPreview(p, missingOnly);}}>Trang sau</button>
                  <button className="btn" onClick={loadByDoc}>Xem theo ảnh</button>
                  <button className="btn" onClick={autoSuggestLabels}>AI gợi ý nhãn (batch 10 ảnh)</button>
                  <button className="btn" onClick={saveLabelUpdates}>Lưu nhãn</button>
                  <div className="tiny">Trang {labelPage}/{totalPages}. Bắt buộc: không được để rỗng cột label</div>
                </div>
                {labelHint && <div className="tiny" style={{marginTop:6}}>{labelHint}</div>}
                <div style={{marginTop:6}} className="tiny">
                  AI Progress: {aiLabelJob?.progress?.current_docs||0}/{aiLabelJob?.progress?.total_docs||0} ảnh ({aiLabelJob?.progress?.percent||0}%)
                  {aiLabelJob?.status ? ` | Trạng thái: ${aiLabelJob.status}` : ""}
                </div>
                <div className="bar" style={{marginTop:4}}><div style={{width:`${aiLabelJob?.progress?.percent||0}%`}}/></div>
                <div className="table-wrap" style={{maxHeight:340, marginTop:8}}>
                  <table className="tbl">
                    <thead>
                      <tr>
                        <th style={{width:80}}>Ảnh</th>
                        <th style={{width:180}}>Doc ID</th>
                        <th style={{width:80}}>Dòng</th>
                        <th>Nội dung OCR trích xuất</th>
                        <th style={{width:220}}>Nhãn gán</th>
                      </tr>
                    </thead>
                    <tbody>
                      {reviewByDoc.map((doc)=>doc.rows.map((r,idx)=>(
                        <tr key={`${doc.doc_id}-${r.row_number}`}>
                          <td>
                            {idx===0 ? (
                              doc.preview_path ? (
                                <img
                                  className="thumb"
                                  src={`/api/files/image?path=${encodeURIComponent(doc.preview_path)}`}
                                  alt={doc.doc_id}
                                  onClick={()=>setPreviewImage(`/api/files/image?path=${encodeURIComponent(doc.preview_path)}`)}
                                />
                              ) : "-"
                            ) : ""}
                          </td>
                          <td>{idx===0 ? doc.doc_id : ""}</td>
                          <td>{r.row_number}</td>
                          <td>{String(r.text||"")}</td>
                          <td>
                            <select className="input" value={r.picked_label||""} onChange={e=>setLabelRows(prev=>prev.map(x=>x.row_number===r.row_number?{...x,picked_label:e.target.value}:x))}>
                              <option value="">-- chọn nhãn --</option>
                              {allowedLabels.map(lb=><option key={lb} value={lb}>{lb}</option>)}
                            </select>
                          </td>
                        </tr>
                      )))}
                      {labelRows.length===0 && (
                        <tr><td colSpan={5} className="tiny">Chưa có dữ liệu hiển thị. Bấm `Lấy dữ liệu` để rà soát.</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>
                <div className="row" style={{gridTemplateColumns:"1fr auto",marginTop:10}}><div className="tiny">POST /api/pipeline/preprocess-gcn-dataset</div><button className="btn dark" onClick={runDatasetCompile}>Compile dataset</button></div>
                <div className="term" style={{marginTop:8}}><h4>LOG BƯỚC 3 (THỰC TẾ)</h4><pre>{lastLog(dsJob)}</pre></div>
              </div>
            </div>

            <div className="step">
              <div className="step-head"><h3>Bước 4: Import dữ liệu mẫu (tùy chọn)</h3><span className={`badge ${statusClass(hfJob?.status)}`}>{(hfJob?.status||"idle").toUpperCase()}</span></div>
              <div className="step-body">
                <div className="row" style={{gridTemplateColumns:"1fr auto"}}><div className="tiny">POST /api/pipeline/convert-hf-cord-to-csv</div><button className="btn" onClick={runCordImport}>Import CORD</button></div>
                <div className="term" style={{marginTop:8}}><h4>LOG BƯỚC 4 (THỰC TẾ)</h4><pre>{lastLog(hfJob)}</pre></div>
              </div>
            </div>

            <div className="step"><div className="step-head"><h3>Output API mới nhất</h3><span className="badge idle">READY</span></div><div className="step-body"><div className="term"><h4>APPLICATION/JSON</h4><pre>{out||"Chưa có phản hồi API"}</pre></div></div></div>
          </>}
        </div>
      </main>
    </div>
    {previewImage && (
      <div className="modal" onClick={()=>setPreviewImage("")}>
        <div className="modal-box" onClick={(e)=>e.stopPropagation()}>
          <button className="modal-close" onClick={()=>setPreviewImage("")}>Đóng</button>
          <img src={previewImage} alt="preview"/>
        </div>
      </div>
    )}
  </>;
}

createRoot(document.getElementById("root")).render(<App />);
