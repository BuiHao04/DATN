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
.modal-wide{width:min(1280px,96vw);max-height:92vh;overflow:auto;padding:14px}
.inspect-grid{display:grid;grid-template-columns:340px 1fr;gap:12px;margin-top:10px}
.inspect-side{display:flex;flex-direction:column;gap:10px}
.inspect-preview{width:100%;max-height:360px;object-fit:contain;background:#f6f6fb;border:1px solid #d8d4e5;border-radius:10px}
.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.stat{border:1px solid #d7d3e3;border-radius:10px;background:#faf9ff;padding:10px}
.stat .k{font-size:11px;color:#6b7280;margin-bottom:4px}.stat .v{font-size:18px;font-weight:800}
.matrix-note{margin:6px 0 0;font-size:12px;color:#6b7280}
.doc-browser{display:grid;grid-template-columns:320px 1fr;gap:12px;margin-top:10px}
.doc-list{border:1px solid #d4d1e0;border-radius:10px;background:#fff;overflow:hidden}
.doc-list-head{padding:10px;border-bottom:1px solid #e7e3ef;background:#faf9ff;display:flex;flex-direction:column;gap:8px}
.doc-list-body{max-height:760px;overflow:auto}
.doc-toolbar{display:grid;grid-template-columns:1.2fr .8fr;gap:8px}
.doc-pager{display:grid;grid-template-columns:84px 1fr auto auto;gap:8px;align-items:center}
.doc-summary{display:flex;justify-content:space-between;gap:8px;align-items:center;padding:8px 10px;border:1px solid #e4e0ee;border-radius:10px;background:#fff}
.doc-summary strong{font-size:13px}
.doc-muted{font-size:11px;color:#6b7280}
.doc-item{display:grid;grid-template-columns:56px 1fr auto;gap:8px;padding:8px 10px;border-bottom:1px solid #efedf5;cursor:pointer}
.doc-item.active{background:#eef2ff}
.doc-meta{display:flex;flex-direction:column;gap:4px;min-width:0}
.doc-id{font-size:13px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.doc-sub{font-size:11px;color:#6b7280}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#eef2ff;color:#3049b9;font-size:11px;font-weight:700}
.inspect-panel{border:1px solid #d4d1e0;border-radius:10px;background:#fff;overflow:hidden}
.inspect-panel-head{padding:10px 12px;border-bottom:1px solid #e7e3ef;background:#faf9ff;display:flex;justify-content:space-between;align-items:center;gap:10px}
.inspect-title{display:flex;flex-direction:column;gap:4px}
.inspect-title h4{margin:0;font-size:18px}
.inspect-tabs{display:flex;gap:8px;flex-wrap:wrap}
.tab-btn{border:1px solid #cbd3f0;background:#fff;color:#3049b9;border-radius:8px;padding:7px 10px;font-size:12px;font-weight:700;cursor:pointer}
.tab-btn.active{background:#3049b9;color:#fff;border-color:#3049b9}
.inspect-layout{display:grid;grid-template-columns:360px 1fr;gap:12px;padding:12px}
.image-stage{position:relative;border:1px solid #ddd8ea;border-radius:10px;background:#f8f7fc;overflow:hidden}
.image-stage img{width:100%;display:block}
.ocr-box{position:absolute;border:2px solid rgba(48,73,185,.55);background:rgba(48,73,185,.08);cursor:pointer}
.ocr-box.active{border-color:#ef4444;background:rgba(239,68,68,.12)}
.node-row{cursor:pointer}
.node-row.active{background:#eef2ff}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:10px}
.card{background:#fff;border:1px solid #d4d0e1;border-radius:12px;padding:12px}
.card .k{font-size:12px;color:#6b7280;margin-bottom:6px}.card .v{font-size:22px;font-weight:800}
.split{display:grid;grid-template-columns:1.15fr .85fr;gap:10px}
.stack{display:flex;flex-direction:column;gap:10px}
.actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.hero-train{display:grid;grid-template-columns:1.2fr .8fr;gap:12px;margin-bottom:12px}
.train-main{background:linear-gradient(135deg,#ffffff 0%,#f7f8ff 100%);border:1px solid #d9d5e7;border-radius:14px;padding:16px}
.train-side{display:flex;flex-direction:column;gap:10px}
.chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{padding:6px 10px;border-radius:999px;background:#eef2ff;color:#3049b9;font-size:12px;font-weight:700}
.preset-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.preset{border:1px solid #cfd6f5;background:#fff;border-radius:12px;padding:10px;cursor:pointer}
.preset.active{border-color:#3049b9;background:#eef2ff}
.field-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.field{display:flex;flex-direction:column;gap:6px}
.field-label{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:700}
.hint{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:999px;background:#eef2ff;color:#3049b9;font-size:11px;cursor:help}
.field-help{font-size:12px;color:#6b7280;line-height:1.4}
.details{border:1px solid #d8d4e5;border-radius:12px;background:#fff;overflow:hidden}
.details summary{list-style:none;cursor:pointer;padding:12px 14px;font-weight:800;background:#faf9ff;border-bottom:1px solid #ebe8f3}
.details summary::-webkit-details-marker{display:none}
.alert{border:1px solid #d9d5e7;border-radius:12px;padding:10px 12px;background:#fff}
.alert.warn{background:#fff7ed;border-color:#fed7aa}
.alert.good{background:#ecfdf5;border-color:#bbf7d0}
.alert.bad{background:#fef2f2;border-color:#fecaca}
.alert-title{font-size:12px;font-weight:800;margin-bottom:4px}
.results-grid{display:grid;grid-template-columns:1.05fr .95fr;gap:12px}
.metrics-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
.json-box{background:#f8f7fd;border:1px solid #d8d4e5;border-radius:10px;padding:10px;max-height:280px;overflow:auto;font-family:Consolas,monospace;font-size:12px;white-space:pre-wrap}
.kv{display:grid;grid-template-columns:220px 1fr;gap:8px;font-size:13px}
.kv div{padding:6px 0;border-bottom:1px solid #edeaf5}
.kv div:nth-last-child(-n+2){border-bottom:none}
.image-compare{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.image-card{border:1px solid #d8d4e5;border-radius:10px;background:#fff;overflow:hidden}
.image-card h4{margin:0;padding:8px 10px;background:#faf9ff;border-bottom:1px solid #e7e3ef;font-size:12px}
.image-card img{display:block;width:100%;max-height:420px;object-fit:contain;background:#f7f6fc}
.scroll-table{max-height:320px;overflow:auto;border:1px solid #d4d1e0;border-radius:8px}
.empty-state{padding:18px;border:1px dashed #d5d0e2;border-radius:10px;background:#fcfcff;color:#6b7280;font-size:13px}
.review-shell{display:grid;grid-template-columns:300px 1.15fr .95fr;gap:12px;margin-top:10px}
.review-pane{border:1px solid #d4d1e0;border-radius:12px;background:#fff;overflow:hidden}
.review-pane-head{padding:10px 12px;border-bottom:1px solid #e8e5f0;background:#faf9ff}
.review-pane-head h4{margin:0;font-size:15px}
.review-pane-body{padding:10px 12px}
.review-list{max-height:820px;overflow:auto}
.review-image{border:1px solid #ddd8ea;border-radius:12px;background:#f8f7fc;overflow:auto;padding:12px;text-align:center}
.review-canvas{position:relative;display:inline-block;line-height:0;max-width:100%}
.review-canvas img{display:block;max-width:100%;height:auto;max-height:680px;background:#f8f7fc}
.review-overlay{position:absolute;inset:0;pointer-events:none}
.review-overlay svg{width:100%;height:100%;display:block}
.ocr-poly{cursor:pointer;pointer-events:auto}
.review-kpis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:10px}
.review-kpi{border:1px solid #ddd8ea;border-radius:10px;background:#f9f8fe;padding:9px}
.review-kpi .k{font-size:11px;color:#6b7280;margin-bottom:4px}.review-kpi .v{font-size:18px;font-weight:800}
.node-filterbar{display:grid;grid-template-columns:1fr 160px 160px;gap:8px;margin-bottom:10px}
.node-detail{border:1px solid #ddd8ea;border-radius:10px;background:#faf9ff;padding:10px;margin-bottom:10px}
.node-detail-grid{display:grid;grid-template-columns:110px 1fr;gap:8px;font-size:12px}
.node-detail-grid div{padding:4px 0;border-bottom:1px solid #ebe8f3}
.node-detail-grid div:nth-last-child(-n+2){border-bottom:none}
.chipline{display:flex;gap:6px;flex-wrap:wrap}
.mini-chip{display:inline-flex;align-items:center;padding:4px 8px;border-radius:999px;background:#eef2ff;color:#3049b9;font-size:11px;font-weight:700;border:1px solid transparent}
button.mini-chip{cursor:pointer}
.mini-chip.active{background:#3049b9;color:#fff;border-color:#3049b9}
.compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.compare-box{border:1px solid #d8d4e5;border-radius:10px;background:#fff;overflow:hidden}
.compare-box h4{margin:0;padding:8px 10px;border-bottom:1px solid #e7e3ef;background:#faf9ff;font-size:12px}
.compare-box .body{padding:10px}
.explain-banner{border:1px solid #cfe1ff;background:linear-gradient(135deg,#f8fbff 0%,#eef4ff 100%);border-radius:12px;padding:12px}
.explain-banner strong{display:block;margin-bottom:4px}
.change-good{color:#166534;font-weight:700}
.change-muted{color:#6b7280}
@media (max-width:1200px){.app{grid-template-columns:220px 1fr}.path{max-width:260px}}
@media (max-width:1100px){.cards,.split,.inspect-grid,.doc-browser,.inspect-layout,.hero-train,.field-grid,.preset-grid,.doc-toolbar,.doc-pager,.results-grid,.metrics-grid,.image-compare,.kv,.review-shell,.node-filterbar,.compare-grid{grid-template-columns:1fr}}
`;

const NAV = [["single", "Kiểm thử 1 ảnh"],["prep", "Chuẩn bị dữ liệu"],["train", "Huấn luyện"],["results", "Kết quả & Đánh giá"]];
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

const LABEL_TEXT = {
  MERCHANT_NAME: "Tên cửa hàng",
  MERCHANT_ADDRESS: "Địa chỉ cửa hàng",
  MERCHANT_PHONE: "Số điện thoại",
  TAX_CODE: "Mã số thuế",
  INVOICE_ID: "Mã / số hóa đơn",
  DATE: "Ngày",
  TIME: "Giờ",
  CASHIER: "Thu ngân / quầy",
  ITEM_NAME: "Tên sản phẩm",
  ITEM_QTY: "Số lượng",
  ITEM_UNIT_PRICE: "Đơn giá",
  ITEM_AMOUNT: "Thành tiền dòng",
  SUBTOTAL: "Tạm tính",
  SERVICE_FEE: "Phí dịch vụ",
  DISCOUNT: "Giảm giá",
  TAX_AMOUNT: "Tiền thuế",
  TOTAL_AMOUNT: "Tổng thanh toán",
  PAYMENT_METHOD: "Phương thức thanh toán",
  OTHER: "Khác",
};

function labelText(v){
  return LABEL_TEXT[v] || v || "Chưa gán";
}

const LABEL_COLORS = {
  MERCHANT_NAME: {border:"#2563eb", fill:"rgba(37,99,235,.12)"},
  MERCHANT_ADDRESS: {border:"#0f766e", fill:"rgba(15,118,110,.12)"},
  MERCHANT_PHONE: {border:"#0891b2", fill:"rgba(8,145,178,.12)"},
  TAX_CODE: {border:"#7c3aed", fill:"rgba(124,58,237,.12)"},
  INVOICE_ID: {border:"#9333ea", fill:"rgba(147,51,234,.12)"},
  DATE: {border:"#ea580c", fill:"rgba(234,88,12,.12)"},
  TIME: {border:"#f97316", fill:"rgba(249,115,22,.12)"},
  CASHIER: {border:"#db2777", fill:"rgba(219,39,119,.12)"},
  ITEM_NAME: {border:"#16a34a", fill:"rgba(22,163,74,.12)"},
  ITEM_QTY: {border:"#65a30d", fill:"rgba(101,163,13,.12)"},
  ITEM_UNIT_PRICE: {border:"#ca8a04", fill:"rgba(202,138,4,.12)"},
  ITEM_AMOUNT: {border:"#d97706", fill:"rgba(217,119,6,.12)"},
  SUBTOTAL: {border:"#b45309", fill:"rgba(180,83,9,.12)"},
  SERVICE_FEE: {border:"#c2410c", fill:"rgba(194,65,12,.12)"},
  DISCOUNT: {border:"#dc2626", fill:"rgba(220,38,38,.12)"},
  TAX_AMOUNT: {border:"#be123c", fill:"rgba(190,18,60,.12)"},
  TOTAL_AMOUNT: {border:"#ef4444", fill:"rgba(239,68,68,.16)"},
  PAYMENT_METHOD: {border:"#0284c7", fill:"rgba(2,132,199,.12)"},
  OTHER: {border:"#64748b", fill:"rgba(100,116,139,.12)"},
};

function labelColor(label){
  return LABEL_COLORS[label] || LABEL_COLORS.OTHER;
}

function nodeHasQuad(node){
  return Array.isArray(node?.quad) && node.quad.length >= 4;
}

function nodeQuadPoints(node){
  if(!nodeHasQuad(node)) return "";
  return node.quad.map((pt)=>`${Number(pt?.[0]||0)},${Number(pt?.[1]||0)}`).join(" ");
}

function isMoneyLike(text){
  const t = String(text||"").trim().toLowerCase();
  return /(\d[\d.,]{2,}|\b(vnd|vnđ|đ)\b)/i.test(t);
}

function isDateLike(text){
  const t = String(text||"").trim();
  return /\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b/.test(t);
}

function isTimeLike(text){
  const t = String(text||"").trim();
  return /\b\d{1,2}:\d{2}\b/.test(t);
}

function isPhoneLike(text){
  const digits = String(text||"").replace(/\D/g,"");
  return digits.length >= 9 && digits.length <= 11;
}

function isTaxCodeLike(text){
  const digits = String(text||"").replace(/\D/g,"");
  return digits.length === 10 || digits.length === 13 || digits.length === 14;
}

function reviewIssueForNode(node){
  const text = String(node?.text||"").trim();
  const label = String(node?.picked_label || node?.label || "").trim();
  if(!label) return "Thiếu nhãn";
  if(isDateLike(text) && label !== "DATE") return "Text giống ngày nhưng chưa gán Ngày";
  if(isTimeLike(text) && label !== "TIME") return "Text giống giờ nhưng chưa gán Giờ";
  if(isPhoneLike(text) && !["MERCHANT_PHONE","TAX_CODE","INVOICE_ID"].includes(label)) return "Text giống số điện thoại / mã nhưng nhãn chưa hợp lý";
  if(isTaxCodeLike(text) && label === "OTHER") return "Text giống mã số thuế / mã hóa đơn nhưng đang là Khác";
  if(isMoneyLike(text) && !["ITEM_UNIT_PRICE","ITEM_AMOUNT","SUBTOTAL","SERVICE_FEE","DISCOUNT","TAX_AMOUNT","TOTAL_AMOUNT","OTHER"].includes(label)) return "Text giống tiền nhưng nhãn chưa hợp lý";
  return "";
}

const TRAIN_HELP = {
  dataset_json: "Đường dẫn file dataset JSON dùng để train. Nếu file này sai hoặc chưa tạo xong ở bước chuẩn bị dữ liệu thì train sẽ không chạy được.",
  val_dataset_json: "Dataset dùng để kiểm tra sau mỗi epoch. Có validation thì bạn sẽ biết model đang tốt lên hay bắt đầu học quá tay.",
  checkpoint: "Nơi lưu trọng số model sau khi train. Nên đặt tên riêng cho từng lần train để dễ so sánh.",
  base_checkpoint: "Checkpoint nền của Stage A. Stage B sẽ fine-tune từ đây nên nếu chọn đúng checkpoint tốt, model học nhanh và ổn định hơn.",
  init_checkpoint: "Checkpoint khởi tạo cho Stage A hoặc full flow. Dùng khi bạn muốn học tiếp thay vì train từ đầu.",
  epochs: "Số vòng quét hết toàn bộ dữ liệu. Epoch cao hơn có thể học kỹ hơn nhưng cũng dễ quá khớp nếu dữ liệu ít.",
  lr: "Learning rate quyết định model cập nhật nhanh hay chậm. Quá lớn dễ học lệch, quá nhỏ thì train rất lâu.",
  early_stop_patience: "Nếu validation không còn cải thiện sau N epoch thì dừng sớm. Giúp tiết kiệm thời gian và tránh overfit.",
  output_eval: "File JSON chứa kết quả đánh giá cuối cùng sau khi chạy full flow.",
  eval_json: "Dataset chỉ dùng để đánh giá cuối. Không bắt buộc nhưng rất nên có nếu bạn muốn đo chất lượng thực tế.",
};

const STAGE_B_PRESETS = {
  an_toan: {label:"An toàn", epochs:15, lr:0.0003, early_stop_patience:3, desc:"Cho người mới, ít rủi ro học lệch, thời gian vừa phải."},
  can_bang: {label:"Cân bằng", epochs:20, lr:0.0005, early_stop_patience:4, desc:"Khuyến nghị mặc định nếu dataset của bạn tương đối ổn."},
  ky_hon: {label:"Học kỹ hơn", epochs:35, lr:0.0002, early_stop_patience:6, desc:"Cho dataset lớn hơn hoặc khi bạn muốn model học chậm nhưng sâu hơn."},
};

const OCR_HELP = {
  lang: "Ngôn ngữ OCR. Với hóa đơn tiếng Việt nên để `vi` để nhận dấu tốt hơn.",
  input_dir: "Thư mục ảnh đầu vào mà bước 2 sẽ quét toàn bộ để OCR.",
  output_dir: "Thư mục lưu kết quả OCR cho bước 3, gồm CSV, ảnh chuẩn hóa, JSON OCR và debug box.",
  ocr_engine: "PaddleOCR là mặc định. VietOCR sẽ dùng Paddle để detect box rồi VietOCR đọc text bên trong box.",
  det_db_box_thresh: "Ngưỡng giữ box detect. Tăng lên thì box chặt hơn nhưng có thể bỏ sót chữ nhỏ.",
  det_db_unclip_ratio: "Mức nới box ra sau detect. Giảm xuống nếu box đang bị phình rộng quá.",
  drop_score: "Ngưỡng bỏ text OCR yếu. Tăng lên nếu đang dính nhiều text rác.",
  det_limit_side_len: "Kích thước cạnh lớn nhất trước khi detect. Tăng lên giúp bắt chữ nhỏ nhưng chậm hơn và tốn VRAM/RAM hơn.",
  upscale_factor: "Phóng to ảnh trước khi OCR. Hữu ích với ảnh mờ, chữ nhỏ hoặc hóa đơn chụp xa.",
  use_dilation: "Nối nét chữ trước detect. Chỉ nên bật khi chữ bị đứt hoặc box đang vỡ vụn.",
  save_every_images: "Cứ mỗi N ảnh sẽ flush CSV/JSON xuống đĩa một lần để hạn chế mất tiến độ nếu dừng giữa chừng.",
  overwrite_existing: "OCR lại toàn bộ từ đầu hoặc chỉ chạy tiếp phần ảnh chưa có kết quả.",
};

function shortName(s){s=String(s||""); return s.length>26?`${s.slice(0,24)}...`:s;}

function TrainField({label, hint, children, help}){
  return (
    <div className="field">
      <div className="field-label">
        <span>{label}</span>
        {hint ? <span className="hint" title={hint}>?</span> : null}
      </div>
      {children}
      {help ? <div className="field-help">{help}</div> : null}
    </div>
  );
}

function FileSelect({value, onChange, options, placeholder}){
  return (
    <select className="input" value={value} onChange={e=>onChange(e.target.value)}>
      <option value="">{placeholder || "-- chọn file --"}</option>
      {options.map((opt)=><option key={opt} value={opt}>{opt}</option>)}
    </select>
  );
}

function classifyTrainProfile(cfg){
  const epochs = Number(cfg.epochs || 0);
  const lr = Number(cfg.lr || 0);
  if (epochs <= 15 && lr >= 0.0004) return {label:"Nhanh", desc:"Ưu tiên chạy nhanh để kiểm tra pipeline hoặc test thử mô hình.", tone:"warn"};
  if (epochs >= 30 || lr <= 0.00025) return {label:"Học kỹ", desc:"Model học chậm và sâu hơn, phù hợp khi bạn có thời gian và dữ liệu tương đối tốt.", tone:"good"};
  return {label:"Cân bằng", desc:"Cấu hình trung dung, hợp lý cho đa số lần train Stage B đầu tiên.", tone:"good"};
}

function validateStageBConfig(cfg){
  const issues = [];
  const notes = [];
  if(!String(cfg.dataset_json||"").trim()) issues.push("Thiếu dataset train Stage B.");
  if(!String(cfg.base_checkpoint||"").trim()) issues.push("Thiếu checkpoint nền Stage A.");
  if(!String(cfg.checkpoint||"").trim()) issues.push("Thiếu đường dẫn lưu checkpoint đầu ra.");
  const epochs = Number(cfg.epochs || 0);
  const lr = Number(cfg.lr || 0);
  const patience = Number(cfg.early_stop_patience || 0);
  if(!Number.isFinite(epochs) || epochs <= 0) issues.push("Epoch phải lớn hơn 0.");
  if(!Number.isFinite(lr) || lr <= 0) issues.push("Learning rate phải lớn hơn 0.");
  if(epochs > 80) notes.push("Epoch đang khá cao. Nếu dữ liệu chưa lớn, model có thể dễ overfit.");
  if(lr > 0.001) notes.push("Learning rate khá lớn. Model có thể học không ổn định.");
  if(lr < 0.0001) notes.push("Learning rate khá nhỏ. Train có thể rất chậm.");
  if(String(cfg.val_dataset_json||"").trim() && patience === 0) notes.push("Bạn đã có validation nhưng early stop đang bằng 0. Nên cân nhắc đặt 3-6.");
  if(!String(cfg.val_dataset_json||"").trim()) notes.push("Chưa có validation dataset. Model vẫn train được nhưng bạn sẽ khó biết khi nào bắt đầu overfit.");
  return {issues, notes};
}

function validateStageBEvalConfig(cfg){
  const issues = [];
  if(!String(cfg.dataset_json||"").trim()) issues.push("Thiếu dataset test để đánh giá.");
  if(!String(cfg.checkpoint||"").trim()) issues.push("Thiếu checkpoint Stage B để đánh giá.");
  if(!String(cfg.output_eval||"").trim()) issues.push("Thiếu đường dẫn output report.");
  return {issues};
}

function App(){
  const [page,setPage]=useState("single");
  const [jobs,setJobs]=useState([]);
  const [out,setOut]=useState("");
  const [files,setFiles]=useState([]);
  const [pickMode,setPickMode]=useState("files");
  const [pickerKey,setPickerKey]=useState(0);
  const [subdir,setSubdir]=useState("");
  const [recentRaw,setRecentRaw]=useState([]);
  const [allRawImages,setAllRawImages]=useState([]);
  const [ocr,setOcr]=useState({
    input_dir:"data/stage_b_raw_images",
    output_dir:"data/labeling_stage_b",
    lang:"vi",
    ocr_engine:"paddle",
    det_db_thresh:0.25,
    det_db_box_thresh:0.6,
    det_db_unclip_ratio:1.25,
    drop_score:0.45,
    use_dilation:0,
    det_limit_side_len:1600,
    upscale_factor:1.6,
    save_debug_images:1,
    copy_images:1,
    save_every_images:10,
    overwrite_existing:1
  });
  const [dataset,setDataset]=useState({input_csv:"data/labeling_stage_b/nodes_to_label.csv",output_json:"data/stage_b_vi_dataset.json"});
  const [labelRows,setLabelRows]=useState([]);
  const [allowedLabels,setAllowedLabels]=useState(LABELS);
  const [labelHint,setLabelHint]=useState("");
  const [labelLimit,setLabelLimit]=useState(100);
  const [labelPage,setLabelPage]=useState(1);
  const [totalPages,setTotalPages]=useState(1);
  const [docView,setDocView]=useState([]);
  const [previewImage,setPreviewImage]=useState("");
  const [graphInspect,setGraphInspect]=useState(null);
  const [graphInspectLoading,setGraphInspectLoading]=useState(false);
  const [docSearch,setDocSearch]=useState("");
  const [docFilter,setDocFilter]=useState("all");
  const [docPage,setDocPage]=useState(1);
  const [docPageSize,setDocPageSize]=useState(40);
  const [docHasNext,setDocHasNext]=useState(false);
  const [inspectTab,setInspectTab]=useState("ocr");
  const [activeNodeIndex,setActiveNodeIndex]=useState(-1);
  const [imageNatural,setImageNatural]=useState({w:1,h:1});
  const [missingOnly,setMissingOnly]=useState(false);
  const [reviewSearch,setReviewSearch]=useState("");
  const [reviewNodeFilter,setReviewNodeFilter]=useState("all");
  const [inferNodeFilter,setInferNodeFilter]=useState("all");
  const [inferActiveNode,setInferActiveNode]=useState(-1);
  const [trainA,setTrainA]=useState({dataset_json:"data/stage_a_dataset.json",val_dataset_json:"",checkpoint:"outputs/checkpoints/gcn_stage_a.pt",init_checkpoint:"",epochs:30,lr:0.001,early_stop_patience:0});
  const [trainB,setTrainB]=useState({dataset_json:"data/stage_b_vi_dataset.json",val_dataset_json:"",base_checkpoint:"outputs/checkpoints/gcn_stage_a.pt",checkpoint:"outputs/checkpoints/gcn_stage_b.pt",epochs:20,lr:0.0005,early_stop_patience:0});
  const [trainFull,setTrainFull]=useState({stage_a_json:"data/stage_a_dataset.json",stage_b_json:"data/stage_b_vi_dataset.json",stage_a_ckpt:"outputs/checkpoints/gcn_stage_a.pt",stage_b_ckpt:"outputs/checkpoints/gcn_stage_b.pt",stage_a_epochs:30,stage_b_epochs:20,stage_a_lr:0.001,stage_b_lr:0.0005,init_checkpoint:"",eval_json:"",output_eval:"outputs/gcn_eval_report.json"});
  const [splitStageB,setSplitStageB]=useState({
    input_json:"data/stage_b_vi_dataset.json",
    output_train_json:"data/stage_b_train.json",
    output_val_json:"data/stage_b_val.json",
    output_test_json:"data/stage_b_test.json",
    train_ratio:0.7,
    val_ratio:0.15,
    test_ratio:0.15,
    seed:42,
  });
  const [splitResult,setSplitResult]=useState(null);
  const [stageBEval,setStageBEval]=useState({
    dataset_json:"data/stage_b_test.json",
    checkpoint:"outputs/checkpoints/gcn_stage_b.pt",
    output_eval:"outputs/gcn_stage_b_test_report.json",
  });
  const [inferOne,setInferOne]=useState({
    image:"",
    lang:"vi",
    ocr_engine:"paddle",
    checkpoint:"outputs/checkpoints/gcn_stage_b.pt",
    ocr_debug_image:"outputs/ocr_boxes_single.jpg",
    output_json:"outputs/gcn_infer_single.json",
  });
  const [inferUploadFile,setInferUploadFile]=useState(null);
  const [singleCheck,setSingleCheck]=useState({
    image:"",
    lang:"vi",
    ocr_engine:"paddle",
    det_db_thresh:0.25,
    det_db_box_thresh:0.6,
    det_db_unclip_ratio:1.25,
    drop_score:0.45,
    use_dilation:0,
    det_limit_side_len:1600,
    upscale_factor:1.6,
    llm_model:"gpt-4.1-mini",
    output_dir:"data/single_image_check",
    save_debug_image:1,
    same_line_ratio:1.2,
    near_threshold:250,
    llm_text_batch_size:10,
  });
  const [singleCheckUploadFile,setSingleCheckUploadFile]=useState(null);
  const [singleInspect,setSingleInspect]=useState(null);
  const [singleInspectLoading,setSingleInspectLoading]=useState(false);
  const [singleInspectTab,setSingleInspectTab]=useState("ocr");
  const [singleActiveNode,setSingleActiveNode]=useState(-1);
  const [singleNatural,setSingleNatural]=useState({w:1,h:1});
  const [evalReport,setEvalReport]=useState(null);
  const [inferReport,setInferReport]=useState(null);
  const [knownDataFiles,setKnownDataFiles]=useState([]);
  const [knownCheckpointFiles,setKnownCheckpointFiles]=useState([]);
  const pollRef=useRef(null);

  const applySingleOcrPreset=(preset)=>{
    if(preset==="tight"){
      setSingleCheck(prev=>({
        ...prev,
        det_db_thresh:0.25,
        det_db_box_thresh:0.62,
        det_db_unclip_ratio:1.18,
        drop_score:0.45,
        use_dilation:0,
        det_limit_side_len:1600,
        upscale_factor:1.8,
      }));
      return;
    }
    if(preset==="balanced"){
      setSingleCheck(prev=>({
        ...prev,
        det_db_thresh:0.25,
        det_db_box_thresh:0.58,
        det_db_unclip_ratio:1.25,
        drop_score:0.40,
        use_dilation:0,
        det_limit_side_len:1536,
        upscale_factor:1.6,
      }));
      return;
    }
    setSingleCheck(prev=>({
      ...prev,
      det_db_thresh:0.22,
      det_db_box_thresh:0.50,
      det_db_unclip_ratio:1.45,
      drop_score:0.35,
      use_dilation:1,
      det_limit_side_len:1792,
      upscale_factor:1.8,
    }));
  };

  const loadJobs=async()=>{try{const r=await fetch("/api/jobs");const d=await r.json();setJobs(Array.isArray(d)?d:[]);}catch{setJobs([])}};
  const loadRecentRaw=async()=>{try{const r=await fetch("/api/files/stage-b-raw-images");const d=await r.json();setAllRawImages(d.files||[]);setRecentRaw((d.files||[]).slice(0,10)); if(d.input_dir) setOcr(s=>({...s,input_dir:d.input_dir}));}catch{setAllRawImages([]);setRecentRaw([])}};
  const loadTrainFileOptions=async()=>{
    try{
      const [dataRes, ckptRes] = await Promise.all([
        fetch("/api/files/list?dir=data"),
        fetch("/api/files/checkpoints"),
      ]);
      const dataJson = await dataRes.json();
      const ckptJson = await ckptRes.json();
      setKnownDataFiles((dataJson.files||[]).filter(x=>/\.(json|csv)$/i.test(String(x))));
      setKnownCheckpointFiles((ckptJson.files||[]).filter(x=>/\.(pt|pth|bin)$/i.test(String(x))));
    }catch{
      setKnownDataFiles([]);
      setKnownCheckpointFiles([]);
    }
  };

  useEffect(()=>{loadJobs(); loadRecentRaw(); loadTrainFileOptions();},[]);
  useEffect(()=>{
    if(knownCheckpointFiles.length===0) return;
    const hasCurrent = knownCheckpointFiles.includes(trainB.base_checkpoint);
    const cordCkpt = knownCheckpointFiles.find(x=>String(x).endsWith("gcn_cord.pt"));
    if(!hasCurrent && cordCkpt){
      setTrainB(prev=>prev.base_checkpoint===trainB.base_checkpoint ? {...prev, base_checkpoint: cordCkpt} : prev);
      setStageBEval(prev=>!knownCheckpointFiles.includes(prev.checkpoint) ? {...prev, checkpoint: cordCkpt} : prev);
      setInferOne(prev=>!knownCheckpointFiles.includes(prev.checkpoint) ? {...prev, checkpoint: cordCkpt} : prev);
    }
  },[knownCheckpointFiles]);

  const latestByMode=useMemo(()=>{const m={}; for(const j of jobs) if(!m[j.mode]) m[j.mode]=j; return m;},[jobs]);
  const ocrJob=latestByMode["prepare_ocr_labeling"]; const dsJob=latestByMode["preprocess_gcn_dataset"]; const hfJob=latestByMode["convert_hf_cord_to_csv"];
  const aiLabelJob=latestByMode["labeling_auto_suggest"];
  const trainAJob=latestByMode["train_gcn_stage_a"];
  const trainBJob=latestByMode["train_gcn_stage_b"];
  const trainFullJob=latestByMode["train_gcn_full"];
  const stageBTestJob=latestByMode["test_gcn"];
  const inferJob=latestByMode["gcn_infer"];
  useEffect(()=>{const run=jobs.some(j=>j.status==="queued"||j.status==="running"); pollRef.current=setTimeout(loadJobs, run?2500:8000); return ()=>clearTimeout(pollRef.current);},[jobs]);
  useEffect(()=>{
    if(stageBTestJob?.status==="success" && stageBEval.output_eval){
      loadEvalReport();
    }
  },[stageBTestJob?.id, stageBTestJob?.status]);
  useEffect(()=>{
    if(inferJob?.status==="success" && inferOne.output_json){
      loadInferReport();
    }
  },[inferJob?.id, inferJob?.status]);
  useEffect(()=>{
    if((graphInspect?.nodes||[]).length && activeNodeIndex < 0){
      setActiveNodeIndex(graphInspect.nodes[0].node_index);
    }
  },[graphInspect?.doc_id]);
  useEffect(()=>{
    if(!graphInspect?.nodes?.length) return;
    if(reviewNodeFilter==="all" || reviewNodeFilter==="missing" || reviewNodeFilter==="labeled") return;
    const matched = graphInspect.nodes.find((node)=>{
      const label = String(node.picked_label || node.label || "").trim() || "OTHER";
      return label === reviewNodeFilter;
    });
    if(matched) setActiveNodeIndex(matched.node_index);
  },[reviewNodeFilter, graphInspect?.doc_id]);
  useEffect(()=>{
    if((inferReport?.node_features||[]).length && inferActiveNode < 0){
      setInferActiveNode(inferReport.node_features[0].node_id);
    }
  },[inferReport]);
  useEffect(()=>{
    if((singleInspect?.nodes||[]).length && singleActiveNode < 0){
      setSingleActiveNode(singleInspect.nodes[0].node_index);
    }
  },[singleInspect]);
  const statusClass=(s)=>s==="success"?"success":s==="failed"?"failed":s==="running"?"running":s==="queued"?"queued":s==="stopping"?"queued":s==="stopped"?"idle":"idle";
  const lastLog=(j)=>{if(!j) return "Chưa có log"; const t=(j.stderr||j.stdout||"").trim(); if(!t) return "Job đã tạo, đang chờ log..."; return t.split(/\r?\n/).slice(-25).join("\n");};

  const apiPost=async(url,body)=>{const r=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});const d=await r.json();setOut(JSON.stringify(d,null,2));await loadJobs();return d;};
  const setOutSummary=(title, data)=>setOut(JSON.stringify({title, ...data}, null, 2));
  const upload=async()=>{if(!files.length) return; const fd=new FormData(); files.forEach(f=>fd.append("files",f)); if(subdir.trim()) fd.append("subdir",subdir.trim()); const r=await fetch("/api/files/upload-images",{method:"POST",body:fd}); const d=await r.json(); setOut(JSON.stringify(d,null,2)); await loadRecentRaw(); await loadJobs();};
  const clearOldFiles=async()=>{const d=await apiPost("/api/files/clear-stage-b-raw-images",{}); setFiles([]); setSubdir(""); setPickMode("files"); setPickerKey(v=>v+1); await loadRecentRaw(); return d;};
  const runOcrBatch=async()=>{await apiPost("/api/pipeline/prepare-ocr-labeling",ocr);};
  const stopOcrBatch=async()=>{
    if(!ocrJob?.id) return;
    const r=await fetch(`/api/jobs/${ocrJob.id}/stop`,{method:"POST"});
    const d=await r.json();
    setOut(JSON.stringify(d,null,2));
    await loadJobs();
    setLabelHint("Đã gửi yêu cầu dừng bước 2. Hệ thống sẽ lưu các ảnh đã OCR xong để bạn sang bước 3 kiểm tra.");
  };
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
          only_empty: onlyMissing ? 1 : 0,
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
      setOutSummary("labeling_sample", {
        page: d.page,
        page_size: d.page_size,
        total_rows: d.total_rows,
        filtered_total_rows: d.filtered_total_rows,
        empty_label_count: d.empty_label_count,
        only_empty: d.only_empty,
        returned_rows: rows.length,
      });
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
  const runTrainB=async()=>{
    const check = validateStageBConfig(trainB);
    const issues = [...check.issues, ...stageBPathChecks.issues];
    const notes = [...check.notes, ...stageBPathChecks.notes];
    if(issues.length){
      setOut(JSON.stringify({status:"blocked",stage:"train_stage_b",issues,notes},null,2));
      return;
    }
    await apiPost("/api/pipeline/train-gcn-stage-b", {...trainB, val_dataset_json: trainB.val_dataset_json || null});
  };
  const applySplitOutputsToForms=(outputs)=>{
    if(!outputs) return;
    setTrainB(prev=>({
      ...prev,
      dataset_json: outputs.train?.path || prev.dataset_json,
      val_dataset_json: outputs.validation?.path || prev.val_dataset_json,
    }));
    setTrainFull(prev=>({
      ...prev,
      stage_b_json: outputs.train?.path || prev.stage_b_json,
      eval_json: outputs.test?.path || prev.eval_json,
    }));
    setStageBEval(prev=>({
      ...prev,
      dataset_json: outputs.test?.path || prev.dataset_json,
    }));
  };
  const runSplitStageB=async()=>{
    const d = await apiPost("/api/pipeline/split-gcn-dataset", splitStageB);
    if(d?.outputs){
      setSplitResult(d);
      applySplitOutputsToForms(d.outputs);
      setLabelHint(`Đã tách dataset Stage B: train=${d.outputs.train.graphs}, val=${d.outputs.validation.graphs}, test=${d.outputs.test.graphs}.`);
      await loadTrainFileOptions();
    }
  };
  const runStageBEval=async()=>{
    const check = validateStageBEvalConfig(stageBEval);
    if(check.issues.length){
      setOut(JSON.stringify({status:"blocked",stage:"eval_stage_b",issues:check.issues},null,2));
      return;
    }
    await apiPost("/api/pipeline/test-gcn", stageBEval);
  };
  const loadJsonOutput=async(path)=>{
    if(!String(path||"").trim()) return null;
    const r = await fetch(`/api/files/json?path=${encodeURIComponent(path)}`);
    const d = await r.json();
    if(!r.ok){
      setOut(JSON.stringify(d,null,2));
      return null;
    }
    setOut(JSON.stringify(d,null,2));
    return d;
  };
  const loadEvalReport=async()=>{
    const d = await loadJsonOutput(stageBEval.output_eval);
    if(d) setEvalReport(d);
  };
  const uploadInferImage=async()=>{
    if(!inferUploadFile) return;
    const fd = new FormData();
    fd.append("files", inferUploadFile);
    fd.append("subdir", "single_infer");
    const r = await fetch("/api/files/upload-images",{method:"POST",body:fd});
    const d = await r.json();
    setOut(JSON.stringify(d,null,2));
    if(r.ok && d.files?.[0]){
      setInferOne(prev=>({...prev,image:d.files[0]}));
      await loadRecentRaw();
    }
  };
  const runInferOne=async()=>{
    if(!String(inferOne.image||"").trim()){
      setOut(JSON.stringify({status:"blocked",stage:"gcn_infer",issues:["Thiếu đường dẫn ảnh để infer."]},null,2));
      return;
    }
    if(!String(inferOne.checkpoint||"").trim()){
      setOut(JSON.stringify({status:"blocked",stage:"gcn_infer",issues:["Thiếu checkpoint Stage B để infer."]},null,2));
      return;
    }
    await apiPost("/api/pipeline/gcn-infer", inferOne);
  };
  const loadInferReport=async()=>{
    const d = await loadJsonOutput(inferOne.output_json);
    if(d) setInferReport(d);
  };
  const uploadSingleCheckImage=async()=>{
    if(!singleCheckUploadFile) return;
    const fd = new FormData();
    fd.append("files", singleCheckUploadFile);
    fd.append("subdir", "single_check");
    const r = await fetch("/api/files/upload-images",{method:"POST",body:fd});
    const d = await r.json();
    setOut(JSON.stringify(d,null,2));
    if(r.ok && d.files?.[0]){
      setSingleCheck(prev=>({...prev,image:d.files[0]}));
      await loadRecentRaw();
    }
  };
  const runSinglePreview=async(withAI=false)=>{
    if(!String(singleCheck.image||"").trim()){
      setOut(JSON.stringify({status:"blocked",stage:"single_image_preview",issues:["Thiếu đường dẫn ảnh để kiểm thử."]},null,2));
      return;
    }
    setSingleInspectLoading(true);
    setSingleInspectTab(withAI ? "ai" : "ocr");
    setSingleActiveNode(-1);
    try{
      const r = await fetch("/api/pipeline/single-image-preview",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({...singleCheck, with_ai: withAI ? 1 : 0}),
      });
      const d = await r.json();
      setOut(JSON.stringify(d,null,2));
      if(!r.ok){
        setSingleInspect(null);
        setLabelHint(`Lỗi kiểm thử 1 ảnh: ${d?.detail || "unknown"}`);
        return;
      }
      setSingleInspect({
        ...d,
        nodes:(d.nodes||[]).map(node=>({
          ...node,
          picked_label: node.picked_label || node.label || "",
        })),
      });
      await loadTrainFileOptions();
    }catch(e){
      setSingleInspect(null);
      setLabelHint("Không gọi được API kiểm thử 1 ảnh.");
    }finally{
      setSingleInspectLoading(false);
    }
  };
  const runTrainFull=async()=>{await apiPost("/api/pipeline/train-gcn-full", {...trainFull, init_checkpoint: trainFull.init_checkpoint || null, eval_json: trainFull.eval_json || null});};
  const loadByDoc=async(pageOverride=null)=>{
    const targetPage = pageOverride ?? Number(docPage||1);
    const d=await apiPost("/api/pipeline/labeling-by-doc",{
      input_csv:dataset.input_csv,
      label_col:"label",
      text_col:"text",
      doc_id_col:"doc_id",
      page: targetPage,
      page_size: Number(docPageSize||40)
    });
    setDocView(d.docs||[]);
    setDocPage(Number(d.page||targetPage));
    setDocHasNext(Boolean(d.has_next));
    setOutSummary("labeling_by_doc", {
      page: d.page,
      page_size: d.page_size,
      has_next: d.has_next,
      returned_docs: (d.docs||[]).length,
    });
  };
  const openGraphInspect=async(docId)=>{
    setGraphInspectLoading(true);
    setInspectTab("ocr");
    setActiveNodeIndex(-1);
    try{
      const r=await fetch("/api/pipeline/labeling-graph-inspect",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({
          input_csv:dataset.input_csv,
          doc_id:docId,
          label_col:"label",
          text_col:"text",
          doc_id_col:"doc_id",
          score_col:"score",
          x1_col:"x1",
          y1_col:"y1",
          x2_col:"x2",
          y2_col:"y2",
          same_line_ratio:1.2,
          near_threshold:250
        })
      });
      const d=await r.json();
      if(!r.ok){
        setLabelHint(`Lỗi xem chi tiết graph của ảnh: ${d?.detail || "unknown"}`);
        setGraphInspect(null);
        return;
      }
      setOutSummary("labeling_graph_inspect", {
        doc_id: d.doc_id,
        num_nodes: d.graph?.num_nodes || 0,
        num_edges: d.graph?.num_edges || 0,
        preview_path: d.preview_path || "",
      });
      setGraphInspect({...d, nodes:(d.nodes||[]).map(x=>({...x, picked_label:x.label||""}))});
    }catch{
      setLabelHint("Không gọi được API xem graph cho ảnh.");
      setGraphInspect(null);
    }finally{
      setGraphInspectLoading(false);
    }
  };
  const saveInspectLabels=async()=>{
    if(!graphInspect?.nodes?.length) return;
    const updates=(graphInspect.nodes||[]).map(n=>({row_number:n.row_number,label:String(n.picked_label||"").trim()})).filter(x=>x.label!=="");
    const d=await apiPost("/api/pipeline/labeling-apply",{input_csv:dataset.input_csv,label_col:"label",updates});
    if(d?.updated>=0){
      setLabelHint(`Đã lưu ${d.updated} nhãn của ảnh ${graphInspect.doc_id}.`);
      await loadByDoc();
      await openGraphInspect(graphInspect.doc_id);
    }
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
  const filteredDocs = useMemo(()=>{
    const q=docSearch.trim().toLowerCase();
    return (docView||[]).filter(doc=>{
      const id=String(doc.doc_id||"").toLowerCase();
      const hit=!q || id.includes(q);
      if(!hit) return false;
      if(docFilter==="missing") return Number(doc.empty_labels||0)>0;
      if(docFilter==="labeled") return Number(doc.empty_labels||0)===0;
      return true;
    });
  },[docView, docSearch, docFilter]);
  const activeInspectNode = useMemo(
    ()=> (graphInspect?.nodes||[]).find(n=>n.node_index===activeNodeIndex) || null,
    [graphInspect, activeNodeIndex]
  );
  const reviewDocChecklist = useMemo(()=>{
    const required = ["MERCHANT_NAME","DATE","TOTAL_AMOUNT"];
    const present = new Set((graphInspect?.nodes||[]).map(n=>String(n.picked_label || n.label || "").trim()).filter(Boolean));
    return required.map((label)=>({
      label,
      ok: present.has(label),
    }));
  },[graphInspect]);
  const suspiciousNodes = useMemo(()=>{
    return (graphInspect?.nodes||[]).map((node)=>({
      ...node,
      issue: reviewIssueForNode(node),
    })).filter((node)=>node.issue);
  },[graphInspect]);
  const reviewNodesFiltered = useMemo(()=>{
    const q = reviewSearch.trim().toLowerCase();
    return (graphInspect?.nodes||[]).filter((node)=>{
      const text = String(node.text||"").toLowerCase();
      const label = String(node.picked_label||node.label||"");
      const missing = !String(node.picked_label||"").trim();
      const issue = reviewIssueForNode(node);
      if(q && !text.includes(q) && !label.toLowerCase().includes(q)) return false;
      if(reviewNodeFilter==="missing") return missing;
      if(reviewNodeFilter==="labeled") return !missing;
      if(reviewNodeFilter==="suspicious") return !!issue;
      if(reviewNodeFilter==="money") return isMoneyLike(node.text);
      if(reviewNodeFilter==="date") return isDateLike(node.text);
      if(reviewNodeFilter!=="all") return label===reviewNodeFilter;
      return true;
    });
  },[graphInspect, reviewSearch, reviewNodeFilter]);
  const reviewImageNodes = useMemo(()=>{
    return (graphInspect?.nodes||[]).filter((node)=>{
      const label = String(node.picked_label || node.label || "").trim() || "OTHER";
      const missing = !String(node.picked_label || "").trim();
      const issue = reviewIssueForNode(node);
      if(reviewNodeFilter==="missing") return missing;
      if(reviewNodeFilter==="labeled") return !missing;
      if(reviewNodeFilter==="suspicious") return !!issue;
      if(reviewNodeFilter==="money") return isMoneyLike(node.text);
      if(reviewNodeFilter==="date") return isDateLike(node.text);
      if(reviewNodeFilter!=="all") return label===reviewNodeFilter;
      return true;
    });
  },[graphInspect, reviewNodeFilter]);
  const nextMissingNode = useMemo(
    ()=> (graphInspect?.nodes||[]).find((n)=>!String(n.picked_label||"").trim()) || null,
    [graphInspect]
  );
  const nextSuspiciousNode = useMemo(
    ()=> (graphInspect?.nodes||[]).find((n)=>!!reviewIssueForNode(n)) || null,
    [graphInspect]
  );
  const inspectLabelSummary = useMemo(()=>{
    const counts = {};
    for(const node of (graphInspect?.nodes||[])){
      const key = String(node.picked_label || node.label || "").trim() || "OTHER";
      counts[key] = (counts[key] || 0) + 1;
    }
    return Object.entries(counts).sort((a,b)=>b[1]-a[1]);
  },[graphInspect]);
  const trainRunning = [trainAJob, trainBJob, trainFullJob].some(j=>j?.status==="running" || j?.status==="queued");
  const trainDoneCount = [trainAJob, trainBJob, trainFullJob].filter(j=>j?.status==="success").length;
  const trainFailCount = [trainAJob, trainBJob, trainFullJob].filter(j=>j?.status==="failed").length;
  const stageBProfile = useMemo(()=>classifyTrainProfile(trainB),[trainB]);
  const stageBValidation = useMemo(()=>validateStageBConfig(trainB),[trainB]);
  const stageBEvalValidation = useMemo(()=>validateStageBEvalConfig(stageBEval),[stageBEval]);
  const knownDataSet = useMemo(()=>new Set(knownDataFiles),[knownDataFiles]);
  const knownCheckpointSet = useMemo(()=>new Set(knownCheckpointFiles),[knownCheckpointFiles]);
  const stageBPathChecks = useMemo(()=>{
    const issues = [];
    const notes = [];
    if(String(trainB.dataset_json||"").trim() && !knownDataSet.has(trainB.dataset_json)){
      issues.push(`Không tìm thấy dataset train: ${trainB.dataset_json}`);
    }
    if(String(trainB.val_dataset_json||"").trim() && !knownDataSet.has(trainB.val_dataset_json)){
      issues.push(`Không tìm thấy dataset validation: ${trainB.val_dataset_json}`);
    }
    if(String(trainB.base_checkpoint||"").trim() && !knownCheckpointSet.has(trainB.base_checkpoint)){
      issues.push(`Không tìm thấy checkpoint nền: ${trainB.base_checkpoint}`);
      const cordCkpt = knownCheckpointFiles.find(x=>String(x).endsWith("gcn_cord.pt"));
      if(cordCkpt){
        notes.push(`Bạn có thể dùng checkpoint có sẵn: ${cordCkpt}`);
      }
    }
    if(String(trainB.checkpoint||"").trim() && !knownCheckpointSet.has(trainB.checkpoint)){
      notes.push("Checkpoint đầu ra chưa tồn tại là bình thường nếu đây là lần train mới. File sẽ được tạo sau khi train xong.");
    }
    return {issues, notes};
  },[trainB, knownDataSet, knownCheckpointSet, knownCheckpointFiles]);
  const evalMetrics = evalReport?.metrics || null;
  const evalPerClassRows = useMemo(()=>{
    const per = evalReport?.per_class || {};
    return Object.entries(per).map(([label, stat])=>({label, ...(stat||{})})).sort((a,b)=>(b.f1||0)-(a.f1||0));
  },[evalReport]);
  const inferInvoiceRows = useMemo(()=>{
    const inv = inferReport?.invoice || {};
    return Object.entries(inv);
  },[inferReport]);
  const inferRuleRows = useMemo(()=>{
    const inv = inferReport?.rule_baseline?.invoice || {};
    return Object.entries(inv);
  },[inferReport]);
  const inferNodes = useMemo(()=>{
    const baseNodes = inferReport?.node_features || [];
    return baseNodes.filter((node)=>{
      if(inferNodeFilter==="changed") return !!node.changed_by_gcn;
      if(inferNodeFilter==="money") return Number(node.feature_vector?.[2] || 0) > 0;
      return true;
    });
  },[inferReport, inferNodeFilter]);
  const inferActiveNodeData = useMemo(
    ()=> (inferReport?.node_features || []).find(n=>n.node_id===inferActiveNode) || null,
    [inferReport, inferActiveNode]
  );
  const inferChangedCount = useMemo(
    ()=> (inferReport?.node_features || []).filter(n=>n.changed_by_gcn).length,
    [inferReport]
  );
  const singleActiveNodeData = useMemo(
    ()=> (singleInspect?.nodes || []).find(n=>n.node_index===singleActiveNode) || null,
    [singleInspect, singleActiveNode]
  );
  const singleLabelSummary = useMemo(
    ()=> Object.entries(singleInspect?.label_summary || {}).sort((a,b)=>b[1]-a[1]),
    [singleInspect]
  );
  const singleExtractedRows = useMemo(
    ()=> Object.entries(singleInspect?.extracted_fields || {}).filter(([,vals])=>Array.isArray(vals) && vals.length>0),
    [singleInspect]
  );
  const singleSuspiciousCount = useMemo(
    ()=> (singleInspect?.nodes || []).filter(node=>reviewIssueForNode(node)).length,
    [singleInspect]
  );

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
          <datalist id="known-data-files">
            {knownDataFiles.map((f)=><option key={f} value={f} />)}
          </datalist>
          <datalist id="known-checkpoint-files">
            {knownCheckpointFiles.map((f)=><option key={f} value={f} />)}
          </datalist>
          {page==="single" ? <>
            <h2 className="title">Kiểm thử 1 ảnh trước khi train B</h2>
            <div className="subtitle">Trang này dành riêng cho việc kiểm tra chất lượng từng bước trên đúng <b>1 ảnh</b>: OCR có bắt box tốt không, text OCR ra sao, AI gán nhãn thế nào, và dữ liệu graph cuối cùng sẽ đi vào train B trông ra sao. Mình dùng cùng logic thật của pipeline để bạn kiểm tra trước khi chạy batch lớn.</div>

            <div className="step">
              <div className="step-head"><h3>Cấu hình ảnh kiểm thử</h3><span className={`badge ${singleInspectLoading?"running":"idle"}`}>{singleInspectLoading?"RUNNING":"READY"}</span></div>
              <div className="step-body">
                <div className="field-grid">
                  <TrainField label="Ảnh đã có trong project" hint="Bạn có thể chọn ảnh đã upload ở bước chuẩn bị dữ liệu hoặc ảnh trong thư mục thử nghiệm riêng.">
                    <FileSelect value={singleCheck.image} onChange={(v)=>setSingleCheck({...singleCheck,image:v})} options={allRawImages} placeholder="-- chọn 1 ảnh để kiểm thử --" />
                  </TrainField>
                  <TrainField label="Ngôn ngữ OCR" hint="Nên để `vi` cho hóa đơn tiếng Việt có dấu.">
                    <input className="input" value={singleCheck.lang} onChange={e=>setSingleCheck({...singleCheck,lang:e.target.value})} placeholder="vi" />
                  </TrainField>
                  <TrainField label="Engine OCR" hint="`paddle` là mặc định. `vietocr` sẽ dùng Paddle để detect box, sau đó VietOCR để đọc text trong từng box.">
                    <select className="input" value={singleCheck.ocr_engine} onChange={e=>setSingleCheck({...singleCheck,ocr_engine:e.target.value})}>
                      <option value="paddle">PaddleOCR</option>
                      <option value="vietocr">Paddle detect + VietOCR</option>
                    </select>
                  </TrainField>
                  <TrainField label="DB box thresh" hint="Tăng lên thì box chặt hơn, nhưng có thể bỏ sót text nhỏ. Với hóa đơn thường thử 0.58 đến 0.68.">
                    <input className="input" type="number" step="0.01" value={singleCheck.det_db_box_thresh} onChange={e=>setSingleCheck({...singleCheck,det_db_box_thresh:e.target.value})} />
                  </TrainField>
                  <TrainField label="Unclip ratio" hint="Giảm xuống thì box ôm sát chữ hơn. Nếu box bị phình rộng, hãy thử 1.10 đến 1.25.">
                    <input className="input" type="number" step="0.01" value={singleCheck.det_db_unclip_ratio} onChange={e=>setSingleCheck({...singleCheck,det_db_unclip_ratio:e.target.value})} />
                  </TrainField>
                  <TrainField label="Drop score" hint="Bỏ các box OCR quá yếu. Tăng lên nếu đang bắt nhiều box rác.">
                    <input className="input" type="number" step="0.01" value={singleCheck.drop_score} onChange={e=>setSingleCheck({...singleCheck,drop_score:e.target.value})} />
                  </TrainField>
                  <TrainField label="Det limit side" hint="Tăng kích thước detect để dễ bắt text nhỏ, đổi lại sẽ chậm hơn.">
                    <input className="input" type="number" step="64" value={singleCheck.det_limit_side_len} onChange={e=>setSingleCheck({...singleCheck,det_limit_side_len:e.target.value})} />
                  </TrainField>
                  <TrainField label="Upscale factor" hint="Phóng ảnh lên trước OCR. Hữu ích khi ảnh nhỏ hoặc chữ mảnh.">
                    <input className="input" type="number" step="0.1" value={singleCheck.upscale_factor} onChange={e=>setSingleCheck({...singleCheck,upscale_factor:e.target.value})} />
                  </TrainField>
                  <TrainField label="Use dilation" hint="Bật nếu chữ đứt nét hoặc box đang bị vỡ đoạn. Bình thường nên để 0.">
                    <select className="input" value={singleCheck.use_dilation} onChange={e=>setSingleCheck({...singleCheck,use_dilation:Number(e.target.value)})}>
                      <option value={0}>0 - Tắt</option>
                      <option value={1}>1 - Bật</option>
                    </select>
                  </TrainField>
                  <TrainField label="Model AI gợi ý nhãn" hint="Chỉ dùng khi bạn bấm chạy OCR + AI. Endpoint sẽ gọi model này để gán nhãn dựa trên OCR text và context layout.">
                    <input className="input" value={singleCheck.llm_model} onChange={e=>setSingleCheck({...singleCheck,llm_model:e.target.value})} placeholder="gpt-4.1-mini" />
                  </TrainField>
                  <TrainField label="Thư mục lưu output preview" hint="Trang này sẽ tự lưu ra CSV preview tương thích train B, ảnh box OCR và file graph JSON vào thư mục này.">
                    <input className="input" value={singleCheck.output_dir} onChange={e=>setSingleCheck({...singleCheck,output_dir:e.target.value})} placeholder="data/single_image_check" />
                  </TrainField>
                </div>
                <div className="chipline" style={{marginTop:10}}>
                  <button className="tab-btn" onClick={()=>applySingleOcrPreset("tight")}>Preset box chặt</button>
                  <button className="tab-btn" onClick={()=>applySingleOcrPreset("balanced")}>Preset cân bằng</button>
                  <button className="tab-btn" onClick={()=>applySingleOcrPreset("hard")}>Preset ảnh khó</button>
                </div>
                <div className="tiny" style={{marginTop:6}}>
                  Mẹo nhanh: nếu box đang phình rộng, hãy giảm <b>Unclip ratio</b> trước. Nếu mất chữ nhỏ, tăng <b>Det limit side</b> hoặc giảm nhẹ <b>DB box thresh</b>.
                </div>
                <div className="actions" style={{marginTop:12}}>
                  <input type="file" accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp" onChange={e=>setSingleCheckUploadFile(e.target.files?.[0] || null)} />
                  <button className="btn" onClick={uploadSingleCheckImage}>Upload ảnh test</button>
                  <button className="btn dark" onClick={()=>runSinglePreview(false)}>Chỉ chạy OCR preview</button>
                  <button className="btn primary" onClick={()=>runSinglePreview(true)}>Chạy OCR + AI gán nhãn</button>
                </div>
                <div className="tiny" style={{marginTop:6}}>
                  {singleCheckUploadFile ? `Đã chọn file local: ${singleCheckUploadFile.name}` : "Bạn có thể upload một ảnh mới hoặc chọn ảnh đã có trong project."}
                </div>
              </div>
            </div>

            <div className="review-shell">
              <div className="review-pane">
                <div className="review-pane-head">
                  <h4>Tóm tắt nhanh</h4>
                </div>
                <div className="review-pane-body">
                  {!singleInspect ? (
                    <div className="empty-state">Chưa có kết quả. Hãy chọn một ảnh rồi chạy OCR preview hoặc OCR + AI.</div>
                  ) : (
                    <>
                      <div className="review-kpis" style={{gridTemplateColumns:"1fr"}}>
                        <div className="review-kpi"><div className="k">Doc ID</div><div className="v" style={{fontSize:18}}>{singleInspect.doc_id}</div></div>
                        <div className="review-kpi"><div className="k">Số node OCR</div><div className="v">{singleInspect.graph?.num_nodes || 0}</div></div>
                        <div className="review-kpi"><div className="k">Số cạnh graph</div><div className="v">{singleInspect.graph?.num_edges || 0}</div></div>
                        <div className="review-kpi"><div className="k">Node nghi ngờ cần rà</div><div className="v">{singleSuspiciousCount}</div></div>
                      </div>
                      <div className="node-detail" style={{marginTop:10}}>
                        <div style={{fontWeight:800, marginBottom:8}}>File output tạo ra từ trang này</div>
                        <div className="node-detail-grid">
                          <div>CSV cho train B</div><div>{singleInspect.artifacts?.train_b_csv_path || "-"}</div>
                          <div>Graph preview JSON</div><div>{singleInspect.artifacts?.graph_json_path || "-"}</div>
                          <div>Ảnh OCR box</div><div>{singleInspect.ocr_boxes_image || "-"}</div>
                          <div>Thư mục output</div><div>{singleInspect.artifacts?.output_dir || "-"}</div>
                        </div>
                      </div>
                      <div className="node-detail" style={{marginTop:10}}>
                        <div style={{fontWeight:800, marginBottom:8}}>Cấu hình OCR vừa chạy</div>
                        <div className="node-detail-grid">
                          <div>Engine</div><div>{singleCheck.ocr_engine}</div>
                          <div>DB box thresh</div><div>{singleInspect.ocr_config?.det_db_box_thresh ?? "-"}</div>
                          <div>Unclip ratio</div><div>{singleInspect.ocr_config?.det_db_unclip_ratio ?? "-"}</div>
                          <div>Drop score</div><div>{singleInspect.ocr_config?.drop_score ?? "-"}</div>
                          <div>Det limit side</div><div>{singleInspect.ocr_config?.det_limit_side_len ?? "-"}</div>
                          <div>Upscale factor</div><div>{singleInspect.ocr_config?.upscale_factor ?? "-"}</div>
                        </div>
                      </div>
                      <div className="step" style={{marginTop:10}}>
                        <div className="step-head"><h3>Trường đã trích xuất / gán nhãn</h3><span className="badge idle">FIELDS</span></div>
                        <div className="step-body">
                          {singleExtractedRows.length===0 ? (
                            <div className="empty-state">Chưa có trường nào được gán nhãn. Hãy chạy `OCR + AI gán nhãn` để xem dữ liệu train B thử nghiệm.</div>
                          ) : (
                            <div className="chipline">
                              {singleLabelSummary.map(([label,count])=>(
                                <span key={label} className="mini-chip">{labelText(label)}: {count}</span>
                              ))}
                            </div>
                          )}
                          <div className="table-wrap" style={{maxHeight:320, marginTop:10}}>
                            <table className="tbl">
                              <thead>
                                <tr>
                                  <th style={{width:180}}>Nhãn</th>
                                  <th>Ví dụ text OCR được gán</th>
                                </tr>
                              </thead>
                              <tbody>
                                {singleExtractedRows.map(([label, values])=>(
                                  <tr key={label}>
                                    <td>{labelText(label)}</td>
                                    <td>{values.slice(0,6).join(" | ")}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      </div>
                      <div className="step" style={{marginTop:10}}>
                        <div className="step-head"><h3>Thống kê AI / heuristic</h3><span className="badge idle">AI</span></div>
                        <div className="step-body">
                          <div className="kv">
                            <div>Chiến lược</div><div>{singleInspect.ai_stats?.strategy_used || "-"}</div>
                            <div>Heuristic ăn chắc</div><div>{singleInspect.ai_stats?.heuristic_hits ?? 0}</div>
                            <div>Node cần AI xử lý</div><div>{singleInspect.ai_stats?.llm_targets ?? 0}</div>
                            <div>Số batch LLM</div><div>{singleInspect.ai_stats?.llm_batches ?? 0}</div>
                          </div>
                          {(singleInspect.ai_stats?.llm_errors || []).length > 0 ? (
                            <div className="alert bad" style={{marginTop:10}}>
                              <div className="alert-title">LLM có lỗi ở một vài batch</div>
                              <div className="tiny">{(singleInspect.ai_stats?.llm_errors || []).join(" | ")}</div>
                            </div>
                          ) : null}
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>

              <div className="review-pane">
                <div className="review-pane-head">
                  <h4>Ảnh OCR và graph preview</h4>
                  <div className="chipline">
                    <button className={`tab-btn ${singleInspectTab==="ocr"?"active":""}`} onClick={()=>setSingleInspectTab("ocr")}>Xem box OCR</button>
                    <button className={`tab-btn ${singleInspectTab==="ai"?"active":""}`} onClick={()=>setSingleInspectTab("ai")}>Xem nhãn AI</button>
                    <button className={`tab-btn ${singleInspectTab==="graph"?"active":""}`} onClick={()=>setSingleInspectTab("graph")}>Xem graph train B</button>
                  </div>
                </div>
                <div className="review-pane-body">
                  {singleInspectLoading ? (
                    <div className="empty-state">Đang chạy OCR / AI trên ảnh này...</div>
                  ) : !singleInspect ? (
                    <div className="empty-state">Chưa có dữ liệu preview 1 ảnh.</div>
                  ) : singleInspectTab==="graph" ? (
                    <>
                      <div className="review-kpis">
                        <div className="review-kpi"><div className="k">Số node</div><div className="v">{singleInspect.graph?.num_nodes || 0}</div></div>
                        <div className="review-kpi"><div className="k">Số cạnh</div><div className="v">{singleInspect.graph?.num_edges || 0}</div></div>
                        <div className="review-kpi"><div className="k">Số feature</div><div className="v">{(singleInspect.feature_names || []).length}</div></div>
                      </div>
                      <div className="node-detail" style={{marginTop:10}}>
                        <div style={{fontWeight:800, marginBottom:6}}>Feature đầu vào train B</div>
                        <div className="tiny">{(singleInspect.feature_names || []).join(", ") || "Không có feature"}</div>
                      </div>
                      <div className="term" style={{marginTop:10}}><h4>EDGE_INDEX</h4><pre>{JSON.stringify(singleInspect.graph?.edge_index || [[],[]], null, 2)}</pre></div>
                    </>
                  ) : (
                    <>
                      <div className="compare-grid" style={{marginBottom:12}}>
                        <div className="step" style={{margin:0}}>
                          <div className="step-head"><h3>Ảnh gốc</h3><span className="badge idle">RAW</span></div>
                          <div className="step-body">
                            {singleInspect.original_image_path ? (
                              <img className="inspect-preview" src={`/api/files/image?path=${encodeURIComponent(singleInspect.original_image_path)}`} alt="Ảnh gốc" />
                            ) : <div className="empty-state">Không có ảnh gốc.</div>}
                          </div>
                        </div>
                        <div className="step" style={{margin:0}}>
                          <div className="step-head"><h3>Ảnh sau normalize + mask</h3><span className="badge idle">NORM</span></div>
                          <div className="step-body">
                            {singleInspect.normalized_image_path ? (
                              <img className="inspect-preview" src={`/api/files/image?path=${encodeURIComponent(singleInspect.normalized_image_path)}`} alt="Ảnh normalize" />
                            ) : <div className="empty-state">Không có ảnh normalized.</div>}
                          </div>
                        </div>
                        <div className="step" style={{margin:0}}>
                          <div className="step-head"><h3>Ảnh debug OCR</h3><span className="badge idle">DEBUG</span></div>
                          <div className="step-body">
                            {singleInspect.ocr_boxes_image ? (
                              <img className="inspect-preview" src={`/api/files/image?path=${encodeURIComponent(singleInspect.ocr_boxes_image)}`} alt="Ảnh OCR debug" />
                            ) : <div className="empty-state">Chưa lưu ảnh debug OCR.</div>}
                          </div>
                        </div>
                      </div>
                      <div className="review-image">
                        {singleInspect.preview_path ? (
                          <>
                                    <div className="review-canvas">
                                      <img
                                        src={`/api/files/image?path=${encodeURIComponent(singleInspect.preview_path)}`}
                                        alt={singleInspect.doc_id}
                                        onLoad={(e)=>setSingleNatural({w:e.target.naturalWidth||1,h:e.target.naturalHeight||1})}
                                      />
                                      <div className="review-overlay">
                                        <svg viewBox={`0 0 ${singleNatural.w} ${singleNatural.h}`} preserveAspectRatio="none">
                                          {(singleInspect.nodes || []).map((node)=>{
                                            const color = labelColor(String((singleInspectTab==="ocr" ? node.heuristic_label : (node.picked_label || node.label || node.suggested_label)) || "OTHER").trim() || "OTHER");
                                            const isActive = singleActiveNode===node.node_index;
                                            return nodeHasQuad(node) ? (
                                              <polygon
                                                key={`single-poly-${node.node_index}`}
                                                className="ocr-poly"
                                                points={nodeQuadPoints(node)}
                                                fill={isActive ? "rgba(239,68,68,.12)" : color.fill}
                                                stroke={isActive ? "#ef4444" : color.border}
                                                strokeWidth="2"
                                                onClick={()=>setSingleActiveNode(node.node_index)}
                                              />
                                            ) : (
                                              <rect
                                                key={`single-rect-${node.node_index}`}
                                                className="ocr-poly"
                                                x={Number(node.bbox?.[0]||0)}
                                                y={Number(node.bbox?.[1]||0)}
                                                width={Math.max(1, Number((node.bbox?.[2]||0) - (node.bbox?.[0]||0)))}
                                                height={Math.max(1, Number((node.bbox?.[3]||0) - (node.bbox?.[1]||0)))}
                                                fill={isActive ? "rgba(239,68,68,.12)" : color.fill}
                                                stroke={isActive ? "#ef4444" : color.border}
                                                strokeWidth="2"
                                                onClick={()=>setSingleActiveNode(node.node_index)}
                                              />
                                            );
                                          })}
                                        </svg>
                                      </div>
                                    </div>
                                  </>
                                ) : <div className="empty-state">Không có ảnh preview.</div>}
                      </div>
                      <div className="tiny" style={{marginTop:8}}>
                        {singleInspectTab==="ocr"
                          ? "Màu box đang đi theo heuristic sơ bộ để bạn kiểm tra OCR và context ban đầu."
                          : "Màu box đang đi theo nhãn cuối cùng sau khi chạy AI gợi ý nhãn."}
                      </div>
                    </>
                  )}
                </div>
              </div>

              <div className="review-pane">
                <div className="review-pane-head">
                  <h4>Node OCR, nhãn và context</h4>
                </div>
                <div className="review-pane-body">
                  {!singleInspect ? (
                    <div className="empty-state">Chưa có node OCR nào để xem.</div>
                  ) : (
                    <>
                      {singleActiveNodeData ? (
                        <div className="node-detail">
                          <div style={{fontWeight:800, marginBottom:8}}>Node đang chọn: #{singleActiveNodeData.node_index}</div>
                          <div className="node-detail-grid">
                            <div>OCR text</div><div>{singleActiveNodeData.text}</div>
                            <div>Dòng CSV preview</div><div>{singleActiveNodeData.row_number}</div>
                            <div>Score OCR</div><div>{Number(singleActiveNodeData.score||0).toFixed(3)}</div>
                            <div>Region dòng</div><div>{singleActiveNodeData.line_region || "-"}</div>
                            <div>Gợi ý heuristic</div><div>{labelText(singleActiveNodeData.heuristic_label)}</div>
                            <div>Nhãn AI / cuối</div><div>{labelText(singleActiveNodeData.picked_label || singleActiveNodeData.label || singleActiveNodeData.suggested_label)}</div>
                            <div>Cột gợi ý</div><div>{singleActiveNodeData.column_hint || "-"}</div>
                            <div>Lý do rà nhanh</div><div>{reviewIssueForNode(singleActiveNodeData) || "Không thấy dấu hiệu bất thường rõ ràng"}</div>
                          </div>
                        </div>
                      ) : null}
                      <div className="table-wrap" style={{maxHeight:820}}>
                        <table className="tbl">
                          <thead>
                            <tr>
                              <th style={{width:56}}>Node</th>
                              <th style={{width:70}}>Dòng</th>
                              <th>Nội dung OCR</th>
                              <th style={{width:160}}>Heuristic</th>
                              <th style={{width:170}}>AI / cuối</th>
                              <th style={{width:80}}>Score</th>
                            </tr>
                          </thead>
                          <tbody>
                            {(singleInspect.nodes || []).map((node)=>(
                              <tr key={node.node_index} className={`node-row ${singleActiveNode===node.node_index?"active":""}`} onClick={()=>setSingleActiveNode(node.node_index)}>
                                <td>{node.node_index}</td>
                                <td>{node.row_number}</td>
                                <td>{node.text}</td>
                                <td>{labelText(node.heuristic_label)}</td>
                                <td>{labelText(node.picked_label || node.label || node.suggested_label)}</td>
                                <td>{Number(node.score || 0).toFixed(3)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>

            <div className="step">
              <div className="step-head"><h3>Output API mới nhất</h3><span className="badge idle">READY</span></div>
              <div className="step-body">
                <div className="term"><h4>APPLICATION/JSON</h4><pre>{out||"Chưa có phản hồi API"}</pre></div>
              </div>
            </div>
          </> : page==="train" ? <>
            <h2 className="title">Huấn luyện GCN</h2>
            <div className="subtitle">Màn này tập trung cho Stage B vì đây là bước bạn sẽ dùng nhiều nhất để huấn luyện mô hình hóa đơn. Mình để sẵn giải thích từng tham số để ngay cả khi chưa rành machine learning bạn vẫn có thể bắt đầu an toàn.</div>

            <div className="cards">
              <div className="card"><div className="k">Job train đã chạy</div><div className="v">{[trainAJob,trainBJob,trainFullJob].filter(Boolean).length}</div></div>
              <div className="card"><div className="k">Đang chạy</div><div className="v">{trainRunning?1:0}</div></div>
              <div className="card"><div className="k">Thành công</div><div className="v">{trainDoneCount}</div></div>
              <div className="card"><div className="k">Thất bại</div><div className="v">{trainFailCount}</div></div>
            </div>

            <div className="hero-train">
              <div className="train-main">
                <div className="chips">
                  <span className="chip">Bước khuyên dùng: Train Stage B</span>
                  <span className="chip">Checkpoint nền: Stage A</span>
                  <span className="chip">Phù hợp cho người mới</span>
                </div>
                <h3 style={{margin:"14px 0 6px",fontSize:26}}>Huấn luyện Stage B cho hóa đơn tiếng Việt</h3>
                <p className="p">Nếu bạn đã chuẩn bị xong `data/stage_b_vi_dataset.json` ở bước dữ liệu, đây là nơi chính để train model nhận diện trường trên hóa đơn. Bạn chỉ cần điền đúng dataset, chọn checkpoint nền Stage A và giữ preset mặc định nếu chưa chắc nên chỉnh gì.</p>
                <div className="preset-grid" style={{marginTop:12}}>
                  {Object.entries(STAGE_B_PRESETS).map(([key,p])=>(
                    <button key={key} className={`preset ${trainB.epochs===p.epochs && Number(trainB.lr)===p.lr && trainB.early_stop_patience===p.early_stop_patience?"active":""}`} onClick={()=>setTrainB({...trainB,epochs:p.epochs,lr:p.lr,early_stop_patience:p.early_stop_patience})}>
                      <div style={{fontWeight:800, marginBottom:4}}>{p.label}</div>
                      <div className="tiny">Epoch: {p.epochs} | LR: {p.lr} | Dừng sớm: {p.early_stop_patience}</div>
                      <div className="field-help" style={{marginTop:6}}>{p.desc}</div>
                    </button>
                  ))}
                </div>
                <div className={`alert ${stageBProfile.tone}`} style={{marginTop:12}}>
                  <div className="alert-title">Mức train hiện tại: {stageBProfile.label}</div>
                  <div className="field-help">{stageBProfile.desc}</div>
                </div>
              </div>
              <div className="train-side">
                <div className="step">
                  <div className="step-head"><h3>Checklist trước khi train</h3><span className="badge idle">READY</span></div>
                  <div className="step-body">
                    <p className="p">1. Đã có `data/stage_b_vi_dataset.json` từ bước chuẩn bị dữ liệu.</p>
                    <p className="p" style={{marginTop:8}}>2. Đã có checkpoint nền Stage A, ví dụ `outputs/checkpoints/gcn_stage_a.pt`.</p>
                    <p className="p" style={{marginTop:8}}>3. Nếu chưa có validation dataset, cứ để trống. Model vẫn train được.</p>
                  </div>
                </div>
                <div className="step">
                  <div className="step-head"><h3>Khi nào nên chỉnh tham số?</h3><span className="badge idle">TIP</span></div>
                  <div className="step-body">
                    <p className="p">Nếu mới dùng lần đầu, hãy giữ preset `Cân bằng`.</p>
                    <p className="p" style={{marginTop:8}}>Chỉ tăng `epoch` khi model còn học được. Chỉ giảm `lr` khi train bị dao động hoặc kết quả không ổn định.</p>
                  </div>
                </div>
                <div className="step">
                  <div className="step-head"><h3>Chọn file như thế nào?</h3><span className="badge idle">INFO</span></div>
                  <div className="step-body">
                    <p className="p">Các ô bên dưới đang nhận <b>đường dẫn file trong project</b>, không phải upload file mới từ máy tính.</p>
                    <p className="p" style={{marginTop:8}}>Mình đã thêm danh sách gợi ý file có sẵn. Bạn chỉ cần bấm vào ô và chọn từ danh sách xổ xuống.</p>
                  </div>
                </div>
                <div className="step">
                  <div className="step-head"><h3>Tách train / val / test</h3><span className="badge idle">NEW</span></div>
                  <div className="step-body">
                    <p className="p">Nếu bạn mới chỉ có `data/stage_b_vi_dataset.json` tổng, hãy tách nó thành 3 tập trước khi train để kết quả đáng tin hơn.</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="split">
              <div className="stack">
                <div className="step">
                  <div className="step-head"><h3>Tách dataset Stage B thành train / validation / test</h3><span className="badge idle">RECOMMENDED</span></div>
                  <div className="step-body">
                    <div className="field-grid">
                      <TrainField label="Dataset tổng đầu vào" hint="Đây là file tổng trước khi tách. Thường là stage_b_vi_dataset.json.">
                        <input list="known-data-files" className="input" value={splitStageB.input_json} onChange={e=>setSplitStageB({...splitStageB,input_json:e.target.value})} placeholder="data/stage_b_vi_dataset.json"/>
                      </TrainField>
                      <TrainField label="Random seed" hint="Seed giúp việc chia tập có thể lặp lại giống nhau giữa các lần split.">
                        <input className="input" type="number" value={splitStageB.seed} onChange={e=>setSplitStageB({...splitStageB,seed:Number(e.target.value)})} placeholder="42"/>
                      </TrainField>
                      <TrainField label="Tập train output" hint="File train sau khi tách. Đây sẽ là file chính để đưa vào train Stage B.">
                        <input list="known-data-files" className="input" value={splitStageB.output_train_json} onChange={e=>setSplitStageB({...splitStageB,output_train_json:e.target.value})} placeholder="data/stage_b_train.json"/>
                      </TrainField>
                      <TrainField label="Tập validation output" hint="File validation dùng để theo dõi chất lượng trong lúc train.">
                        <input list="known-data-files" className="input" value={splitStageB.output_val_json} onChange={e=>setSplitStageB({...splitStageB,output_val_json:e.target.value})} placeholder="data/stage_b_val.json"/>
                      </TrainField>
                      <TrainField label="Tập test output" hint="File test dùng để đánh giá cuối cùng sau khi train xong.">
                        <input list="known-data-files" className="input" value={splitStageB.output_test_json} onChange={e=>setSplitStageB({...splitStageB,output_test_json:e.target.value})} placeholder="data/stage_b_test.json"/>
                      </TrainField>
                    </div>
                    <div className="field-grid" style={{marginTop:10}}>
                      <TrainField label="Tỉ lệ train" hint="Phần dữ liệu để model học. Thường lớn nhất, ví dụ 0.7 hoặc 0.8.">
                        <input className="input" type="number" step="0.01" value={splitStageB.train_ratio} onChange={e=>setSplitStageB({...splitStageB,train_ratio:Number(e.target.value)})} placeholder="0.7"/>
                      </TrainField>
                      <TrainField label="Tỉ lệ validation" hint="Phần dữ liệu để kiểm tra trong lúc train.">
                        <input className="input" type="number" step="0.01" value={splitStageB.val_ratio} onChange={e=>setSplitStageB({...splitStageB,val_ratio:Number(e.target.value)})} placeholder="0.15"/>
                      </TrainField>
                      <TrainField label="Tỉ lệ test" hint="Phần dữ liệu để đánh giá cuối cùng sau khi train xong.">
                        <input className="input" type="number" step="0.01" value={splitStageB.test_ratio} onChange={e=>setSplitStageB({...splitStageB,test_ratio:Number(e.target.value)})} placeholder="0.15"/>
                      </TrainField>
                    </div>
                    <div className="actions" style={{marginTop:12}}>
                      <div className="tiny">POST /api/pipeline/split-gcn-dataset</div>
                      <button className="btn dark" onClick={runSplitStageB}>Tách dataset Stage B</button>
                      {splitResult?.outputs ? <button className="btn" onClick={()=>applySplitOutputsToForms(splitResult.outputs)}>Dùng ngay cho train</button> : null}
                    </div>
                    <div className="field-help" style={{marginTop:8}}>Sau khi tách xong, màn hình sẽ tự đổ `train` vào ô Dataset train, `validation` vào ô Dataset validation và `test` vào ô eval của full flow.</div>
                    {splitResult?.outputs ? (
                      <div className="cards" style={{marginTop:10}}>
                        <div className="card"><div className="k">Train graphs</div><div className="v">{splitResult.outputs.train?.graphs || 0}</div><div className="tiny">{splitResult.outputs.train?.path}</div></div>
                        <div className="card"><div className="k">Validation graphs</div><div className="v">{splitResult.outputs.validation?.graphs || 0}</div><div className="tiny">{splitResult.outputs.validation?.path}</div></div>
                        <div className="card"><div className="k">Test graphs</div><div className="v">{splitResult.outputs.test?.graphs || 0}</div><div className="tiny">{splitResult.outputs.test?.path}</div></div>
                      </div>
                    ) : null}
                  </div>
                </div>
                <div className="step">
                  <div className="step-head"><h3>Thiết lập train Stage B</h3><span className={`badge ${statusClass(trainBJob?.status)}`}>{(trainBJob?.status||"idle").toUpperCase()}</span></div>
                  <div className="step-body">
                    {stageBValidation.issues.length > 0 && (
                      <div className="alert bad" style={{marginBottom:10}}>
                        <div className="alert-title">Chưa nên bấm train ngay</div>
                        {stageBValidation.issues.map((msg, idx)=><div key={idx} className="field-help">- {msg}</div>)}
                      </div>
                    )}
                    {stageBPathChecks.issues.length > 0 && (
                      <div className="alert bad" style={{marginBottom:10}}>
                        <div className="alert-title">Đường dẫn file đang có vấn đề</div>
                        {stageBPathChecks.issues.map((msg, idx)=><div key={idx} className="field-help">- {msg}</div>)}
                        {stageBPathChecks.notes.map((msg, idx)=><div key={`note-${idx}`} className="field-help">- {msg}</div>)}
                      </div>
                    )}
                    {stageBValidation.issues.length === 0 && stageBValidation.notes.length > 0 && (
                      <div className="alert warn" style={{marginBottom:10}}>
                        <div className="alert-title">Cảnh báo nhẹ trước khi train</div>
                        {stageBValidation.notes.map((msg, idx)=><div key={idx} className="field-help">- {msg}</div>)}
                      </div>
                    )}
                    {stageBValidation.issues.length === 0 && stageBPathChecks.issues.length === 0 && stageBValidation.notes.length === 0 && (
                      <div className="alert good" style={{marginBottom:10}}>
                        <div className="alert-title">Cấu hình ổn để bắt đầu</div>
                        <div className="field-help">Bạn có thể bấm train ngay. Đây là cấu hình sạch, không có cảnh báo quan trọng.</div>
                      </div>
                    )}
                    {stageBPathChecks.issues.length === 0 && stageBPathChecks.notes.length > 0 && (
                      <div className="alert warn" style={{marginBottom:10}}>
                        <div className="alert-title">Gợi ý về file checkpoint</div>
                        {stageBPathChecks.notes.map((msg, idx)=><div key={idx} className="field-help">- {msg}</div>)}
                      </div>
                    )}
                    <div className="field-grid">
                      <TrainField label="Dataset train" hint={TRAIN_HELP.dataset_json} help="Đây là file dữ liệu chính để học. Thường là `data/stage_b_vi_dataset.json`.">
                        <input list="known-data-files" className="input" value={trainB.dataset_json} onChange={e=>setTrainB({...trainB,dataset_json:e.target.value})} placeholder="data/stage_b_vi_dataset.json"/>
                      </TrainField>
                      <TrainField label="Dataset validation (tùy chọn)" hint={TRAIN_HELP.val_dataset_json} help="Có file này thì bạn sẽ biết model có đang tốt lên thật hay không. Nếu chưa có thì để trống.">
                        <input list="known-data-files" className="input" value={trainB.val_dataset_json} onChange={e=>setTrainB({...trainB,val_dataset_json:e.target.value})} placeholder="Ví dụ: data/stage_b_val_dataset.json"/>
                      </TrainField>
                      <TrainField label="Checkpoint nền Stage A" hint={TRAIN_HELP.base_checkpoint} help="Stage B không học từ số 0 mà học tiếp từ checkpoint này. Đây là tham số rất quan trọng.">
                        <FileSelect value={trainB.base_checkpoint} onChange={(v)=>setTrainB({...trainB,base_checkpoint:v})} options={knownCheckpointFiles} placeholder="-- chọn checkpoint nền Stage A --" />
                      </TrainField>
                      <TrainField label="Checkpoint đầu ra Stage B" hint={TRAIN_HELP.checkpoint} help="Model sau khi train xong sẽ được lưu ở đây để bạn test, infer hoặc train tiếp.">
                        <input list="known-checkpoint-files" className="input" value={trainB.checkpoint} onChange={e=>setTrainB({...trainB,checkpoint:e.target.value})} placeholder="outputs/checkpoints/gcn_stage_b.pt"/>
                      </TrainField>
                      <TrainField label="Epoch" hint={TRAIN_HELP.epochs} help="Gợi ý: 15-20 cho chạy nhanh, 20-35 khi bạn muốn model học kỹ hơn.">
                        <input className="input" type="number" value={trainB.epochs} onChange={e=>setTrainB({...trainB,epochs:Number(e.target.value)})} placeholder="20"/>
                      </TrainField>
                      <TrainField label="Learning rate" hint={TRAIN_HELP.lr} help="Gợi ý an toàn: 0.0003 đến 0.0005. Nếu chưa biết, cứ dùng 0.0005.">
                        <input className="input" type="number" step="0.0001" value={trainB.lr} onChange={e=>setTrainB({...trainB,lr:Number(e.target.value)})} placeholder="0.0005"/>
                      </TrainField>
                      <TrainField label="Early stop patience" hint={TRAIN_HELP.early_stop_patience} help="Nếu có validation dataset thì nên để 3-6. Nếu chưa có validation thì có thể để 0.">
                        <input className="input" type="number" value={trainB.early_stop_patience} onChange={e=>setTrainB({...trainB,early_stop_patience:Number(e.target.value)})} placeholder="4"/>
                      </TrainField>
                    </div>
                    <div className="actions" style={{marginTop:12}}>
                      <div className="tiny">POST /api/pipeline/train-gcn-stage-b</div>
                      <button className="btn primary" onClick={runTrainB} title={[...stageBValidation.issues, ...stageBPathChecks.issues].length ? [...stageBValidation.issues, ...stageBPathChecks.issues].join(" | ") : "Cấu hình hợp lệ, có thể bắt đầu train"}>Bắt đầu train Stage B</button>
                    </div>
                  </div>
                </div>

                <details className="details">
                  <summary>Thiết lập nâng cao: Stage A và full flow</summary>
                  <div className="step-body">
                    <div className="step" style={{marginBottom:10}}>
                      <div className="step-head"><h3>Đánh giá checkpoint Stage B trên tập test</h3><span className="badge idle">TEST</span></div>
                      <div className="step-body">
                        {stageBEvalValidation.issues.length > 0 ? (
                          <div className="alert bad" style={{marginBottom:10}}>
                            <div className="alert-title">Chưa thể chạy đánh giá</div>
                            {stageBEvalValidation.issues.map((msg, idx)=><div key={idx} className="field-help">- {msg}</div>)}
                          </div>
                        ) : null}
                        <div className="field-grid">
                          <TrainField label="Dataset test" hint="Tập test chỉ dùng để chấm điểm cuối cùng sau khi train xong.">
                            <input list="known-data-files" className="input" value={stageBEval.dataset_json} onChange={e=>setStageBEval({...stageBEval,dataset_json:e.target.value})} placeholder="data/stage_b_test.json"/>
                          </TrainField>
                          <TrainField label="Checkpoint Stage B" hint="Checkpoint bạn muốn đem đi đánh giá. Có thể là file .best.pt hoặc .pt cuối cùng.">
                            <FileSelect value={stageBEval.checkpoint} onChange={(v)=>setStageBEval({...stageBEval,checkpoint:v})} options={knownCheckpointFiles} placeholder="-- chọn checkpoint để đánh giá --" />
                          </TrainField>
                          <TrainField label="Output report JSON" hint="File báo cáo kết quả đánh giá sẽ được lưu ra đây.">
                            <input className="input" value={stageBEval.output_eval} onChange={e=>setStageBEval({...stageBEval,output_eval:e.target.value})} placeholder="outputs/gcn_stage_b_test_report.json"/>
                          </TrainField>
                        </div>
                        <div className="actions" style={{marginTop:10}}>
                          <div className="tiny">POST /api/pipeline/test-gcn</div>
                          <button className="btn dark" onClick={runStageBEval}>Đánh giá trên tập test</button>
                        </div>
                      </div>
                    </div>
                    <div className="step" style={{marginBottom:10}}>
                      <div className="step-head"><h3>Stage A: pretrain / train nền</h3><span className={`badge ${statusClass(trainAJob?.status)}`}>{(trainAJob?.status||"idle").toUpperCase()}</span></div>
                      <div className="step-body">
                        <div className="field-grid">
                          <TrainField label="Dataset train" hint={TRAIN_HELP.dataset_json}><input className="input" value={trainA.dataset_json} onChange={e=>setTrainA({...trainA,dataset_json:e.target.value})} /></TrainField>
                          <TrainField label="Validation (tùy chọn)" hint={TRAIN_HELP.val_dataset_json}><input className="input" value={trainA.val_dataset_json} onChange={e=>setTrainA({...trainA,val_dataset_json:e.target.value})} /></TrainField>
                          <TrainField label="Checkpoint output" hint={TRAIN_HELP.checkpoint}><input className="input" value={trainA.checkpoint} onChange={e=>setTrainA({...trainA,checkpoint:e.target.value})} /></TrainField>
                          <TrainField label="Init checkpoint (tùy chọn)" hint={TRAIN_HELP.init_checkpoint}><FileSelect value={trainA.init_checkpoint || ""} onChange={(v)=>setTrainA({...trainA,init_checkpoint:v})} options={knownCheckpointFiles} placeholder="-- chọn checkpoint khởi tạo --" /></TrainField>
                          <TrainField label="Epoch" hint={TRAIN_HELP.epochs}><input className="input" type="number" value={trainA.epochs} onChange={e=>setTrainA({...trainA,epochs:Number(e.target.value)})} /></TrainField>
                          <TrainField label="Learning rate" hint={TRAIN_HELP.lr}><input className="input" type="number" step="0.0001" value={trainA.lr} onChange={e=>setTrainA({...trainA,lr:Number(e.target.value)})} /></TrainField>
                          <TrainField label="Early stop patience" hint={TRAIN_HELP.early_stop_patience}><input className="input" type="number" value={trainA.early_stop_patience} onChange={e=>setTrainA({...trainA,early_stop_patience:Number(e.target.value)})} /></TrainField>
                        </div>
                        <div className="actions" style={{marginTop:10}}><div className="tiny">POST /api/pipeline/train-gcn-stage-a</div><button className="btn dark" onClick={runTrainA}>Train Stage A</button></div>
                      </div>
                    </div>

                    <div className="step">
                      <div className="step-head"><h3>Full Flow: A {"->"} B {"->"} Eval</h3><span className={`badge ${statusClass(trainFullJob?.status)}`}>{(trainFullJob?.status||"idle").toUpperCase()}</span></div>
                      <div className="step-body">
                        <div className="field-grid">
                          <TrainField label="Dataset Stage A" hint={TRAIN_HELP.dataset_json}><input className="input" value={trainFull.stage_a_json} onChange={e=>setTrainFull({...trainFull,stage_a_json:e.target.value})} /></TrainField>
                          <TrainField label="Dataset Stage B" hint={TRAIN_HELP.dataset_json}><input className="input" value={trainFull.stage_b_json} onChange={e=>setTrainFull({...trainFull,stage_b_json:e.target.value})} /></TrainField>
                          <TrainField label="Checkpoint Stage A" hint={TRAIN_HELP.checkpoint}><FileSelect value={trainFull.stage_a_ckpt} onChange={(v)=>setTrainFull({...trainFull,stage_a_ckpt:v})} options={knownCheckpointFiles} placeholder="-- chọn checkpoint Stage A --" /></TrainField>
                          <TrainField label="Checkpoint Stage B" hint={TRAIN_HELP.checkpoint}><FileSelect value={trainFull.stage_b_ckpt} onChange={(v)=>setTrainFull({...trainFull,stage_b_ckpt:v})} options={knownCheckpointFiles} placeholder="-- chọn checkpoint Stage B --" /></TrainField>
                          <TrainField label="Dataset eval (tùy chọn)" hint={TRAIN_HELP.eval_json}><input className="input" value={trainFull.eval_json} onChange={e=>setTrainFull({...trainFull,eval_json:e.target.value})} /></TrainField>
                          <TrainField label="Output eval JSON" hint={TRAIN_HELP.output_eval}><input className="input" value={trainFull.output_eval} onChange={e=>setTrainFull({...trainFull,output_eval:e.target.value})} /></TrainField>
                          <TrainField label="Init checkpoint Stage A" hint={TRAIN_HELP.init_checkpoint}><FileSelect value={trainFull.init_checkpoint || ""} onChange={(v)=>setTrainFull({...trainFull,init_checkpoint:v})} options={knownCheckpointFiles} placeholder="-- chọn checkpoint khởi tạo --" /></TrainField>
                          <TrainField label="Epoch Stage A" hint={TRAIN_HELP.epochs}><input className="input" type="number" value={trainFull.stage_a_epochs} onChange={e=>setTrainFull({...trainFull,stage_a_epochs:Number(e.target.value)})} /></TrainField>
                          <TrainField label="Epoch Stage B" hint={TRAIN_HELP.epochs}><input className="input" type="number" value={trainFull.stage_b_epochs} onChange={e=>setTrainFull({...trainFull,stage_b_epochs:Number(e.target.value)})} /></TrainField>
                          <TrainField label="LR Stage A" hint={TRAIN_HELP.lr}><input className="input" type="number" step="0.0001" value={trainFull.stage_a_lr} onChange={e=>setTrainFull({...trainFull,stage_a_lr:Number(e.target.value)})} /></TrainField>
                          <TrainField label="LR Stage B" hint={TRAIN_HELP.lr}><input className="input" type="number" step="0.0001" value={trainFull.stage_b_lr} onChange={e=>setTrainFull({...trainFull,stage_b_lr:Number(e.target.value)})} /></TrainField>
                        </div>
                        <div className="actions" style={{marginTop:10}}><div className="tiny">POST /api/pipeline/train-gcn-full</div><button className="btn dark" onClick={runTrainFull}>Chạy full flow</button></div>
                      </div>
                    </div>
                  </div>
                </details>
              </div>

              <div className="stack">
                <div className="step">
                  <div className="step-head"><h3>Theo dõi job train</h3><span className={`badge ${statusClass(trainBJob?.status || trainAJob?.status || trainFullJob?.status)}`}>{((trainBJob?.status || trainAJob?.status || trainFullJob?.status || "idle")).toUpperCase()}</span></div>
                  <div className="step-body">
                    <div className="tiny">Stage B là log quan trọng nhất</div>
                    <div className="term" style={{marginTop:6}}><h4>TRAIN STAGE B</h4><pre>{lastLog(trainBJob)}</pre></div>
                    <div className="tiny" style={{marginTop:10}}>Stage A</div>
                    <div className="term" style={{marginTop:6}}><h4>TRAIN STAGE A</h4><pre>{lastLog(trainAJob)}</pre></div>
                    <div className="tiny" style={{marginTop:10}}>Full flow</div>
                    <div className="term" style={{marginTop:6}}><h4>TRAIN FULL FLOW</h4><pre>{lastLog(trainFullJob)}</pre></div>
                  </div>
                </div>

                <div className="step">
                  <div className="step-head"><h3>Giải thích nhanh cho người mới</h3><span className="badge idle">EASY</span></div>
                  <div className="step-body">
                    <p className="p"><b>Epoch</b>: model học đi học lại dữ liệu bao nhiêu vòng.</p>
                    <p className="p" style={{marginTop:8}}><b>Learning rate</b>: model sửa sai nhanh hay chậm ở mỗi lần cập nhật.</p>
                    <p className="p" style={{marginTop:8}}><b>Validation</b>: bộ dữ liệu để kiểm tra xem model có đang học tốt thật hay không.</p>
                    <p className="p" style={{marginTop:8}}><b>Early stop</b>: dừng sớm khi model không tiến bộ nữa để đỡ tốn thời gian.</p>
                    <p className="p" style={{marginTop:8}}><b>Checkpoint</b>: file trọng số model được lưu ra sau khi train.</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="step"><div className="step-head"><h3>Output API mới nhất</h3><span className="badge idle">READY</span></div><div className="step-body"><div className="term"><h4>APPLICATION/JSON</h4><pre>{out||"Chưa có phản hồi API"}</pre></div></div></div>
          </> : page==="results" ? <>
            <h2 className="title">Kết quả & Đánh giá</h2>
            <div className="subtitle">Màn này dành cho 2 việc sau huấn luyện: chạy tập test để xem chỉ số thật, và infer một ảnh lẻ để thấy rõ GCN đã dùng graph như thế nào để sửa hoặc giữ nhãn so với rule/OCR cơ bản.</div>

            <div className="cards">
              <div className="card"><div className="k">Job đánh giá</div><div className="v">{stageBTestJob?1:0}</div></div>
              <div className="card"><div className="k">Job infer 1 ảnh</div><div className="v">{inferJob?1:0}</div></div>
              <div className="card"><div className="k">Checkpoint infer hiện tại</div><div className="v" style={{fontSize:14}}>{shortName(inferOne.checkpoint || "chưa chọn")}</div></div>
              <div className="card"><div className="k">Tập test hiện tại</div><div className="v" style={{fontSize:14}}>{shortName(stageBEval.dataset_json || "chưa chọn")}</div></div>
            </div>

            <div className="results-grid">
              <div className="stack">
                <div className="step">
                  <div className="step-head"><h3>1. Đánh giá trên tập test</h3><span className={`badge ${statusClass(stageBTestJob?.status)}`}>{(stageBTestJob?.status||"idle").toUpperCase()}</span></div>
                  <div className="step-body">
                    <div className="field-grid">
                      <TrainField label="Dataset test" hint="Tập test dùng để chấm điểm cuối cùng sau khi train xong.">
                        <input list="known-data-files" className="input" value={stageBEval.dataset_json} onChange={e=>setStageBEval({...stageBEval,dataset_json:e.target.value})} placeholder="data/stage_b_test.json"/>
                      </TrainField>
                      <TrainField label="Checkpoint Stage B" hint="Chọn checkpoint bạn muốn đem đi đánh giá trên tập test.">
                        <FileSelect value={stageBEval.checkpoint} onChange={(v)=>setStageBEval({...stageBEval,checkpoint:v})} options={knownCheckpointFiles} placeholder="-- chọn checkpoint để test --" />
                      </TrainField>
                      <TrainField label="File report output" hint="Báo cáo JSON sẽ được lưu ở đây để bạn mở lại sau này.">
                        <input className="input" value={stageBEval.output_eval} onChange={e=>setStageBEval({...stageBEval,output_eval:e.target.value})} placeholder="outputs/gcn_stage_b_test_report.json"/>
                      </TrainField>
                    </div>
                    <div className="actions" style={{marginTop:12}}>
                      <div className="tiny">POST /api/pipeline/test-gcn</div>
                      <button className="btn dark" onClick={runStageBEval}>Chạy test set</button>
                      <button className="btn" onClick={loadEvalReport}>Đọc report</button>
                    </div>
                    <div className="term" style={{marginTop:8}}><h4>LOG TEST GCN</h4><pre>{lastLog(stageBTestJob)}</pre></div>
                  </div>
                </div>

                <div className="step">
                  <div className="step-head"><h3>2. Infer một ảnh lẻ</h3><span className={`badge ${statusClass(inferJob?.status)}`}>{(inferJob?.status||"idle").toUpperCase()}</span></div>
                  <div className="step-body">
                    <div className="field-grid">
                      <TrainField label="Ảnh đã có trong project" hint="Có thể chọn luôn ảnh đã upload ở bước chuẩn bị dữ liệu.">
                        <FileSelect value={inferOne.image} onChange={(v)=>setInferOne({...inferOne,image:v})} options={allRawImages} placeholder="-- chọn ảnh đã có --" />
                      </TrainField>
                      <TrainField label="Checkpoint dùng để infer" hint="Checkpoint Stage B sẽ dùng để gán nhãn cho các node OCR trên ảnh này.">
                        <FileSelect value={inferOne.checkpoint} onChange={(v)=>setInferOne({...inferOne,checkpoint:v})} options={knownCheckpointFiles} placeholder="-- chọn checkpoint infer --" />
                      </TrainField>
                      <TrainField label="Ngôn ngữ OCR" hint="Hiện tại nên để `vi` cho hóa đơn tiếng Việt.">
                        <input className="input" value={inferOne.lang} onChange={e=>setInferOne({...inferOne,lang:e.target.value})} placeholder="vi"/>
                      </TrainField>
                      <TrainField label="Engine OCR" hint="`vietocr` dùng Paddle để bắt box rồi VietOCR để đọc chữ. Hữu ích khi bạn muốn so chất lượng text tiếng Việt với PaddleOCR thuần.">
                        <select className="input" value={inferOne.ocr_engine} onChange={e=>setInferOne({...inferOne,ocr_engine:e.target.value})}>
                          <option value="paddle">PaddleOCR</option>
                          <option value="vietocr">Paddle detect + VietOCR</option>
                        </select>
                      </TrainField>
                      <TrainField label="Ảnh OCR debug output" hint="Ảnh này sẽ vẽ box OCR để bạn kiểm tra OCR có bắt đúng vùng chữ hay không.">
                        <input className="input" value={inferOne.ocr_debug_image} onChange={e=>setInferOne({...inferOne,ocr_debug_image:e.target.value})} placeholder="outputs/ocr_boxes_single.jpg"/>
                      </TrainField>
                      <TrainField label="JSON kết quả infer" hint="File JSON chứa toàn bộ trường trích xuất, node OCR và graph info của ảnh lẻ.">
                        <input className="input" value={inferOne.output_json} onChange={e=>setInferOne({...inferOne,output_json:e.target.value})} placeholder="outputs/gcn_infer_single.json"/>
                      </TrainField>
                    </div>
                    <div className="actions" style={{marginTop:12}}>
                      <input type="file" accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp" onChange={e=>setInferUploadFile(e.target.files?.[0] || null)} />
                      <button className="btn" onClick={uploadInferImage}>Upload ảnh lẻ</button>
                      <button className="btn dark" onClick={runInferOne}>Chạy infer ảnh này</button>
                      <button className="btn" onClick={loadInferReport}>Đọc kết quả infer</button>
                    </div>
                    <div className="tiny" style={{marginTop:6}}>
                      {inferUploadFile ? `Đã chọn file local: ${inferUploadFile.name}` : "Bạn có thể chọn ảnh sẵn có trong project hoặc upload một ảnh lẻ mới."}
                    </div>
                    <div className="term" style={{marginTop:8}}><h4>LOG GCN INFER</h4><pre>{lastLog(inferJob)}</pre></div>
                  </div>
                </div>
              </div>

              <div className="stack">
                <div className="step">
                  <div className="step-head"><h3>Bảng chỉ số test gần nhất</h3><span className="badge idle">REPORT</span></div>
                  <div className="step-body">
                    {!evalMetrics ? (
                      <div className="empty-state">Chưa có report đánh giá. Hãy chạy `Chạy test set` rồi bấm `Đọc report` để xem chỉ số.</div>
                    ) : (
                      <>
                        <div className="metrics-grid">
                          <div className="card"><div className="k">Accuracy</div><div className="v">{((evalMetrics.accuracy||0)*100).toFixed(2)}%</div></div>
                          <div className="card"><div className="k">F1 macro</div><div className="v">{(evalMetrics.f1_macro||0).toFixed(4)}</div></div>
                          <div className="card"><div className="k">Loss avg</div><div className="v">{(evalMetrics.loss_avg||0).toFixed(4)}</div></div>
                          <div className="card"><div className="k">Số graph test</div><div className="v">{evalMetrics.num_graphs||0}</div></div>
                        </div>
                        <div className="kv" style={{marginTop:10}}>
                          <div>Precision macro</div><div>{(evalMetrics.precision_macro||0).toFixed(4)}</div>
                          <div>Recall macro</div><div>{(evalMetrics.recall_macro||0).toFixed(4)}</div>
                          <div>Số node test</div><div>{evalMetrics.num_nodes||0}</div>
                          <div>Thời gian infer TB / graph</div><div>{(evalMetrics.inference_time_ms_avg_per_graph||0).toFixed(2)} ms</div>
                        </div>
                        <div className="scroll-table" style={{marginTop:10}}>
                          <table className="tbl">
                            <thead>
                              <tr>
                                <th>Nhãn</th>
                                <th>TP</th>
                                <th>FP</th>
                                <th>FN</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1</th>
                              </tr>
                            </thead>
                            <tbody>
                              {evalPerClassRows.map((row)=>(
                                <tr key={row.label}>
                                  <td>{row.label}</td>
                                  <td>{row.tp}</td>
                                  <td>{row.fp}</td>
                                  <td>{row.fn}</td>
                                  <td>{Number(row.precision||0).toFixed(4)}</td>
                                  <td>{Number(row.recall||0).toFixed(4)}</td>
                                  <td>{Number(row.f1||0).toFixed(4)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </>
                    )}
                  </div>
                </div>

                <div className="step">
                  <div className="step-head"><h3>Kết quả infer ảnh lẻ</h3><span className="badge idle">INFER</span></div>
                  <div className="step-body">
                    {!inferReport ? (
                      <div className="empty-state">Chưa có kết quả infer. Hãy chọn ảnh, chọn checkpoint rồi bấm `Chạy infer ảnh này`.</div>
                    ) : (
                      <>
                        <div className="explain-banner">
                          <strong>GCN đang giúp ở đâu?</strong>
                          <div className="tiny">GCN không chỉ nhìn từng text OCR riêng lẻ. Nó còn nhìn quan hệ giữa các node gần nhau trong graph. Ở ảnh này, có <span className="change-good">{inferChangedCount} node</span> đã được đổi nhãn so với baseline rule-based.</div>
                        </div>
                        <div className="image-compare">
                          <div className="image-card">
                            <h4>Ảnh gốc</h4>
                            {inferReport.image_path ? <img src={`/api/files/image?path=${encodeURIComponent(inferReport.image_path)}`} alt="Ảnh gốc"/> : <div className="empty-state">Không có ảnh gốc.</div>}
                          </div>
                          <div className="image-card">
                            <h4>Ảnh OCR debug</h4>
                            {inferReport.ocr_boxes_image ? <img src={`/api/files/image?path=${encodeURIComponent(inferReport.ocr_boxes_image)}`} alt="OCR debug"/> : <div className="empty-state">Không có ảnh debug OCR.</div>}
                          </div>
                        </div>
                        <div className="metrics-grid" style={{marginTop:10}}>
                          <div className="card"><div className="k">Classifier mode</div><div className="v" style={{fontSize:14}}>{shortName(inferReport.classifier_mode||"-")}</div></div>
                          <div className="card"><div className="k">Số node OCR</div><div className="v">{inferReport.graph?.num_nodes||0}</div></div>
                          <div className="card"><div className="k">Số cạnh graph</div><div className="v">{inferReport.graph?.num_edges||0}</div></div>
                          <div className="card"><div className="k">Node bị GCN đổi nhãn</div><div className="v">{inferChangedCount}</div></div>
                        </div>
                        <div className="step" style={{marginTop:10}}>
                          <div className="step-head"><h3>So sánh trước và sau GCN</h3></div>
                          <div className="step-body">
                            <div className="compare-grid">
                              <div className="compare-box">
                                <h4>Baseline rule/OCR</h4>
                                <div className="body">
                                  <div className="kv">
                                    {inferRuleRows.map(([k,v])=>(
                                      <React.Fragment key={k}>
                                        <div>{k}</div>
                                        <div>{Array.isArray(v) ? v.join(" | ") : String(v ?? "-")}</div>
                                      </React.Fragment>
                                    ))}
                                  </div>
                                </div>
                              </div>
                              <div className="compare-box">
                                <h4>Kết quả cuối sau GCN</h4>
                                <div className="body">
                                  <div className="kv">
                                    {inferInvoiceRows.map(([k,v])=>(
                                      <React.Fragment key={k}>
                                        <div>{k}</div>
                                        <div>{Array.isArray(v) ? v.join(" | ") : String(v ?? "-")}</div>
                                      </React.Fragment>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="step" style={{marginTop:10}}>
                          <div className="step-head"><h3>Node OCR, feature và nhãn model gán</h3></div>
                          <div className="step-body">
                            <div className="actions" style={{marginBottom:10}}>
                              <select className="input" style={{maxWidth:220}} value={inferNodeFilter} onChange={e=>setInferNodeFilter(e.target.value)}>
                                <option value="all">Tất cả node</option>
                                <option value="changed">Chỉ node bị GCN đổi nhãn</option>
                                <option value="money">Chỉ node có dấu hiệu tiền</option>
                              </select>
                              <div className="tiny">Feature graph: {(inferReport.graph?.feature_names || []).join(", ")}</div>
                            </div>
                            {inferActiveNodeData ? (
                              <div className="node-detail">
                                <div style={{fontWeight:800, marginBottom:8}}>Node đang xem: #{inferActiveNodeData.node_id}</div>
                                <div className="node-detail-grid">
                                  <div>OCR text</div><div>{inferActiveNodeData.text}</div>
                                  <div>Baseline rule</div><div>{labelText(inferActiveNodeData.rule_label)}</div>
                                  <div>Nhãn sau GCN</div><div>{labelText(inferActiveNodeData.predicted_label)}</div>
                                  <div>GCN có đổi không?</div><div>{inferActiveNodeData.changed_by_gcn ? "Có, GCN đã đổi nhãn" : "Không, GCN giữ nguyên"}</div>
                                  <div>Số hàng xóm graph</div><div>{(inferActiveNodeData.neighbors||[]).length}</div>
                                  <div>Feature vector</div><div>{JSON.stringify(inferActiveNodeData.feature_vector || [])}</div>
                                </div>
                              </div>
                            ) : (
                              <div className="empty-state" style={{marginBottom:10}}>Bấm vào một dòng bên dưới để xem node đó được baseline và GCN xử lý như thế nào.</div>
                            )}
                            <div className="scroll-table">
                              <table className="tbl">
                                <thead>
                                  <tr>
                                    <th>Node</th>
                                    <th>Text</th>
                                    <th>Rule</th>
                                    <th>GCN</th>
                                    <th>GCN đổi?</th>
                                    <th>Score OCR</th>
                                    <th>Số hàng xóm</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {inferNodes.map((row, idx)=>(
                                    <tr key={idx} className={`node-row ${inferActiveNode===row.node_id?"active":""}`} onClick={()=>setInferActiveNode(row.node_id)}>
                                      <td>{row.node_id}</td>
                                      <td>{row.text}</td>
                                      <td>{labelText(row.rule_label)}</td>
                                      <td>{labelText(row.predicted_label)}</td>
                                      <td className={row.changed_by_gcn ? "change-good" : "change-muted"}>{row.changed_by_gcn ? "Có" : "Không"}</td>
                                      <td>{Number(row.confidence||0).toFixed(3)}</td>
                                      <td>{(row.neighbors||[]).length}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>

            <div className="step"><div className="step-head"><h3>Output API mới nhất</h3><span className="badge idle">READY</span></div><div className="step-body"><div className="term"><h4>APPLICATION/JSON</h4><pre>{out||"Chưa có phản hồi API"}</pre></div></div></div>
          </> : <>
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
                <div className="grid3">
                  <TrainField label="Ngôn ngữ OCR" hint={OCR_HELP.lang} help="Với hóa đơn tiếng Việt, gần như luôn nên để `vi`.">
                    <input className="input" value={ocr.lang} onChange={e=>setOcr({...ocr,lang:e.target.value})}/>
                  </TrainField>
                  <TrainField label="Thư mục ảnh đầu vào" hint={OCR_HELP.input_dir} help="Thường là `data/stage_b_raw_images`.">
                    <input className="input" value={ocr.input_dir} onChange={e=>setOcr({...ocr,input_dir:e.target.value})}/>
                  </TrainField>
                  <TrainField label="Thư mục output OCR" hint={OCR_HELP.output_dir} help="Bước 3 sẽ đọc dữ liệu từ thư mục này để rà nhãn và compile dataset.">
                    <input className="input" value={ocr.output_dir} onChange={e=>setOcr({...ocr,output_dir:e.target.value})}/>
                  </TrainField>
                </div>
                <div className="row" style={{gridTemplateColumns:"220px 1fr", marginTop:8}}>
                  <TrainField label="Engine OCR" hint={OCR_HELP.ocr_engine} help="Nếu bạn đang ưu tiên tốc độ thì thử PaddleOCR trước. Nếu ưu tiên text tiếng Việt có dấu, có thể so thêm với VietOCR.">
                    <select className="input" value={ocr.ocr_engine} onChange={e=>setOcr({...ocr,ocr_engine:e.target.value})}>
                      <option value="paddle">PaddleOCR</option>
                      <option value="vietocr">Paddle detect + VietOCR</option>
                    </select>
                  </TrainField>
                  <div className="tiny">Nếu chọn `vietocr`, hệ thống sẽ giữ box của Paddle nhưng dùng VietOCR để đọc text trong từng box.</div>
                </div>
                <div className="grid3" style={{marginTop:8}}>
                  <TrainField label="DB box thresh" hint={OCR_HELP.det_db_box_thresh} help="Gợi ý dễ bắt đầu: 0.55-0.60. Nếu box đang nuốt cả nền, tăng dần lên 0.62-0.68.">
                    <input className="input" type="number" step="0.01" value={ocr.det_db_box_thresh} onChange={e=>setOcr({...ocr,det_db_box_thresh:e.target.value})} placeholder="0.58" />
                  </TrainField>
                  <TrainField label="Unclip ratio" hint={OCR_HELP.det_db_unclip_ratio} help="Gợi ý dễ bắt đầu: 1.15-1.25. Nếu box bị chật cắt chữ, tăng nhẹ; nếu box phình, giảm xuống.">
                    <input className="input" type="number" step="0.01" value={ocr.det_db_unclip_ratio} onChange={e=>setOcr({...ocr,det_db_unclip_ratio:e.target.value})} placeholder="1.25" />
                  </TrainField>
                  <TrainField label="Drop score" hint={OCR_HELP.drop_score} help="Gợi ý dễ bắt đầu: 0.40-0.50. Tăng lên nếu đang nhận nhiều box rác ở nền.">
                    <input className="input" type="number" step="0.01" value={ocr.drop_score} onChange={e=>setOcr({...ocr,drop_score:e.target.value})} placeholder="0.45" />
                  </TrainField>
                </div>
                <div className="grid3" style={{marginTop:8}}>
                  <TrainField label="Det limit side" hint={OCR_HELP.det_limit_side_len} help="Thường 1280-1792 là hợp lý. Ảnh càng to thì detect càng kỹ nhưng chậm hơn.">
                    <input className="input" type="number" step="64" value={ocr.det_limit_side_len} onChange={e=>setOcr({...ocr,det_limit_side_len:e.target.value})} placeholder="1536" />
                  </TrainField>
                  <TrainField label="Upscale factor" hint={OCR_HELP.upscale_factor} help="Thường 1.4-1.8. Ảnh quá mờ hoặc chụp xa thì nên tăng, nhưng càng tăng càng tốn tài nguyên.">
                    <input className="input" type="number" step="0.1" value={ocr.upscale_factor} onChange={e=>setOcr({...ocr,upscale_factor:e.target.value})} placeholder="1.6" />
                  </TrainField>
                  <TrainField label="Use dilation" hint={OCR_HELP.use_dilation} help="Để `tắt` nếu box đang tương đối ổn. Chỉ bật khi chữ bị đứt, mảnh hoặc detect bị vỡ nhiều mảnh nhỏ.">
                    <select className="input" value={ocr.use_dilation} onChange={e=>setOcr({...ocr,use_dilation:Number(e.target.value)})}>
                      <option value={0}>Dilation: tắt</option>
                      <option value={1}>Dilation: bật</option>
                    </select>
                  </TrainField>
                </div>
                <div className="grid2" style={{marginTop:8}}>
                  <TrainField label="Lưu mỗi N ảnh" hint={OCR_HELP.save_every_images} help="Nếu hay dừng giữa chừng để kiểm tra, nên để 5-10. Đây là mốc auto-lưu để bạn dừng xong vẫn có dữ liệu sang bước 3.">
                    <input className="input" type="number" min="1" max="100" value={ocr.save_every_images} onChange={e=>setOcr({...ocr,save_every_images:Number(e.target.value||10)})} placeholder="10" />
                  </TrainField>
                  <TrainField label="Chế độ chạy lại" hint={OCR_HELP.overwrite_existing} help="`OCR lại từ đầu` dùng khi bạn vừa đổi tham số OCR. `Chạy tiếp phần còn lại` dùng khi lần trước dừng giữa chừng và muốn OCR tiếp.">
                    <select className="input" value={ocr.overwrite_existing} onChange={e=>setOcr({...ocr,overwrite_existing:Number(e.target.value)})}>
                      <option value={1}>OCR lại từ đầu</option>
                      <option value={0}>Chạy tiếp phần còn lại</option>
                    </select>
                  </TrainField>
                </div>
                <div className="tiny" style={{marginTop:6}}>
                  Gợi ý hiện tại cho box chặt hơn: <b>box_thresh 0.58</b>, <b>unclip 1.25</b>, <b>drop_score 0.45</b>. Nếu box vẫn phình rộng, giảm unclip xuống 1.15-1.20.
                </div>
                <div className="tiny" style={{marginTop:4}}>
                  Hệ thống hiện OCR tuần tự để ưu tiên ổn định. Nếu cần dừng giữa chừng để kiểm tra, hãy dùng nút <b>Dừng bước 2</b>, dữ liệu đã auto-lưu vẫn dùng được ở bước 3.
                </div>
                <div style={{marginTop:8}} className="p">Tiến độ xử lý file: <b>{ocrCurrent}</b> / <b>{ocrTotal}</b> ({ocrPct}%)</div>
                {ocrCurrentFile && <div className="tiny" style={{marginTop:4}}>Đang xử lý: <b>{ocrCurrentFile}</b></div>}
                <div className="bar" style={{marginTop:6}}><div style={{width:`${ocrPct}%`}}/></div>
                <div className="row" style={{gridTemplateColumns:"1fr auto auto",marginTop:10}}>
                  <div className="tiny">POST /api/pipeline/prepare-ocr-labeling</div>
                  <button className="btn dark" onClick={runOcrBatch}>Chạy bước 2</button>
                  <button className="btn" onClick={stopOcrBatch} disabled={!ocrJob || !["queued","running","stopping"].includes(String(ocrJob.status||""))}>
                    {ocrJob?.status==="stopping" ? "Đang dừng..." : "Dừng bước 2"}
                  </button>
                </div>
                {(ocrJob?.status==="stopped" || ocrJob?.status==="stopping") && (
                  <div className="tiny" style={{marginTop:6}}>
                    {ocrJob?.status==="stopped"
                      ? "Bước 2 đã dừng an toàn. Bạn có thể sang bước 3 để tải dữ liệu partial và kiểm tra ngay."
                      : "Đã gửi yêu cầu dừng. Hệ thống sẽ hoàn tất ảnh đang OCR rồi lưu kết quả đã có."}
                  </div>
                )}
                <div className="term" style={{marginTop:8}}><h4>LOG BƯỚC 2 (THỰC TẾ)</h4><pre>{lastLog(ocrJob)}</pre></div>
              </div>
            </div>

            <div className="step">
              <div className="step-head"><h3>Bước 3: Hoàn tất nhãn và compile dataset</h3><span className={`badge ${statusClass(dsJob?.status)}`}>{(dsJob?.status||"idle").toUpperCase()}</span></div>
              <div className="step-body">
                <div className="grid2"><input className="input" value={dataset.input_csv} onChange={e=>setDataset({...dataset,input_csv:e.target.value})}/><input className="input" value={dataset.output_json} onChange={e=>setDataset({...dataset,output_json:e.target.value})}/></div>
                <div className="row" style={{gridTemplateColumns:"100px 90px auto auto auto auto 1fr",marginTop:8}}>
                  <input className="input" type="number" min="10" max="500" value={labelLimit} onChange={e=>setLabelLimit(e.target.value)} />
                  <input className="input" type="number" min="1" value={labelPage} onChange={e=>setLabelPage(e.target.value)} />
                  <button className="btn" onClick={()=>{setMissingOnly(false); loadLabelPreview();}}>Lấy dữ liệu</button>
                  <button className="btn" onClick={()=>{setMissingOnly(true); loadLabelPreview(1, true);}}>Xem dữ liệu thiếu nhãn</button>
                  <button className="btn dark" onClick={()=>loadByDoc(1)}>Mở workspace theo ảnh</button>
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
                <div className="review-shell">
                  <div className="review-pane">
                    <div className="review-pane-head">
                      <h4>Danh sách hóa đơn để rà</h4>
                      <div className="tiny">Trang {docPage} | tải {docPageSize} ảnh mỗi lần | {docHasNext ? "còn trang sau" : "đang là trang cuối hiện có"}</div>
                    </div>
                    <div className="review-pane-body">
                      <div className="doc-toolbar">
                        <input className="input" placeholder="Tìm theo doc id..." value={docSearch} onChange={e=>setDocSearch(e.target.value)} />
                        <select className="input" value={docFilter} onChange={e=>setDocFilter(e.target.value)}>
                          <option value="all">Tất cả ảnh</option>
                          <option value="missing">Ảnh còn thiếu nhãn</option>
                          <option value="labeled">Ảnh đã đủ nhãn</option>
                        </select>
                      </div>
                      <div className="doc-pager" style={{marginTop:8}}>
                        <input className="input" type="number" min="10" max="100" value={docPageSize} onChange={e=>setDocPageSize(e.target.value)} />
                        <button className="btn" onClick={()=>loadByDoc(1)}>Tải danh sách ảnh</button>
                        <button className="btn" onClick={()=>{const p=Math.max(1, Number(docPage)-1); loadByDoc(p);}} disabled={Number(docPage)<=1}>Trang trước</button>
                        <button className="btn" onClick={()=>{loadByDoc(Number(docPage)+1);}} disabled={!docHasNext}>Trang sau</button>
                      </div>
                    </div>
                    <div className="review-list">
                      {filteredDocs.map((doc)=>(
                        <div key={doc.doc_id} className={`doc-item ${graphInspect?.doc_id===doc.doc_id?"active":""}`} onClick={()=>openGraphInspect(doc.doc_id)}>
                          <div>
                            {doc.preview_path ? <img className="thumb" loading="lazy" decoding="async" src={`/api/files/image?path=${encodeURIComponent(doc.preview_path)}`} alt={doc.doc_id}/> : <div className="thumb" />}
                          </div>
                          <div className="doc-meta">
                            <div className="doc-id">{doc.doc_id}</div>
                            <div className="doc-sub">{doc.total_nodes || 0} node | thiếu nhãn: {doc.empty_labels || 0}</div>
                            <div className="doc-sub">{Object.entries(doc.labels||{}).slice(0,2).map(([k,v])=>`${labelText(k)}:${v}`).join(" | ")}</div>
                          </div>
                          <div><span className={`badge ${Number(doc.empty_labels||0)>0?"queued":"success"}`}>{Number(doc.empty_labels||0)>0?"THIẾU":"OK"}</span></div>
                        </div>
                      ))}
                      {filteredDocs.length===0 && <div className="tiny" style={{padding:12}}>Chưa có ảnh nào. Bấm `Mở workspace theo ảnh` để tải dữ liệu từ CSV.</div>}
                    </div>
                  </div>

                  <div className="review-pane">
                    <div className="review-pane-head">
                      <h4>Hóa đơn đang kiểm tra</h4>
                      <div className="chipline">
                        <span className="mini-chip">{graphInspect?.doc_id || "Chưa chọn ảnh"}</span>
                        <span className="mini-chip">{inspectTab==="ocr" ? "Chế độ rà nhãn" : "Chế độ graph train B"}</span>
                      </div>
                    </div>
                    <div className="review-pane-body">
                      {graphInspectLoading ? (
                        <div className="empty-state">Đang tải ảnh, node OCR và graph...</div>
                      ) : !graphInspect ? (
                        <div className="empty-state">Chọn một hóa đơn ở cột bên trái để bắt đầu rà nhãn theo ảnh.</div>
                      ) : (
                        <>
                          <div className="actions" style={{marginBottom:10}}>
                            <button className={`tab-btn ${inspectTab==="ocr"?"active":""}`} onClick={()=>setInspectTab("ocr")}>Rà nhãn theo ảnh</button>
                            <button className={`tab-btn ${inspectTab==="graph"?"active":""}`} onClick={()=>setInspectTab("graph")}>Xem graph train B</button>
                            <button className="btn" onClick={()=>nextMissingNode && setActiveNodeIndex(nextMissingNode.node_index)} disabled={!nextMissingNode}>Tới node thiếu nhãn</button>
                            <button className="btn" onClick={()=>nextSuspiciousNode && setActiveNodeIndex(nextSuspiciousNode.node_index)} disabled={!nextSuspiciousNode}>Tới node nghi ngờ</button>
                            <button className="btn" onClick={saveInspectLabels}>Lưu nhãn ảnh này</button>
                          </div>
                          {inspectTab==="ocr" ? (
                            <>
                              <div className="review-image">
                                {graphInspect.preview_path ? (
                                  <>
                                    <div className="review-canvas">
                                      <img
                                        src={`/api/files/image?path=${encodeURIComponent(graphInspect.preview_path)}`}
                                        alt={graphInspect.doc_id}
                                        onLoad={(e)=>setImageNatural({w:e.target.naturalWidth||1,h:e.target.naturalHeight||1})}
                                      />
                                      <div className="review-overlay">
                                        <svg viewBox={`0 0 ${imageNatural.w} ${imageNatural.h}`} preserveAspectRatio="none">
                                          {reviewImageNodes.map((node)=>{
                                            const color = labelColor(String(node.picked_label || node.label || "").trim() || "OTHER");
                                            const isActive = activeNodeIndex===node.node_index;
                                            return nodeHasQuad(node) ? (
                                              <polygon
                                                key={`box-poly-${node.node_index}`}
                                                className="ocr-poly"
                                                points={nodeQuadPoints(node)}
                                                fill={isActive ? "rgba(239,68,68,.12)" : color.fill}
                                                stroke={isActive ? "#ef4444" : color.border}
                                                strokeWidth="2"
                                                onClick={()=>setActiveNodeIndex(node.node_index)}
                                              />
                                            ) : (
                                              <rect
                                                key={`box-rect-${node.node_index}`}
                                                className="ocr-poly"
                                                x={Number(node.bbox?.[0]||0)}
                                                y={Number(node.bbox?.[1]||0)}
                                                width={Math.max(1, Number((node.bbox?.[2]||0) - (node.bbox?.[0]||0)))}
                                                height={Math.max(1, Number((node.bbox?.[3]||0) - (node.bbox?.[1]||0)))}
                                                fill={isActive ? "rgba(239,68,68,.12)" : color.fill}
                                                stroke={isActive ? "#ef4444" : color.border}
                                                strokeWidth="2"
                                                onClick={()=>setActiveNodeIndex(node.node_index)}
                                              />
                                            );
                                          })}
                                        </svg>
                                      </div>
                                    </div>
                                  </>
                                ) : <div className="empty-state">Không có ảnh preview.</div>}
                              </div>
                              <div className="review-kpis">
                                <div className="review-kpi"><div className="k">Số node OCR</div><div className="v">{graphInspect.graph?.num_nodes || 0}</div></div>
                                <div className="review-kpi"><div className="k">Số node chưa gán</div><div className="v">{(graphInspect.nodes||[]).filter(n=>!String(n.picked_label||"").trim()).length}</div></div>
                                <div className="review-kpi"><div className="k">Số cạnh graph</div><div className="v">{graphInspect.graph?.num_edges || 0}</div></div>
                              </div>
                              <div className="chipline" style={{marginTop:10}}>
                                {reviewDocChecklist.map((item)=>(
                                  <span key={item.label} className={`badge ${item.ok ? "success" : "failed"}`}>{labelText(item.label)}: {item.ok ? "OK" : "Thiếu"}</span>
                                ))}
                                {suspiciousNodes.length>0 ? <span className="badge failed">Nghi ngờ: {suspiciousNodes.length}</span> : <span className="badge success">Không thấy node nghi ngờ</span>}
                              </div>
                              <div className="chipline" style={{marginTop:10}}>
                                <button className={`mini-chip ${reviewNodeFilter==="all"?"active":""}`} onClick={()=>setReviewNodeFilter("all")}>Tất cả</button>
                                <button className={`mini-chip ${reviewNodeFilter==="missing"?"active":""}`} onClick={()=>setReviewNodeFilter(reviewNodeFilter==="missing" ? "all" : "missing")}>Thiếu nhãn</button>
                                <button className={`mini-chip ${reviewNodeFilter==="suspicious"?"active":""}`} onClick={()=>setReviewNodeFilter(reviewNodeFilter==="suspicious" ? "all" : "suspicious")}>Nghi ngờ sai</button>
                                <button className={`mini-chip ${reviewNodeFilter==="money"?"active":""}`} onClick={()=>setReviewNodeFilter(reviewNodeFilter==="money" ? "all" : "money")}>Giống tiền</button>
                                <button className={`mini-chip ${reviewNodeFilter==="date"?"active":""}`} onClick={()=>setReviewNodeFilter(reviewNodeFilter==="date" ? "all" : "date")}>Giống ngày</button>
                                {inspectLabelSummary.slice(0,6).map(([label,count])=>(
                                  <button
                                    key={label}
                                    className={`mini-chip ${reviewNodeFilter===label?"active":""}`}
                                    onClick={()=>{
                                      setReviewNodeFilter(reviewNodeFilter===label ? "all" : label);
                                      setReviewSearch("");
                                    }}
                                  >
                                    {labelText(label)}: {count}
                                  </button>
                                ))}
                              </div>
                            </>
                          ) : (
                            <>
                              <div className="review-kpis">
                                <div className="review-kpi"><div className="k">Số node</div><div className="v">{graphInspect.graph?.num_nodes || 0}</div></div>
                                <div className="review-kpi"><div className="k">Số cạnh</div><div className="v">{graphInspect.graph?.num_edges || 0}</div></div>
                                <div className="review-kpi"><div className="k">Số feature</div><div className="v">{(graphInspect.feature_names || []).length}</div></div>
                              </div>
                              <div className="node-detail" style={{marginTop:10}}>
                                <div style={{fontWeight:800, marginBottom:6}}>Dữ liệu graph sẽ đi vào train B</div>
                                <div className="tiny">Feature đang dùng: {(graphInspect.feature_names || []).join(", ") || "Không có feature"}.</div>
                              </div>
                              <div className="term"><h4>EDGE_INDEX</h4><pre>{JSON.stringify(graphInspect.graph?.edge_index || [[],[]], null, 2)}</pre></div>
                              <div className="table-wrap" style={{maxHeight:360, marginTop:10}}>
                                <table className="tbl">
                                  <thead>
                                    <tr>
                                      <th style={{minWidth:70}}>Node</th>
                                      {((graphInspect.graph?.adjacency_matrix || [])[0] || []).map((_, idx)=><th key={idx} style={{minWidth:42, textAlign:"center"}}>{idx}</th>)}
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {(graphInspect.graph?.adjacency_matrix || []).map((row, rowIdx)=>(
                                      <tr key={rowIdx}>
                                        <th style={{background:"#f7f6fc"}}>{rowIdx}</th>
                                        {row.map((cell, colIdx)=><td key={colIdx} style={{textAlign:"center"}}>{cell}</td>)}
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            </>
                          )}
                        </>
                      )}
                    </div>
                  </div>

                  <div className="review-pane">
                    <div className="review-pane-head">
                      <h4>Node OCR và nhãn của hóa đơn này</h4>
                    </div>
                    <div className="review-pane-body">
                      {!graphInspect ? (
                        <div className="empty-state">Chưa có hóa đơn nào được chọn.</div>
                      ) : (
                        <>
                          <div className="node-filterbar">
                            <input className="input" placeholder="Lọc theo text hoặc mã nhãn..." value={reviewSearch} onChange={e=>setReviewSearch(e.target.value)} />
                            <select className="input" value={reviewNodeFilter} onChange={e=>setReviewNodeFilter(e.target.value)}>
                              <option value="all">Tất cả node</option>
                              <option value="missing">Chỉ node thiếu nhãn</option>
                              <option value="labeled">Chỉ node đã có nhãn</option>
                              <option value="suspicious">Chỉ node nghi ngờ sai nhãn</option>
                              <option value="money">Chỉ node giống tiền</option>
                              <option value="date">Chỉ node giống ngày</option>
                              {allowedLabels.map(lb=><option key={lb} value={lb}>{labelText(lb)}</option>)}
                            </select>
                            <button className="btn" onClick={()=>{setReviewSearch(""); setReviewNodeFilter("missing");}}>Lọc node thiếu nhãn</button>
                          </div>
                          {activeInspectNode ? (
                            <div className="node-detail">
                              <div style={{fontWeight:800, marginBottom:8}}>Node đang chọn: #{activeInspectNode.node_index}</div>
                              <div className="node-detail-grid">
                                <div>OCR text</div><div>{activeInspectNode.text}</div>
                                <div>Dòng CSV</div><div>{activeInspectNode.row_number}</div>
                                <div>Score OCR</div><div>{Number(activeInspectNode.score||0).toFixed(3)}</div>
                                <div>BBox</div><div>{Array.isArray(activeInspectNode.bbox) ? activeInspectNode.bbox.map(v=>Number(v).toFixed(1)).join(", ") : ""}</div>
                                <div>Nhãn hiện tại</div><div>{labelText(activeInspectNode.picked_label || activeInspectNode.label)}</div>
                                <div>Cờ kiểm tra nhanh</div><div>{reviewIssueForNode(activeInspectNode) || "Không thấy dấu hiệu bất thường rõ ràng"}</div>
                              </div>
                            </div>
                          ) : (
                            <div className="empty-state" style={{marginBottom:10}}>Bấm vào box trên ảnh hoặc một dòng OCR bên dưới để xem chi tiết node đó.</div>
                          )}
                          <div className="table-wrap" style={{maxHeight:760}}>
                            <table className="tbl">
                              <thead>
                                <tr>
                                  <th style={{width:56}}>Node</th>
                                  <th style={{width:70}}>Dòng</th>
                                  <th>Nội dung OCR trích xuất</th>
                                  <th style={{width:180}}>Nhãn</th>
                                  <th style={{width:88}}>Score</th>
                                </tr>
                              </thead>
                              <tbody>
                                {reviewNodesFiltered.map((node)=>(
                                  <tr key={node.node_index} className={`node-row ${activeNodeIndex===node.node_index?"active":""}`} onClick={()=>setActiveNodeIndex(node.node_index)}>
                                    <td>{node.node_index}</td>
                                    <td>{node.row_number}</td>
                                    <td>{node.text}</td>
                                    <td>
                                      <select className="input" value={node.picked_label||""} onChange={e=>setGraphInspect(prev=>({...prev,nodes:prev.nodes.map(x=>x.node_index===node.node_index?{...x,picked_label:e.target.value}:x)}))}>
                                        <option value="">-- chọn nhãn --</option>
                                        {allowedLabels.map(lb=><option key={lb} value={lb}>{labelText(lb)}</option>)}
                                      </select>
                                    </td>
                                    <td>{Number(node.score || 0).toFixed(3)}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </>
                      )}
                    </div>
                  </div>
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
