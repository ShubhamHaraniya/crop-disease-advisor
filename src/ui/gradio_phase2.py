"""
Crop Disease Advisor — Phase 2 UI
Clean, professional SaaS-dashboard aesthetic.
"""

import io, os, requests
from pathlib import Path
from datetime import datetime
from PIL import Image

import gradio as gr
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import tempfile

API_URL = os.getenv("API_URL", "http://localhost:8000")
REGIONS = ["North India", "South India", "East India", "West India", "Central India"]
SEASONS = ["Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)"]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PDF GENERATOR — Agronomist Clinical Report Style                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def generate_pdf(result: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    W = 170*mm
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=14*mm, bottomMargin=18*mm,
    )

    # Palette
    INK       = colors.HexColor("#0f172a")
    SLATE     = colors.HexColor("#334155")
    MUTED     = colors.HexColor("#64748b")
    HAIRLINE  = colors.HexColor("#e2e8f0")
    SURFACE   = colors.HexColor("#f8fafc")
    SURFACE2  = colors.HexColor("#f1f5f9")
    EMERALD   = colors.HexColor("#059669")
    EMERALD_L = colors.HexColor("#d1fae5")
    TEAL      = colors.HexColor("#0d9488")
    AMBER     = colors.HexColor("#d97706")
    AMBER_L   = colors.HexColor("#fef3c7")
    RED       = colors.HexColor("#dc2626")
    RED_L     = colors.HexColor("#fee2e2")
    VIOLET    = colors.HexColor("#7c3aed")
    VIOLET_L  = colors.HexColor("#ede9fe")
    BLUE      = colors.HexColor("#2563eb")
    BLUE_L    = colors.HexColor("#dbeafe")
    ORANGE    = colors.HexColor("#ea580c")
    WHITE     = colors.white

    severity_str = result.get("severity", "Mild")
    sev_key = severity_str.split()[0]
    sev_color = {"Severe": RED, "Moderate": AMBER, "Mild": EMERALD, "None": EMERALD}.get(sev_key, TEAL)
    sev_bg    = {"Severe": RED_L, "Moderate": AMBER_L, "Mild": EMERALD_L, "None": EMERALD_L}.get(sev_key, BLUE_L)

    tp = result.get("treatment_plan") or {}
    urgency_str = tp.get("action_urgency", "—")
    urg_color = {"Immediate": RED, "Within 3 days": ORANGE, "Within a week": AMBER, "Monitor": EMERALD}.get(urgency_str, BLUE)

    ss = getSampleStyleSheet()
    def S(name, **kw):
        return ParagraphStyle(name, parent=ss["Normal"], **kw)

    # ── Styles ─────────────────────────────────────────────────────────────────
    styles = dict(
        cover_title   = S("ct", fontSize=22, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_LEFT, spaceAfter=2),
        cover_sub     = S("cs", fontSize=10, fontName="Helvetica", textColor=colors.HexColor("#bbf7d0"), alignment=TA_LEFT),
        cover_meta    = S("cm", fontSize=8,  fontName="Helvetica", textColor=colors.HexColor("#6ee7b7"), alignment=TA_LEFT),
        section_title = S("st", fontSize=11, fontName="Helvetica-Bold", textColor=WHITE, leading=16),
        label         = S("lb", fontSize=7.5,fontName="Helvetica-Bold", textColor=MUTED, spaceAfter=1, leading=10),
        value         = S("vl", fontSize=10, fontName="Helvetica-Bold", textColor=INK, spaceAfter=2),
        value_c       = S("vc", fontSize=10, fontName="Helvetica-Bold", textColor=sev_color),
        body          = S("bd", fontSize=9.5,fontName="Helvetica", textColor=SLATE, leading=14, alignment=TA_JUSTIFY),
        bullet        = S("bu", fontSize=9.5,fontName="Helvetica", textColor=SLATE, leading=14, leftIndent=5*mm),
        th            = S("th", fontSize=9,  fontName="Helvetica-Bold", textColor=WHITE),
        td            = S("td", fontSize=9,  fontName="Helvetica", textColor=SLATE, leading=12),
        td_bold       = S("tb", fontSize=9,  fontName="Helvetica-Bold", textColor=INK, leading=12),
        footer        = S("ft", fontSize=7.5,fontName="Helvetica", textColor=MUTED, alignment=TA_CENTER),
        footer_b      = S("fb", fontSize=7.5,fontName="Helvetica-Bold", textColor=SLATE, alignment=TA_CENTER),
        badge_text    = S("bg", fontSize=9,  fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER),
        disclaimer    = S("di", fontSize=8.5,fontName="Helvetica-Oblique", textColor=SLATE, leading=12, alignment=TA_JUSTIFY),
    )

    # ── Helpers ────────────────────────────────────────────────────────────────
    def divider(c=None, t=0.5):
        return HRFlowable(width=W, thickness=t, color=c or HAIRLINE)

    def section_bar(icon, title, bg=None):
        t = Table([[Paragraph(f"{icon}  {title}", styles["section_title"])]], colWidths=[W])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), bg or EMERALD),
            ("TOPPADDING",    (0,0),(-1,-1), 7),
            ("BOTTOMPADDING", (0,0),(-1,-1), 7),
            ("LEFTPADDING",   (0,0),(-1,-1), 14),
        ]))
        return t

    def meta_grid(items, n_cols=4):
        col_w = W / n_cols
        hdr = [Paragraph(k.upper(), styles["label"]) for k,_ in items]
        val = [Paragraph(str(v), styles["value"]) for _,v in items]
        t = Table([hdr, val], colWidths=[col_w]*n_cols)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), SURFACE2),
            ("BACKGROUND",    (0,1),(-1,1), SURFACE),
            ("BOX",           (0,0),(-1,-1), 0.5, HAIRLINE),
            ("INNERGRID",     (0,0),(-1,-1), 0.3, HAIRLINE),
            ("TOPPADDING",    (0,0),(-1,-1), 7),
            ("BOTTOMPADDING", (0,0),(-1,-1), 7),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ]))
        return t

    def treatment_tbl(rows, headers, widths, hdr_bg=None):
        data = [[Paragraph(h, styles["th"]) for h in headers]]
        for row in rows:
            data.append([Paragraph(str(c), styles["td"]) for c in row])
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), hdr_bg or EMERALD),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [SURFACE, SURFACE2]),
            ("BOX",           (0,0),(-1,-1), 0.5, HAIRLINE),
            ("INNERGRID",     (0,0),(-1,-1), 0.25, HAIRLINE),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 9),
            ("VALIGN",        (0,0),(-1,-1), "TOP"),
        ]))
        return t

    # ── Story ──────────────────────────────────────────────────────────────────
    story = []
    now = datetime.now()
    conf_val = result.get("confidence", 0)
    disease_clean = result.get("disease","N/A").replace("___"," — ").replace("_"," ")

    # 1. Cover Banner
    banner_data = [
        [Paragraph("🌾  CROP DISEASE ADVISOR", styles["cover_title"])],
        [Paragraph("Clinical Agronomic Diagnosis &amp; Treatment Report", styles["cover_sub"])],
        [Paragraph(
            f"Report&nbsp;&nbsp;CDA-{now.strftime('%Y%m%d-%H%M%S')} &nbsp;·&nbsp; "
            f"Issued&nbsp;{now.strftime('%d %b %Y, %I:%M %p')}",
            styles["cover_meta"])],
    ]
    banner = Table(banner_data, colWidths=[W])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), colors.HexColor("#064e3b")),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 16),
        ("RIGHTPADDING",  (0,0),(-1,-1), 16),
    ]))
    story += [banner, Spacer(1, 5*mm)]

    # 2. Diagnosis Summary
    story += [section_bar("📋", "Diagnosis Summary"), Spacer(1, 2*mm)]
    story.append(meta_grid([
        ("Disease Detected", disease_clean),
        ("Crop Species",     result.get("crop","N/A")),
        ("AI Confidence",    f"{conf_val:.1f}%"),
        ("Severity Level",   severity_str),
    ]))
    story += [Spacer(1, 2*mm)]
    story.append(meta_grid([
        ("Region",      result.get("region","—")),
        ("Season",      result.get("season","—")),
        ("Action Urgency", urgency_str),
        ("Status",      "COMPLETED"),
    ]))
    story += [Spacer(1, 4*mm)]

    # 3. Severity + Urgency Badge Row
    badge_row = [[
        Paragraph(f"● SEVERITY: {severity_str.upper()}", styles["badge_text"]),
        Paragraph(f"⏱  ACTION: {urgency_str.upper()}", styles["badge_text"]),
    ]]
    bt = Table(badge_row, colWidths=[W/2, W/2])
    bt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(0,-1), sev_color),
        ("BACKGROUND",    (1,0),(1,-1), urg_color),
        ("TOPPADDING",    (0,0),(-1,-1), 9),
        ("BOTTOMPADDING", (0,0),(-1,-1), 9),
        ("INNERGRID",     (0,0),(-1,-1), 2, WHITE),
    ]))
    story += [bt, Spacer(1, 6*mm)]

    # 4. Organic Treatments
    organics = tp.get("organic_treatments", [])
    if organics:
        story.append(KeepTogether([
            section_bar("🌿", "Organic / Biological Treatments", TEAL),
            Spacer(1, 2*mm),
            treatment_tbl(
                [[t.get("method",""), t.get("application",""), t.get("frequency","")]
                  for t in organics],
                ["Method / Agent", "Application Instructions", "Frequency"],
                [44*mm, 90*mm, 36*mm], hdr_bg=TEAL,
            ),
            Spacer(1, 5*mm),
        ]))

    # 5. Chemical Treatments
    chemicals = tp.get("chemical_treatments", [])
    if chemicals:
        story.append(KeepTogether([
            section_bar("⚗️", "Chemical Treatments", RED),
            Spacer(1, 2*mm),
            treatment_tbl(
                [[t.get("product",""), t.get("dosage",""), t.get("safety_note","")]
                  for t in chemicals],
                ["Product / Fungicide", "Dosage &amp; Dilution", "Safety Note"],
                [52*mm, 44*mm, 74*mm], hdr_bg=RED,
            ),
            Spacer(1, 5*mm),
        ]))

    # 6. Preventive Measures
    preventive = tp.get("preventive_measures", [])
    if preventive:
        prev_rows = [[Paragraph(f"✓  {m}", styles["bullet"])] for m in preventive]
        prev_t = Table(prev_rows, colWidths=[W])
        prev_t.setStyle(TableStyle([
            ("ROWBACKGROUNDS",(0,0),(-1,-1), [SURFACE, VIOLET_L]),
            ("BOX",           (0,0),(-1,-1), 0.5, colors.HexColor("#c4b5fd")),
            ("INNERGRID",     (0,0),(-1,-1), 0.25, HAIRLINE),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ]))
        story.append(KeepTogether([
            section_bar("🛡️", "Preventive Measures", VIOLET),
            Spacer(1, 2*mm), prev_t, Spacer(1, 5*mm),
        ]))

    # 7. Yield Impact
    yield_val = tp.get("yield_impact_estimate","")
    if yield_val:
        yt = Table([[Paragraph(yield_val, styles["body"])]], colWidths=[W])
        yt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), AMBER_L),
            ("BOX",           (0,0),(-1,-1), 0.6, AMBER),
            ("TOPPADDING",    (0,0),(-1,-1), 9),
            ("BOTTOMPADDING", (0,0),(-1,-1), 9),
            ("LEFTPADDING",   (0,0),(-1,-1), 12),
        ]))
        story.append(KeepTogether([
            section_bar("📉", "Estimated Yield Impact", AMBER),
            Spacer(1, 2*mm), yt, Spacer(1, 5*mm),
        ]))

    # 8. Regional & Seasonal Advisory
    regional = tp.get("regional_notes","")
    seasonal = tp.get("seasonal_notes","")
    if regional or seasonal:
        adv = Table([
            [Paragraph("🌍  Regional Notes", styles["section_title"]),
             Paragraph("🌦  Seasonal Notes", styles["section_title"])],
            [Paragraph(regional or "—", styles["body"]),
             Paragraph(seasonal or "—", styles["body"])],
        ], colWidths=[W/2, W/2])
        adv.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#1e3a5f")),
            ("BACKGROUND",    (0,1),(-1,1), BLUE_L),
            ("BOX",           (0,0),(-1,-1), 0.5, colors.HexColor("#93c5fd")),
            ("INNERGRID",     (0,0),(-1,-1), 0.3, HAIRLINE),
            ("TOPPADDING",    (0,0),(-1,-1), 8),
            ("BOTTOMPADDING", (0,0),(-1,-1), 8),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
            ("VALIGN",        (0,0),(-1,-1), "TOP"),
        ]))
        story += [adv, Spacer(1, 6*mm)]

    # 9. Disclaimer
    disc = Table([[Paragraph(
        "⚠ DISCLAIMER: This report is AI-generated for advisory purposes only. "
        "Consult a licensed agronomist or plant pathologist before applying any treatment. "
        "Validate dosages against current local regulations and ICAR guidelines.",
        styles["disclaimer"],
    )]], colWidths=[W])
    disc.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), RED_L),
        ("BOX",        (0,0),(-1,-1), 0.8, RED),
        ("TOPPADDING", (0,0),(-1,-1), 9),
        ("BOTTOMPADDING",(0,0),(-1,-1), 9),
        ("LEFTPADDING",(0,0),(-1,-1), 12),
    ]))
    story += [disc, Spacer(1, 7*mm)]

    # 10. Footer
    story.append(divider(HAIRLINE, 0.5))
    story.append(Spacer(1, 3*mm))
    ft = Table([[
        Paragraph("Crop Disease Advisor v2.0", styles["footer_b"]),
        Paragraph("Shubham Haraniya &nbsp;·&nbsp; Vidhan Savaliya", styles["footer"]),
        Paragraph("AI-Powered Diagnostics", styles["footer_b"]),
    ]], colWidths=[W/3, W/3, W/3])
    ft.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story.append(ft)
    story.append(Paragraph(
        "Vision: EfficientNet-B4 &nbsp;·&nbsp; LLM: Qwen2.5-3B QLoRA",
        styles["footer"],
    ))

    doc.build(story)
    return path


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  INFERENCE                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def predict(image: Image.Image, region: str, season: str, severity: str):
    empty = (None, "","","","","","","", None)
    if image is None:
        return empty

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    try:
        resp = requests.post(
            f"{API_URL}/predict",
            files={"file": ("leaf.png", buf, "image/png")},
            data={"region": region, "season": season, "severity": severity},
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        return (None, "❌ **API not reachable** — is the backend running?", "","","","","","", None)
    except Exception as e:
        return (None, f"❌ **Error:** {e}", "","","","","","", None)

    label_dict = {
        p["label"].replace("___"," — ").replace("_"," "): p["confidence"]/100
        for p in data["top5"]
    }

    tp         = data.get("treatment_plan") or {}
    confidence = data["confidence"]
    severity   = data["severity"]
    disease    = data["disease"].replace("___"," — ").replace("_"," ")
    crop_name  = data["crop"]

    sev_key = severity.split()[0] if severity else "Mild"
    sev_hex = {"Severe":"#dc2626","Moderate":"#d97706","Mild":"#059669","None":"#059669"}.get(sev_key,"#2563eb")
    sev_bg  = {"Severe":"#fee2e2","Moderate":"#fef3c7","Mild":"#d1fae5","None":"#d1fae5"}.get(sev_key,"#dbeafe")

    urg = tp.get("action_urgency","—")
    urg_hex = {"Immediate":"#dc2626","Within 3 days":"#ea580c","Within a week":"#d97706","Monitor":"#059669"}.get(urg,"#2563eb")

    diag_html = f"""
<div style="font-family:'Inter',sans-serif;padding:4px 0 12px">

  <!-- Result Banner -->
  <div style="
    background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
    border:1px solid #334155;border-radius:16px;
    padding:20px 24px;margin-bottom:16px;
  ">
    <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#64748b;text-transform:uppercase;margin-bottom:8px">
      AI Diagnosis Complete
    </div>
    <div style="font-size:1.35rem;font-weight:800;color:#f1f5f9;margin-bottom:4px">{disease}</div>
    <div style="font-size:0.9rem;color:#94a3b8">Detected in <strong style="color:#34d399">{crop_name}</strong></div>

    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:16px">
      <span style="background:{sev_bg};color:{sev_hex};border:1.5px solid {sev_hex};
            font-weight:700;font-size:0.78rem;padding:5px 14px;border-radius:20px;letter-spacing:0.05em">
        ⚠&nbsp; Severity: {severity}
      </span>
      <span style="background:#dbeafe;color:#1d4ed8;border:1.5px solid #93c5fd;
            font-weight:700;font-size:0.78rem;padding:5px 14px;border-radius:20px;letter-spacing:0.05em">
        🔬&nbsp; Confidence: {confidence:.1f}%
      </span>
      <span style="background:#0f172a;color:{urg_hex};border:1.5px solid {urg_hex}44;
            font-weight:700;font-size:0.78rem;padding:5px 14px;border-radius:20px;letter-spacing:0.05em">
        ⏱&nbsp; {urg}
      </span>
    </div>
  </div>

  <!-- Stats Row -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:14px 16px">
      <div style="font-size:0.68rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px">Region</div>
      <div style="font-size:0.95rem;font-weight:600;color:#e2e8f0">📍 {region}</div>
    </div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:14px 16px">
      <div style="font-size:0.68rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px">Season</div>
      <div style="font-size:0.95rem;font-weight:600;color:#e2e8f0">🗓 {season}</div>
    </div>
  </div>
</div>"""

    organic_md = "\n\n---\n\n".join(
        f"**🌿 {t.get('method','')}**\n\n"
        f"**How:** {t.get('application','')}\n\n"
        f"**When:** _{t.get('frequency','')}_"
        for t in tp.get("organic_treatments",[])
    ) or "_No organic treatments recommended._"

    chemical_md = "\n\n---\n\n".join(
        f"**💊 {t.get('product','')}**\n\n"
        f"**Dosage:** `{t.get('dosage','')}`\n\n"
        f"**⚠ Safety:** _{t.get('safety_note','')}_"
        for t in tp.get("chemical_treatments",[])
    ) or "_No chemical treatments recommended._"

    preventive_md = "\n".join(
        f"- ✅ {m}" for m in tp.get("preventive_measures",[])
    ) or "_No preventive measures listed._"

    yield_impact = tp.get("yield_impact_estimate","—")
    urgency      = tp.get("action_urgency","—")
    notes_md = (
        f"### 🌍 Regional Notes\n{tp.get('regional_notes','—')}\n\n"
        f"### 🌦 Seasonal Notes\n{tp.get('seasonal_notes','—')}"
    )

    pdf_path = generate_pdf(dict(
        disease=data["disease"], crop=crop_name,
        confidence=confidence, severity=severity,
        region=region, season=season,
        treatment_plan=tp,
    ))

    return (
        label_dict, diag_html,
        organic_md, chemical_md, preventive_md,
        yield_impact, urgency, notes_md,
        pdf_path,
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  EXAMPLES                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
EXAMPLES = [
    ["notebooks/sample_augmentations/sample_aug_00.png","North India","Kharif (Monsoon)","Moderate"],
    ["notebooks/sample_augmentations/sample_aug_01.png","South India","Rabi (Winter)","Moderate"],
    ["notebooks/sample_augmentations/sample_aug_02.png","East India","Zaid (Summer)","Moderate"],
]
EXAMPLES = [e for e in EXAMPLES if Path(e[0]).exists()]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CSS                                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

/* ── Foundation ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #060d1f !important;
    color: #cbd5e1 !important;
}

/* ── Subtle grid background ─────────────────────────────────── */
.gradio-container {
    background-image:
        linear-gradient(rgba(16,185,129,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(16,185,129,0.03) 1px, transparent 1px) !important;
    background-size: 40px 40px !important;
}

/* ── Scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 99px; }

/* ── Gradio containers ──────────────────────────────────────── */
.gr-block, .gr-form, .gr-panel,
.gradio-container .block,
div[data-testid="block"] {
    background: #0d1526 !important;
    border: 1px solid #1a2744 !important;
    border-radius: 14px !important;
}

/* ── Labels ─────────────────────────────────────────────────── */
label, .gr-label, .label-wrap span {
    color: #475569 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Inputs ─────────────────────────────────────────────────── */
input, textarea, select {
    background: #070e20 !important;
    border: 1px solid #1a2744 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color .15s, box-shadow .15s !important;
}
input:focus, textarea:focus, select:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,.12) !important;
    outline: none !important;
}

/* ── Dropdowns ──────────────────────────────────────────────── */
.gradio-dropdown, ul[role="listbox"] {
    background: #0d1526 !important;
    border-color: #1a2744 !important;
}
li[role="option"]:hover { background: #10b98115 !important; color: #10b981 !important; }
li[role="option"][aria-selected="true"] { background: #10b98120 !important; color: #10b981 !important; }

/* ── Hero ───────────────────────────────────────────────────── */
#hero {
    background: linear-gradient(180deg, #0a2215 0%, #071a10 100%);
    border: 1px solid #10b98118;
    border-radius: 18px;
    padding: 36px 32px 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 4px;
}
#hero::before {
    content: '';
    position: absolute; top: -60px; left: 50%; transform: translateX(-50%);
    width: 500px; height: 200px;
    background: radial-gradient(ellipse, #10b98118 0%, transparent 70%);
    pointer-events: none;
}
#hero h1 {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    color: #fff !important;
    letter-spacing: -0.5px;
    margin: 0 0 8px !important;
}
#hero h1 span {
    background: linear-gradient(135deg, #10b981, #34d399, #6ee7b7);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
#hero p { font-size: 0.95rem !important; color: #94a3b8 !important; margin: 0 0 18px !important; }
.hero-pills { display: flex; flex-wrap: wrap; justify-content: center; gap: 7px; }
.pill {
    background: rgba(16,185,129,.08);
    border: 1px solid rgba(16,185,129,.2);
    color: #6ee7b7;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 13px;
    border-radius: 99px;
    letter-spacing: 0.04em;
}

/* ── Panel cards ────────────────────────────────────────────── */
.panel-card {
    background: #0d1526;
    border: 1px solid #1a2744;
    border-radius: 16px;
    padding: 0;
    overflow: hidden;
}
.panel-header {
    background: linear-gradient(90deg, #071a10, #0a2215);
    border-bottom: 1px solid #10b98120;
    padding: 12px 18px;
}
.panel-header-title {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #10b981;
    text-transform: uppercase;
}
.panel-body { padding: 18px; }

/* ── Analyze button ─────────────────────────────────────────── */
#analyze-btn {
    background: linear-gradient(135deg, #065f46, #059669, #10b981) !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 13px !important;
    width: 100% !important;
    cursor: pointer !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 24px rgba(16,185,129,.25), 0 0 0 0 rgba(16,185,129,.2) !important;
    transition: all 0.2s cubic-bezier(.4,0,.2,1) !important;
    position: relative !important;
    overflow: hidden !important;
}
#analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(16,185,129,.4) !important;
}
#analyze-btn:active { transform: translateY(0) !important; }

/* ── Section label bars ─────────────────────────────────────── */
.sec-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #10b981;
    padding: 8px 0 6px;
    border-top: 1px solid #1a2744;
    margin-top: 20px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-label::after { content: ''; flex: 1; height: 1px; background: #1a2744; }

/* ── Tabs ───────────────────────────────────────────────────── */
.gr-tabs .tab-nav { background: transparent !important; border-bottom: 1px solid #1a2744 !important; }
.gr-tabs .tab-nav button {
    background: transparent !important;
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 10px 18px !important;
    transition: all .15s !important;
}
.gr-tabs .tab-nav button:hover { color: #94a3b8 !important; }
.gr-tabs .tab-nav button.selected {
    color: #10b981 !important;
    border-bottom-color: #10b981 !important;
    background: transparent !important;
}
.gr-tabs .tabitem {
    border: none !important;
    background: transparent !important;
    padding: 14px 4px !important;
}

/* ── Markdown ───────────────────────────────────────────────── */
.gr-markdown, .prose {
    color: #94a3b8 !important;
    line-height: 1.75 !important;
    font-size: 0.9rem !important;
}
.gr-markdown strong, .prose strong { color: #e2e8f0 !important; font-weight: 600 !important; }
.gr-markdown h3, .prose h3 {
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    margin: 1rem 0 0.4rem !important;
    letter-spacing: 0.02em !important;
}
.gr-markdown hr { border-color: #1a2744 !important; margin: 14px 0 !important; }
.gr-markdown code {
    background: #0f172a !important;
    color: #34d399 !important;
    padding: 2px 7px !important;
    border-radius: 5px !important;
    font-size: 0.85em !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}
.gr-markdown ul li { color: #94a3b8 !important; }
.gr-markdown ul li::marker { color: #10b981 !important; }
.gr-markdown em { color: #64748b !important; }

/* ── Label bar chart ────────────────────────────────────────── */
.label-container { background: #0d1526 !important; border-radius: 8px !important; padding: 4px !important; }
.label-container .bar { background: linear-gradient(90deg, #065f46, #10b981) !important; border-radius: 3px !important; }
.label-container .label { color: #cbd5e1 !important; font-size: 0.82rem !important; }
.label-container .value { color: #10b981 !important; font-weight: 700 !important; }

/* ── Textboxes ──────────────────────────────────────────────── */
.gr-textbox textarea, .gr-textbox input {
    background: #070e20 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1a2744 !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
}

/* ── Accordion ──────────────────────────────────────────────── */
details summary, .gr-accordion summary {
    background: #0d1526 !important;
    border: 1px solid #1a2744 !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    padding: 11px 16px !important;
    transition: all .15s !important;
}
details summary:hover, .gr-accordion summary:hover {
    border-color: #10b98140 !important;
    color: #e2e8f0 !important;
}
details .inner, .gr-accordion .inner {
    background: #070e20 !important;
    border: 1px solid #1a2744 !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 14px !important;
}

/* ── File widget ────────────────────────────────────────────── */
.gr-file-preview, .file-preview {
    background: #0d1526 !important;
    border: 1px dashed #1a2744 !important;
    border-radius: 12px !important;
    transition: all .2s !important;
}
.gr-file-preview:hover {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,.07) !important;
}

/* ── Image upload ───────────────────────────────────────────── */
.gr-image, [data-testid="image"] {
    border: 1px solid #1a2744 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    background: #070e20 !important;
}

/* ── Footer ─────────────────────────────────────────────────── */
#footer {
    text-align: center;
    padding: 16px 0 8px;
    color: #1e293b;
    font-size: 0.75rem;
    border-top: 1px solid #1a2744;
    margin-top: 24px;
}
#footer a { color: #10b981 !important; text-decoration: none; }
#footer a:hover { text-decoration: underline; }
"""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  UI LAYOUT                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with gr.Blocks(
    title="🌾 Crop Disease Advisor",
    theme=gr.themes.Base(
        primary_hue="green",
        secondary_hue="emerald",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
) as demo:

    gr.HTML(f"<style>{CSS}</style>")

    # ── Hero ──────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="hero">
      <h1>🌾 Crop Disease <span>Advisor</span></h1>
      <p>AI-powered plant disease diagnosis &amp; personalized treatment planning for Indian farmers</p>
      <div class="hero-pills">
        <span class="pill">🤖 EfficientNet-B4</span>
        <span class="pill">🧠 Qwen2.5-3B QLoRA</span>
        <span class="pill">📍 Region-Aware</span>
        <span class="pill">🗓 Season-Aware</span>
        <span class="pill">🇮🇳 Indian Farming</span>
      </div>
    </div>
    """)

    # ── Body Layout ───────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── Left: Inputs (narrow) ───────────────────────────────────────────
        with gr.Column(scale=3, min_width=300):

            gr.HTML('<div class="sec-label">📷 Upload</div>')
            input_image = gr.Image(
                type="pil", label="Leaf Photo",
                sources=["upload", "webcam"], height=280,
            )

            gr.HTML('<div class="sec-label">⚙ Configuration</div>')
            region_dd = gr.Dropdown(
                choices=REGIONS, value="North India",
                label="Region",
            )
            season_dd = gr.Dropdown(
                choices=SEASONS, value="Kharif (Monsoon)",
                label="Season",
            )
            severity_dd = gr.Dropdown(
                choices=["Mild","Moderate","Severe"], value="Moderate",
                label="Observed Severity",
            )

            analyze_btn = gr.Button(
                "🔍  Analyze Disease",
                variant="primary", elem_id="analyze-btn",
            )

            if EXAMPLES:
                with gr.Accordion("📂 Example Images", open=False):
                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=[input_image, region_dd, season_dd, severity_dd],
                    )

        # ── Right: Results (wide) ───────────────────────────────────────────
        with gr.Column(scale=7, min_width=480):

            # Diagnosis card
            gr.HTML('<div class="sec-label">🦠 Diagnosis</div>')
            diag_out = gr.HTML(
                "<div style='color:#1e293b;font-style:italic;padding:20px 0;text-align:center;font-size:0.88rem'>"
                "Upload a leaf image and click Analyze to begin...</div>"
            )

            # Top-5 predictions
            gr.HTML('<div class="sec-label">📊 Top-5 Predictions</div>')
            label_out = gr.Label(num_top_classes=5, label="")

            # Treatment tabs
            gr.HTML('<div class="sec-label">💊 Treatment Plan</div>')
            with gr.Tabs():
                with gr.Tab("🌿 Organic"):
                    organic_out = gr.Markdown("_Run analysis to see recommendations..._")
                with gr.Tab("⚗️ Chemical"):
                    chemical_out = gr.Markdown("_Run analysis to see recommendations..._")
                with gr.Tab("🛡️ Preventive"):
                    preventive_out = gr.Markdown("_Run analysis to see recommendations..._")
                with gr.Tab("📋 Advisory"):
                    notes_out = gr.Markdown("_Regional and seasonal advisory will appear here._")

            # Impact row
            gr.HTML('<div class="sec-label">📈 Impact & Urgency</div>')
            with gr.Row():
                yield_out = gr.Textbox(
                    label="Yield Impact Estimate",
                    interactive=False, placeholder="Awaiting analysis...", lines=2,
                )
                urgency_out = gr.Textbox(
                    label="Action Urgency",
                    interactive=False, placeholder="Awaiting analysis...",
                )

            # Download
            gr.HTML('<div class="sec-label">📄 Report</div>')
            pdf_out = gr.File(label="Download Clinical PDF Report")

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="footer">
      Crop Disease Advisor v2.0 &nbsp;·&nbsp;
      Shubham Haraniya &amp; Vidhan Savaliya
    </div>
    """)

    # ── Wiring ────────────────────────────────────────────────────────────────
    analyze_btn.click(
        fn=predict,
        inputs=[input_image, region_dd, season_dd, severity_dd],
        outputs=[
            label_out, diag_out,
            organic_out, chemical_out, preventive_out,
            yield_out, urgency_out, notes_out,
            pdf_out,
        ],
        show_progress="full",
        api_name="predict",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
