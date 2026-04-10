"""
Crop Disease Advisor — Streamlit Frontend
Premium dark dashboard, replaces the Gradio UI entirely.
"""

import io, os, base64, requests, tempfile
from pathlib import Path
from datetime import datetime
from PIL import Image

import streamlit as st

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Crop Disease Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
REGIONS = ["North India", "South India", "East India", "West India", "Central India"]
SEASONS = ["Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)"]
SEVERITIES = ["Mild", "Moderate", "Severe"]


# ═══════════════════════════════════════════════════════════════════════════
# THEME & CSS
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global reset ─── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: #03091a;
}

/* ── Sidebar ─── */
section[data-testid="stSidebar"] {
    background: #060e22 !important;
    border-right: 1px solid #0f2040 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stFileUploader label,
section[data-testid="stSidebar"] p {
    color: #64748b !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Sidebar selectbox ─── */
div[data-baseweb="select"] > div {
    background: #060e22 !important;
    border-color: #0f2040 !important;
    color: #e2e8f0 !important;
}

/* ── File uploader ─── */
div[data-testid="stFileUploadDropzone"] {
    background: #060e22 !important;
    border: 1.5px dashed #0f2040 !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
}
div[data-testid="stFileUploadDropzone"]:hover {
    border-color: #10b981 !important;
}

/* ── Buttons ─── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #065f46, #059669, #10b981) !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    box-shadow: 0 4px 20px rgba(16,185,129,.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(16,185,129,.45) !important;
}

/* ── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    background: #060e22 !important;
    border-bottom: 1px solid #0f2040 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 20px !important;
}
.stTabs [aria-selected="true"] {
    color: #10b981 !important;
    border-bottom-color: #10b981 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #060e22 !important;
    border: 1px solid #0f2040 !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 1.2rem !important;
}

/* ── Metrics ─── */
div[data-testid="metric-container"] {
    background: #060e22;
    border: 1px solid #0f2040;
    border-radius: 12px;
    padding: 1rem;
}
div[data-testid="metric-container"] label {
    color: #475569 !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
}

/* ── Dividers ─── */
hr { border-color: #0f2040 !important; }

/* ── Markdown text ─── */
.stMarkdown p, .stMarkdown li, .stText {
    color: #94a3b8 !important;
    line-height: 1.75 !important;
}
.stMarkdown strong { color: #e2e8f0 !important; }
.stMarkdown h3 { color: #cbd5e1 !important; font-size: 0.9rem !important; }
.stMarkdown code {
    background: #0f172a !important;
    color: #34d399 !important;
    padding: 2px 8px !important;
    border-radius: 5px !important;
    font-size: 0.85em !important;
}

/* ── Download button ─── */
.stDownloadButton > button {
    width: 100% !important;
    background: #060e22 !important;
    color: #10b981 !important;
    border: 1.5px solid #10b981 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: #10b981 !important;
    color: #fff !important;
}

/* ── Spinner ─── */
.stSpinner > div { border-top-color: #10b981 !important; }

/* ── Image ─── */
img { border-radius: 12px; }

/* ── Scrollbar ─── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #0f2040; border-radius: 99px; }

/* ── Expanders ─── */
.streamlit-expanderHeader {
    background: #060e22 !important;
    border: 1px solid #0f2040 !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent {
    background: #03091a !important;
    border: 1px solid #0f2040 !important;
    border-top: none !important;
}
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════
# PDF GENERATOR  (identical to Gradio version)
# ═══════════════════════════════════════════════════════════════════════════

def generate_pdf(result: dict) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    W = 170*mm
    doc = SimpleDocTemplate(path, pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=14*mm, bottomMargin=18*mm)

    INK=colors.HexColor("#0f172a"); SLATE=colors.HexColor("#334155")
    MUTED=colors.HexColor("#64748b"); HAIRLINE=colors.HexColor("#e2e8f0")
    SURFACE=colors.HexColor("#f8fafc"); SURFACE2=colors.HexColor("#f1f5f9")
    EMERALD=colors.HexColor("#059669"); EMERALD_L=colors.HexColor("#d1fae5")
    TEAL=colors.HexColor("#0d9488"); AMBER=colors.HexColor("#d97706")
    AMBER_L=colors.HexColor("#fef3c7"); RED=colors.HexColor("#dc2626")
    RED_L=colors.HexColor("#fee2e2"); VIOLET=colors.HexColor("#7c3aed")
    VIOLET_L=colors.HexColor("#ede9fe"); BLUE_L=colors.HexColor("#dbeafe")
    ORANGE=colors.HexColor("#ea580c"); WHITE=colors.white

    severity_str = result.get("severity","Mild")
    sev_key = severity_str.split()[0]
    sev_color={
        "Severe":RED,"Moderate":AMBER,"Mild":EMERALD,"None":EMERALD
    }.get(sev_key,TEAL)

    tp = result.get("treatment_plan") or {}
    urgency_str = tp.get("action_urgency","—")
    urg_color={
        "Immediate":RED,"Within 3 days":ORANGE,"Within a week":AMBER,"Monitor":EMERALD
    }.get(urgency_str,colors.HexColor("#2563eb"))

    ss=getSampleStyleSheet()
    def S(n,**k): return ParagraphStyle(n,parent=ss["Normal"],**k)

    styles=dict(
        ct=S("ct",fontSize=22,fontName="Helvetica-Bold",textColor=WHITE,alignment=1,spaceAfter=2),
        cs=S("cs",fontSize=10,fontName="Helvetica",textColor=colors.HexColor("#bbf7d0"),alignment=1),
        cm=S("cm",fontSize=8,fontName="Helvetica",textColor=colors.HexColor("#6ee7b7"),alignment=1),
        st=S("st",fontSize=11,fontName="Helvetica-Bold",textColor=WHITE,leading=16),
        lb=S("lb",fontSize=7.5,fontName="Helvetica-Bold",textColor=MUTED,spaceAfter=1,leading=10),
        vl=S("vl",fontSize=10,fontName="Helvetica-Bold",textColor=INK,spaceAfter=2),
        bd=S("bd",fontSize=9.5,fontName="Helvetica",textColor=SLATE,leading=14,alignment=4),
        bu=S("bu",fontSize=9.5,fontName="Helvetica",textColor=SLATE,leading=14,leftIndent=5*mm),
        th=S("th",fontSize=9,fontName="Helvetica-Bold",textColor=WHITE),
        td=S("td",fontSize=9,fontName="Helvetica",textColor=SLATE,leading=12),
        ft=S("ft",fontSize=7.5,fontName="Helvetica",textColor=MUTED,alignment=1),
        fb=S("fb",fontSize=7.5,fontName="Helvetica-Bold",textColor=SLATE,alignment=1),
        bg=S("bg",fontSize=9,fontName="Helvetica-Bold",textColor=WHITE,alignment=1),
        di=S("di",fontSize=8.5,fontName="Helvetica-Oblique",textColor=SLATE,leading=12,alignment=4),
    )

    def divider(c=None):
        return HRFlowable(width=W, thickness=0.5, color=c or HAIRLINE)

    def sec_bar(icon, title, bg=None):
        t=Table([[Paragraph(f"{icon}  {title}",styles["st"])]],colWidths=[W])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),bg or EMERALD),
            ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
            ("LEFTPADDING",(0,0),(-1,-1),14),
        ]))
        return t

    def meta_grid(items, n=4):
        cw=W/n
        hdr=[Paragraph(k.upper(),styles["lb"]) for k,_ in items]
        val=[Paragraph(str(v),styles["vl"]) for _,v in items]
        t=Table([hdr,val],colWidths=[cw]*n)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),SURFACE2),("BACKGROUND",(0,1),(-1,1),SURFACE),
            ("BOX",(0,0),(-1,-1),0.5,HAIRLINE),("INNERGRID",(0,0),(-1,-1),0.3,HAIRLINE),
            ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
            ("LEFTPADDING",(0,0),(-1,-1),10),
        ]))
        return t

    def treat_tbl(rows,headers,widths,hbg=None):
        data=[[Paragraph(h,styles["th"]) for h in headers]]
        for row in rows: data.append([Paragraph(str(c),styles["td"]) for c in row])
        t=Table(data,colWidths=widths,repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),hbg or EMERALD),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[SURFACE,SURFACE2]),
            ("BOX",(0,0),(-1,-1),0.5,HAIRLINE),("INNERGRID",(0,0),(-1,-1),0.25,HAIRLINE),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),9),("VALIGN",(0,0),(-1,-1),"TOP"),
        ]))
        return t

    now=datetime.now()
    story=[]
    conf_val=result.get("confidence",0)
    disease_clean=result.get("disease","N/A").replace("___"," — ").replace("_"," ")

    # Banner
    banner=Table([
        [Paragraph("🌾  CROP DISEASE ADVISOR",styles["ct"])],
        [Paragraph("Clinical Agronomic Diagnosis &amp; Treatment Report",styles["cs"])],
        [Paragraph(f"CDA-{now.strftime('%Y%m%d-%H%M%S')} · {now.strftime('%d %b %Y, %I:%M %p')}",styles["cm"])],
    ],colWidths=[W])
    banner.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#064e3b")),
        ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("LEFTPADDING",(0,0),(-1,-1),16),("RIGHTPADDING",(0,0),(-1,-1),16),
    ]))
    story+=[banner,Spacer(1,5*mm)]

    story+=[sec_bar("📋","Diagnosis Summary"),Spacer(1,2*mm)]
    story.append(meta_grid([
        ("Disease Detected",disease_clean),("Crop Species",result.get("crop","N/A")),
        ("AI Confidence",f"{conf_val:.1f}%"),("Severity Level",severity_str),
    ]))
    story+=[Spacer(1,2*mm)]
    story.append(meta_grid([
        ("Region",result.get("region","—")),("Season",result.get("season","—")),
        ("Action Urgency",urgency_str),("Status","COMPLETED"),
    ]))
    story+=[Spacer(1,4*mm)]

    bt=Table([[
        Paragraph(f"● SEVERITY: {severity_str.upper()}",styles["bg"]),
        Paragraph(f"⏱  ACTION: {urgency_str.upper()}",styles["bg"]),
    ]],colWidths=[W/2,W/2])
    bt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1),sev_color),("BACKGROUND",(1,0),(1,-1),urg_color),
        ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
        ("INNERGRID",(0,0),(-1,-1),2,WHITE),
    ]))
    story+=[bt,Spacer(1,6*mm)]

    organics=tp.get("organic_treatments",[])
    if organics:
        story.append(KeepTogether([
            sec_bar("🌿","Organic / Biological Treatments",TEAL),Spacer(1,2*mm),
            treat_tbl([[t.get("method",""),t.get("application",""),t.get("frequency","")] for t in organics],
                ["Method / Agent","Application Instructions","Frequency"],[44*mm,90*mm,36*mm],hbg=TEAL),
            Spacer(1,5*mm),
        ]))

    chemicals=tp.get("chemical_treatments",[])
    if chemicals:
        story.append(KeepTogether([
            sec_bar("⚗️","Chemical Treatments",RED),Spacer(1,2*mm),
            treat_tbl([[t.get("product",""),t.get("dosage",""),t.get("safety_note","")] for t in chemicals],
                ["Product / Fungicide","Dosage &amp; Dilution","Safety Note"],[52*mm,44*mm,74*mm],hbg=RED),
            Spacer(1,5*mm),
        ]))

    preventive=tp.get("preventive_measures",[])
    if preventive:
        prev_t=Table([[Paragraph(f"✓  {m}",styles["bu"])] for m in preventive],colWidths=[W])
        prev_t.setStyle(TableStyle([
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[SURFACE,VIOLET_L]),
            ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#c4b5fd")),
            ("INNERGRID",(0,0),(-1,-1),0.25,HAIRLINE),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),10),
        ]))
        story.append(KeepTogether([sec_bar("🛡️","Preventive Measures",VIOLET),Spacer(1,2*mm),prev_t,Spacer(1,5*mm)]))

    yield_val=tp.get("yield_impact_estimate","")
    if yield_val:
        yt=Table([[Paragraph(yield_val,styles["bd"])]],colWidths=[W])
        yt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),AMBER_L),("BOX",(0,0),(-1,-1),0.6,AMBER),
            ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),("LEFTPADDING",(0,0),(-1,-1),12),
        ]))
        story.append(KeepTogether([sec_bar("📉","Estimated Yield Impact",AMBER),Spacer(1,2*mm),yt,Spacer(1,5*mm)]))

    regional=tp.get("regional_notes",""); seasonal=tp.get("seasonal_notes","")
    if regional or seasonal:
        adv=Table([
            [Paragraph("🌍  Regional Notes",styles["st"]),Paragraph("🌦  Seasonal Notes",styles["st"])],
            [Paragraph(regional or "—",styles["bd"]),Paragraph(seasonal or "—",styles["bd"])],
        ],colWidths=[W/2,W/2])
        adv.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1e3a5f")),
            ("BACKGROUND",(0,1),(-1,1),BLUE_L),
            ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#93c5fd")),
            ("INNERGRID",(0,0),(-1,-1),0.3,HAIRLINE),
            ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
            ("LEFTPADDING",(0,0),(-1,-1),10),("VALIGN",(0,0),(-1,-1),"TOP"),
        ]))
        story+=[adv,Spacer(1,6*mm)]

    disc=Table([[Paragraph(
        "⚠ DISCLAIMER: AI-generated advisory only. Consult a licensed agronomist before applying any treatment.",
        styles["di"],
    )]],colWidths=[W])
    disc.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),RED_L),("BOX",(0,0),(-1,-1),0.8,RED),
        ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),("LEFTPADDING",(0,0),(-1,-1),12),
    ]))
    story+=[disc,Spacer(1,7*mm),divider(HAIRLINE),Spacer(1,3*mm)]

    ft=Table([[
        Paragraph("Crop Disease Advisor v2.0",styles["fb"]),
        Paragraph("Shubham Haraniya · Vidhan Savaliya",styles["ft"]),
        Paragraph("AI-Powered Diagnostics",styles["fb"]),
    ]],colWidths=[W/3,W/3,W/3])
    ft.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story.append(ft)

    doc.build(story)
    with open(path,"rb") as f: data=f.read()
    os.unlink(path)
    return data

# Handle missing VIOLET_L in pdf scope
try:
    from reportlab.lib import colors as _rc
    VIOLET_L = _rc.HexColor("#ede9fe")
    VIOLET    = _rc.HexColor("#7c3aed")
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def call_api(image: Image.Image, region: str, season: str, severity: str) -> dict | None:
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
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach backend API. Make sure `python app.py` is running.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None


def sev_badge_html(severity: str, confidence: float) -> str:
    sev_key = severity.split()[0]
    col = {"Severe":"#dc2626","Moderate":"#d97706","Mild":"#059669","None":"#059669"}.get(sev_key,"#2563eb")
    bg  = {"Severe":"#fee2e220","Moderate":"#fef3c720","Mild":"#d1fae520","None":"#d1fae520"}.get(sev_key,"#dbeafe20")
    return (
        f'<span style="background:{bg};color:{col};border:1.5px solid {col};'
        f'font-weight:700;font-size:0.78rem;padding:4px 14px;border-radius:99px;'
        f'letter-spacing:0.05em">⚠ {severity}</span>&nbsp;&nbsp;'
        f'<span style="background:#1d4ed820;color:#60a5fa;border:1.5px solid #3b82f6;'
        f'font-weight:700;font-size:0.78rem;padding:4px 14px;border-radius:99px;'
        f'letter-spacing:0.05em">🔬 {confidence:.1f}% Confidence</span>'
    )


def stat_card(col, label: str, value: str, icon: str = ""):
    col.markdown(
        f"""<div style="background:#060e22;border:1px solid #0f2040;border-radius:13px;
        padding:16px 18px;text-align:center">
        <div style="font-size:1.4rem;margin-bottom:6px">{icon}</div>
        <div style="font-size:0.68rem;font-weight:700;color:#475569;text-transform:uppercase;
        letter-spacing:0.08em;margin-bottom:4px">{label}</div>
        <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    # Logo / brand
    st.markdown("""
    <div style="text-align:center;padding:20px 0 24px">
      <div style="font-size:2.2rem;margin-bottom:6px">🌾</div>
      <div style="font-size:1.1rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px">Crop Disease Advisor</div>
      <div style="font-size:0.72rem;color:#475569;margin-top:4px">Plant Health Monitoring System</div>
    </div>
    <hr style="border-color:#0f2040;margin:0 0 20px">
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "📷 Upload Leaf Photo",
        type=["jpg","jpeg","png","webp"],
        help="JPG, PNG or WebP — up to 10 MB",
    )

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    region   = st.selectbox("🌍 Region",   REGIONS,   index=0)
    season   = st.selectbox("🌦 Season",   SEASONS,   index=0)
    severity = st.selectbox("⚠ Observed Severity", SEVERITIES, index=1)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    analyze_btn = st.button("🔍  Analyze Disease", use_container_width=True)

    st.markdown("""
    <hr style="border-color:#0f2040;margin:20px 0 14px">
    <div style="font-size:0.68rem;color:#1e293b;text-align:center;line-height:1.8">
      Vision: EfficientNet-B4<br>
      LLM: Qwen2.5-3B QLoRA<br>
      <span style="color:#10b981">38 PlantVillage Classes</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
  background:linear-gradient(135deg,#062a17 0%,#083a20 50%,#062a17 100%);
  border:1px solid #10b98120;border-radius:18px;
  padding:32px 36px 26px;margin-bottom:28px;
  position:relative;overflow:hidden
">
  <div style="
    position:absolute;top:-40px;left:30%;
    width:400px;height:200px;
    background:radial-gradient(ellipse,#10b98112 0%,transparent 70%);
    pointer-events:none
  "></div>
  <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.12em;color:#10b981;
       text-transform:uppercase;margin-bottom:10px">AI-Powered Agricultural Diagnostics</div>
  <div style="font-size:2rem;font-weight:800;color:#fff;letter-spacing:-0.5px;margin-bottom:8px">
    Plant Disease Detection &amp; Treatment Planning
  </div>
  <div style="font-size:0.92rem;color:#64748b;max-width:600px">
    Upload a leaf photo, select your region and season, and get an instant AI diagnosis plus a
    personalised treatment plan — powered by EfficientNet-B4 and Qwen2.5-3B QLoRA.
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:18px">
    <span style="background:#10b98110;border:1px solid #10b98130;color:#6ee7b7;
          font-size:0.72rem;font-weight:600;padding:4px 12px;border-radius:99px">🤖 EfficientNet-B4</span>
    <span style="background:#10b98110;border:1px solid #10b98130;color:#6ee7b7;
          font-size:0.72rem;font-weight:600;padding:4px 12px;border-radius:99px">🧠 Qwen2.5-3B QLoRA</span>
    <span style="background:#10b98110;border:1px solid #10b98130;color:#6ee7b7;
          font-size:0.72rem;font-weight:600;padding:4px 12px;border-radius:99px">📍 Region & Season Aware</span>
    <span style="background:#10b98110;border:1px solid #10b98130;color:#6ee7b7;
          font-size:0.72rem;font-weight:600;padding:4px 12px;border-radius:99px">🇮🇳 Built for Indian Farmers</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── State ─────────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "image"  not in st.session_state:
    st.session_state.image  = None


# ── Run analysis ──────────────────────────────────────────────────────────
if analyze_btn and uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.session_state.image = image
    with st.spinner("🔬 Running AI diagnosis — please wait…"):
        st.session_state.result = call_api(image, region, season, severity)

elif analyze_btn and not uploaded:
    st.warning("📷 Please upload a leaf image first.")


# ── Show image preview if uploaded ───────────────────────────────────────
if uploaded and st.session_state.image is None:
    st.session_state.image = Image.open(uploaded).convert("RGB")


# ── Results ───────────────────────────────────────────────────────────────
result = st.session_state.result

if result is None:
    # Empty state
    left, right = st.columns([1, 1])
    with left:
        if st.session_state.image:
            st.image(st.session_state.image, caption="Uploaded Leaf", use_container_width=True)
        else:
            st.markdown("""
            <div style="background:#060e22;border:1.5px dashed #0f2040;border-radius:16px;
            padding:60px 20px;text-align:center;color:#1e293b">
              <div style="font-size:3rem;margin-bottom:12px">🌿</div>
              <div style="font-size:0.88rem;font-weight:600">Upload a leaf photo from the sidebar to begin</div>
            </div>
            """, unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div style="background:#060e22;border:1px solid #0f2040;border-radius:16px;
        padding:40px 28px;text-align:center">
          <div style="font-size:2.5rem;margin-bottom:16px">🦠</div>
          <div style="font-size:0.95rem;font-weight:700;color:#1e293b;margin-bottom:8px">
            Diagnosis results will appear here
          </div>
          <div style="font-size:0.82rem;color:#1a2744">
            Configure region and season in the sidebar, then click Analyze.
          </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Layout ──────────────────────────────────────────────────────────────
    img_col, res_col = st.columns([4, 6], gap="large")

    with img_col:
        st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#10b981;text-transform:uppercase;margin-bottom:8px">📷 Analyzed Leaf</p>', unsafe_allow_html=True)
        if st.session_state.image:
            st.image(st.session_state.image, use_container_width=True)

        # Confidence bar chart
        st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#10b981;text-transform:uppercase;margin:16px 0 8px">📊 Top-5 Predictions</p>', unsafe_allow_html=True)
        for p in result.get("top5", []):
            label = p["label"].replace("___", " — ").replace("_", " ")
            conf  = p["confidence"]
            is_top = (p == result["top5"][0])
            bar_col = "#10b981" if is_top else "#1e3a5f"
            st.markdown(
                f'<div style="margin-bottom:6px">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.78rem;color:{"#e2e8f0" if is_top else "#475569"};'
                f'font-weight:{"700" if is_top else "500"};margin-bottom:3px">'
                f'<span>{label}</span><span>{conf:.1f}%</span></div>'
                f'<div style="height:5px;background:#0f2040;border-radius:99px;overflow:hidden">'
                f'<div style="height:100%;width:{conf}%;background:{bar_col};'
                f'border-radius:99px;transition:width 0.4s"></div></div></div>',
                unsafe_allow_html=True,
            )

    with res_col:
        disease_clean = result["disease"].replace("___"," — ").replace("_"," ")
        tp = result.get("treatment_plan") or {}
        confidence = result["confidence"]
        detected_severity = result["severity"]
        crop_name = result["crop"]
        urgency = tp.get("action_urgency","—")

        # ── Diagnosis card ────────────────────────────────────────────────
        st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#10b981;text-transform:uppercase;margin-bottom:8px">🦠 Diagnosis</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#060e22;border:1px solid #0f2040;border-radius:16px;padding:22px 24px;margin-bottom:16px">
          <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;color:#475569;text-transform:uppercase;margin-bottom:6px">Disease Identified</div>
          <div style="font-size:1.3rem;font-weight:800;color:#f1f5f9;margin-bottom:4px">{disease_clean}</div>
          <div style="font-size:0.88rem;color:#64748b;margin-bottom:16px">Detected in <strong style="color:#34d399">{crop_name}</strong></div>
          <div style="display:flex;gap:8px;flex-wrap:wrap">
            {sev_badge_html(detected_severity, confidence)}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 4-stat row ────────────────────────────────────────────────────
        s1, s2, s3, s4 = st.columns(4)
        stat_card(s1, "Region", region, "📍")
        stat_card(s2, "Season", season.split()[0], "🗓")
        stat_card(s3, "Severity", detected_severity, "⚠")
        stat_card(s4, "Urgency", urgency, "⏱")

    # ── Treatment Tabs ───────────────────────────────────────────────────
    st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#10b981;text-transform:uppercase;margin:24px 0 10px">💊 AI Treatment Plan</p>', unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["🌿 Organic", "⚗️ Chemical", "🛡️ Preventive", "📋 Advisory"])

    with t1:
        organics = tp.get("organic_treatments", [])
        if organics:
            for t in organics:
                with st.expander(f"🌿 {t.get('method','')}", expanded=True):
                    st.markdown(f"**How to apply:** {t.get('application','')}")
                    st.markdown(f"**Frequency:** _{t.get('frequency','')}_")
        else:
            st.info("No organic treatments listed.")

    with t2:
        chemicals = tp.get("chemical_treatments", [])
        if chemicals:
            for t in chemicals:
                with st.expander(f"💊 {t.get('product','')}", expanded=True):
                    st.markdown(f"**Dosage:** `{t.get('dosage','')}`")
                    st.warning(f"⚠ Safety: {t.get('safety_note','')}", icon="⚠️")
        else:
            st.info("No chemical treatments listed.")

    with t3:
        preventive = tp.get("preventive_measures", [])
        if preventive:
            for m in preventive:
                st.markdown(f"- ✅ {m}")
        else:
            st.info("No preventive measures listed.")

    with t4:
        col_r, col_s = st.columns(2)
        with col_r:
            st.markdown("**🌍 Regional Notes**")
            st.markdown(tp.get("regional_notes","—"))
        with col_s:
            st.markdown("**🌦 Seasonal Notes**")
            st.markdown(tp.get("seasonal_notes","—"))
        if tp.get("yield_impact_estimate"):
            st.markdown("---")
            st.markdown(f"**📉 Yield Impact:** {tp['yield_impact_estimate']}")

    # ── PDF Download ─────────────────────────────────────────────────────
    st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#10b981;text-transform:uppercase;margin:24px 0 10px">📄 Download Report</p>', unsafe_allow_html=True)
    col_dl, col_info = st.columns([3, 7])
    with col_dl:
        pdf_bytes = generate_pdf(dict(
            disease=result["disease"], crop=crop_name,
            confidence=confidence, severity=detected_severity,
            region=region, season=season,
            treatment_plan=tp,
        ))
        fname = f"crop_disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button(
            label="📥 Download Clinical PDF Report",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
        )
    with col_info:
        st.markdown(
            '<span style="font-size:0.78rem;color:#334155">Full clinical report with diagnosis, '
            'treatment tables, yield impact estimate, regional advisory and legal disclaimer.</span>',
            unsafe_allow_html=True,
        )


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #0f2040;margin-top:36px;padding:16px 0 8px;
text-align:center;font-size:0.75rem;color:#1e293b">
  Crop Disease Advisor v2.0 &nbsp;·&nbsp;
  Shubham Haraniya &amp; Vidhan Savaliya
</div>
""", unsafe_allow_html=True)
