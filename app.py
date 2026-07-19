import os
import json
import uuid as _uuid
import pandas as pd
import numpy as np
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

# ---------- config ----------
VERDICTS = "workspace.default.facility_verdicts"
FACILITIES = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities"
OVERRIDES = "workspace.default.planner_overrides"
RECEIPTS = "workspace.default.llm_receipts"
CAPS = ["icu", "maternity", "emergency", "oncology", "trauma", "nicu"]

# Databricks-hosted LLM used for Tier-2 receipts. Verify this exact name on the
# Serving page (left nav). A wrong name will NOT crash the app - falls back to OpenAI.
SERVING_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# Lakebase (transactional persistence for planner overrides). Set both in app.yaml.
# If unset/unreachable, overrides fall back to the Delta table - nothing breaks.
LAKEBASE_HOST = os.getenv("LAKEBASE_HOST")
LAKEBASE_INSTANCE = os.getenv("LAKEBASE_INSTANCE", "trust-desk-db")

# Mosaic AI Vector Search (semantic retrieval - additive, never alters verdicts)
VS_ENDPOINT = "trust_desk_vs"
VS_INDEX = "workspace.default.facility_desc_index"

st.set_page_config(page_title="Mimir - Intelligence you can trust", layout="wide", page_icon="🧿")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

/* ========= MIMIR design tokens =========
   base #0A0908 · surface rgba(255,255,255,.03) · hairline rgba(224,168,46,.16)
   text #F5F1E8 / muted #9A948A · gold #F4C95D / #E0A82E / #B8860B / #F7E7CE */

/* hide streamlit chrome, let the hero sit high */
#MainMenu, footer {visibility: hidden;}
[data-testid="stHeader"] {background: transparent !important;}
[data-testid="stHeader"] * {background: transparent;}
.block-container {padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px;}

/* app canvas: warm near-black, radial gold glow behind the hero, soft vignette */
.stApp {
  background:
    radial-gradient(ellipse 55% 34% at 28% -6%, rgba(244,201,93,0.14), rgba(224,168,46,0.05) 45%, transparent 70%),
    radial-gradient(ellipse 80% 60% at 50% 45%, transparent 55%, rgba(0,0,0,0.38) 100%),
    #0A0908;
  background-attachment: fixed;
}
/* faint grain for depth */
.stApp::after {
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: 0.028;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='160' height='160' filter='url(%23n)'/%3E%3C/svg%3E");
}

/* type system */
html, body, [class*="css"] {font-family: 'Inter', sans-serif; color: #F5F1E8;}
h1, h2, h3 {font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; color: #F5F1E8;}
[data-testid="stCaptionContainer"], .stCaption, small {color: #9A948A !important;}

/* ---- Mimir hero ---- */
.mimir-stage {padding: 0.6rem 0 0.4rem 0;}
.mimir-hero {display: flex; align-items: center; gap: 20px; margin-bottom: 6px;}
.mimir-well {
  width: 58px; height: 58px; border-radius: 50%; flex: 0 0 58px;
  background: radial-gradient(circle at 35% 30%, #0A0908 28%, transparent 30%),
              conic-gradient(from 210deg, #F4C95D, #B8860B, #F7E7CE, #E0A82E, #F4C95D);
  box-shadow: 0 0 28px rgba(224,168,46,0.38), 0 0 64px rgba(244,201,93,0.14), inset 0 0 12px rgba(0,0,0,0.6);
}
.mimir-name {
  font-family: 'Space Grotesk', sans-serif; font-size: 3.1rem; font-weight: 700;
  letter-spacing: 0.05em; line-height: 1;
  background: linear-gradient(100deg, #F7E7CE 0%, #F4C95D 42%, #E0A82E 70%, #B8860B 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.mimir-tag {color: #9A948A; font-size: 0.92rem; letter-spacing: 0.22em;
  text-transform: uppercase; margin-top: 8px; font-weight: 500;}
.mimir-sub {color: #9A948A; font-size: 1.02rem; line-height: 1.6; margin: 14px 0 6px 0; max-width: 62rem;}

/* ---- summary stat cards + verdict pills (status palette, never gold) ---- */
.stat-card {
  background: rgba(255,255,255,0.03); border: 1px solid rgba(224,168,46,0.16);
  border-radius: 14px; padding: 18px 20px 16px 20px;
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
}
.stat-num {font-family: 'Space Grotesk', sans-serif; font-size: 2.35rem; font-weight: 600;
  color: #F5F1E8; line-height: 1.15; margin-top: 10px;}
.stat-sub {color: #9A948A; font-size: 0.8rem; margin-top: 2px;}
.pill {display: inline-block; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.09em;
  text-transform: uppercase; padding: 4px 12px; border-radius: 999px; line-height: 1.4;}
.pill-emerald {background: #17B877; color: #05261A;}
.pill-slate   {background: #6B7686; color: #10151D;}
.pill-red     {background: #E5484D; color: #2B0709;}

/* metric cards (kept on-theme for any st.metric) */
div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.03); border: 1px solid rgba(224,168,46,0.16);
  border-radius: 14px; padding: 16px 18px;
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
}
div[data-testid="stMetricValue"] {font-family: 'Space Grotesk', sans-serif; font-size: 2rem;}

/* expander rows as glass cards */
div[data-testid="stExpander"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(224,168,46,0.16) !important; border-radius: 14px !important;
  margin-bottom: 12px;
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
  transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
}
div[data-testid="stExpander"]:hover {
  border-color: rgba(224,168,46,0.38) !important;
  transform: translateY(-1px);
  box-shadow: 0 8px 28px rgba(0,0,0,0.45), 0 0 18px rgba(224,168,46,0.07), inset 0 1px 0 rgba(255,255,255,0.05);
}
div[data-testid="stExpander"] details {border: none !important; background: transparent !important;}
div[data-testid="stExpander"] summary {font-size: 0.95rem; padding: 15px 18px; color: #F5F1E8;}
div[data-testid="stExpander"] summary:hover {color: #F4C95D;}

/* buttons: gold gradient primary, dark text, soft glow */
.stButton > button {
  border-radius: 10px; border: 1px solid rgba(224,168,46,0.45);
  background: linear-gradient(135deg, #F4C95D 0%, #E0A82E 55%, #B8860B 100%);
  color: #1A1207; font-family: 'Inter', sans-serif; font-weight: 600;
  transition: all .15s ease; box-shadow: 0 1px 10px rgba(224,168,46,0.16);
}
.stButton > button:hover {
  border-color: #F4C95D; color: #0A0908; transform: translateY(-1px);
  box-shadow: 0 0 22px rgba(244,201,93,0.42), 0 4px 14px rgba(0,0,0,0.35);
}
.stButton > button:active {transform: translateY(0);}
.stButton > button:focus:not(:active) {border-color: #F4C95D; color: #1A1207;}

/* progress bar: gold ramp on a dark track */
div[data-testid="stProgress"] > div > div {background: linear-gradient(90deg, #F4C95D, #B8860B); border-radius: 999px;}
div[data-testid="stProgress"] > div {background: rgba(255,255,255,0.06); border-radius: 999px;}

/* tables: hairline grid on glass */
div[data-testid="stTable"] table {font-size: 0.85rem; border-collapse: collapse;}
div[data-testid="stTable"] td, div[data-testid="stTable"] th {
  padding: 7px 12px; border-color: rgba(255,255,255,0.07) !important;}
div[data-testid="stTable"] th {
  color: #9A948A !important; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
  background: rgba(255,255,255,0.02);}
div[data-testid="stTable"] {border: 1px solid rgba(224,168,46,0.14); border-radius: 12px; overflow: hidden;}

/* dataframe container */
div[data-testid="stDataFrame"] {
  border: 1px solid rgba(224,168,46,0.16); border-radius: 12px; overflow: hidden;}

/* tabs (on-theme if used) */
.stTabs [data-baseweb="tab-list"] {gap: 4px; border-bottom: 1px solid rgba(224,168,46,0.16);}
.stTabs [data-baseweb="tab"] {color: #9A948A; font-family: 'Space Grotesk', sans-serif;}
.stTabs [aria-selected="true"] {color: #F4C95D !important;}
.stTabs [data-baseweb="tab-highlight"] {background: linear-gradient(90deg, #F4C95D, #B8860B);}

/* inputs + selects: dark glass with gold focus */
div[data-baseweb="select"] > div, .stTextInput input {
  border-radius: 10px !important; background: rgba(255,255,255,0.03) !important;
  border-color: rgba(255,255,255,0.10) !important;}
.stTextInput input:focus, div[data-baseweb="select"] > div:focus-within {
  border-color: rgba(224,168,46,0.6) !important; box-shadow: 0 0 0 1px rgba(224,168,46,0.3);}
.stTextInput input::placeholder {color: #9A948A;}

/* radio pills: ghost glass, gold when active */
div[role="radiogroup"] label {
  background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.09);
  border-radius: 999px; padding: 4px 14px; margin-right: 8px;
  transition: border-color .15s ease, background .15s ease;
}
div[role="radiogroup"] label:hover {border-color: rgba(224,168,46,0.5);}
div[role="radiogroup"] label:has(input:checked) {
  border-color: rgba(224,168,46,0.65); background: rgba(224,168,46,0.09);}

/* alerts: soften into the theme */
div[data-testid="stAlert"] {border-radius: 12px;}

/* dividers as gold hairlines */
hr {border: none; border-top: 1px solid rgba(224,168,46,0.16); margin: 1.4rem 0;}

/* map container framed like the cards */
div[data-testid="stDeckGlJsonChart"] {
  border: 1px solid rgba(224,168,46,0.16); border-radius: 14px; overflow: hidden;}

/* why-ranked line */
.why-rank {color: #9A948A; font-size: 0.82rem; margin: -4px 0 10px 2px;
  border-left: 2px solid #E0A82E; padding-left: 10px;}
</style>
""", unsafe_allow_html=True)

# ---------- db helpers (Delta via SQL warehouse) ----------
cfg = Config()

def get_conn():
    return sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
        credentials_provider=lambda: cfg.authenticate,
    )

@st.cache_data(ttl=600)
def q(query: str) -> pd.DataFrame:
    with get_conn() as c, c.cursor() as cur:
        cur.execute(query)
        return cur.fetchall_arrow().to_pandas()

def exec_sql(query: str, params=None):
    with get_conn() as c, c.cursor() as cur:
        cur.execute(query, params or {})

def ensure_overrides_table():
    exec_sql(f"""CREATE TABLE IF NOT EXISTS {OVERRIDES} (
        unique_id STRING, capability STRING, original_verdict STRING,
        new_verdict STRING, note STRING, created_at TIMESTAMP)""")

# ---------- Lakebase (transactional persistence, Delta fallback) ----------
def _lakebase_conn():
    """Connect to Lakebase Postgres as the app's own service principal.
    New projects-style Lakebase: the password is the SP's own OAuth token,
    minted at call time (ambient auth - no key stored anywhere).
    Note: generate_database_credential is for old-style Database Instances
    and fails against projects-style Lakebase."""
    import psycopg2
    from databricks.sdk import WorkspaceClient
    if not LAKEBASE_HOST:
        raise RuntimeError("LAKEBASE_HOST not configured")
    w = WorkspaceClient()
    token = w.config.oauth_token().access_token
    return psycopg2.connect(
        host=LAKEBASE_HOST, port=5432, dbname="databricks_postgres",
        user=os.getenv("DATABRICKS_CLIENT_ID", ""),
        password=token, sslmode="require", connect_timeout=15)

def _lakebase_ensure_table(pg):
    with pg.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS planner_overrides (
            unique_id TEXT, capability TEXT, original_verdict TEXT,
            new_verdict TEXT, note TEXT, created_at TIMESTAMPTZ DEFAULT now())""")
    pg.commit()

def save_override(uid: str, cap_key: str, original: str, new_v: str, note: str) -> str:
    """Write the planner's review. Lakebase first (rubric-native transactional store);
    Delta table as the kill-switch fallback. Returns which store accepted the write."""
    try:
        pg = _lakebase_conn()
        try:
            _lakebase_ensure_table(pg)
            with pg.cursor() as cur:
                cur.execute(
                    "INSERT INTO planner_overrides (unique_id, capability, original_verdict, new_verdict, note) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (uid, cap_key, original, new_v, note or ""))
            pg.commit()
            return "Lakebase"
        finally:
            pg.close()
    except Exception as e:
        print(f"[Lakebase save failed] {type(e).__name__}: {e}")

    ensure_overrides_table()
    safe_note = (note or "").replace("'", "''")
    exec_sql(f"""INSERT INTO {OVERRIDES} VALUES (
        '{uid}', '{cap_key}', '{original}', '{new_v}', '{safe_note}', current_timestamp())""")
    return "Delta (fallback)"

def load_overrides() -> tuple:
    """Read saved reviews - Lakebase first, Delta fallback. Returns (df, source)."""
    try:
        pg = _lakebase_conn()
        try:
            _lakebase_ensure_table(pg)
            ov = pd.read_sql("SELECT * FROM planner_overrides ORDER BY created_at DESC LIMIT 50", pg)
            return ov, "Lakebase"
        finally:
            pg.close()
    except Exception as e:
        print(f"[Lakebase load failed] {type(e).__name__}: {e}")
    ov = q(f"SELECT * FROM {OVERRIDES} ORDER BY created_at DESC LIMIT 50")
    return ov, "Delta (fallback)"

# ---------- Tier 2: LLM receipts (on-demand, cached) ----------
def _call_llm(prompt: str) -> str:
    """Databricks Foundation Model serving first (ambient service-principal auth,
    no API key); OpenAI as the kill-switch fallback so a live demo never dies."""
    try:
        from databricks.sdk import WorkspaceClient
        db_client = WorkspaceClient().serving_endpoints.get_open_ai_client()
        resp = db_client.chat.completions.create(
            model=SERVING_ENDPOINT,
            max_tokens=250,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        pass

    from openai import OpenAI
    oa_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = oa_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=250,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

def get_receipt(uid: str, cap_key: str, verdict: str, det: pd.Series) -> str:
    exec_sql(f"""CREATE TABLE IF NOT EXISTS {RECEIPTS} (
        unique_id STRING, capability STRING, receipt STRING, created_at TIMESTAMP)""")
    safe_uid = uid.replace("'", "")
    cached = q(f"SELECT receipt FROM {RECEIPTS} WHERE unique_id='{safe_uid}' AND capability='{cap_key}' LIMIT 1")
    if not cached.empty:
        return cached.iloc[0]["receipt"]

    evidence_text = "\n".join(
        f"[{f}] {str(det[f])[:1200]}" for f in ["description", "equipment", "procedure", "capability"]
        if det is not None and pd.notna(det.get(f))
    )
    prompt = (
        f"A healthcare facility claims the capability: {cap_key.upper()}. "
        f"Our deterministic scorer marked this claim: {verdict}.\n"
        f"From ONLY the facility's own fields below, quote the 1-3 exact sentences or list items "
        f"that most directly SUPPORT or CONTRADICT the {cap_key.upper()} claim, each prefixed by its [field]. "
        f"If nothing is relevant, say exactly: 'No directly relevant sentence found.' "
        f"Never invent text that is not present.\n\n{evidence_text}"
    )
    receipt = _call_llm(prompt)
    safe_r = receipt.replace("'", "''")
    exec_sql(f"""INSERT INTO {RECEIPTS} VALUES ('{safe_uid}', '{cap_key}', '{safe_r}', current_timestamp())""")
    return receipt

# ---------- Mosaic AI Vector Search (semantic retrieval - additive only) ----------
def semantic_search(query_text: str, k: int = 5) -> pd.DataFrame:
    from databricks.vector_search.client import VectorSearchClient
    vsc = VectorSearchClient(disable_notice=True)
    idx = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
    res = idx.similarity_search(
        query_text=query_text,
        columns=["unique_id", "name", "city", "state", "description"],
        num_results=k,
    )
    cols = [c["name"] for c in res["manifest"]["columns"]]
    rows = res["result"]["data_array"]
    return pd.DataFrame(rows, columns=cols)

# ---------- ranking explanation ----------
def why_ranked(r) -> str:
    """One-line, human-readable reason this facility sits where it does.
    Mirrors the ORDER BY exactly: verdict class -> evidence tier -> data completeness."""
    if r.verdict == "Contradicted":
        pos = "Pinned to top: the data conflicts - review before relying on it"
    elif r.verdict == "Corroborated":
        pos = "Ranked by proof: corroborated claims sort above unverified ones"
    else:
        pos = "Below corroborated: no evidence either way (data desert, not a red flag)"
    return (f"{pos} · evidence {int(r.evidence_tier)}/4 fields support the claim · "
            f"{int(r.data_completeness*100)}% of fields filled")

# ---------- ui ----------
st.markdown("""
<div class="mimir-stage">
  <div class="mimir-hero">
    <div class="mimir-well"></div>
    <div>
      <div class="mimir-name">MIMIR</div>
      <div class="mimir-tag">Intelligence you can trust</div>
    </div>
  </div>
  <p class="mimir-sub">
    Can this facility actually do what it claims? Every verdict cites its evidence.
    Missing data is reported, never punished -
    <span style='color:#e6b34d;'>amber</span> means <i>we don't know</i>,
    <span style='color:#e5484d;'>red</span> means <i>the data conflicts</i>.
  </p>
</div>
""", unsafe_allow_html=True)

regions = q(f"""SELECT DISTINCT region FROM (
    SELECT state AS region FROM {VERDICTS} WHERE state IS NOT NULL
    UNION SELECT city AS region FROM {VERDICTS} WHERE city IS NOT NULL
) WHERE region IS NOT NULL ORDER BY region""")["region"].tolist()

c1, c2, c3 = st.columns([2, 2, 1])
cap = c1.selectbox("Capability", CAPS, format_func=str.upper)
state = c2.selectbox("Region (state or city - type to search)", ["All India"] + regions)
c3.write("")

# --- semantic search (Mosaic AI Vector Search) - additive, never alters verdicts ---
with st.expander("🔎 Semantic search - describe the care you need (Mosaic AI Vector Search)"):
    sq = st.text_input("e.g. 'newborn intensive care with ventilators' or 'cancer radiation therapy'",
                       key="vs_query")
    if sq:
        try:
            with st.spinner("Searching descriptions by meaning..."):
                hits = semantic_search(sq)
            for _, h in hits.iterrows():
                st.markdown(f"**{h['name']}** - {h.get('city') or ''}, {h.get('state') or ''}")
                st.caption(str(h["description"])[:280] + "...")
            st.caption("Semantic retrieval via Mosaic AI Vector Search. Retrieval only - "
                       "verdicts always come from the deterministic evidence scorer.")
        except Exception:
            st.info("Semantic index not available yet - run the Vector Search setup cell in the "
                    "scorer notebook and grant the app SELECT on the index. Everything else works without it.")

where = f"capability = '{cap}'"
if state != "All India":
    sv = state.replace(chr(39), chr(39)*2)
    where += f" AND (state = '{sv}' OR city = '{sv}')"

df = q(f"""
    SELECT * FROM {VERDICTS} WHERE {where}
    ORDER BY CASE verdict WHEN 'Contradicted' THEN 0 WHEN 'Corroborated' THEN 1 ELSE 2 END,
             evidence_tier DESC, data_completeness DESC
    LIMIT 400
""")

if df.empty:
    st.info("No facilities claim this capability in this region.")
    st.stop()

# summary strip - true totals across the whole filter, not just the fetched page
tot = q(f"SELECT verdict, COUNT(*) n FROM {VERDICTS} WHERE {where} GROUP BY verdict")
cnt = dict(zip(tot.verdict, tot.n))
n_c = cnt.get("Corroborated", 0)
n_u = cnt.get("Unverified", 0)
n_x = cnt.get("Contradicted", 0)
m1, m2, m3 = st.columns(3)
m1.markdown(f"""<div class="stat-card"><span class="pill pill-emerald">Corroborated</span>
<div class="stat-num">{int(n_c)}</div>
<div class="stat-sub">claims backed by the facility's own records</div></div>""", unsafe_allow_html=True)
m2.markdown(f"""<div class="stat-card"><span class="pill pill-slate">Unverified</span>
<div class="stat-num">{int(n_u)}</div>
<div class="stat-sub">no evidence either way</div></div>""", unsafe_allow_html=True)
m3.markdown(f"""<div class="stat-card"><span class="pill pill-red">Contradicted</span>
<div class="stat-num">{int(n_x)}</div>
<div class="stat-sub">the data conflicts - review first</div></div>""", unsafe_allow_html=True)

# --- how the ranking works (transparent by design) ---
with st.expander("⚖️ How this ranking works - no black box"):
    st.markdown("""
Facilities are ordered by three transparent rules, applied in order:

1. **Contradicted first** - conflicts are pinned to the top so a planner sees risk before anything else.
2. **Evidence tier** - among the rest, facilities whose own records support the claim in more fields (0-4) rank higher.
3. **Data completeness** - ties break toward facilities with fuller records.

No AI decides this order. The LLM only *extracts citations* on demand; the ranking is a deterministic,
auditable rule you can read above. Missing data lowers certainty, never the verdict -
that's the difference between a *data desert* and a *medical desert*.
    """)

BADGE = {"Corroborated": "🟢", "Unverified": "🟡", "Contradicted": "🔴"}

show = st.radio("Show", ["All (contradictions pinned first)", "Contradicted only", "Unverified only"], horizontal=True)
view = df
if show == "Contradicted only":
    view = df[df.verdict == "Contradicted"]
elif show == "Unverified only":
    view = df[df.verdict == "Unverified"]

view_mode = st.radio("View", ["List", "🗺️ Trust map"], horizontal=True, label_visibility="collapsed")

if view_mode == "🗺️ Trust map":
    import pydeck as pdk
    where_v = where.replace("capability =", "v.capability =").replace("state =", "v.state =").replace("city =", "v.city =")
    geo = q(f"""
        SELECT v.name, v.verdict, v.evidence_tier, f.latitude, f.longitude
        FROM {VERDICTS} v JOIN {FACILITIES} f ON v.unique_id = f.unique_id
        WHERE {where_v} AND f.latitude IS NOT NULL AND f.longitude IS NOT NULL
    """)
    total_geo = len(geo)
    geo = geo[(geo.latitude.between(6.0, 37.5)) & (geo.longitude.between(68.0, 97.5))].copy()
    dropped = total_geo - len(geo)

    # Duplicate city-centroid coordinates stack into one blob when zoomed in - spread
    # them with a small deterministic jitter (~0.4 km), seeded for stable reruns.
    if not geo.empty:
        geo["latitude"] = pd.to_numeric(geo["latitude"], errors="coerce")
        geo["longitude"] = pd.to_numeric(geo["longitude"], errors="coerce")
        geo = geo.dropna(subset=["latitude", "longitude"])
        dup = geo.duplicated(subset=["latitude", "longitude"], keep=False)
        n_dup = int(dup.sum())
        if n_dup:
            rng = np.random.default_rng(42)
            geo.loc[dup, "latitude"] = geo.loc[dup, "latitude"] + rng.uniform(-0.004, 0.004, n_dup)
            geo.loc[dup, "longitude"] = geo.loc[dup, "longitude"] + rng.uniform(-0.004, 0.004, n_dup)

    COLORS = {"Corroborated": [45, 212, 167, 220], "Unverified": [230, 179, 77, 210], "Contradicted": [224, 93, 93, 240]}
    geo["color"] = geo["verdict"].map(COLORS)
    # Small meters radius as base; pixel clamps (3-10px) keep dots visible zoomed out
    # and small zoomed in - works on every pydeck version.
    geo["radius"] = geo["verdict"].map({"Corroborated": 1200, "Unverified": 1200, "Contradicted": 1800})

    st.pydeck_chart(pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        initial_view_state=pdk.ViewState(latitude=22.5, longitude=80.0, zoom=3.7),
        layers=[pdk.Layer(
            "ScatterplotLayer", data=geo,
            get_position=["longitude", "latitude"],
            get_fill_color="color", get_radius="radius",
            radius_min_pixels=3, radius_max_pixels=10,
            stroked=True, get_line_color=[15, 15, 15, 200], line_width_min_pixels=1,
            pickable=True, opacity=0.85,
        )],
        tooltip={"text": "{name}\n{verdict} · evidence {evidence_tier}/4"},
    ), use_container_width=True)

    st.caption(
        f"🟢 corroborated · 🟡 unverified (data desert - we don't know, not nothing's there) · 🔴 contradicted. "
        f"{dropped} record(s) excluded for impossible coordinates - flagged, not plotted."
    )
    st.stop()

st.divider()

@st.cache_data(ttl=600)
def facility_details(uids: tuple) -> pd.DataFrame:
    if not uids:
        return pd.DataFrame()
    in_list = ",".join("'" + str(u).replace("'", "") + "'" for u in uids)
    d = q(f"SELECT unique_id, description, capability, procedure, equipment, specialties, numberDoctors, capacity "
          f"FROM {FACILITIES} WHERE unique_id IN ({in_list})")
    return d.set_index("unique_id")

visible = view.head(40)
dets_all = facility_details(tuple(visible.unique_id))

for _, r in visible.iterrows():
    badge = BADGE[r.verdict]
    label = (f"{badge} **{r['name']}** - {r.city or ''}, {r.state or ''}  ·  "
             f"{r.verdict}  ·  evidence {int(r.evidence_tier)}/4  ·  data {int(r.data_completeness*100)}%")
    with st.expander(label):
        det = dets_all.loc[r.unique_id] if r.unique_id in dets_all.index else None

        st.markdown(f"<div class='why-rank'>{why_ranked(r)}</div>", unsafe_allow_html=True)

        st.progress(float(r.data_completeness), text=f"Data completeness {int(r.data_completeness*100)}% - "
                    f"{'blank fields are reported, not counted against the facility' if r.data_completeness < 1 else 'full data available'}")

        rows = []
        for field, present, match, raw in [
            ("description", r.desc_present, r.desc_match, det["description"] if det is not None else None),
            ("equipment",   r.equip_present, r.equip_match, det["equipment"] if det is not None else None),
            ("procedure",   r.proc_present, r.proc_match, det["procedure"] if det is not None else None),
            ("staffing",    r.staff_present, r.staff_match, f"{r.num_doctors} doctors" if pd.notna(r.num_doctors) else None),
        ]:
            if not present:
                status, evidence = "- no data (not counted against)", ""
            elif match:
                status, evidence = "✓ supports the claim", str(raw)[:400]
            else:
                status, evidence = "present, no match found", str(raw)[:200]
            rows.append({"field": field, "status": status, "evidence": evidence})
        st.table(pd.DataFrame(rows))

        if r.conflict_reason:
            st.error(f"Conflict: {r.conflict_reason}")

        # AI-extracted citation (Tier 2, on-demand, cached)
        if st.button("✨ Extract cited evidence (AI)", key=f"ai_{r.unique_id}_{cap}"):
            try:
                with st.spinner("Extracting exact sentences from this facility's own records..."):
                    receipt = get_receipt(r.unique_id, cap, r.verdict, det)
                st.info(receipt)
            except Exception as e:
                st.warning(f"AI receipt unavailable ({e}); the raw evidence above stands on its own.")

        # override + note -> persisted (Lakebase first, Delta fallback)
        st.markdown("**Planner review**")
        oc1, oc2, oc3 = st.columns([2, 3, 1])
        new_v = oc1.selectbox("Override verdict", ["(keep)", "Corroborated", "Unverified", "Contradicted"],
                              key=f"v_{r.unique_id}_{cap}")
        note = oc2.text_input("Note", placeholder="e.g. Called facility - confirmed 6 ICU beds",
                              key=f"n_{r.unique_id}_{cap}")
        if oc3.button("Save", key=f"s_{r.unique_id}_{cap}"):
            nv = r.verdict if new_v == "(keep)" else new_v
            store = save_override(r.unique_id, cap, r.verdict, nv, note)
            st.success(f"Saved to {store} - this review persists beyond the session.")

st.divider()
with st.expander("Saved planner reviews"):
    try:
        ov, source = load_overrides()
        st.dataframe(ov, use_container_width=True)
        st.caption(f"Persistence: {source}")
    except Exception:
        st.caption("No reviews saved yet.")

st.caption("Mimir · verdicts from a deterministic evidence scorer · AI extracts citations, never decides trust")