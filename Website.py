# ============================================================
# Annals Explorer: Interactive Knowledge Graph Website
# ============================================================
#
# Written by : Dr. Sudhansu Bala Das
# Email      : baladas.sudhansu@gmail.com
#
# Objective:
# This application provides an interactive exploration platform
# for the Annals of Ulster (600–700 CE) dataset. It enables users
# to explore historical people, places, events, source evidence,
# and entity relationships through searchable pages and
# knowledge graph visualisations.
#
# Main Features:
# - People and Places exploration with historical evidence
# - Interactive knowledge graph representation
# - Source-based question answering interface
# - Network visualisation of annal relationships
#
# Setup:
# 1. Keep the project data files inside the "data" folder.
# 2. Update PROJECT_ROOT below according to your local system path.
# 3. Install required packages:
#
#       pip install streamlit pandas pyvis graphviz
#
# 4. Run the website:
#
#       streamlit run website.py
#
# ============================================================

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


# ============================================================
# PATHS
# ============================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

DATA_DIR = PROJECT_ROOT
READY_DIR = PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "_graph_html"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PEOPLE_CSV = READY_DIR / "website_people.csv"
PLACES_CSV = READY_DIR / "website_places.csv"
EVENTS_CSV = READY_DIR / "website_events.csv"
EDGES_CSV = READY_DIR / "website_edges.csv"
SUMMARY_CSV = READY_DIR / "website_summary.csv"

RAW_EVENTS_CSV = PROJECT_ROOT / "annals_events_U600_U700.csv"


# ============================================================
# STREAMLIT SETUP
# ============================================================

st.set_page_config(
    page_title="Annals Explorer",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.stApp { background:#FBF7ED; }
header[data-testid="stHeader"] { display:none; }
[data-testid="stSidebar"] { display:none; }

.block-container {
    max-width:1200px;
    padding-top:1rem;
    padding-bottom:2rem;
}

.topbar {
    background:#072B57;
    color:white;
    padding:1rem 1.2rem;
    border-radius:16px;
    margin-bottom:.8rem;
}

.title { font-size:2rem; font-weight:900; }
.subtitle { color:#DCEBFF; font-size:.95rem; }
.small { color:#667085; font-size:.95rem; }

.evidence {
    background:#FFFDF7;
    border-left:5px solid #B98525;
    padding:.85rem 1rem;
    border-radius:10px;
    margin:.7rem 0;
}

.legend-box {
    background:white;
    border:1px solid #E3D8C8;
    border-radius:12px;
    padding:.85rem 1rem;
    margin:.7rem 0;
}

.metric-card {
    padding:1rem;
    border-radius:16px;
    color:white;
    min-height:115px;
    box-shadow:0 8px 20px rgba(0,0,0,0.08);
}
.metric-title {
    font-size:.95rem;
    opacity:.95;
}
.metric-value {
    font-size:2.3rem;
    font-weight:900;
    margin-top:.4rem;
}
.metric-events { background:#2563EB; }
.metric-people { background:#DC2626; }
.metric-places { background:#16A34A; }
.metric-edges { background:#7C3AED; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    people = pd.read_csv(PEOPLE_CSV)
    places = pd.read_csv(PLACES_CSV)
    events = pd.read_csv(EVENTS_CSV)
    edges = pd.read_csv(EDGES_CSV)
    summary = pd.read_csv(SUMMARY_CSV)

    raw_events = pd.read_csv(RAW_EVENTS_CSV) if RAW_EVENTS_CSV.exists() else pd.DataFrame()

    for df in [people, places, events, edges, summary, raw_events]:
        for c in df.columns:
            if df[c].dtype == "object":
                df[c] = df[c].fillna("").astype(str)

    events["year"] = pd.to_numeric(events["year"], errors="coerce")
    edges["year"] = pd.to_numeric(edges["year"], errors="coerce")

    if not raw_events.empty and "year" in raw_events.columns:
        raw_events["year"] = pd.to_numeric(raw_events["year"], errors="coerce")

    return people, places, events, edges, summary, raw_events


people_df, places_df, events_df, edges_df, summary_df, raw_events_df = load_data()

DATA_MIN_YEAR = int(events_df["year"].min())
DATA_MAX_YEAR = int(events_df["year"].max())


# ============================================================
# HELPERS
# ============================================================

def clean(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def html_escape(s: str) -> str:
    return (
        clean(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def simple_name(name: str) -> str:
    s = clean(name)
    s = re.sub(r"\(\?\)", "", s)
    s = re.sub(r"\s+the learned$", "", s, flags=re.I)
    s = re.sub(r"\s+according to some$", "", s, flags=re.I)
    s = re.sub(r"\s+according to others$", "", s, flags=re.I)
    s = re.sub(r"\s*[—–-]\s*the king of Ireland$", "", s, flags=re.I)
    s = re.sub(r"\s+the king of Ireland$", "", s, flags=re.I)
    return clean(s)


def is_bad_person_name(name: str) -> bool:
    s = simple_name(name).lower()
    bad_bits = ["descendant of", " years", "and of", "penda's son", "flight", "victor"]
    return (not s) or any(b in s for b in bad_bits) or bool(re.search(r"\d", s))


def simple_place(place: str) -> str:
    s = clean(place)
    s = re.sub(r"^the battle of\s+", "", s, flags=re.I)
    s = re.sub(r"^battle of\s+", "", s, flags=re.I)
    return clean(s)


# ============================================================
# DISPLAY NORMALISATION FOR PUBLIC BUTTON LABELS
# ============================================================

PERSON_ROLE_PATTERNS = [
    (r"^(.+?)\s*[—–-]\s*the\s+king\s+of\s+(.+)$", "king of"),
    (r"^(.+?)\s+from\s+(.+)$", "from"),
    (r"^(.+?)\s+abbot\s+of\s+(.+)$", "abbot of"),
    (r"^(.+?)\s+bishop\s+of\s+(.+)$", "bishop of"),
    (r"^(.+?)\s+king\s+of\s+(.+)$", "king of"),
    (r"^(.+?)\s+grandson\s+of\s+(.+)$", "grandson of"),
]


def split_compound_person_label(name: str) -> list[str]:
    """
    Split only clear two-person labels used as a single extracted display phrase.
    Example: 'Finnguine the Tall and Feradach Méth' -> two public buttons.
    Do not split formulaic names such as 'Aed son of X and Y' here.
    """
    s = clean(name)
    if not s:
        return []

    # Safe special case visible in the data.
    if re.search(r"\bthe\s+Tall\s+and\s+", s, flags=re.I):
        return [clean(x) for x in re.split(r"\s+and\s+", s, maxsplit=1, flags=re.I) if clean(x)]

    return [s]


def display_person_name(name: str) -> str:
    """
    Clean public People-page labels only. Source evidence remains unchanged.
    Keeps historically useful identifiers like 'Caemgein of Glenn dá Locha',
    but removes role/context phrases like 'abbot of Í' or 'grandson of Rónán'.
    """
    s = simple_name(name)
    if not s:
        return ""

    s = s.replace("—", " - ").replace("–", " - ")

    for pat, _rel in PERSON_ROLE_PATTERNS:
        m = re.search(pat, s, flags=re.I)
        if m:
            return clean(m.group(1))

    return clean(s)


def person_context_from_name(name: str) -> list[str]:
    """Return context removed from a person button label, for display on the detail page."""
    s = simple_name(name)
    if not s:
        return []
    s2 = s.replace("—", " - ").replace("–", " - ")
    notes = []
    for pat, rel in PERSON_ROLE_PATTERNS:
        m = re.search(pat, s2, flags=re.I)
        if m:
            target = clean(m.group(2))
            if target:
                notes.append(f"{rel}: {target}")
    return notes


def expand_person_display_names(name: str) -> list[str]:
    """Return all clean public names represented by one extracted people row."""
    out = []
    for part in split_compound_person_label(name):
        label = display_person_name(part)
        if label and not is_bad_person_name(label):
            out.append(label)
    return list(dict.fromkeys(out))


def display_place_name(place: str) -> str:
    """
    Clean public Places-page labels only.
    Example: 'Muiresc between Cenél Cairpri' -> 'Muiresc'.
    Example: 'Dál Riata or nAraide' -> 'Dál Riata' for first label.
    """
    s = simple_place(place)
    if not s:
        return ""

    if re.search(r"\s+between\s+", s, flags=re.I):
        s = re.split(r"\s+between\s+", s, flags=re.I)[0]

    if re.search(r"\s+or\s+", s, flags=re.I):
        s = re.split(r"\s+or\s+", s, flags=re.I)[0]

    return clean(s)


def place_context_from_name(place: str) -> list[str]:
    """Return context removed from a place button label."""
    s = simple_place(place)
    notes = []
    m = re.search(r"^(.+?)\s+between\s+(.+)$", s, flags=re.I)
    if m:
        notes.append(f"between: {clean(m.group(2))}")
    m = re.search(r"^(.+?)\s+or\s+(.+)$", s, flags=re.I)
    if m:
        notes.append(f"alternative reading: {clean(m.group(2))}")
    return notes


def expand_place_display_names(place: str) -> list[str]:
    """Return clean public place labels represented by one extracted place phrase."""
    s = simple_place(place)
    if not s:
        return []

    if re.search(r"\s+or\s+", s, flags=re.I):
        left, right = [clean(x) for x in re.split(r"\s+or\s+", s, maxsplit=1, flags=re.I)]
        # If the right side is abbreviated, e.g. 'nAraide', retain both readable forms.
        if left.lower().startswith("dál ") and right and not right.lower().startswith("dál "):
            right = "Dál " + right
        return [display_place_name(left), display_place_name(right)]

    label = display_place_name(s)
    return [label] if label else []


def nice_relation(rel: str) -> str:
    return {
        "has_event": "has event",
        "victim": "victim",
        "death": "death",
        "participant": "participant",
        "agent_in": "agent",
        "killed": "killed",
        "occurred_at": "place",
        "supported_by": "evidence",
    }.get(clean(rel), clean(rel).replace("_", " "))


def person_event_relation_label(event_type: str) -> str:
    """
    Label the selected person in an event graph.

    For peaceful/natural death records, avoid "victim".
    "Victim" is used only for violent death / slaying-style events.
    """
    et = clean(event_type).lower()
    if et in {"death", "repose"}:
        return "death"
    if et in {"slaying", "killing", "murder"}:
        return "victim"
    if et == "battle":
        return "participant"
    return "mentioned"


def dot_id(x: str) -> str:
    x = re.sub(r"[^A-Za-z0-9_]", "_", clean(x))
    if not x:
        x = "node"
    if x[0].isdigit():
        x = "n_" + x
    return x


def short_label(x: str, n: int = 36) -> str:
    x = clean(x)
    return x if len(x) <= n else x[: n - 3] + "..."


def extract_year_range_any(q: str):
    q = clean(q).lower()
    years = [int(y) for y in re.findall(r"\b([1-9]\d{2,3})\b", q)]

    if len(years) >= 2:
        return min(years), max(years), "range"

    if len(years) == 1:
        y = years[0]
        if "after" in q or "since" in q or "from" in q:
            return y, DATA_MAX_YEAR, "after"
        if "before" in q or "until" in q or "up to" in q:
            return DATA_MIN_YEAR, y, "before"
        return y, y, "single"

    return None, None, None


def overlap_with_dataset(start, end):
    if start is None or end is None:
        return None, None, ""

    if end < DATA_MIN_YEAR or start > DATA_MAX_YEAR:
        msg = (
            f"The loaded website dataset covers only {DATA_MIN_YEAR}–{DATA_MAX_YEAR} CE. "
            f"Your question asks about {start}–{end} CE, which is outside the loaded data."
        )
        return None, None, msg

    clipped_start = max(start, DATA_MIN_YEAR)
    clipped_end = min(end, DATA_MAX_YEAR)

    msg = ""
    if clipped_start != start or clipped_end != end:
        msg = (
            f"Note: the loaded website dataset covers only {DATA_MIN_YEAR}–{DATA_MAX_YEAR} CE. "
            f"So I can answer only the overlapping part: {clipped_start}–{clipped_end} CE."
        )

    return clipped_start, clipped_end, msg


STOPWORDS = {
    "who", "what", "where", "when", "which", "why", "how",
    "tell", "about", "show", "give", "find", "search",
    "is", "was", "were", "are", "the", "of", "in", "to",
    "from", "and", "or", "me", "please", "happened",
    "happen", "events", "event", "records", "record",
    "between", "during", "after", "before", "father",
    "grandfather", "killed", "kill"
}


def extract_query_terms(q: str):
    q = clean(q).lower()
    words = re.findall(r"[a-zA-ZÁÉÍÓÚáéíóúÍíḃḂṅṄ]+", q)
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def extract_name_from_question(q: str) -> str:
    q = clean(q)
    q = re.sub(r"\b[1-9]\d{2,3}\b", " ", q)
    q = re.sub(
        r"(?i)\b(who|what|where|when|which|tell|show|give|find|search|is|was|were|are|the|of|in|to|about|me|please|father|grandfather|killed|kill|happened|happen|between|from|and|after|before|during|events|event)\b",
        " ",
        q,
    )
    q = re.sub(r"[?.,:;!]", " ", q)
    return clean(q)


# ============================================================
# HEADER / EVIDENCE
# ============================================================

def header():
    st.markdown(
        """
<div class="topbar">
  <div class="title">📜 Annals Explorer</div>
  <div class="subtitle">Annals of Ulster · 600–700 CE · People, Places, Evidence, and Knowledge Graph</div>
</div>
""",
        unsafe_allow_html=True,
    )

    pages = ["Home", "People", "Places", "Full Graph", "About"]
    cols = st.columns(len(pages))

    for i, p in enumerate(pages):
        if cols[i].button(p, width="stretch", key=f"nav_{p}"):
            st.session_state.page = p
            st.session_state.selected_person_name = ""
            st.session_state.selected_person_id = ""
            st.session_state.selected_place_name = ""
            st.rerun()


def evidence_box(entry, year, event_type, texts):
    if isinstance(texts, str):
        texts = [texts]

    evidence_html = "".join(f"<p>“{html_escape(t)}”</p>" for t in texts if clean(t))

    st.markdown(
        f"""
<div class="evidence">
<b>Entry:</b> {html_escape(entry)}
&nbsp; | &nbsp;
<b>Year:</b> {html_escape(year)}
&nbsp; | &nbsp;
<b>Event:</b> {html_escape(event_type)}
<br>{evidence_html}
</div>
""",
        unsafe_allow_html=True,
    )


def grouped_evidence_rows(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    temp = events.copy()
    temp["raw_clause"] = temp["raw_clause"].astype(str).map(clean)
    temp = temp[temp["raw_clause"] != ""].drop_duplicates(
        subset=["entry_id", "year", "event_type", "raw_clause"]
    )

    grouped = (
        temp.groupby(["entry_id", "year", "event_type"], dropna=False)["raw_clause"]
        .apply(lambda x: list(dict.fromkeys([clean(v) for v in x if clean(v)])))
        .reset_index()
    )

    return grouped.sort_values(["year", "entry_id"])


def show_evidence_with_entry_graphs(events: pd.DataFrame, key_prefix: str):
    """
    Legacy helper kept for compatibility.

    It now shows source evidence only, without per-entry graph buttons.
    Person and Place pages show their main graph automatically above the evidence.
    """
    if events.empty:
        st.info("No source evidence found.")
        return

    grouped = grouped_evidence_rows(events)

    for _, r in grouped.iterrows():
        year = int(r["year"]) if not pd.isna(r["year"]) else ""
        entry_id = clean(r["entry_id"])
        event_type = clean(r["event_type"])
        evidence_box(entry_id, year, event_type, r["raw_clause"])


# ============================================================
# PLACES
# ============================================================

def extract_evidence_places() -> list[str]:
    candidates = set()

    for text in events_df["raw_clause"].dropna().astype(str):
        patterns = [
            r"\bking of ([A-ZÁÉÍÓÚÍ][A-Za-zÁÉÍÓÚáéíóúÍíÁÉÓÚḃḂṅṄ \-]+)",
            r"\bin ([A-ZÁÉÍÓÚÍ][A-Za-zÁÉÍÓÚáéíóúÍíÁÉÓÚḃḂṅṄ \-]+)",
            r"\bat ([A-ZÁÉÍÓÚÍ][A-Za-zÁÉÍÓÚáéíóúÍíÁÉÓÚḃḂṅṄ \-]+)",
            r"\bfrom ([A-ZÁÉÍÓÚÍ][A-Za-zÁÉÍÓÚáéíóúÍíÁÉÓÚḃḂṅṄ \-]+)",
        ]

        for pat in patterns:
            for m in re.finditer(pat, text):
                val = clean(m.group(1))
                val = re.split(r"\s+by\s+|\s+and\s+|\s+in which\s+|,|\.|;", val)[0]
                val = display_place_name(val)

                bad = {"Ireland", "December", "Sunday", "Saxons", "Picts", "Britons"}
                if val and len(val) <= 40 and val not in bad:
                    candidates.add(val)

    candidates.add("Temair")
    extracted = set()
    for x in places_df["display_place"].tolist():
        for label in expand_place_display_names(x):
            if clean(label):
                extracted.add(label)
    return sorted(extracted.union(candidates), key=lambda x: x.lower())


ALL_PLACE_NAMES = extract_evidence_places()


# ============================================================
# PERSON DISAMBIGUATION
# ============================================================

def get_people_variants(simple: str) -> pd.DataFrame:
    tmp = people_df.copy()
    tmp["public_names"] = tmp["display_name"].apply(expand_person_display_names)
    mask = tmp["public_names"].apply(lambda names: simple.lower() in [n.lower() for n in names])
    return tmp[mask].copy()


def person_variant_label(row) -> str:
    name = display_person_name(row.get("display_name", ""))
    full = clean(row.get("full_names", ""))
    years = clean(row.get("years", ""))
    roles = clean(row.get("roles", ""))

    label = name
    if full and full.lower() != name.lower():
        label += f" — {short_label(full, 60)}"
    if years:
        label += f" ({years})"
    if roles:
        label += f" [{roles}]"
    return label


# ============================================================
# PROFILE GRAPH
# ============================================================

def dot_header():
    return [
        "digraph G {",
        "rankdir=TB;",
        "splines=ortho;",
        "overlap=false;",
        'graph [bgcolor="white", pad="0.06", nodesep="0.28", ranksep="0.38", margin="0"];',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=8, margin="0.06,0.04", height=0.22, width=0.55];',
        'edge [fontname="Arial", fontsize=7, arrowsize=0.55, color="#4B5563", fontcolor="#111827"];',
    ]


def add_dot_node(lines, node_id, label, node_type):
    fill = {
        "Year": "#EDE9FE",
        "Event": "#DBEAFE",
        "Person": "#FEE2E2",
        "Place": "#DCFCE7",
    }.get(clean(node_type), "#FFFFFF")

    color = {
        "Year": "#7C3AED",
        "Event": "#2563EB",
        "Person": "#DC2626",
        "Place": "#16A34A",
    }.get(clean(node_type), "#444444")

    label = short_label(label, 34)

    lines.append(
        f'{node_id} [label="{html_escape(label)}", fillcolor="{fill}", color="{color}", penwidth=1.4];'
    )


def add_dot_edge(lines, src, tgt, label):
    lines.append(f'{src} -> {tgt} [label="{html_escape(nice_relation(label))}", penwidth=1.0];')


def extract_killers_from_clause(text: str) -> list[str]:
    text = clean(text)

    if "killed him" not in text.lower() and "killed her" not in text.lower():
        return []

    before = re.split(r"\bkilled him\b|\bkilled her\b", text, flags=re.I)[0]
    before = clean(before)

    before = re.sub(r",\s*foster-brother[^,]+,", ",", before, flags=re.I)
    before = re.sub(r",\s*[^,]+?,", ",", before)

    parts = re.split(r"\s+and\s+|,", before)
    names = []

    for p in parts:
        p = clean(p)
        if not p:
            continue
        if len(p.split()) > 5:
            continue
        if p.lower().startswith(("the ", "and ")):
            continue
        names.append(p)

    return list(dict.fromkeys(names))


def profile_graph_from_events(related_events: pd.DataFrame, max_events=4) -> str:
    df = related_events.copy()
    df["raw_clause"] = df["raw_clause"].astype(str).map(clean)

    df_main = (
        df.drop_duplicates(subset=["entry_id", "year", "event_type"])
        .head(max_events)
        .copy()
    )

    lines = dot_header()
    added = set()
    added_edges = set()

    def node(node_id, label, node_type):
        did = dot_id(node_id)
        if did not in added:
            add_dot_node(lines, did, label, node_type)
            added.add(did)
        return did

    def edge(src, tgt, label):
        key = (src, tgt, label)
        if key not in added_edges:
            add_dot_edge(lines, src, tgt, label)
            added_edges.add(key)

    for _, r in df_main.iterrows():
        year = int(r["year"]) if not pd.isna(r["year"]) else ""
        entry = clean(r["entry_id"])
        event_type = clean(r["event_type"])
        event_id = f"{entry}_{year}_{event_type}"
        event_label = f"{event_type} {entry}"

        yid = node(f"year_{year}", f"{year} CE", "Year")
        eid = node(f"event_{event_id}", event_label, "Event")
        edge(yid, eid, "has event")

        same_entry = df[
            (df["entry_id"].astype(str) == entry)
            & (df["event_type"].astype(str) == event_type)
        ]

        victim_node = None

        for _, s in same_entry.iterrows():
            victim = clean(s.get("victim_full", "")) or clean(s.get("victim_display", ""))
            killer = clean(s.get("killer_full", "")) or clean(s.get("killer_display", ""))
            place = clean(s.get("place_display", "")) or clean(s.get("place_full", ""))

            if victim:
                victim_node = node(f"person_v_{event_id}_{victim}", victim, "Person")
                edge(eid, victim_node, person_event_relation_label(event_type))

            if killer:
                kid = node(f"person_k_{event_id}_{killer}", killer, "Person")
                edge(kid, eid, "agent in")
                if victim_node:
                    edge(kid, victim_node, "killed")

            if place:
                pid = node(f"place_{event_id}_{place}", place, "Place")
                edge(eid, pid, "place")

            for kname in extract_killers_from_clause(s["raw_clause"]):
                if victim_node:
                    kid = node(f"person_k_clause_{entry}_{kname}", kname, "Person")
                    edge(kid, victim_node, "killed")

    lines.append("}")
    return "\n".join(lines)


def show_profile_graph(related_events: pd.DataFrame, max_events=4):
    st.markdown("### Knowledge Graph")

    if related_events.empty:
        st.info("No graph available.")
        return

    graph_col, key_col = st.columns([0.62, 0.38])

    with graph_col:
        st.graphviz_chart(
            profile_graph_from_events(related_events, max_events=max_events),
            width="stretch",
        )

    with key_col:
        st.markdown(
            """
<div class="legend-box">
<b>How to read this graph</b><br>
<span style="color:#7C3AED;">■</span> Year &nbsp;
<span style="color:#2563EB;">■</span> Event &nbsp;
<span style="color:#DC2626;">■</span> Person &nbsp;
<span style="color:#16A34A;">■</span> Place<br><br>
Arrows show the direction of the relationship.
</div>
""",
            unsafe_allow_html=True,
        )


# ============================================================
# FULL GRAPH: COMPLETE INTERCONNECTION GRAPH
# ============================================================


def safe_id(prefix, value):
    """Internal node ID. This is never shown as the public label."""
    v = re.sub(r"[^A-Za-z0-9_]+", "_", clean(value))
    v = re.sub(r"_+", "_", v).strip("_")
    return f"{prefix}_{v or 'node'}"[:160]


def year_label(values, max_items=5):
    years = sorted(set(int(x) for x in values if not pd.isna(x)))
    if not years:
        return ""
    if len(years) <= max_items:
        return ", ".join(map(str, years))
    return ", ".join(map(str, years[:max_items])) + "…"


def full_years(values):
    years = sorted(set(int(x) for x in values if not pd.isna(x)))
    return ", ".join(map(str, years))


def clean_place_for_graph(place):
    """
    Clean only the display used in the full graph.
    Examples:
      Muiresc between Cenél Cairpri -> Muiresc
      Dál Riata or nAraide -> Dál Riata
    """
    s = simple_place(place)
    if not s:
        return ""

    if re.search(r"\s+between\s+", s, flags=re.I):
        s = re.split(r"\s+between\s+", s, flags=re.I)[0]

    if re.search(r"\s+or\s+", s, flags=re.I):
        s = re.split(r"\s+or\s+", s, flags=re.I)[0]

    return clean(s)


def clean_person_for_graph(name):
    """
    Clean only the display used in the full graph.

    Returns:
      clean_person, place_relations, person_relations

    Examples:
      Suibne Menn—the king of Ireland -> Suibne Menn + (king of, Ireland)
      Ségéne from Achad Claidib -> Ségéne + (from, Achad Claidib)
      Suibne moccu Urthrí abbot of Í -> Suibne moccu Urthrí + (abbot of, Í)
      Dúnchad grandson of Rónán -> Dúnchad + (grandson of, Rónán)
    """
    s = clean(name)
    place_relations = []
    person_relations = []
    if not s:
        return "", place_relations, person_relations

    s = s.replace("—", " - ").replace("–", " - ")

    # Role/title after dash: Suibne Menn—the king of Ireland
    m = re.search(r"^(.+?)\s*-\s*the\s+king\s+of\s+(.+)$", s, flags=re.I)
    if m:
        person = clean(m.group(1))
        place = clean_place_for_graph(m.group(2))
        if place:
            place_relations.append(("king of", place))
        return simple_name(person), place_relations, person_relations

    # Person from place
    m = re.search(r"^(.+?)\s+from\s+(.+)$", s, flags=re.I)
    if m:
        person = clean(m.group(1))
        place = clean_place_for_graph(m.group(2))
        if place:
            place_relations.append(("from", place))
        return simple_name(person), place_relations, person_relations

    # Person abbot/bishop/king of place
    for rel in ["abbot of", "bishop of", "king of"]:
        m = re.search(rf"^(.+?)\s+{rel}\s+(.+)$", s, flags=re.I)
        if m:
            person = clean(m.group(1))
            place = clean_place_for_graph(m.group(2))
            if place:
                place_relations.append((rel, place))
            return simple_name(person), place_relations, person_relations

    # Genealogy in name: represent as person-to-person relation, not as a long person label.
    m = re.search(r"^(.+?)\s+grandson\s+of\s+(.+)$", s, flags=re.I)
    if m:
        person = simple_name(m.group(1))
        ancestor = simple_name(m.group(2))
        if ancestor:
            person_relations.append(("grandson of", ancestor))
        return person, place_relations, person_relations

    return simple_name(s), place_relations, person_relations


def extract_lineage_from_name_for_graph(name):
    """Extract child -> father -> grandfather from phrases like 'Aed son of X son of Y'."""
    s = clean(name)
    if not s or " son of " not in s.lower():
        return []

    parts = re.split(r"\s+son of\s+", s, flags=re.I)
    parts = [simple_name(p) for p in parts if clean(p)]
    if len(parts) < 2:
        return []

    rels = []
    child = parts[0]
    father = parts[1]
    if child and father and not is_bad_person_name(child) and not is_bad_person_name(father):
        rels.append((child, father, "father"))

    if len(parts) >= 3:
        grandfather = parts[2]
        if child and grandfather and not is_bad_person_name(grandfather):
            rels.append((child, grandfather, "grandfather"))
        if father and grandfather and not is_bad_person_name(father) and not is_bad_person_name(grandfather):
            rels.append((father, grandfather, "father"))

    return rels


def extract_roles_places_for_graph(text):
    """Extract person-place style relations from the annal sentence for full graph links."""
    text = clean(text)
    out = []

    patterns = [
        (r"\bking of ([A-ZÁÉÍÓÚÍ][A-Za-zÁÉÍÓÚáéíóúÍíÁÉÓÚḃḂṅṄ \-]+)", "king of"),
        (r"\babbot of ([A-ZÁÉÍÓÚÍ][A-Za-zÁÉÍÓÚáéíóúÍíÁÉÓÚḃḂṅṄ \-]+)", "abbot of"),
        (r"\bbishop of ([A-ZÁÉÍÓÚÍ][A-Za-zÁÉÍÓÚáéíóúÍíÁÉÓÚḃḂṅṄ \-]+)", "bishop of"),
        (r"\bin ([A-ZÁÉÍÓÚÍ][A-Za-zÁÉÍÓÚáéíóúÍíÁÉÓÚḃḂṅṄ \-]+)", "at"),
        (r"\bat ([A-ZÁÉÍÓÚÍ][A-Za-zÁÉÍÓÚáéíóúÍíÁÉÓÚḃḂṅṄ \-]+)", "at"),
        (r"\bfrom ([A-ZÁÉÍÓÚÍ][A-Za-zÁÉÍÓÚáéíóúÍíÁÉÓÚḃḂṅṄ \-]+)", "from"),
    ]

    bad_places = {"December", "Sunday", "Saxons", "Picts", "Britons"}

    for pat, rel in patterns:
        for m in re.finditer(pat, text):
            place = clean(m.group(1))
            place = re.split(r"\s+by\s+|\s+and\s+|\s+in which\s+|,|\.|;", place)[0]
            place = clean_place_for_graph(place)
            if place and len(place) <= 45 and place not in bad_places:
                out.append((rel, place))

    return out


def plain_evidence_title(label, node_type, rows):
    """Plain text for click/hover. No HTML tags, so <b> never appears."""
    lines = [label, f"Type: {node_type}"]

    if rows is None or rows.empty:
        return "\n".join(lines)

    years = sorted(set(int(y) for y in rows["year"].dropna()))
    entries = list(dict.fromkeys(clean(x) for x in rows["entry_id"].dropna()))
    event_types = list(dict.fromkeys(clean(x) for x in rows["event_type"].dropna()))

    if years:
        lines.append(f"Year(s): {', '.join(map(str, years))}")
    if entries:
        lines.append(f"Entry ID(s): {', '.join(entries[:20])}")
    if event_types:
        lines.append(f"Event type(s): {', '.join(event_types[:12])}")

    lines.append("")
    lines.append("Source evidence")

    evidence_rows = rows.drop_duplicates(["entry_id", "year", "event_type", "raw_clause"])
    for _, r in evidence_rows.head(15).iterrows():
        lines.append("")
        lines.append(
            f"Entry: {clean(r['entry_id'])} | "
            f"Year: {int(r['year'])} | "
            f"Event: {clean(r['event_type'])}"
        )
        lines.append(f"“{clean(r['raw_clause'])}”")

    if len(evidence_rows) > 15:
        lines.append("")
        lines.append("More evidence is available on the People or Places page.")

    return "\n".join(lines)


def inject_click_panel(html):
    """
    Adds one fixed right-side click panel to the PyVis graph.
    Important: the graph does NOT use PyVis hover titles, so the old stuck tooltip box disappears.
    """
    panel = """
<style>
#kg-info-panel {
 position:absolute; right:18px; top:18px; width:340px; max-height:720px; overflow:auto;
 background:white; border:1px solid #d6d3d1; border-radius:14px; padding:13px 15px;
 box-shadow:0 8px 24px rgba(0,0,0,.16); font-family:Arial,sans-serif; font-size:13px;
 line-height:1.42; z-index:9999; white-space:normal;
}
#kg-info-panel .kg-text { white-space:pre-wrap; }
#kg-info-panel .hint { color:#64748b; }
</style>

<div id="kg-info-panel">
  <b>Explore the graph</b>
  <p class="hint">Click a person, place, or relationship to read the annal entry, year, event type, and source sentence.</p>
</div>

<script>
function kgEscapeHTML(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
setTimeout(function() {
 if (typeof network !== "undefined") {
   network.on("click", function(params) {
     var panel = document.getElementById("kg-info-panel");
     if (!panel) return;
     if (params.nodes.length > 0) {
       var node = network.body.data.nodes.get(params.nodes[0]);
       var text = node.kg_info || node.label || "Node";
       panel.innerHTML = '<div class="kg-text">' + kgEscapeHTML(text) + '</div>';
     } else if (params.edges.length > 0) {
       var edge = network.body.data.edges.get(params.edges[0]);
       var text = edge.kg_info || edge.label || "Relationship";
       panel.innerHTML = '<div class="kg-text">' + kgEscapeHTML(text) + '</div>';
     }
   });
 }
}, 900);
</script>
"""
    return html.replace("</body>", panel + "\n</body>")




def pyvis_full_graph(height="900px"):
    """
    Single Full Knowledge Graph for Annals Explorer.

    Design:
      - No artificial Annals of Ulster centre node.
      - No visible event / entry nodes.
      - Only real entities are shown as nodes: Person and Place / kingdom / monastery / group.
      - Edges are direct relationships: Person -> Person and Person -> Place.
      - Year, entry ID, event type, and source sentence are kept in the click panel.
      - Same person/place is merged into one node.
    """

    ev = events_df[
        (events_df["year"] >= DATA_MIN_YEAR)
        & (events_df["year"] <= DATA_MAX_YEAR)
    ].copy()

    net = Network(
        height=height,
        width="100%",
        bgcolor="#FBF7ED",
        font_color="#111111",
        directed=True,
        notebook=False,
    )

    net.force_atlas_2based(
        gravity=-185,
        central_gravity=0.030,
        spring_length=260,
        spring_strength=0.045,
        damping=0.88,
        overlap=1.25,
    )

    def canonical_person(name: str) -> str:
        person, _place_rels, _person_rels = clean_person_for_graph(name)
        person = display_person_name(person)
        if not person or is_bad_person_name(person):
            return ""
        return person

    def canonical_place(place: str) -> str:
        p = clean_place_for_graph(place)
        if not p:
            return ""
        if p.strip().lower() in {"i", "í", "hi", "hí"}:
            return "Í / Iona"
        return p

    people_rows: dict[str, list[int]] = {}
    place_rows: dict[str, list[int]] = {}
    edge_map: dict[tuple, list[int]] = {}

    def remember_person(person: str, idx: int) -> str:
        person = canonical_person(person)
        if person:
            people_rows.setdefault(person, []).append(idx)
        return person

    def remember_place(place: str, idx: int) -> str:
        place = canonical_place(place)
        if place:
            place_rows.setdefault(place, []).append(idx)
        return place

    def add_edge_record(src_type: str, src: str, dst_type: str, dst: str, rel: str, kind: str, idx: int):
        if not src or not dst or src == dst:
            return
        rel = clean(rel)
        key = (src_type, src, dst_type, dst, rel, kind)
        edge_map.setdefault(key, []).append(idx)

    # Build entity nodes and direct relationships from all 600–700 rows.
    for idx, r in ev.iterrows():
        if pd.isna(r.get("year", None)):
            continue

        raw = clean(r.get("raw_clause", ""))
        event_type = clean(r.get("event_type", "")).lower()

        victim_raw = clean(r.get("victim_full", "")) or clean(r.get("victim_display", ""))
        killer_raw = clean(r.get("killer_full", "")) or clean(r.get("killer_display", ""))
        place_raw = clean(r.get("place_display", "")) or clean(r.get("place_full", ""))

        victim = ""
        killer = ""
        people_in_entry = []

        for role, raw_person in [("victim", victim_raw), ("agent", killer_raw)]:
            if not raw_person:
                continue

            base_person, derived_place_rels, derived_person_rels = clean_person_for_graph(raw_person)
            person = remember_person(base_person, idx)
            if not person:
                continue

            people_in_entry.append(person)

            if role == "victim":
                victim = person
            if role == "agent":
                killer = person

            # Person -> Place from embedded role/context phrases.
            for rel, dplace in derived_place_rels:
                dplace = remember_place(dplace, idx)
                if dplace:
                    add_edge_record("PERSON", person, "PLACE", dplace, rel, "place", idx)

            # Person -> Person from embedded context.
            for rel, target_person in derived_person_rels:
                target_person = remember_person(target_person, idx)
                if target_person:
                    add_edge_record("PERSON", person, "PERSON", target_person, rel, "family", idx)

            # Full lineage chain, e.g. "Aedán son of Gabrán son of Domangart".
            for child, parent, rel in extract_lineage_from_name_for_graph(raw_person):
                child = remember_person(child, idx)
                parent = remember_person(parent, idx)
                if child and parent:
                    add_edge_record("PERSON", child, "PERSON", parent, rel, "family", idx)

        # Person -> Person conflict.
        if killer and victim and killer != victim:
            add_edge_record("PERSON", killer, "PERSON", victim, "killed", "conflict", idx)

        # Clause-based killers such as "X and Y killed him".
        for k in extract_killers_from_clause(raw):
            kperson = remember_person(k, idx)
            if kperson:
                people_in_entry.append(kperson)
                if victim:
                    add_edge_record("PERSON", kperson, "PERSON", victim, "killed", "conflict", idx)

        people_in_entry = list(dict.fromkeys([p for p in people_in_entry if p]))

        # Structured event place as a direct Person -> Place relation.
        place = remember_place(place_raw, idx)
        if place and people_in_entry:
            rel = "at" if event_type in {"battle", "slaying"} else "associated with"
            for person in people_in_entry:
                add_edge_record("PERSON", person, "PLACE", place, rel, "place", idx)

        # Relations directly visible in the source sentence.
        for rel, role_place in extract_roles_places_for_graph(raw):
            role_place = remember_place(role_place, idx)
            if role_place and people_in_entry:
                for person in people_in_entry:
                    add_edge_record("PERSON", person, "PLACE", role_place, rel, "place", idx)

    def node_info(label: str, node_type: str, rows: pd.DataFrame) -> str:
        rows = rows.drop_duplicates(["entry_id", "year", "event_type", "raw_clause"])
        years = sorted(set(int(y) for y in rows["year"].dropna()))
        entries = list(dict.fromkeys(clean(x) for x in rows["entry_id"].dropna()))
        event_types = list(dict.fromkeys(clean(x) for x in rows["event_type"].dropna()))

        lines = [label, f"Type: {node_type}"]
        if years:
            lines.append(f"Year(s): {', '.join(map(str, years))}")
        if entries:
            lines.append(f"Entry ID(s): {', '.join(entries[:18])}")
        if event_types:
            lines.append(f"Event type(s): {', '.join(event_types[:10])}")

        lines.append("")
        lines.append("Source evidence")
        for _, rr in rows.head(10).iterrows():
            lines.append("")
            lines.append(
                f"Entry: {clean(rr['entry_id'])} | "
                f"Year: {int(rr['year']) if not pd.isna(rr['year']) else ''} | "
                f"Event: {clean(rr['event_type'])}"
            )
            lines.append(f"“{clean(rr['raw_clause'])}”")

        if len(rows) > 10:
            lines.append("")
            lines.append("More evidence is available on the People or Places page.")

        return "\n".join(lines)

    def edge_info(src: str, dst: str, rel: str, indices: list[int]) -> str:
        rows = ev.loc[list(dict.fromkeys(indices))].copy()
        rows = rows.drop_duplicates(["entry_id", "year", "event_type", "raw_clause"])
        years = sorted(set(int(y) for y in rows["year"].dropna()))
        entries = list(dict.fromkeys(clean(x) for x in rows["entry_id"].dropna()))
        event_types = list(dict.fromkeys(clean(x) for x in rows["event_type"].dropna()))

        lines = [
            f"Relationship: {src} → {dst}",
            f"Relation: {rel}",
        ]
        if years:
            lines.append(f"Year(s): {', '.join(map(str, years))}")
        if entries:
            lines.append(f"Entry ID(s): {', '.join(entries[:12])}")
        if event_types:
            lines.append(f"Event type(s): {', '.join(event_types[:8])}")

        lines.append("")
        lines.append("Source evidence")
        for _, rr in rows.head(10).iterrows():
            lines.append("")
            lines.append(
                f"Entry: {clean(rr['entry_id'])} | "
                f"Year: {int(rr['year']) if not pd.isna(rr['year']) else ''} | "
                f"Event: {clean(rr['event_type'])}"
            )
            lines.append(f"“{clean(rr['raw_clause'])}”")

        if len(rows) > 10:
            lines.append("")
            lines.append("More evidence is available on the People or Places page.")

        return "\n".join(lines)

    connected = set()
    for (src_type, src, dst_type, dst, rel, kind), indices in edge_map.items():
        connected.add((src_type, src))
        connected.add((dst_type, dst))

    added_nodes = set()
    visible_nodes = set()
    seen_edges = set()

    def evidence_count_size(base: int, count: int, cap: int) -> int:
        return min(cap, base + int(min(count, 12) * 1.45))

    def add_node(node_id: str, label: str, node_type: str, kg_info: str, count: int = 1):
        if node_id in added_nodes:
            return

        if node_type == "Person":
            color = {
                "background": "#16A34A",
                "border": "#065F46",
                "highlight": {"background": "#22C55E", "border": "#064E3B"},
            }
            size = evidence_count_size(24, count, 44)
            font_size = 16
        else:
            color = {
                "background": "#FACC15",
                "border": "#A16207",
                "highlight": {"background": "#FDE047", "border": "#854D0E"},
            }
            size = evidence_count_size(26, count, 46)
            font_size = 16

        net.add_node(
            node_id,
            label=short_label(label, 28),
            kg_info=kg_info,
            color=color,
            size=size,
            shape="dot",
            borderWidth=2,
            font={
                "size": font_size,
                "face": "arial",
                "color": "#111111",
                "strokeWidth": 4,
                "strokeColor": "#FBF7ED",
            },
        )
        added_nodes.add(node_id)
        visible_nodes.add(node_id)

    def edge_style(rel: str, kind: str):
        rel_l = clean(rel).lower()
        if kind == "conflict" or rel_l == "killed":
            return "#DC2626", 2.5, False, "killed"
        if kind == "family" or rel_l in {"father", "grandfather", "grandson of"}:
            return "#7C3AED", 2.0, True, rel
        if rel_l == "king of":
            return "#2563EB", 2.2, False, "king of"
        if rel_l in {"abbot of", "bishop of"}:
            return "#0F766E", 2.0, False, rel
        if rel_l == "from":
            return "#EA580C", 1.8, False, "from"
        if rel_l == "at":
            return "#EA580C", 1.5, False, ""
        return "#EA580C", 1.3, False, ""

    def add_edge(src_id: str, dst_id: str, rel: str, kg_info: str, kind: str, weight: int = 1):
        if src_id not in visible_nodes or dst_id not in visible_nodes or src_id == dst_id:
            return

        key = (src_id, dst_id, rel, kind)
        if key in seen_edges:
            return
        seen_edges.add(key)

        color, width, dashes, visible_label = edge_style(rel, kind)

        net.add_edge(
            src_id,
            dst_id,
            label=visible_label,
            kg_info=kg_info,
            arrows="to",
            color={"color": color, "highlight": color, "hover": color, "opacity": 0.82},
            width=width + min(weight, 5) * 0.10,
            dashes=dashes,
            font={
                "size": 11 if visible_label else 0,
                "color": "#111111",
                "strokeWidth": 4,
                "strokeColor": "#FBF7ED",
            },
        )

    for person, indices in sorted(people_rows.items(), key=lambda x: x[0].lower()):
        if ("PERSON", person) in connected:
            rows = ev.loc[list(dict.fromkeys(indices))]
            add_node(
                safe_id("PERSON", person),
                person,
                "Person",
                node_info(person, "Person", rows),
                len(rows),
            )

    for place, indices in sorted(place_rows.items(), key=lambda x: x[0].lower()):
        if ("PLACE", place) in connected:
            rows = ev.loc[list(dict.fromkeys(indices))]
            add_node(
                safe_id("PLACE", place),
                place,
                "Place",
                node_info(place, "Place / kingdom / monastery / group", rows),
                len(rows),
            )

    priority = {"conflict": 0, "family": 1, "place": 2}
    for (src_type, src, dst_type, dst, rel, kind), indices in sorted(
        edge_map.items(),
        key=lambda x: priority.get(x[0][5], 9),
    ):
        add_edge(
            safe_id(src_type, src),
            safe_id(dst_type, dst),
            rel,
            edge_info(src, dst, rel, indices),
            kind,
            weight=len(indices),
        )

    net.set_options("""
{
  "nodes": {
    "borderWidth": 2,
    "shadow": { "enabled": true, "size": 7, "x": 1, "y": 1 },
    "font": {
      "face": "arial",
      "color": "#111111",
      "strokeWidth": 4,
      "strokeColor": "#FBF7ED"
    }
  },
  "edges": {
    "arrows": { "to": { "enabled": true, "scaleFactor": 0.30 } },
    "smooth": { "enabled": true, "type": "dynamic", "roundness": 0.20 },
    "selectionWidth": 3,
    "hoverWidth": 2
  },
  "physics": {
    "enabled": true,
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {
      "gravitationalConstant": -185,
      "centralGravity": 0.030,
      "springLength": 260,
      "springConstant": 0.045,
      "damping": 0.88,
      "avoidOverlap": 1.25
    },
    "stabilization": { "enabled": true, "iterations": 1500, "fit": true }
  },
  "interaction": {
    "hover": false,
    "tooltipDelay": 200,
    "hideEdgesOnDrag": true,
    "navigationButtons": true,
    "keyboard": true,
    "multiselect": true,
    "zoomView": true,
    "dragView": true
  }
}
""")

    out_html = OUT_DIR / "full_annals_entity_relationship_graph.html"
    net.save_graph(str(out_html))
    return inject_click_panel(out_html.read_text(encoding="utf-8"))

# ============================================================
# ============================================================
# CHATBOT: EVIDENCE-FIRST QA
# ============================================================

EVENT_WORDS = {
    "battle": ["battle", "battles", "war", "wars", "fought"],
    "death": ["death", "deaths", "died", "dead"],
    "slaying": ["slaying", "slayings", "slain", "killed", "killing", "murder"],
    "repose": ["repose", "reposes", "abbot", "bishop", "saint"],
}

QUESTION_WORDS = {
    "who", "what", "where", "when", "which", "why", "how", "tell", "about",
    "show", "give", "find", "search", "is", "was", "were", "are", "the", "of",
    "in", "to", "from", "and", "or", "me", "please", "happened", "happen",
    "events", "event", "records", "record", "between", "during", "after", "before",
    "father", "grandfather", "killed", "kill", "did", "does", "record", "annals",
    "annal", "ulster", "ce", "year", "years"
}


def normalise_query_text(q: str) -> str:
    q = clean(q).lower()
    q = q.replace("–", "-").replace("—", "-")
    return q


def title_event(event_type: str) -> str:
    e = clean(event_type)
    return e[:1].upper() + e[1:] if e else "Record"


def get_public_people_names() -> list[str]:
    names = sorted(
        set(
            public_name
            for x in people_df["display_name"].tolist()
            for public_name in expand_person_display_names(x)
            if clean(public_name) and not is_bad_person_name(public_name)
        ),
        key=lambda x: len(x),
        reverse=True,
    )
    return names


def get_public_place_names() -> list[str]:
    return sorted(set(ALL_PLACE_NAMES), key=lambda x: len(x), reverse=True)


def exact_or_contained_entity(q: str, names: list[str]) -> str:
    """Prefer exact entity match, then whole-phrase containment."""
    qn = normalise_query_text(q)
    qn_clean = re.sub(r"[^a-záéíóúíḃṅ\s-]", " ", qn, flags=re.I)
    qn_clean = clean(qn_clean).lower()

    for name in names:
        if qn_clean == name.lower():
            return name

    for name in names:
        n = name.lower()
        if len(n) >= 4 and re.search(rf"\b{re.escape(n)}\b", qn_clean):
            return name

    return ""


def detect_event_type(q: str) -> str:
    qn = normalise_query_text(q)
    for event_type, words in EVENT_WORDS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", qn) for w in words):
            # But do not interpret "who killed X" as event-type listing.
            if event_type == "slaying" and qn.startswith("who killed"):
                continue
            return event_type
    return ""


def rows_for_public_person(name: str) -> pd.DataFrame:
    if not name:
        return pd.DataFrame()
    matching_people = people_df[
        people_df["display_name"].apply(
            lambda x: name.lower() in [p.lower() for p in expand_person_display_names(x)]
        )
    ]
    ids = matching_people["person_id"].tolist()
    if not ids:
        return pd.DataFrame()
    rows = events_df[
        events_df["victim_id"].isin(ids)
        | events_df["killer_id"].isin(ids)
        | events_df["raw_clause"].str.lower().str.contains(re.escape(name.lower()), na=False)
    ].copy()
    return rows.drop_duplicates(subset=["entry_id", "year", "event_type", "raw_clause"])


def rows_for_public_place(place: str) -> pd.DataFrame:
    if not place:
        return pd.DataFrame()
    ids = places_df[
        places_df["display_place"].apply(
            lambda x: place.lower() in [p.lower() for p in expand_place_display_names(x)]
        )
    ]["place_id"].tolist()
    rows = events_df[
        events_df["place_id"].isin(ids)
        | events_df["place_display"].str.lower().str.contains(re.escape(place.lower()), na=False)
        | events_df["place_full"].str.lower().str.contains(re.escape(place.lower()), na=False)
        | events_df["raw_clause"].str.lower().str.contains(re.escape(place.lower()), na=False)
    ].copy()
    return rows.drop_duplicates(subset=["entry_id", "year", "event_type", "raw_clause"])


def format_source_rows(rows: pd.DataFrame, max_rows: int = 8) -> str:
    if rows is None or rows.empty:
        return ""

    rows = rows.copy().drop_duplicates(subset=["entry_id", "year", "event_type", "raw_clause"])
    rows = rows.sort_values(["year", "entry_id", "event_type"])

    lines = []
    for _, r in rows.head(max_rows).iterrows():
        year = int(r["year"]) if not pd.isna(r["year"]) else ""
        event_type = title_event(r.get("event_type", ""))
        entry = clean(r.get("entry_id", ""))
        clause = clean(r.get("raw_clause", ""))
        lines.append(f"- **{entry} ({year}) — {event_type}:** {clause}")

    if len(rows) > max_rows:
        lines.append(f"- …and {len(rows) - max_rows} more matching record(s).")

    return "\n".join(lines)


def answer_period_range(start: int, end: int, note: str = "") -> str:
    sub = events_df[(events_df["year"] >= start) & (events_df["year"] <= end)].copy()

    if sub.empty:
        return note or f"I could not find records for **{start}–{end} CE** in the loaded Annals data."

    years = sorted(set(int(y) for y in sub["year"].dropna()))

    lines = []
    if note:
        lines.append(note)
        lines.append("")

    if start == end:
        lines.append(f"In **{start} CE**, the Annals record the following evidence:")
    else:
        lines.append(f"Between **{start} and {end} CE**, the Annals record evidence across **{len(years)} year(s)**.")

    # Group by year so it reads historically, not like a data report.
    for y in years[:12]:
        yrows = sub[sub["year"] == y].copy()
        lines.append(f"\n**{y} CE**")
        lines.append(format_source_rows(yrows, max_rows=8))

    if len(years) > 12:
        lines.append(f"\nI have shown the first 12 years. There are more records in this range.")

    return "\n".join(lines)



def find_people_by_name(name: str, exact_simple=False) -> pd.DataFrame:
    """
    Find people rows by the public/normalised display name.
    This is used by the chatbot and keeps names like "Aed" working.
    """
    name = display_person_name(name).lower()
    if not name:
        return pd.DataFrame()

    temp = people_df.copy()
    temp["public_names"] = temp["display_name"].apply(expand_person_display_names)
    temp["public_names_lower"] = temp["public_names"].apply(lambda xs: [x.lower() for x in xs])
    temp["search_text"] = (
        temp["display_name"].astype(str).str.lower() + " "
        + temp.get("full_names", pd.Series([""] * len(temp))).astype(str).str.lower() + " "
        + temp["public_names_lower"].apply(lambda xs: " ".join(xs))
    )

    if exact_simple:
        return temp[temp["public_names_lower"].apply(lambda xs: name in xs)].copy()

    return temp[
        temp["public_names_lower"].apply(lambda xs: any(name == x or name in x for x in xs))
        | temp["search_text"].str.contains(re.escape(name), na=False)
    ].copy()

def answer_person(name: str) -> str:
    name = clean(name)
    rows = rows_for_public_person(name)
    people = find_people_by_name(name, exact_simple=True)
    if people.empty:
        people = find_people_by_name(name, exact_simple=False)

    if rows.empty and people.empty:
        return ""

    label = name or display_person_name(people.iloc[0]["display_name"])
    lines = [f"**{label}** is mentioned in the loaded Annals evidence."]

    if not people.empty:
        years = sorted(set(str(y) for y in people["years"].astype(str) if clean(y)))
        roles = sorted(set(r for r in people["roles"].astype(str) if clean(r)))
        contexts = []
        for raw_name in people["display_name"].tolist():
            contexts.extend(person_context_from_name(raw_name))
        contexts = list(dict.fromkeys([c for c in contexts if c]))
        if years:
            lines.append(f"**Year(s):** {', '.join(years)}")
        if roles:
            lines.append(f"**Role(s):** {', '.join(roles)}")
        if contexts:
            lines.append(f"**Context:** {'; '.join(contexts)}")

    if not rows.empty:
        lines.append("\n**Source evidence:**")
        lines.append(format_source_rows(rows, max_rows=10))

    return "\n".join(lines)


def answer_place(place: str) -> str:
    place = clean(place)
    rows = rows_for_public_place(place)
    if rows.empty:
        return ""

    lines = [f"**{place}** appears in the loaded Annals evidence."]

    # Context from original place labels, if any.
    context_notes = []
    for raw_place in places_df["display_place"].tolist():
        if place.lower() in [p.lower() for p in expand_place_display_names(raw_place)]:
            context_notes.extend(place_context_from_name(raw_place))
    context_notes = list(dict.fromkeys([c for c in context_notes if c]))
    if context_notes:
        lines.append(f"**Context:** {'; '.join(context_notes)}")

    years = sorted(set(int(y) for y in rows["year"].dropna()))
    if years:
        lines.append(f"**Year(s):** {', '.join(map(str, years))}")

    lines.append("\n**Source evidence:**")
    lines.append(format_source_rows(rows, max_rows=10))
    return "\n".join(lines)


def parse_father_grandfather_from_text(person: str, text: str):
    p = re.escape(simple_name(person))
    pat = rf"\b{p}\b\s+son of\s+([^,.;]+?)(?:\s+son of\s+([^,.;]+?))?(?:,|\.|;|$)"
    m = re.search(pat, text, flags=re.I)

    if not m:
        return "", ""

    father = clean(m.group(1))
    grandfather = clean(m.group(2)) if m.group(2) else ""
    father = re.split(r"\s+son of\s+", father)[0].strip()
    grandfather = re.split(r"\s+son of\s+", grandfather)[0].strip()
    return father, grandfather


def answer_lineage(question: str, want="father") -> str:
    person = extract_name_from_question(question)
    if not person:
        return "Please mention a person name."

    results = []

    # Use structured people table first.
    candidates = find_people_by_name(person, exact_simple=True)
    if candidates.empty:
        candidates = find_people_by_name(person, exact_simple=False)

    for _, p in candidates.iterrows():
        value = clean(p.get("father", "")) if want == "father" else clean(p.get("grandfather", ""))
        if value:
            results.append((value, clean(p.get("source_entries", "")), clean(p.get("years", "")), clean(p.get("evidence_preview", ""))))

    # Also scan source text for explicit “son of”.
    for _, r in events_df.iterrows():
        text = clean(r.get("raw_clause", ""))
        father, grandfather = parse_father_grandfather_from_text(person, text)
        value = father if want == "father" else grandfather
        if value:
            results.append((value, clean(r["entry_id"]), str(int(r["year"])), text))

    if not results:
        return f"I could not find a recorded **{want}** for **{person}** in the loaded Annals data."

    lines = [f"Recorded **{want}(s)** of **{person}**:"]
    seen = set()
    for value, entry, year, evidence in results:
        key = (value, entry, year, evidence)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- **{value}** — {entry} ({year}). Evidence: {evidence}")
    return "\n".join(lines)


def answer_killed(question: str) -> str:
    person = extract_name_from_question(question)
    if not person:
        return "Please mention the person you are asking about."

    person_name = exact_or_contained_entity(person, get_public_people_names()) or person
    person_low = simple_name(person_name).lower()

    rows = events_df[
        events_df["victim_display"].str.lower().eq(person_low)
        | events_df["victim_full"].str.lower().str.contains(rf"\b{re.escape(person_low)}\b", na=False)
        | events_df["raw_clause"].str.lower().str.contains(rf"\b{re.escape(person_low)}\b", na=False)
    ].copy()

    if rows.empty:
        return f"I could not find a killing/slaying record for **{person_name}** in the loaded Annals data."

    lines = [f"I found the following killing/slaying evidence connected to **{person_name}**:"]
    for _, r in rows.head(10).iterrows():
        killer = clean(r.get("killer_display", "")) or clean(r.get("killer_full", ""))
        prefix = f"{killer} is recorded as killer/agent. " if killer else ""
        lines.append(
            f"- **{clean(r['entry_id'])} ({int(r['year'])}) — {title_event(r.get('event_type', ''))}:** "
            f"{prefix}{clean(r['raw_clause'])}"
        )
    return "\n".join(lines)


def answer_event_type(question: str) -> str:
    et = detect_event_type(question)
    if not et:
        return ""

    start, end, _ = extract_year_range_any(question)
    rows = events_df[events_df["event_type"].str.lower() == et].copy()

    note = ""
    if start is not None and end is not None:
        start2, end2, note = overlap_with_dataset(start, end)
        if start2 is None:
            return note
        rows = rows[(rows["year"] >= start2) & (rows["year"] <= end2)]

    if rows.empty:
        return f"I could not find **{et}** records for that question in the loaded Annals data."

    lines = []
    if note:
        lines.append(note)
        lines.append("")
    lines.append(f"I found the following **{et}** evidence:")
    lines.append(format_source_rows(rows, max_rows=12))
    return "\n".join(lines)


def build_search_events() -> pd.DataFrame:
    ev = events_df.copy()
    ev["search_text"] = (
        ev["entry_id"].astype(str) + " "
        + ev["event_type"].astype(str) + " "
        + ev["victim_display"].astype(str) + " "
        + ev["victim_full"].astype(str) + " "
        + ev["killer_display"].astype(str) + " "
        + ev["killer_full"].astype(str) + " "
        + ev["place_display"].astype(str) + " "
        + ev["place_full"].astype(str) + " "
        + ev["raw_clause"].astype(str)
    ).str.lower()
    return ev


def extract_keyword_terms(q: str) -> list[str]:
    q = normalise_query_text(q)
    words = re.findall(r"[a-zA-ZÁÉÍÓÚáéíóúÍíḃḂṅṄ]+", q)
    return [w for w in words if len(w) > 2 and w not in QUESTION_WORDS]


def answer_keyword_search(question: str) -> str:
    q = clean(question)
    if not q:
        return "Please ask a question."

    ev = build_search_events()
    q_low = q.lower()

    # For short direct queries like "Ireland", exact raw evidence containment should work.
    exact_rows = ev[ev["search_text"].str.contains(re.escape(q_low), na=False)].copy()

    if not exact_rows.empty:
        lines = [f"I found **{q}** in the Annals evidence:"]
        lines.append(format_source_rows(exact_rows, max_rows=10))
        return "\n".join(lines)

    words = extract_keyword_terms(q)
    if not words:
        return "I could not identify a searchable term in that question. Try a person, place, year, event, or relationship."

    ev["score"] = ev["search_text"].apply(lambda txt: sum(1 for w in words if w in txt))
    rows = ev[ev["score"] > 0].sort_values(["score", "year"], ascending=[False, True]).head(10)

    if rows.empty:
        return "I could not find matching evidence in the current Annals data."

    lines = ["I found the following relevant Annals evidence:"]
    lines.append(format_source_rows(rows, max_rows=10))
    return "\n".join(lines)


def answer_chatbot(question: str) -> str:
    q = clean(question)
    q_low = normalise_query_text(q)
    if not q:
        return "Please ask a question."

    # 1) Year / range questions.
    start, end, _ = extract_year_range_any(q_low)
    if start is not None and end is not None:
        start2, end2, note = overlap_with_dataset(start, end)
        if start2 is None:
            return note
        return answer_period_range(start2, end2, note)

    # 2) Relationship questions.
    if "grandfather" in q_low:
        return answer_lineage(q, want="grandfather")
    if "father" in q_low:
        return answer_lineage(q, want="father")
    if q_low.startswith("who killed") or q_low.startswith("who slew") or " killed " in q_low:
        return answer_killed(q)

    # 3) Exact entity detection. This prevents "Ireland" becoming a person answer.
    place_match = exact_or_contained_entity(q, get_public_place_names())
    person_match = exact_or_contained_entity(q, get_public_people_names())

    # If the user asks a bare word/phrase, prefer place if it is a known place; otherwise person; otherwise evidence search.
    bare = len(extract_keyword_terms(q)) <= 3 and not any(w in q_low for w in ["who", "where", "what", "tell", "about"])
    if bare and place_match:
        return answer_place(place_match)
    if bare and person_match:
        return answer_person(person_match)

    # 4) Explicit person/place style questions.
    if any(w in q_low for w in ["where", "place", "location"]):
        if place_match:
            return answer_place(place_match)
        return answer_keyword_search(q)

    if q_low.startswith("who is") or q_low.startswith("who was") or "tell me about" in q_low:
        if person_match:
            return answer_person(person_match)
        # Try extracted name only after exact matching.
        extracted = extract_name_from_question(q)
        pm = exact_or_contained_entity(extracted, get_public_people_names())
        if pm:
            return answer_person(pm)
        return answer_keyword_search(q)

    # 5) Event type questions.
    event_answer = answer_event_type(q)
    if event_answer:
        return event_answer

    # 6) Place/person contained in broader questions.
    if place_match and any(w in q_low for w in ["happened", "recorded", "mention", "mentions", "about"]):
        return answer_place(place_match)
    if person_match and any(w in q_low for w in ["happened", "recorded", "mention", "mentions", "about"]):
        return answer_person(person_match)

    # 7) General evidence search fallback.
    return answer_keyword_search(q)




# ============================================================
# COMPACT MERGED PROFILE GRAPHS FOR PEOPLE / PLACES PAGES
# ============================================================

def show_evidence_only(events: pd.DataFrame):
    """Show source evidence without adding a separate graph button for every entry."""
    if events.empty:
        st.info("No source evidence found.")
        return

    grouped = grouped_evidence_rows(events)
    for _, r in grouped.iterrows():
        year = int(r["year"]) if not pd.isna(r["year"]) else ""
        entry_id = clean(r["entry_id"])
        event_type = clean(r["event_type"])
        evidence_box(entry_id, year, event_type, r["raw_clause"])


def compact_profile_graph_from_events(label: str, related_events: pd.DataFrame, entity_type: str = "Person", max_records: int = 12) -> str:
    """
    Compact merged graph for one selected person/place.

    Purpose:
      - merge all records for the selected name into one readable graph
      - avoid the old large Year -> Event -> Person graph for every single entry
      - keep year/event/entry visible inside the event node label
    """
    df = related_events.copy()
    if df.empty:
        return "digraph G { empty [label=\"No graph available\"]; }"

    df["raw_clause"] = df["raw_clause"].astype(str).map(clean)
    df = df.drop_duplicates(subset=["entry_id", "year", "event_type", "raw_clause"])
    df = df.sort_values(["year", "entry_id", "event_type"]).head(max_records)

    lines = [
        "digraph G {",
        "rankdir=LR;",
        "splines=curved;",
        "overlap=false;",
        'graph [bgcolor="white", pad="0.08", nodesep="0.35", ranksep="0.55", margin="0"];',
        'node [fontname="Arial", fontsize=10, margin="0.08,0.05"];',
        'edge [fontname="Arial", fontsize=9, arrowsize=0.55, color="#4B5563", fontcolor="#111827"];',
    ]

    added_nodes = set()
    added_edges = set()

    def q(s: str) -> str:
        return html_escape(short_label(s, 46))

    def node(node_id: str, label_text: str, fill: str, border: str, shape: str = "box", penwidth: float = 1.6):
        node_id = dot_id(node_id)
        if node_id in added_nodes:
            return node_id
        lines.append(
            f'{node_id} [label="{q(label_text)}", shape={shape}, style="rounded,filled", '
            f'fillcolor="{fill}", color="{border}", penwidth={penwidth}];'
        )
        added_nodes.add(node_id)
        return node_id

    def edge(src: str, dst: str, label_text: str, color: str = "#4B5563", style: str = "solid"):
        key = (src, dst, label_text, color, style)
        if key in added_edges:
            return
        lines.append(
            f'{src} -> {dst} [label="{html_escape(label_text)}", color="{color}", '
            f'fontcolor="#111827", style="{style}", penwidth=1.2];'
        )
        added_edges.add(key)

    center_fill = "#FEE2E2" if entity_type == "Person" else "#DCFCE7"
    center_border = "#DC2626" if entity_type == "Person" else "#16A34A"
    center_id = node(f"center_{label}", label, center_fill, center_border, shape="ellipse", penwidth=2.2)

    for i, (_, r) in enumerate(df.iterrows(), start=1):
        year = int(r["year"]) if not pd.isna(r["year"]) else ""
        entry = clean(r.get("entry_id", ""))
        event_type = clean(r.get("event_type", ""))
        event_label = f"{entry}\\n{year} · {event_type}"
        event_id = node(f"event_{entry}_{event_type}_{i}", event_label, "#DBEAFE", "#2563EB")

        if entity_type == "Person":
            role = "mentioned"
            victim = display_person_name(clean(r.get("victim_display", "")) or clean(r.get("victim_full", "")))
            killer = display_person_name(clean(r.get("killer_display", "")) or clean(r.get("killer_full", "")))

            if victim.lower() == label.lower():
                role = "victim"
                edge(event_id, center_id, role, "#DC2626")
                if killer and killer.lower() != label.lower() and not is_bad_person_name(killer):
                    kid = node(f"person_killer_{killer}_{i}", killer, "#FEE2E2", "#DC2626", shape="ellipse")
                    edge(kid, event_id, "agent", "#DC2626")
            elif killer.lower() == label.lower():
                role = "agent"
                edge(center_id, event_id, role, "#DC2626")
                if victim and victim.lower() != label.lower() and not is_bad_person_name(victim):
                    vid = node(f"person_victim_{victim}_{i}", victim, "#FEE2E2", "#DC2626", shape="ellipse")
                    edge(event_id, vid, "victim", "#DC2626")
            else:
                edge(event_id, center_id, role, "#4B5563", "dashed")

            place = display_place_name(clean(r.get("place_display", "")) or clean(r.get("place_full", "")))
            if place:
                pid = node(f"place_{place}_{i}", place, "#DCFCE7", "#16A34A", shape="ellipse")
                edge(event_id, pid, "place", "#16A34A")

        else:
            edge(event_id, center_id, "place", "#16A34A")

            victim = display_person_name(clean(r.get("victim_display", "")) or clean(r.get("victim_full", "")))
            killer = display_person_name(clean(r.get("killer_display", "")) or clean(r.get("killer_full", "")))

            if victim and not is_bad_person_name(victim):
                vid = node(f"person_victim_{victim}_{i}", victim, "#FEE2E2", "#DC2626", shape="ellipse")
                edge(event_id, vid, "victim", "#DC2626")
            if killer and not is_bad_person_name(killer):
                kid = node(f"person_killer_{killer}_{i}", killer, "#FEE2E2", "#DC2626", shape="ellipse")
                edge(kid, event_id, "agent", "#DC2626")

    if len(related_events.drop_duplicates(subset=["entry_id", "year", "event_type", "raw_clause"])) > max_records:
        more_id = node("more_records", f"+ more records\\nshown below as evidence", "#F9FAFB", "#9CA3AF")
        edge(center_id, more_id, "more evidence", "#9CA3AF", "dashed")

    lines.append("}")
    return "\n".join(lines)


def show_compact_profile_graph(label: str, related_events: pd.DataFrame, entity_type: str, key: str):
    """One small merged graph button for selected person/place."""
    if related_events.empty:
        return

    btn_text = f"Show complete knowledge graph for {label}"
    if st.button(btn_text, key=f"compact_graph_btn_{key}", width="stretch"):
        st.session_state[f"show_compact_graph_{key}"] = not st.session_state.get(f"show_compact_graph_{key}", False)

    if st.session_state.get(f"show_compact_graph_{key}", False):
        st.markdown("### Complete knowledge graph")
        graph_col, legend_col = st.columns([0.72, 0.28])
        with graph_col:
            st.graphviz_chart(
                compact_profile_graph_from_events(label, related_events, entity_type=entity_type),
                width="stretch",
            )
        with legend_col:
            st.markdown(
                """
<div class="legend-box">
<b>How to read this graph</b><br>
<span style="color:#DC2626;">■</span> Person &nbsp;
<span style="color:#2563EB;">■</span> Event &nbsp;
<span style="color:#16A34A;">■</span> Place<br><br>
This graph merges the selected records into one small view. Use the evidence button only when you need the original sentences.
</div>
""",
                unsafe_allow_html=True,
            )


def show_optional_evidence(label: str, related_events: pd.DataFrame, key: str):
    """Hide source evidence behind a button so graph and evidence do not repeat on the same page."""
    if related_events.empty:
        st.info("No source evidence found.")
        return

    if st.button(f"Show source evidence for {label}", key=f"evidence_btn_{key}", width="stretch"):
        st.session_state[f"show_evidence_{key}"] = not st.session_state.get(f"show_evidence_{key}", False)

    if st.session_state.get(f"show_evidence_{key}", False):
        st.markdown("### Source evidence")
        show_evidence_only(related_events)



# ============================================================
# CLEAN PEOPLE / PLACES PROFILE GRAPHS
# ============================================================

def profile_exact_event_rows_for_person(person_ids: list[str]) -> pd.DataFrame:
    """Rows where the selected person is structurally victim or killer/agent."""
    if not person_ids:
        return pd.DataFrame()

    return events_df[
        events_df["victim_id"].isin(person_ids)
        | events_df["killer_id"].isin(person_ids)
    ].copy().drop_duplicates(subset=["entry_id", "year", "event_type", "raw_clause"])


def profile_mention_event_rows(label: str, exact_rows: pd.DataFrame) -> pd.DataFrame:
    """Rows where the label appears in source text but is not the selected exact record."""
    if not label:
        return pd.DataFrame()

    all_mentions = events_df[
        events_df["raw_clause"].str.lower().str.contains(re.escape(label.lower()), na=False)
    ].copy()

    if exact_rows is not None and not exact_rows.empty:
        exact_keys = set(
            zip(
                exact_rows["entry_id"].astype(str),
                exact_rows["year"].astype(str),
                exact_rows["event_type"].astype(str),
                exact_rows["raw_clause"].astype(str),
            )
        )
        all_mentions = all_mentions[
            ~all_mentions.apply(
                lambda r: (
                    str(r["entry_id"]),
                    str(r["year"]),
                    str(r["event_type"]),
                    str(r["raw_clause"]),
                )
                in exact_keys,
                axis=1,
            )
        ]

    return all_mentions.drop_duplicates(subset=["entry_id", "year", "event_type", "raw_clause"])


def profile_rows_for_place(place: str, place_ids: list[str]) -> pd.DataFrame:
    """Rows where selected place appears structurally or in evidence text."""
    rows = events_df[
        events_df["place_id"].isin(place_ids)
        | events_df["raw_clause"].str.lower().str.contains(re.escape(place.lower()), na=False)
        | events_df["place_full"].str.lower().str.contains(re.escape(place.lower()), na=False)
        | events_df["place_display"].str.lower().str.contains(re.escape(place.lower()), na=False)
    ].copy()

    return rows.drop_duplicates(subset=["entry_id", "year", "event_type", "raw_clause"])


def show_evidence_only(events: pd.DataFrame):
    """Show source evidence only; no extra graph buttons inside evidence."""
    if events.empty:
        st.info("No source evidence found.")
        return

    grouped = grouped_evidence_rows(events)
    for _, r in grouped.iterrows():
        year = int(r["year"]) if not pd.isna(r["year"]) else ""
        evidence_box(clean(r["entry_id"]), year, clean(r["event_type"]), r["raw_clause"])


def build_profile_relation_graph(
    label: str,
    exact_rows: pd.DataFrame,
    mention_rows: pd.DataFrame | None = None,
    entity_type: str = "Person",
    max_mentions: int = 4,
) -> str:
    """
    Small relationship graph for People/Places pages.
    It avoids entry-heavy repetition. Source text is shown only under source evidence.
    """
    exact_rows = exact_rows.copy() if exact_rows is not None else pd.DataFrame()
    mention_rows = mention_rows.copy() if mention_rows is not None else pd.DataFrame()

    lines = [
        "digraph G {",
        "rankdir=LR;",
        "splines=curved;",
        "overlap=false;",
        'graph [bgcolor="white", pad="0.05", nodesep="0.32", ranksep="0.50", margin="0"];',
        'node [fontname="Arial", fontsize=10, margin="0.07,0.04"];',
        'edge [fontname="Arial", fontsize=9, arrowsize=0.50, color="#4B5563", fontcolor="#111827"];',
    ]

    added_nodes = set()
    added_edges = set()

    def nid(prefix: str, value: str) -> str:
        return dot_id(f"{prefix}_{value}")

    def add_node(prefix: str, value: str, node_type: str):
        value = clean(value)
        if not value:
            return ""

        node_id = nid(prefix, value)
        if node_id in added_nodes:
            return node_id

        if node_type == "focus_person":
            fill, border, shape, pen = "#FEE2E2", "#DC2626", "ellipse", 2.2
        elif node_type == "focus_place":
            fill, border, shape, pen = "#DCFCE7", "#16A34A", "ellipse", 2.2
        elif node_type == "person":
            fill, border, shape, pen = "#FEE2E2", "#DC2626", "ellipse", 1.5
        elif node_type == "place":
            fill, border, shape, pen = "#DCFCE7", "#16A34A", "ellipse", 1.5
        else:
            fill, border, shape, pen = "#DBEAFE", "#2563EB", "box", 1.3

        lines.append(
            f'{node_id} [label="{html_escape(short_label(value, 34))}", shape={shape}, '
            f'style="rounded,filled", fillcolor="{fill}", color="{border}", penwidth={pen}];'
        )
        added_nodes.add(node_id)
        return node_id

    def add_edge(src: str, dst: str, rel: str, color="#4B5563", style="solid"):
        if not src or not dst or src == dst:
            return

        key = (src, dst, rel, color, style)
        if key in added_edges:
            return

        lines.append(
            f'{src} -> {dst} [label="{html_escape(rel)}", color="{color}", '
            f'fontcolor="#111827", style="{style}", penwidth=1.15];'
        )
        added_edges.add(key)

    focus_type = "focus_person" if entity_type == "Person" else "focus_place"
    focus_id = add_node("focus", label, focus_type)

    if entity_type == "Person":
        for _, r in exact_rows.sort_values(["year", "entry_id"]).iterrows():
            year = int(r["year"]) if not pd.isna(r["year"]) else ""

            victim = display_person_name(clean(r.get("victim_display", "")) or clean(r.get("victim_full", "")))
            killer = display_person_name(clean(r.get("killer_display", "")) or clean(r.get("killer_full", "")))
            place = display_place_name(clean(r.get("place_display", "")) or clean(r.get("place_full", "")))

            if victim and victim.lower() == label.lower():
                event_type_here = clean(r.get("event_type", ""))
                relation_here = person_event_relation_label(event_type_here)
                relation_color = "#2563EB" if relation_here == "death" else "#DC2626"

                if killer and killer.lower() != label.lower() and not is_bad_person_name(killer):
                    kid = add_node("person", killer, "person")
                    add_edge(kid, focus_id, f"killed / agent ({year})", "#DC2626")
                else:
                    eid = add_node("event", f"{event_type_here} {year}", "event")
                    add_edge(eid, focus_id, f"{relation_here} ({year})", relation_color)

            elif killer and killer.lower() == label.lower():
                if victim and victim.lower() != label.lower() and not is_bad_person_name(victim):
                    vid = add_node("person", victim, "person")
                    add_edge(focus_id, vid, f"killed ({year})", "#DC2626")
                else:
                    eid = add_node("event", f"{clean(r['event_type'])} {year}", "event")
                    add_edge(focus_id, eid, f"agent ({year})", "#DC2626")

            if place:
                pid = add_node("place", place, "place")
                add_edge(focus_id, pid, f"place / associated ({year})", "#16A34A")

            # lineage from full exact record names
            for raw_person in [clean(r.get("victim_full", "")), clean(r.get("killer_full", ""))]:
                if not raw_person or label.lower() not in raw_person.lower():
                    continue

                for child, parent, rel in extract_lineage_from_name_for_graph(raw_person):
                    child = display_person_name(child)
                    parent = display_person_name(parent)

                    if child and parent and child.lower() == label.lower():
                        pid = add_node("person", parent, "person")
                        add_edge(focus_id, pid, rel, "#7C3AED", "dashed")

        # Keep source-only mentions visually small and limited.
        for _, r in mention_rows.sort_values(["year", "entry_id"]).head(max_mentions).iterrows():
            year = int(r["year"]) if not pd.isna(r["year"]) else ""
            text = clean(r.get("raw_clause", ""))

            # Example: "Cuanu son of Cellach" -> Cuanu -> Cellach
            m = re.search(
                rf"\b([A-ZÁÉÍÓÚÍ][A-Za-zÁÉÍÓÚáéíóúÍíḃḂṅṄ' -]+?)\s+son of\s+{re.escape(label)}\b",
                text,
            )
            if m:
                child = display_person_name(m.group(1))
                if child and not is_bad_person_name(child):
                    cid = add_node("person", child, "person")
                    add_edge(cid, focus_id, f"son of ({year})", "#7C3AED", "dashed")
                    continue

            eid = add_node("event", f"mentioned {year}", "event")
            add_edge(eid, focus_id, f"mentioned ({year})", "#64748B", "dashed")

    else:
        for _, r in exact_rows.sort_values(["year", "entry_id"]).iterrows():
            year = int(r["year"]) if not pd.isna(r["year"]) else ""

            victim = display_person_name(clean(r.get("victim_display", "")) or clean(r.get("victim_full", "")))
            killer = display_person_name(clean(r.get("killer_display", "")) or clean(r.get("killer_full", "")))

            connected = []
            event_relation = person_event_relation_label(clean(r.get("event_type", "")))
            if victim and not is_bad_person_name(victim):
                connected.append((victim, event_relation))
            if killer and not is_bad_person_name(killer):
                connected.append((killer, "agent"))

            if not connected:
                eid = add_node("event", f"{clean(r['event_type'])} {year}", "event")
                add_edge(eid, focus_id, f"recorded at / in ({year})", "#16A34A")
            else:
                for person, role in connected:
                    pid = add_node("person", person, "person")
                    add_edge(pid, focus_id, f"{role} / place ({year})", "#16A34A")

            text = clean(r.get("raw_clause", ""))
            for rel in ["king of", "abbot of", "bishop of", "from"]:
                if re.search(rf"\b{rel}\s+{re.escape(label)}\b", text, flags=re.I):
                    for person, _role in connected:
                        pid = add_node("person", person, "person")
                        add_edge(pid, focus_id, f"{rel} ({year})", "#EA580C")

    if len(added_nodes) <= 1:
        note = add_node("event", "No structured links found", "event")
        add_edge(note, focus_id, "evidence only", "#64748B", "dashed")

    lines.append("}")
    return "\n".join(lines)


def show_profile_graph_button(label: str, exact_rows: pd.DataFrame, mention_rows: pd.DataFrame, entity_type: str, key: str):
    """
    Show the profile knowledge graph by default.

    Earlier versions required a button click. This version shows the graph immediately
    when a user opens a Person or Place page, because many users will not click
    an extra graph button.
    """
    if exact_rows.empty and (mention_rows is None or mention_rows.empty):
        st.info("No graph available.")
        return

    st.markdown("### Knowledge Graph")

    graph_col, legend_col = st.columns([0.72, 0.28])
    with graph_col:
        st.graphviz_chart(
            build_profile_relation_graph(label, exact_rows, mention_rows, entity_type=entity_type),
            width="stretch",
        )
    with legend_col:
        st.markdown(
            """
<div class="legend-box">
<b>How to read this graph</b><br>
<span style="color:#DC2626;">■</span> Person &nbsp;
<span style="color:#16A34A;">■</span> Place &nbsp;
<span style="color:#2563EB;">■</span> Context<br><br>
This graph shows the main relationships for the selected person or place.
Source evidence is shown below.
</div>
""",
            unsafe_allow_html=True,
        )


def show_evidence_button(label: str, events: pd.DataFrame, key: str):
    """
    Show source evidence by default.

    The function name is kept so the rest of the code does not need to change,
    but it no longer displays a button.
    """
    if events.empty:
        st.info("No source evidence found.")
        return

    st.markdown("### Source evidence")
    show_evidence_only(events)


# ============================================================
# PAGES
# ============================================================

def metric_card(title, value, css_class):
    st.markdown(
        f"""
<div class="metric-card {css_class}">
  <div class="metric-title">{title}</div>
  <div class="metric-value">{value}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def page_home():
    st.markdown("## Ask the Annals")
    st.markdown(
        """
<div class="small">
Explore the people, places, events, and relationships recorded in the
<i>Annals of Ulster</i> between 600 and 700 CE.
</div>
""",
        unsafe_allow_html=True,
    )

    q = st.text_input(
        "Ask the Annals",
        placeholder="Ask anything about the Annals...",
        label_visibility="collapsed",
        key="home_question",
    )

    if st.button("Ask", type="primary"):
        st.session_state.answer = answer_chatbot(q)

    if st.session_state.get("answer"):
        st.markdown("### Answer")
        st.markdown(st.session_state.answer)

    st.markdown("### Dataset summary")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Events", len(events_df), "metric-events")
    with c2:
        metric_card(
            "People",
            len([n for n in people_df["display_name"].unique() if not is_bad_person_name(n)]),
            "metric-people",
        )
    with c3:
        metric_card("Places", len(ALL_PLACE_NAMES), "metric-places")
    with c4:
        metric_card("Graph edges", len(edges_df), "metric-edges")


def page_people():
    st.markdown("## People")

    if not st.session_state.selected_person_name:
        search = st.text_input("Search people", placeholder="Type a name...")

        names = sorted(
            set(
                public_name
                for x in people_df["display_name"].tolist()
                for public_name in expand_person_display_names(x)
                if clean(public_name) and not is_bad_person_name(public_name)
            ),
            key=lambda x: x.lower(),
        )

        if search.strip():
            s = search.lower().strip()
            names = [n for n in names if s in n.lower()]

        st.write(f"**{len(names)} people shown**")

        cols = st.columns(4)
        for i, name in enumerate(names):
            with cols[i % 4]:
                if st.button(name, key=f"person_name_{name}", width="stretch"):
                    st.session_state.selected_person_name = name
                    st.session_state.selected_person_id = ""
                    st.rerun()

    else:
        name = st.session_state.selected_person_name

        if st.button("← Back to all people"):
            st.session_state.selected_person_name = ""
            st.session_state.selected_person_id = ""
            st.rerun()

        st.markdown(f"## 👤 {name}")

        # Merge all exact records for this public name.
        related_people = get_people_variants(name)
        related_ids = related_people["person_id"].tolist()

        exact_rows = profile_exact_event_rows_for_person(related_ids)
        mention_rows = profile_mention_event_rows(name, exact_rows)

        all_rows = pd.concat([exact_rows, mention_rows], ignore_index=True)
        all_rows = all_rows.drop_duplicates(subset=["entry_id", "year", "event_type", "raw_clause"])

        page_key = f"person_{dot_id(name)}_merged"

        show_profile_graph_button(
            name,
            exact_rows,
            mention_rows,
            entity_type="Person",
            key=page_key,
        )

        show_evidence_button(name, all_rows, key=page_key)

def page_places():
    st.markdown("## Places")

    if not st.session_state.selected_place_name:
        search = st.text_input("Search places", placeholder="Type a place...")
        place_names = ALL_PLACE_NAMES.copy()

        if search.strip():
            s = search.lower().strip()
            place_names = [p for p in place_names if s in p.lower()]

        st.write(f"**{len(place_names)} places shown**")

        cols = st.columns(3)
        for i, place in enumerate(place_names):
            with cols[i % 3]:
                if st.button(place, key=f"place_name_{place}", width="stretch"):
                    st.session_state.selected_place_name = place
                    st.rerun()

    else:
        place = st.session_state.selected_place_name

        if st.button("← Back to all places"):
            st.session_state.selected_place_name = ""
            st.rerun()

        st.markdown(f"## 📍 {place}")

        place_ids = places_df[
            places_df["display_place"].apply(
                lambda x: place.lower() in [p.lower() for p in expand_place_display_names(x)]
            )
        ]["place_id"].tolist()

        related_events = profile_rows_for_place(place, place_ids)

        page_key = f"place_{dot_id(place)}"

        show_profile_graph_button(
            place,
            related_events,
            pd.DataFrame(),
            entity_type="Place",
            key=page_key,
        )

        show_evidence_button(place, related_events, key=page_key)


def page_full_graph():
    st.markdown("## Full Knowledge Graph")

    st.markdown(
        """
<div class="legend-box">
<b>People and places in the Annals, 600–700 CE</b><br><br>
This graph shows the Annals as a relationship map. It connects real historical
entities directly: <b>people to people</b> and <b>people to places</b>.
<b>In this knowedge graph:</b> green circles are people. Yellow circles are places,
kingdoms, monasteries, or groups. Lines show relationships such as
family, killing, kingship, religious office, or place association.
Click any circle or line to see the entry ID, year, event type, and original source sentence.
<br><br>
<span style="color:#DC2626;"><b>Red</b></span> = killing/conflict.
<span style="color:#7C3AED;"><b>Purple dotted</b></span> = family/lineage.
<span style="color:#2563EB;"><b>Blue</b></span> = kingship/power.
<span style="color:#0F766E;"><b>Teal</b></span> = religious office.
<span style="color:#EA580C;"><b>Orange</b></span> = place association.
</div>
""",
        unsafe_allow_html=True,
    )

    html_graph = pyvis_full_graph(height="900px")
    components.html(html_graph, height=940, scrolling=True)

def page_about():
    st.markdown("## About Annals Explorer")

    st.markdown(
        """
<style>
.about-hero {
    background: linear-gradient(135deg, #FFFFFF 0%, #FFF8EA 100%);
    border: 1px solid #E3D8C8;
    border-radius: 18px;
    padding: 1.25rem 1.35rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 18px rgba(0,0,0,.05);
}
.about-kicker {
    color:#9A5B13;
    font-size:.78rem;
    font-weight:900;
    letter-spacing:.05em;
    text-transform:uppercase;
    margin-bottom:.3rem;
}
.about-title {
    color:#072B57;
    font-size:1.55rem;
    font-weight:950;
    margin-bottom:.45rem;
}
.about-text {
    color:#374151;
    font-size:1rem;
    line-height:1.55;
    max-width:950px;
}
.about-flow {
    background:white;
    border:1px solid #E3D8C8;
    border-radius:16px;
    padding:1rem;
    margin:1rem 0;
    text-align:center;
    color:#072B57;
    font-weight:900;
    box-shadow:0 5px 14px rgba(0,0,0,.04);
}
.about-grid {
    display:grid;
    grid-template-columns: repeat(2, 1fr);
    gap:.85rem;
    margin:1rem 0;
}
.about-card {
    background:white;
    border:1px solid #E3D8C8;
    border-radius:16px;
    padding:1rem;
    min-height:135px;
    box-shadow:0 5px 14px rgba(0,0,0,.04);
}
.about-card-title {
    color:#072B57;
    font-weight:900;
    font-size:1rem;
    margin-bottom:.35rem;
}
.about-card-text {
    color:#4B5563;
    font-size:.93rem;
    line-height:1.48;
}
.about-example {
    background:#FFFDF7;
    border-left:5px solid #B98525;
    border-radius:14px;
    padding:1rem 1.1rem;
    margin:1rem 0;
    color:#374151;
    font-size:.95rem;
    line-height:1.55;
}
.about-credit {
    background:white;
    border:1px solid #E3D8C8;
    border-radius:16px;
    padding:1rem 1.1rem;
    margin-top:1rem;
    color:#374151;
    font-size:.95rem;
    line-height:1.55;
}
.about-credit b { color:#072B57; }
.about-credit a {
    color:#0B4A8B;
    font-weight:700;
    text-decoration:none;
}
.about-credit a:hover { text-decoration:underline; }
@media (max-width: 900px) {
    .about-grid { grid-template-columns: 1fr; }
}
</style>

<div class="about-hero">
  <div class="about-kicker">Digital humanities · AI · NLP · Knowledge graphs</div>
  <div class="about-title">Exploring the Annals of Ulster through AI and Data Science</div>
  <div class="about-text">
    <i>Annals Explorer</i> is a digital humanities prototype that applies
    Artificial Intelligence (AI), Natural Language Processing (NLP), data science,
    and knowledge graph techniques to selected entries from the
    <i>Annals of Ulster</i> between 600 and 700 CE. It transforms historical text
    into structured data and interactive visualisations, allowing users to explore
    people, places, historical events, and relationships while remaining connected
    to the original source evidence.
  </div>
</div>

<div class="about-flow">
  Historical Text → Data Processing → Natural Language Processing → Knowledge Graph → Interactive Exploration
</div>

<div class="about-grid">
  <div class="about-card">
    <div class="about-card-title">🎯 Project Goal</div>
    <div class="about-card-text">
      To make historical annal entries easier to search, explore, and compare
      through people, places, events, dates, relationships, and interactive
      visualisations.
    </div>
  </div>

  <div class="about-card">
    <div class="about-card-title">🧠 Methods</div>
    <div class="about-card-text">
      A processing pipeline was developed using NLP and data science methods to
      identify people, places, historical events, family relationships,
      person-place associations, conflict relations, dates, and source evidence
      from the annal text.
    </div>
  </div>

  <div class="about-card">
    <div class="about-card-title">📊 Why it Matters</div>
    <div class="about-card-text">
      Historical information is often spread across many short annal entries.
      This website helps reveal connections that are difficult to recognise
      through reading individual entries alone.
    </div>
  </div>

  <div class="about-card">
    <div class="about-card-title">🔮 Current Coverage and Future Directions</div>
    <div class="about-card-text">
      This prototype currently focuses on selected entries from 600–700 CE.
      Future work could extend the approach to additional centuries, improve
      historical name disambiguation, enrich relationship extraction, and compare
      the data with other medieval Irish annals.
    </div>
  </div>
</div>

<div class="about-example">
  <b>Example:</b> an entry such as <i>“Death of Colum Cille...”</i> can be
  represented as a person, an event, and a source sentence. In the website,
  this information becomes searchable data and can also appear in a knowledge
  graph. Every graph, relationship, person, place, and search result remains
  linked to the original annal evidence.
</div>

<div class="about-credit">
  <b>Developed by:</b>
  <a href="https://sudhansu1991.github.io/" target="_blank">Dr. Sudhansu Bala Das</a><br>
  Postdoctoral Researcher, University of Galway<br>
  Insight Research Ireland Centre for Data Analytics<br>
  <b>Email:</b> baladas.sudhansu@gmail.com<br><br>

  <b>Academic supervision:</b>
  <a href="https://research.universityofgalway.ie/en/persons/p%C3%A1draic-moran/" target="_blank">Prof. Pádraic Moran</a><br>
  Classics and Celtic Studies, School of Languages, Literatures and Cultures<br>
  University of Galway
</div>
""",
        unsafe_allow_html=True,
    )
# ============================================================
# ROUTER
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "selected_person_name" not in st.session_state:
    st.session_state.selected_person_name = ""

if "selected_person_id" not in st.session_state:
    st.session_state.selected_person_id = ""

if "selected_place_name" not in st.session_state:
    st.session_state.selected_place_name = ""

header()

if st.session_state.page == "Home":
    page_home()
elif st.session_state.page == "People":
    page_people()
elif st.session_state.page == "Places":
    page_places()
elif st.session_state.page == "Full Graph":
    page_full_graph()
elif st.session_state.page == "About":
    page_about()
