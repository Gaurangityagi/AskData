import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.messages import SystemMessage, HumanMessage
import re
import io
import datetime
import numpy as np
import json

st.set_page_config(
    page_title="AskData",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — warm minimal aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg: #fdf6f0;
    --bg-card: #ffffff;
    --bg-dark: #1a1714;
    --border: #e8e0d8;
    --text: #2c2825;
    --text-muted: #9a918a;
    --accent: #c47a5a;
    --accent-dark: #a36145;
    --red: #c45a5a;
}

.stApp {
    background: var(--bg) !important;
}

[data-testid="stSidebar"] {
    background: #f7efe8 !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label {
    color: var(--text-muted) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'DM Serif Display', Georgia, serif !important;
    color: var(--text) !important;
    font-weight: 400 !important;
}

p, li, span, div, label {
    font-family: 'Inter', sans-serif !important;
}

/* Hero */
.hero {
    background: var(--bg-dark);
    border-radius: 16px;
    padding: 4rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(26,23,20,0.95) 0%, rgba(60,50,40,0.7) 100%);
    pointer-events: none;
}

.hero-brand {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--accent);
    letter-spacing: 0.05em;
    margin: 0 0 2rem 0;
    position: relative;
}

.hero-title {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 3rem;
    font-weight: 400;
    color: #f5f0eb;
    line-height: 1.15;
    margin: 0 0 1.2rem 0;
    position: relative;
    max-width: 600px;
}

.hero-title em {
    font-style: italic;
    color: var(--accent);
}

.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: #a09890;
    margin: 0;
    position: relative;
    font-weight: 300;
    max-width: 480px;
    line-height: 1.6;
}

/* Step labels */
.step-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--accent);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin: 2rem 0 0.6rem 0;
}

/* Section headings */
.section-title {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 1.6rem;
    font-weight: 400;
    color: var(--text);
    margin: 0 0 0.3rem 0;
}

.section-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: var(--text-muted);
    margin: 0 0 1.5rem 0;
    font-weight: 300;
}

/* Field labels */
.field-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 1rem 0 0.4rem 0;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-card {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.2rem;
}

.metric-value {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 1.8rem;
    color: var(--text);
    margin: 0;
    line-height: 1;
}

.metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 0.5rem 0 0 0;
}

/* Status badge */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.04em;
}

.badge-ok {
    background: rgba(196,122,90,0.1);
    color: var(--accent);
    border: 1px solid rgba(196,122,90,0.2);
}

.badge-err {
    background: rgba(196,90,90,0.08);
    color: var(--red);
    border: 1px solid rgba(196,90,90,0.15);
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* Example chips */
.chip {
    display: inline-block;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 5px 14px;
    margin: 3px 2px;
    font-size: 0.78rem;
    color: var(--text-muted);
    font-family: 'Inter', sans-serif;
    transition: all 0.2s;
}

.chip:hover {
    border-color: var(--accent);
    color: var(--accent);
}

/* Inputs */
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: none !important;
}

div[data-testid="stFileUploader"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.8rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: background 0.2s !important;
}

.stButton > button:hover {
    background: var(--accent-dark) !important;
}

.stDownloadButton > button {
    background: transparent !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
}

.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Expander */
div[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 20px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.82rem;
    color: var(--text-muted);
    letter-spacing: 0.03em;
    border-bottom: 2px solid transparent;
}

.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Code */
.stCodeBlock {
    border-radius: 8px !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
@st.cache_resource
def initialize_llm():
    try:
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=st.secrets["GROQ_API_KEY"],
            temperature=0.1,
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return None

llm = initialize_llm()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def generate_pdf(content, title="AskData Report"):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 12, title.encode("latin-1", "ignore").decode("latin-1"), ln=True)
        pdf.set_font("Arial", "", 9)
        pdf.set_text_color(120, 120, 140)
        pdf.cell(0, 8, f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(6)
        pdf.set_text_color(30, 30, 40)
        pdf.set_font("Arial", "", 10)

        if isinstance(content, pd.DataFrame):
            col_widths = [min(max(len(str(c)) * 3.5, 20), 50) for c in content.columns]
            pdf.set_font("Arial", "B", 9)
            for i, col in enumerate(content.columns):
                pdf.cell(col_widths[i], 8, str(col)[:20].encode("latin-1", "ignore").decode("latin-1"), border=1)
            pdf.ln()
            pdf.set_font("Arial", "", 8)
            for _, row in content.head(50).iterrows():
                for i, val in enumerate(row):
                    pdf.cell(col_widths[i], 7, str(val)[:25].encode("latin-1", "ignore").decode("latin-1"), border=1)
                pdf.ln()
        else:
            text = json.dumps(content, indent=2, default=str) if isinstance(content, dict) else str(content)
            for line in text.split("\n"):
                safe = line.encode("latin-1", "ignore").decode("latin-1")
                pdf.multi_cell(0, 6, safe)

        buf = io.BytesIO()
        pdf.output(buf)
        buf.seek(0)
        return buf
    except Exception:
        return None


def apply_user_preprocessing(df, choice, threshold=40):
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns
    obj_cols = df.select_dtypes(include="object").columns

    if choice == "Fill with Mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif choice == "Fill with Median":
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif choice == "Fill with Mode":
        for c in obj_cols:
            mode = df[c].mode()
            df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    elif choice == "Fill with 'Unknown'":
        df[obj_cols] = df[obj_cols].fillna("Unknown")
    elif choice == "Drop Rows with Missing":
        df = df.dropna()
    elif choice == "Drop Columns (above threshold)":
        limit = threshold / 100
        df = df.loc[:, df.isnull().mean() < limit]
    return df


def preprocess_dataframe(df):
    try:
        df.columns = [c.strip().upper() for c in df.columns]
        rename_map = {
            "DATE": "ORDER_DATE", "CUSTOMER_GENDER": "GENDER", "SEX": "GENDER",
            "CUST_NAME": "CUSTOMER_NAME", "CUSTOMER": "CUSTOMER_NAME",
            "USER_ID": "CUSTOMER_ID", "USERID": "CUSTOMER_ID",
            "PRODUCT_CATEGIC": "PRODUCT_CATEGORY", "PRODUCT_CAT": "PRODUCT_CATEGORY",
            "REGION": "ZONE", "AREA": "ZONE", "ORDERID": "ORDER_ID",
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})

        if "PRODUCT" in df.columns:
            if "PRODUCT_CATEGORY" in df.columns:
                df = df.drop(columns=["PRODUCT"])
            else:
                df = df.rename(columns={"PRODUCT": "PRODUCT_CATEGORY"})

        if "ORDER_DATE" in df.columns:
            df["ORDER_DATE"] = pd.to_datetime(df["ORDER_DATE"], errors="coerce", dayfirst=True)
            df = df.dropna(subset=["ORDER_DATE"])

        for col in ["AMOUNT", "ORDERS", "QUANTITY", "UNIT_COST", "UNIT_PRICE", "PROFIT", "COST", "REVENUE", "CUSTOMER_AGE"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].fillna("Unknown")

        return df
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return pd.DataFrame()


def execute_complex_query(df, query, llm):
    try:
        cols = ", ".join(df.columns.tolist())
        dtypes = ", ".join([f"{c}({d})" for c, d in zip(df.columns, df.dtypes)])
        prompt = f"""You are a data analyst. The DataFrame `df` has columns: {cols}
Dtypes: {dtypes}
First rows: {df.head(3).to_string()}

Write Python code using pandas (and numpy if needed) to answer: "{query}"
Store the final output in a variable called `result`.
Use only pandas operations. Do not use print(). Do not import anything."""

        response = llm.invoke([
            SystemMessage(content="You are a Python data analysis expert. Return ONLY executable Python code, no markdown fences."),
            HumanMessage(content=prompt),
        ])
        code = re.sub(r"```(?:python)?\n?", "", response.content).replace("```", "").strip()
        env = {"df": df, "pd": pd, "np": np}
        exec(code, env)
        return env.get("result", "No result produced."), code
    except Exception as e:
        return f"Error: {e}", ""


def execute_chart_query(df, query, llm):
    try:
        cols = ", ".join(df.columns.tolist())
        dtypes = ", ".join([f"{c}({d})" for c, d in zip(df.columns, df.dtypes)])
        prompt = f"""You are a data visualization expert. The DataFrame `df` has columns: {cols}
Dtypes: {dtypes}
First rows: {df.head(3).to_string()}

Write Python code using plotly (px or go) to create a chart for: "{query}"
Store the figure in a variable called `fig`. Do not call fig.show(). Do not import anything."""

        response = llm.invoke([
            SystemMessage(content="You are a Plotly visualization expert. Return ONLY executable Python code, no markdown fences."),
            HumanMessage(content=prompt),
        ])
        code = re.sub(r"```(?:python)?\n?", "", response.content).replace("```", "").strip()
        return code
    except Exception:
        return ""


def format_result_output(result):
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.to_frame("Value")
    if isinstance(result, dict):
        return pd.DataFrame([result])
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.8rem 0 1.2rem 0;">
            <div style="font-family: 'DM Serif Display', Georgia, serif; font-size: 1.4rem; color: #2c2825;">
                ✦ AskData
            </div>
        </div>
        """, unsafe_allow_html=True)

        if llm:
            st.markdown('<span class="badge badge-ok">● Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-err">● Unavailable</span>', unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        st.markdown('<p class="field-label">MISSING VALUE STRATEGY</p>', unsafe_allow_html=True)
        cleaning_option = st.selectbox(
            "strategy",
            ["None", "Fill with Mean", "Fill with Median", "Fill with Mode",
             "Fill with 'Unknown'", "Drop Rows with Missing", "Drop Columns (above threshold)"],
            label_visibility="collapsed",
        )
        threshold = 40
        if cleaning_option == "Drop Columns (above threshold)":
            st.markdown('<p class="field-label">THRESHOLD (%)</p>', unsafe_allow_html=True)
            threshold = st.slider("threshold", 1, 100, 40, label_visibility="collapsed")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="field-label">TRY ASKING</p>', unsafe_allow_html=True)

        examples = [
            "Total revenue by category",
            "Monthly sales trend",
            "Top 10 customers by spend",
            "Average order value by region",
        ]
        chips_html = "".join([f'<span class="chip">{ex}</span>' for ex in examples])
        st.markdown(chips_html, unsafe_allow_html=True)

    # --- Hero ---
    st.markdown("""
    <div class="hero">
        <p class="hero-brand">✦ AskData</p>
        <h1 class="hero-title">Ask your data<br><em>anything.</em></h1>
        <p class="hero-sub">Upload a CSV, ask questions in plain English. Get instant analysis, charts, and downloadable reports.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Upload ---
    st.markdown('<p class="step-label">✦ STEP 01</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Upload your dataset.</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Drop a CSV file below to begin exploring your data.</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded is None:
        return

    # Read CSV
    df_raw = None
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            uploaded.seek(0)
            df_raw = pd.read_csv(uploaded, encoding=enc)
            break
        except Exception:
            continue

    if df_raw is None or df_raw.empty:
        st.error("Could not read the file.")
        return

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-value">{df_raw.shape[0]:,}</p>
            <p class="metric-label">Rows</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-value">{df_raw.shape[1]}</p>
            <p class="metric-label">Columns</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        missing_pct = round(df_raw.isnull().sum().sum() / (df_raw.shape[0] * df_raw.shape[1]) * 100, 1)
        st.markdown(f"""<div class="metric-card">
            <p class="metric-value">{missing_pct}%</p>
            <p class="metric-label">Missing</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        num_count = len(df_raw.select_dtypes(include="number").columns)
        st.markdown(f"""<div class="metric-card">
            <p class="metric-value">{num_count}</p>
            <p class="metric-label">Numeric</p>
        </div>""", unsafe_allow_html=True)

    # Preview
    with st.expander("Preview raw data", expanded=False):
        st.dataframe(df_raw.head(10), use_container_width=True)

    # Clean
    if cleaning_option != "None":
        df_raw = apply_user_preprocessing(df_raw, cleaning_option, threshold)
    df = preprocess_dataframe(df_raw)

    if df.empty:
        st.warning("No data left after preprocessing.")
        return

    with st.expander("Preview cleaned data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    # --- Query ---
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="step-label">✦ STEP 02</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Ask a question.</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Type your question below. Choose a mode to get a table, chart, or simple answer.</p>', unsafe_allow_html=True)

    query = st.text_input(
        "query",
        placeholder="e.g., What is the total revenue by product category?",
        label_visibility="collapsed",
    )

    q1, q2, q3 = st.columns([2, 2, 1])
    with q1:
        st.markdown('<p class="field-label">MODE</p>', unsafe_allow_html=True)
        mode = st.selectbox("mode", ["Smart Analysis", "Chart", "Simple Query"], label_visibility="collapsed")
    with q2:
        st.markdown('<p class="field-label">OPTIONS</p>', unsafe_allow_html=True)
        show_code = st.checkbox("Show generated code", value=False)
    with q3:
        pass

    if not query or not llm:
        if query and not llm:
            st.error("AI is not available. Check your GROQ_API_KEY in Streamlit secrets.")
        return

    # --- Results ---
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="step-label">✦ RESULTS</p>', unsafe_allow_html=True)

    if mode == "Smart Analysis":
        with st.spinner("Analyzing..."):
            result, code = execute_complex_query(df, query, llm)

        if show_code and code:
            st.code(code, language="python")

        result = format_result_output(result)

        if isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)
            csv_data = result.to_csv(index=False)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            c1, c2, _ = st.columns([1, 1, 3])
            with c1:
                st.download_button("Download CSV", csv_data, f"analysis_{ts}.csv", "text/csv")
            with c2:
                pdf_buf = generate_pdf(result, f"AskData — {query[:50]}")
                if pdf_buf:
                    st.download_button("Export PDF", pdf_buf, f"report_{ts}.pdf", "application/pdf")
        else:
            st.write(result)

    elif mode == "Chart":
        with st.spinner("Generating chart..."):
            code = execute_chart_query(df, query, llm)

        if code:
            if show_code:
                st.code(code, language="python")
            try:
                exec_globals = {"df": df, "pd": pd, "px": px, "go": go, "np": np}
                local_vars = {}
                exec(code, exec_globals, local_vars)
                fig = local_vars.get("fig") or exec_globals.get("fig")
                if fig:
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#fdf6f0",
                        font=dict(family="Inter", color="#2c2825"),
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Chart could not be generated. Try rephrasing.")
            except Exception as e:
                st.error(f"Chart error: {e}")
        else:
            st.warning("No chart generated. Try a different query.")

    elif mode == "Simple Query":
        with st.spinner("Processing..."):
            try:
                agent = create_pandas_dataframe_agent(
                    llm, df, verbose=False, allow_dangerous_code=True
                )
                response = agent.invoke(query)
                if isinstance(response, dict) and "output" in response:
                    st.markdown(response["output"])
                else:
                    st.write(response)
            except Exception as e:
                st.error(f"Query error: {e}")


if __name__ == "__main__":
    main()
