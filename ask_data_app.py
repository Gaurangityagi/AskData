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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark SaaS aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: #1a1a26;
    --bg-hover: #22222e;
    --border: #2a2a3a;
    --text-primary: #f0f0f5;
    --text-secondary: #8888a0;
    --accent: #6c5ce7;
    --accent-light: #a29bfe;
    --accent-glow: rgba(108, 92, 231, 0.15);
    --success: #00cec9;
    --warning: #fdcb6e;
    --danger: #ff7675;
}

.stApp {
    background: var(--bg-primary) !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-family: 'Inter', sans-serif !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
}

p, li, span, div, label {
    font-family: 'Inter', sans-serif !important;
}

.hero-container {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 50%, var(--bg-secondary) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, var(--accent-glow) 0%, transparent 50%);
    pointer-events: none;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.4rem 0;
    position: relative;
    letter-spacing: -0.02em;
}

.hero-title span {
    background: linear-gradient(135deg, var(--accent-light), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: var(--text-secondary);
    margin: 0;
    position: relative;
    font-weight: 400;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent-light);
    margin: 0;
}

.metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0.3rem 0 0 0;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
}

.status-connected {
    background: rgba(0, 206, 201, 0.12);
    color: var(--success);
    border: 1px solid rgba(0, 206, 201, 0.25);
}

.status-disconnected {
    background: rgba(255, 118, 117, 0.12);
    color: var(--danger);
    border: 1px solid rgba(255, 118, 117, 0.25);
}

.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}

.query-chip {
    display: inline-block;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 14px;
    margin: 3px 4px;
    font-size: 0.8rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s;
    font-family: 'Inter', sans-serif;
}

.query-chip:hover {
    border-color: var(--accent);
    color: var(--accent-light);
    background: var(--accent-glow);
}

.code-block {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: var(--accent-light);
    overflow-x: auto;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stFileUploader"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5a4bd1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px var(--accent-glow) !important;
}

.stDownloadButton > button {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}

div[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    color: var(--text-secondary);
}

.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

.stSpinner > div {
    border-top-color: var(--accent) !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# LLM initialization — Gemini
# ---------------------------------------------------------------------------
@st.cache_resource
def initialize_llm():
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=st.secrets["GROQ_API_KEY"],
            temperature=0.1,
        )
        return llm
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
# Main app
# ---------------------------------------------------------------------------
def main():
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 700; color: #f0f0f5;">
                ⚡ AskData
            </div>
            <div style="font-size: 0.75rem; color: #8888a0; margin-top: 2px;">AI-powered data analysis</div>
        </div>
        """, unsafe_allow_html=True)

        if llm:
            st.markdown('<span class="status-badge status-connected">● AI Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-disconnected">● AI Unavailable</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-header">Data Cleaning</div>', unsafe_allow_html=True)

        cleaning_option = st.selectbox(
            "Missing value strategy",
            ["None", "Fill with Mean", "Fill with Median", "Fill with Mode",
             "Fill with 'Unknown'", "Drop Rows with Missing", "Drop Columns (above threshold)"],
            label_visibility="collapsed",
        )
        threshold = 40
        if cleaning_option == "Drop Columns (above threshold)":
            threshold = st.slider("Missing % threshold", 1, 100, 40)

        st.markdown("---")
        st.markdown('<div class="section-header">Example Queries</div>', unsafe_allow_html=True)
        examples = [
            "Total revenue by category",
            "Monthly sales trend",
            "Top 10 customers by spend",
            "Average order value by region",
            "Show a bar chart of sales by month",
        ]
        for ex in examples:
            st.markdown(f'<div class="query-chip">{ex}</div>', unsafe_allow_html=True)

    # --- Hero ---
    st.markdown("""
    <div class="hero-container">
        <p class="hero-title">Ask your data <span>anything</span></p>
        <p class="hero-subtitle">Upload a CSV, ask questions in plain English, and get instant analysis, charts, and reports.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Upload ---
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded is None:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 4rem 2rem;
            border: 2px dashed #2a2a3a;
            border-radius: 16px;
            margin: 1rem 0;
        ">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; color: #f0f0f5; font-weight: 600;">
                Drop your CSV here to get started
            </div>
            <div style="font-size: 0.85rem; color: #8888a0; margin-top: 0.5rem;">
                Supports UTF-8, Latin-1, and CP1252 encodings
            </div>
        </div>
        """, unsafe_allow_html=True)
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
        st.error("Could not read the CSV file. Check the encoding or format.")
        return

    # Metrics row
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
            <p class="metric-label">Missing Data</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        num_count = len(df_raw.select_dtypes(include="number").columns)
        st.markdown(f"""<div class="metric-card">
            <p class="metric-value">{num_count}</p>
            <p class="metric-label">Numeric Cols</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    # Data preview tabs
    with st.expander("📋 Data Preview", expanded=False):
        tab_raw, tab_clean = st.tabs(["Raw Data", "Cleaned Data"])
        with tab_raw:
            st.dataframe(df_raw.head(10), use_container_width=True)

    # Apply cleaning
    if cleaning_option != "None":
        df_raw = apply_user_preprocessing(df_raw, cleaning_option, threshold)
    df = preprocess_dataframe(df_raw)

    if df.empty:
        st.warning("No data left after preprocessing.")
        return

    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    # --- Query section ---
    st.markdown('<div class="section-header">🔍 Query</div>', unsafe_allow_html=True)

    query = st.text_input(
        "Ask a question about your data",
        placeholder="e.g., What is the total revenue by product category?",
        label_visibility="collapsed",
    )

    qcol1, qcol2, qcol3 = st.columns([2, 2, 1])
    with qcol1:
        mode = st.selectbox("Mode", ["Smart Analysis", "Chart", "Simple Query"], label_visibility="collapsed")
    with qcol2:
        use_ai = st.checkbox("Use AI", value=llm is not None, disabled=llm is None)
    with qcol3:
        show_code = st.checkbox("Show Code", value=False)

    if query and use_ai and llm:
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        if mode == "Smart Analysis":
            with st.spinner("Analyzing..."):
                result, code = execute_complex_query(df, query, llm)

            if show_code and code:
                st.markdown('<div class="section-header">Generated Code</div>', unsafe_allow_html=True)
                st.code(code, language="python")

            st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
            result = format_result_output(result)

            if isinstance(result, pd.DataFrame):
                st.dataframe(result, use_container_width=True)
                csv_data = result.to_csv(index=False)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                dl_col, pdf_col, _ = st.columns([1, 1, 3])
                with dl_col:
                    st.download_button(
                        "⬇ Download CSV",
                        csv_data,
                        f"analysis_{ts}.csv",
                        "text/csv",
                    )
                with pdf_col:
                    pdf_buf = generate_pdf(result, f"AskData — {query[:50]}")
                    if pdf_buf:
                        st.download_button(
                            "📄 Export PDF",
                            pdf_buf,
                            f"report_{ts}.pdf",
                            "application/pdf",
                        )
            else:
                st.write(result)

        elif mode == "Chart":
            with st.spinner("Generating chart..."):
                code = execute_chart_query(df, query, llm)

            if code:
                if show_code:
                    st.markdown('<div class="section-header">Generated Code</div>', unsafe_allow_html=True)
                    st.code(code, language="python")

                try:
                    exec_globals = {"df": df, "pd": pd, "px": px, "go": go, "np": np}
                    local_vars = {}
                    exec(code, exec_globals, local_vars)
                    fig = local_vars.get("fig") or exec_globals.get("fig")
                    if fig:
                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(26,26,38,1)",
                            font=dict(family="Inter", color="#f0f0f5"),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Chart could not be generated. Try rephrasing your query.")
                except Exception as e:
                    st.error(f"Chart error: {e}")
            else:
                st.warning("No chart code was generated. Try a different query.")

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

    elif query and not llm:
        st.error("AI is not available. Check your API key in Streamlit secrets.")


if __name__ == "__main__":
    main()
