import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF

from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.messages import SystemMessage, HumanMessage

import re
import io
from datetime import datetime
import numpy as np
import json

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(page_title="AskData — E-Commerce Assistant", layout="wide")

# ---------------------------------------------------------------
# CSS - Black + Beige Theme with Floating Navbar
# ---------------------------------------------------------------
CSS = """
<style>

:root {
    --bg: #0f0f0f;
    --card: #1d1d1d;
    --text: #f5e6c8;
    --muted: #cfc1a8;
    --accent: #f5e6c8;
    --nav-bg: rgba(255,255,255,0.05);
}

/* Light Mode */
.light :root {
    --bg: #f7f2e7;
    --card: #ffffff;
    --text: #000000;
    --muted: #5c5c5c;
    --accent: #2a2a2a;
    --nav-bg: rgba(0,0,0,0.05);
}

body {
    background-color: var(--bg);
}

/* Floating navbar */
.navbar {
    position: fixed;
    top: 12px;
    left: 50%;
    transform: translateX(-50%);
    width: calc(100% - 48px);
    max-width: 1250px;
    background: var(--nav-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 12px 20px;
    border-radius: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0px 6px 25px rgba(0,0,0,0.4);
    z-index: 999;
}

/* Title */
.nav-title {
    font-size: 20px;
    font-weight: 700;
    color: var(--text);
}

.nav-sub {
    font-size: 13px;
    color: var(--muted);
}

/* Center dropdown */
.nav-center select {
    padding: 6px 10px;
    font-size: 14px;
    color: var(--text);
    background-color: var(--card);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
}

/* Toggle */
.nav-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--muted);
    font-size: 14px;
}

/* Fix spacing below navbar */
.stApp {
    padding-top: 120px !important;
}

/* Card */
.card {
    background-color: var(--card);
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.25);
    color: var(--text);
    margin-bottom: 20px;
}

/* Section title */
.section-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--text);
}

/* Buttons */
.stButton > button {
    background-color: var(--accent) !important;
    color: var(--bg) !important;
    border: none !important;
    font-weight: 500;
}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------
# LLM Init
# ---------------------------------------------------------------
@st.cache_resource
def initialize_llm():
    try:
        return AzureChatOpenAI(
            openai_api_version=st.secrets.get("OPENAI_API_VERSION"),
            azure_deployment=st.secrets.get("AZURE_DEPLOYMENT"),
            azure_endpoint=st.secrets.get("AZURE_ENDPOINT"),
            api_key=st.secrets.get("AZURE_API_KEY"),
        )
    except:
        return None

llm = initialize_llm()

# ---------------------------------------------------------------
# THEME TOGGLE SYSTEM
# ---------------------------------------------------------------
if "light_mode" not in st.session_state:
    st.session_state["light_mode"] = False

if st.session_state["light_mode"]:
    st.markdown("<script>document.body.classList.add('light');</script>", unsafe_allow_html=True)
else:
    st.markdown("<script>document.body.classList.remove('light');</script>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# FLOATING NAVBAR
# ---------------------------------------------------------------
st.markdown(
    """
    <div class="navbar">
        <div>
            <div class="nav-title">AskData — E-Commerce Assistant</div>
            <div class="nav-sub">Interactive data queries & visualizations</div>
        </div>

        <div class="nav-center">
            <select id="exampleSelect">
                <option value="">Example Queries</option>
                <option value="Find the product with highest frequency in each region">Product frequency by region</option>
                <option value="Top 5 customers by total spending">Top 5 customers</option>
                <option value="Monthly sales trend by category">Sales trend</option>
                <option value="Average order value by gender">AOV by gender</option>
                <option value="Most popular product category by zone">Popular category by zone</option>
            </select>
        </div>

        <div class="nav-toggle">
            <label>Light Mode</label>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Real toggle under navbar
st.session_state["light_mode"] = st.checkbox("Toggle Light Mode", value=st.session_state["light_mode"])

if st.session_state["light_mode"]:
    st.markdown("<script>document.body.classList.add('light');</script>", unsafe_allow_html=True)
else:
    st.markdown("<script>document.body.classList.remove('light');</script>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Functions (unchanged logic)
# ---------------------------------------------------------------

def generate_pdf(content, title="E-Commerce Insight"):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, title, ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", size=10)
        if isinstance(content, dict):
            content = json.dumps(content, indent=2)
        for line in str(content).split('\n'):
            try:
                clean_line = line.encode('latin-1', 'ignore').decode('latin-1')
                pdf.cell(0, 6, clean_line, ln=True)
            except:
                pdf.cell(0, 6, "Encoding issue.", ln=True)
        buffer = io.BytesIO()
        buffer.write(pdf.output(dest="S").encode("latin-1"))
        buffer.seek(0)
        return buffer
    except:
        return None


# ---------------------------------------------------------------
# PREPROCESSOR (unchanged)
# ---------------------------------------------------------------
def preprocess_dataframe(df):
    try:
        df = df.copy()
        df.columns = df.columns.str.strip().str.upper()

        rename_map = {
            'DATE':'ORDER_DATE','SEX':'GENDER','CUST_NAME':'CUSTOMER_NAME',
            'CUSTOMER':'CUSTOMER_NAME','USER_ID':'CUSTOMER_ID','USERID':'CUSTOMER_ID',
            'PRODUCT_CAT':'PRODUCT_CATEGORY','PRODUCT_CATEGIC':'PRODUCT_CATEGORY',
            'REGION':'ZONE','AREA':'ZONE','ORDERID':'ORDER_ID'
        }

        df.rename(columns={c:rename_map[c] for c in rename_map if c in df.columns}, inplace=True)

        if "PRODUCT" in df.columns and "PRODUCT_CATEGORY" not in df.columns:
            df.rename(columns={"PRODUCT":"PRODUCT_CATEGORY"}, inplace=True)
        if "PRODUCT" in df.columns and "PRODUCT_CATEGORY" in df.columns:
            df.drop(columns=["PRODUCT"], inplace=True)

        if "ORDER_DATE" in df.columns:
            df["ORDER_DATE"] = pd.to_datetime(df["ORDER_DATE"], errors='coerce', dayfirst=True)
            df.dropna(subset=["ORDER_DATE"], inplace=True)

        numeric_cols = [
            'AMOUNT','ORDERS','QUANTITY','UNIT_COST','UNIT_PRICE',
            'PROFIT','COST','REVENUE','CUSTOMER_AGE'
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        if "GENDER" in df.columns:
            df["GENDER"] = df["GENDER"].astype(str).str.upper()
            df["GENDER"].replace({"MALE":"M","FEMALE":"F"}, inplace=True)
            df["GENDER"].fillna("Unknown", inplace=True)

        for c in df.columns:
            if df[c].dtype == "object":
                df[c].fillna("Unknown", inplace=True)

        return df
    except:
        return pd.DataFrame()


# ---------------------------------------------------------------
# SMART ANALYSIS EXECUTOR
# ---------------------------------------------------------------
def execute_complex_query(df, query, llm):
    code = ""
    prompt = f"""
    Generate Python code to answer this analytical query about the dataframe 'df'.
    Rules:
    - Use exact column names.
    - Final output must be stored in a variable named 'result'.
    - No markdown. No explanations.
    - Only Python code.
    Columns: {df.columns.tolist()}
    Sample: {df.head(3).to_dict('records')}
    """

    try:
        messages = [SystemMessage(content=prompt), HumanMessage(content=query)]
        res = llm.invoke(messages)
        code = (res.content or "").strip()
        code = re.sub(r"^```[\s\S]*?python", "", code)
        code = re.sub(r"```$", "", code)

        local_vars = {"df": df, "pd": pd, "np": np}
        exec(code, local_vars)

        if "result" in local_vars:
            return local_vars["result"], code
        return "No variable named 'result' found.", code
    except Exception as e:
        return f"Execution error: {e}", code


# ---------------------------------------------------------------
# MAIN APP UI
# ---------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload your CSV</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload sales data (CSV)", type=["csv"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        df = preprocess_dataframe(df_raw)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10))
        st.markdown('</div>', unsafe_allow_html=True)

        # Query section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Ask a Question</div>', unsafe_allow_html=True)

        query = st.text_input("Enter your data question")
        mode = st.selectbox("Select response type", ["Smart Analysis", "Chart", "Simple Query"])
        use_ai = st.checkbox("Use AI Analysis", value=(llm is not None))
        show_code = st.checkbox("Show generated code")

        st.markdown('</div>', unsafe_allow_html=True)

        # Handle query
        if query and use_ai and llm:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

            if mode == "Smart Analysis":
                with st.spinner("Analyzing..."):
                    result, code = execute_complex_query(df, query, llm)

                    if show_code:
                        st.subheader("Generated Code")
                        st.code(code, language="python")

                    formatted = result
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                    else:
                        st.write(result)

            elif mode == "Chart":
                with st.spinner("Generating chart..."):
                    chart_prompt = f"""
                    Generate Python code using pandas + plotly.express.
                    Use df.
                    Put final chart in variable 'fig'.
                    Only code, no markdown.
                    Query: {query}
                    """
                    messages = [
                        SystemMessage(content=chart_prompt),
                        HumanMessage(content="Generate the code now.")
                    ]

                    try:
                        response = llm.invoke(messages)
                        chart_code = (response.content or "").strip()
                        chart_code = re.sub(r"```.*?python", "", chart_code)
                        chart_code = chart_code.replace("```", "")

                        if show_code:
                            st.code(chart_code, language="python")

                        local_vars = {}
                        exec(chart_code, {"df":df, "px":px, "pd":pd, "go":go, "np":np}, local_vars)

                        if "fig" in local_vars:
                            st.plotly_chart(local_vars["fig"], use_container_width=True)
                        else:
                            st.warning("Figure 'fig' not generated by code.")
                    except Exception as e:
                        st.error(f"Chart generation error: {e}")

            elif mode == "Simple Query":
                try:
                    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
                    out = agent.invoke(query)
                    out = out.get('output', str(out)) if isinstance(out, dict) else str(out)
                    st.write(out)
                except Exception as e:
                    st.error(f"Simple query error: {e}")

            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")

