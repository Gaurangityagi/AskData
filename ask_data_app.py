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


# -------------------------------------------------------
# PREMIUM DARK THEME UI (Glass + Smooth UI)
# -------------------------------------------------------
premium_css = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, div, input, textarea, label, span, p {
    font-family: 'Inter', sans-serif !important;
}

/* Main app background */
[data-testid="stAppViewContainer"] {
    background-color: #000000;
    color: white;
    padding: 2rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: rgba(17, 17, 17, 0.9);
    backdrop-filter: blur(10px);
    color: white;
}

/* Headings */
h1, h2, h3 {
    text-align: center;
    font-weight: 700;
    color: #ffffff !important;
}

/* Pretty cards */
.block-container {
    padding-top: 1rem;
}

.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 1.2rem 1.5rem;
    border-radius: 16px;
    margin-bottom: 1.4rem;
}

/* Selectbox styling */
div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
}

/* Inputs */
input, textarea {
    background-color: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 8px;
}

/* Buttons */
button[kind="secondary"] {
    border-radius: 10px !important;
}

/* Divider line */
.hr-line {
    border-bottom: 1px solid rgba(255,255,255,0.15);
    margin: 1.5rem 0;
}

/* Hover glow */
.stButton>button:hover {
    transform: scale(1.01);
    background-color: #222 !important;
}

</style>
"""

st.markdown(premium_css, unsafe_allow_html=True)

st.set_page_config(page_title="📊 E-Commerce Assistant", layout="wide")
st.title("✨ AskData — Your Smart E-Commerce Insights Assistant")


# -------------------------------------------------------
# Initialize LLM
# -------------------------------------------------------
@st.cache_resource
def initialize_llm():
    try:
        return AzureChatOpenAI(
            openai_api_version=st.secrets["OPENAI_API_VERSION"],
            azure_deployment=st.secrets["AZURE_DEPLOYMENT"],
            azure_endpoint=st.secrets["AZURE_ENDPOINT"],
            api_key=st.secrets["AZURE_API_KEY"]
        )
    except:
        return None

llm = initialize_llm()


# -------------------------------------------------------
# PDF Generator
# -------------------------------------------------------
def generate_pdf(content):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, content)
        buf = io.BytesIO()
        buf.write(pdf.output(dest='S').encode('latin-1'))
        buf.seek(0)
        return buf
    except:
        return None


# -------------------------------------------------------
# Preprocess DataFrame
# -------------------------------------------------------
def preprocess_dataframe(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.upper()
    return df


# -------------------------------------------------------
# Execute LLM Query
# -------------------------------------------------------
def execute_complex_query(df, query, llm):
    prompt = f"""
    You are a Python data expert.
    Only output Python code, no text.
    Use dataframe df.
    Final output MUST be in variable `result`.
    Query: {query}
    """

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Generate code.")
    ])

    code = re.sub(r"```(python)?", "", response.content).replace("```", "").strip()

    local_vars = {"df": df, "pd": pd, "np": np}
    exec(code, local_vars)

    return local_vars.get("result"), code


# Format Output
def format_output(result):
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.to_frame("Value")
    return str(result)


# -------------------------------------------------------
# Main UI
# -------------------------------------------------------
def main():

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.header("💡 Example Queries")

        example_queries = [
            "Top 5 customers by spending",
            "Best selling product category each month",
            "Average order amount by gender",
            "Total revenue by zone",
            "Monthly sales trend"
        ]

        selected_example = st.selectbox("Choose a query:", example_queries)

        st.markdown("<div class='hr-line'></div>", unsafe_allow_html=True)

        st.info("Upload a CSV to begin analysis.")


    # ---------- FILE UPLOAD ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("📁 Upload your CSV file", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        df = preprocess_dataframe(df_raw)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📄 Data Preview")
        st.dataframe(df.head())
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- QUERY INPUT ----------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        query = st.text_input("🔍 Enter your question", value=selected_example)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- MODE OPTIONS ----------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        mode = col1.selectbox("Response Type", ["Smart Analysis", "Chart", "Simple Query"])
        use_ai = col2.checkbox("Use AI", value=True)
        show_code = col3.checkbox("Show Code", value=False)

        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------------------------------
        # SMART ANALYSIS
        # -------------------------------------------------------
        if query and use_ai and llm:

            if mode == "Smart Analysis":
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("📊 Analysis Result")

                result, code = execute_complex_query(df, query, llm)
                formatted = format_output(result)

                st.dataframe(formatted)

                if show_code:
                    st.code(code, language="python")

                st.markdown("</div>", unsafe_allow_html=True)


            # ---------------------------------------------------
            # CHART MODE
            # ---------------------------------------------------
            elif mode == "Chart":
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("📈 Chart Output")

                chart_prompt = f"""
                Use pandas & plotly.express.
                DataFrame is df.
                Create a chart named fig.
                Query: {query}
                Only return Python code.
                """

                response = llm.invoke([
                    SystemMessage(content=chart_prompt),
                    HumanMessage(content="Generate chart code.")
                ])

                chart_code = re.sub(r"```(python)?", "", response.content).replace("```", "").strip()

                if show_code:
                    st.code(chart_code, language="python")

                local_vars = {}
                exec(chart_code, {"df": df, "px": px, "pd": pd, "go": go}, local_vars)

                if "fig" in local_vars:
                    st.plotly_chart(local_vars["fig"], use_container_width=True)
                else:
                    st.error("No figure named 'fig' found.")

                st.markdown("</div>", unsafe_allow_html=True)


            # ---------------------------------------------------
            # SIMPLE QUERY
            # ---------------------------------------------------
            elif mode == "Simple Query":
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("📄 Simple Answer")

                agent = create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True)
                result = agent.invoke(query)
                st.markdown(result["output"])

                st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
