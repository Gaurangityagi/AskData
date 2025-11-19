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
# DARK THEME (Inline CSS)
# -------------------------------------------------------
dark_css = """
<style>
body {
    background-color: #000000 !important;
}
[data-testid="stAppViewContainer"] {
    background-color: #000000;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #111111;
    color: white;
}
h1, h2, h3, h4, h5, h6, p, label, span, div, input, textarea {
    color: white !important;
}
.stCheckbox > label {
    color: white !important;
}
.stSelectbox label {
    color: white !important;
}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

st.set_page_config(page_title="📊 E-Commerce Assistant", layout="wide")
st.title("📊 AskData - Data answers made easy")


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
    except Exception as e:
        st.error(f"LLM init error: {e}")
        return None

llm = initialize_llm()


# -------------------------------------------------------
# PDF Generator
# -------------------------------------------------------
def generate_pdf(content, title="E-Commerce Insight"):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, title, ln=True, align='C')

        pdf.ln(10)
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 10, f"Generated on: {datetime.now()}", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", size=10)

        if isinstance(content, dict):
            content = json.dumps(content, indent=2)

        for line in str(content).split('\n'):
            try:
                clean = line.encode('latin-1', 'ignore').decode('latin-1')
                pdf.cell(0, 6, clean, ln=True)
            except:
                pdf.cell(0, 6, "Encoding issue", ln=True)

        buf = io.BytesIO()
        buf.write(pdf.output(dest='S').encode('latin-1'))
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"PDF error: {e}")
        return None


# -------------------------------------------------------
# Preprocess DataFrame
# -------------------------------------------------------
def preprocess_dataframe(df):
    try:
        df_processed = df.copy()

        df_processed.columns = df_processed.columns.str.strip().str.upper()

        rename_map = {
            'DATE': 'ORDER_DATE',
            'CUSTOMER_GENDER': 'GENDER',
            'SEX': 'GENDER',
            'CUSTOMER': 'CUSTOMER_NAME',
            'CUST_NAME': 'CUSTOMER_NAME',
            'USER_ID': 'CUSTOMER_ID',
            'USERID': 'CUSTOMER_ID',
            'PRODUCT_CATEGIC': 'PRODUCT_CATEGORY',
            'PRODUCT_CAT': 'PRODUCT_CATEGORY',
            'REGION': 'ZONE',
            'AREA': 'ZONE',
            'ORDERID': 'ORDER_ID'
        }

        cols_to_rename = {}
        for old, new in rename_map.items():
            if old in df_processed.columns and new not in df_processed.columns:
                cols_to_rename[old] = new

        if cols_to_rename:
            df_processed.rename(columns=cols_to_rename, inplace=True)

        if "PRODUCT_CATEGORY" in df_processed.columns and "PRODUCT" in df_processed.columns:
            df_processed.drop(columns=["PRODUCT"], inplace=True)

        if "ORDER_DATE" in df_processed.columns:
            df_processed["ORDER_DATE"] = pd.to_datetime(
                df_processed["ORDER_DATE"], errors="coerce", dayfirst=True
            )
            df_processed.dropna(subset=["ORDER_DATE"], inplace=True)

        numeric_cols = [
            'AMOUNT', 'ORDERS', 'QUANTITY', 'UNIT_COST', 'UNIT_PRICE',
            'PROFIT', 'COST', 'REVENUE', 'CUSTOMER_AGE'
        ]
        for col in numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

        if "GENDER" in df_processed.columns:
            df_processed["GENDER"] = df_processed["GENDER"].astype(str).str.upper().replace({
                "MALE": "M", "FEMALE": "F"
            })

        for col in df_processed.columns:
            if df_processed[col].dtype == object:
                df_processed[col] = df_processed[col].fillna("Unknown")

        return df_processed

    except Exception as e:
        st.error(f"Preprocess error: {e}")
        return pd.DataFrame()


# -------------------------------------------------------
# Execute LLM Complex Query
# -------------------------------------------------------
def execute_complex_query(df, query, llm):
    system_prompt = f"""
    You are a senior data analyst.
    Generate Python code ONLY (no markdown).
    DataFrame = df
    Columns = {df.columns.tolist()}
    Final result MUST be stored in variable: result
    """

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}")
        ]

        response = llm.invoke(messages)
        code = response.content.strip()

        code = re.sub(r"```(python)?", "", code).replace("```", "").strip()

        local_vars = {"df": df, "pd": pd, "np": np}
        exec(code, local_vars)

        return local_vars.get("result"), code

    except Exception as e:
        return f"Execution error: {e}", ""


# -------------------------------------------------------
# Format result for display
# -------------------------------------------------------
def format_result_output(result):
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.to_frame("Value")
    if isinstance(result, dict):
        return pd.DataFrame([result])
    return str(result)


# -------------------------------------------------------
# Main App
# -------------------------------------------------------
def main():

    with st.sidebar:
        st.header("Example Queries")

        example_queries = [
            "Product with highest frequency in each region",
            "Top 5 customers by total spending",
            "Monthly sales trend by category",
            "Average order value by gender",
            "Most popular product category by zone"
        ]

        selected_example = st.selectbox("Choose an example query:", example_queries)

    uploaded = st.file_uploader("📁 Upload CSV", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        df = preprocess_dataframe(df_raw)

        st.subheader("Data Preview")
        st.dataframe(df.head())

        query = st.text_input("Enter your question", value=selected_example)

        mode = st.selectbox("Response Type", ["Smart Analysis", "Chart", "Simple Query"])
        use_ai = st.checkbox("Use AI Analysis", value=True)
        show_code = st.checkbox("Show Generated Code", value=False)

        if query and use_ai and llm:

            # ---------------- SMART ANALYSIS -------------------
            if mode == "Smart Analysis":
                result, code = execute_complex_query(df, query, llm)

                if show_code:
                    st.code(code, language="python")

                formatted = format_result_output(result)
                st.subheader("Result")
                st.dataframe(formatted)

            # ---------------- CHART MODE ------------------------
            elif mode == "Chart":
                system_prompt = f"""
                Use pandas + plotly.express.
                DataFrame: df
                Columns: {df.columns.tolist()}
                Define plot as 'fig'
                Return ONLY code.
                """

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Generate chart code for: {query}")
                ]

                response = llm.invoke(messages)
                chart_code = re.sub(r"```(python)?", "", response.content).replace("```", "").strip()

                if show_code:
                    st.code(chart_code, language="python")

                local_vars = {}
                exec(chart_code, {"df": df, "px": px, "pd": pd, "go": go}, local_vars)

                if "fig" in local_vars:
                    st.plotly_chart(local_vars["fig"], use_container_width=True)
                else:
                    st.error("No figure named fig found.")

            # ---------------- SIMPLE QUERY ----------------------
            elif mode == "Simple Query":
                agent = create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True)
                result = agent.invoke(query)
                st.markdown(result["output"])

        elif query and not use_ai:
            st.warning("Enable AI to use analysis.")


if __name__ == "__main__":
    main()
