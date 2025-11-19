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
import base64
from datetime import datetime
import numpy as np
import json
import ast


# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AskData – E-Commerce Assistant", layout="wide")


# -------------------- DARK MODE (NATIVE) --------------------
# No CSS required. User's Streamlit theme handles dark mode automatically.


# -------------------- INITIALIZE LLM --------------------
@st.cache_resource
def initialize_llm():
    try:
        llm = AzureChatOpenAI(
            openai_api_version=st.secrets["OPENAI_API_VERSION"],
            azure_deployment=st.secrets["AZURE_DEPLOYMENT"],
            azure_endpoint=st.secrets["AZURE_ENDPOINT"],
            api_key=st.secrets["AZURE_API_KEY"]
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return None

llm = initialize_llm()


# -------------------- PDF GENERATION --------------------
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
                pdf.cell(0, 6, "Encoding issue - line skipped.", ln=True)

        buffer = io.BytesIO()
        pdf_output = pdf.output(dest='S').encode('latin-1')
        buffer.write(pdf_output)
        buffer.seek(0)
        return buffer

    except Exception as e:
        st.error(f"PDF Error: {e}")
        return None


# -------------------- PREPROCESS DATA --------------------
def preprocess_dataframe(df):
    try:
        df_processed = df.copy()
        df_processed.columns = df_processed.columns.str.strip().str.upper()

        rename_map = {
            'DATE': 'ORDER_DATE',
            'CUSTOMER_GENDER': 'GENDER',
            'SEX': 'GENDER',
            'CUST_NAME': 'CUSTOMER_NAME',
            'CUSTOMER': 'CUSTOMER_NAME',
            'USER_ID': 'CUSTOMER_ID',
            'USERID': 'CUSTOMER_ID',
            'PRODUCT_CATEGIC': 'PRODUCT_CATEGORY',
            'PRODUCT_CAT': 'PRODUCT_CATEGORY',
            'REGION': 'ZONE',
            'AREA': 'ZONE',
            'ORDERID': 'ORDER_ID'
        }

        cols_to_rename = {}
        for k, v in rename_map.items():
            if k in df_processed.columns and v not in df_processed.columns:
                cols_to_rename[k] = v

        if cols_to_rename:
            df_processed.rename(columns=cols_to_rename, inplace=True)

        if 'PRODUCT_CATEGORY' in df_processed.columns and 'PRODUCT' in df_processed.columns:
            df_processed.drop(columns=['PRODUCT'], inplace=True, errors='ignore')
        elif 'PRODUCT' in df_processed.columns:
            df_processed.rename(columns={'PRODUCT': 'PRODUCT_CATEGORY'}, inplace=True)

        if 'ORDER_DATE' in df_processed.columns:
            df_processed['ORDER_DATE'] = pd.to_datetime(
                df_processed['ORDER_DATE'], errors='coerce', dayfirst=True
            )
            df_processed.dropna(subset=['ORDER_DATE'], inplace=True)

        numeric_cols = [
            'AMOUNT', 'ORDERS', 'QUANTITY', 'UNIT_COST',
            'UNIT_PRICE', 'PROFIT', 'COST', 'REVENUE', 'CUSTOMER_AGE'
        ]

        for col in numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

        if 'GENDER' in df_processed.columns:
            df_processed['GENDER'] = (
                df_processed['GENDER'].astype(str).str.upper()
                    .replace({'MALE': 'M', 'FEMALE': 'F'})
            )

        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].fillna("Unknown")

        return df_processed

    except Exception as e:
        st.error(f"Preprocess error: {e}")
        return pd.DataFrame()


# -------------------- EXECUTE LLM QUERY --------------------
def execute_complex_query(df, query, llm):
    system_prompt = f"""
    You are an expert data analyst. Generate Python code (ONLY CODE, NO MARKDOWN)
    that uses the dataframe 'df' to answer the question.
    Columns: {df.columns.tolist()}
    """

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate Python code for: {query}")
        ]

        response = llm.invoke(messages)
        code = response.content.strip()

        code = re.sub(r"```python", "", code)
        code = re.sub(r"```", "", code)

        local_vars = {'df': df, 'pd': pd, 'np': np}
        exec(code, local_vars)

        if "result" in local_vars:
            return local_vars["result"], code
        return "Error: No result variable returned.", code

    except Exception as e:
        return f"Execution error: {e}", ""


# -------------------- MAIN APP --------------------
def main():

    st.title("AskData — E-Commerce Assistant")
    st.caption("Interactive data queries and visual insights")

    # -------------------- EXAMPLE QUERY DROPDOWN --------------------
    example_query = st.selectbox(
        "Example Queries",
        [
            "",
            "Find the product with highest frequency in each region",
            "Top 5 customers by total spending",
            "Monthly sales trend by category",
            "Average order value by gender",
            "Most popular product category by zone"
        ]
    )

    if example_query:
        query = example_query
    else:
        query = st.text_input("Enter your analytical question")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        try:
            df_raw = pd.read_csv(uploaded_file)
            df = preprocess_dataframe(df_raw)

            st.subheader("Data Preview")
            st.dataframe(df.head())

            st.subheader("Response Type")
            col1, col2, col3 = st.columns(3)
            with col1:
                mode = st.selectbox("Mode", ["Smart Analysis", "Chart", "Simple Query"])
            with col2:
                use_ai = st.checkbox("Use AI", value=True)
            with col3:
                show_code = st.checkbox("Show Code")

            if query and use_ai:

                # ---------------- SMART ANALYSIS ----------------
                if mode == "Smart Analysis":
                    st.subheader("Results")
                    with st.spinner("Analyzing..."):
                        result, code = execute_complex_query(df, query, llm)

                    if show_code:
                        st.code(code)

                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                        st.download_button(
                            "Download CSV",
                            result.to_csv(index=False),
                            file_name="result.csv"
                        )
                    else:
                        st.write(result)

                # ---------------- CHART MODE ----------------
                elif mode == "Chart":
                    system_prompt = f"""
                    Generate Python code using pandas + plotly.express to create a chart.
                    Use dataframe 'df'. Only output valid code that creates a figure named fig.
                    """

                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=query)
                    ]

                    with st.spinner("Generating chart..."):
                        code = llm.invoke(messages).content

                    code = re.sub(r"```python", "", code)
                    code = re.sub(r"```", "", code)

                    if show_code:
                        st.code(code)

                    local = {}
                    exec(code, {"df": df, "px": px, "pd": pd, "np": np, "go": go}, local)

                    if "fig" in local:
                        st.plotly_chart(local["fig"], use_container_width=True)
                    else:
                        st.error("Generated code did not produce a figure.")

                # ---------------- SIMPLE QUERY ----------------
                elif mode == "Simple Query":
                    agent = create_pandas_dataframe_agent(
                        llm, df, verbose=False, allow_dangerous_code=True
                    )
                    response = agent.invoke(query)
                    st.write(response["output"])

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
