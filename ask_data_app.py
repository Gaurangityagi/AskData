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

st.set_page_config(page_title="Ask Data", layout="wide")
st.title(" AskData - Data answers made easy")

# ---------------------------
# Initialize LLM
# ---------------------------
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


# ---------------------------
# PDF Generator
# ---------------------------
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
            clean_line = line.encode('latin-1', 'ignore').decode('latin-1')
            pdf.cell(0, 6, clean_line, ln=True)

        pdf_buffer = io.BytesIO()
        pdf_output = pdf.output(dest='S').encode('latin-1')
        pdf_buffer.write(pdf_output)
        pdf_buffer.seek(0)
        return pdf_buffer

    except Exception as e:
        st.error(f"PDF generation error: {e}")
        return None


# ------------------------------------------------------
# EXTRA PREPROCESSING OPTIONS (NEWLY ADDED)
# ------------------------------------------------------
def apply_user_preprocessing(df, preprocessing_choice, missing_threshold):
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    if preprocessing_choice == "Fill numeric with Mean":
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)

    elif preprocessing_choice == "Fill numeric with Median":
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)

    elif preprocessing_choice == "Fill categorical with Mode":
        for col in cat_cols:
            if df[col].mode().empty:
                df[col].fillna("Unknown", inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    elif preprocessing_choice == "Fill all with NULL/Unknown":
        df.fillna("Unknown", inplace=True)

    elif preprocessing_choice == "Drop rows with missing values":
        df.dropna(inplace=True)

    elif preprocessing_choice == "Drop columns with missing % > threshold":
        threshold = missing_threshold / 100
        missing_percent = df.isna().mean()
        cols_to_drop = missing_percent[missing_percent > threshold].index
        df.drop(columns=cols_to_drop, inplace=True)

    return df


# ---------------------------
# Main Preprocess Function
# ---------------------------
def preprocess_dataframe(df):
    try:
        df_processed = df.copy()

        # Standardize Columns
        df_processed.columns = df_processed.columns.str.strip().str.upper()

        flexible_rename_map = {
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

        rename_dict = {k: v for k, v in flexible_rename_map.items() if k in df_processed.columns}
        df_processed.rename(columns=rename_dict, inplace=True)

        # Handle PRODUCT column
        if 'PRODUCT_CATEGORY' in df_processed.columns and 'PRODUCT' in df_processed.columns:
            df_processed.drop(columns=['PRODUCT'], inplace=True)
        elif 'PRODUCT' in df_processed.columns:
            df_processed.rename(columns={'PRODUCT': 'PRODUCT_CATEGORY'}, inplace=True)

        # Convert Dates
        if 'ORDER_DATE' in df_processed.columns:
            df_processed['ORDER_DATE'] = pd.to_datetime(df_processed['ORDER_DATE'], errors='coerce', dayfirst=True)
            df_processed.dropna(subset=['ORDER_DATE'], inplace=True)

        # Convert numerics
        for col in ['AMOUNT', 'ORDERS', 'QUANTITY', 'UNIT_COST', 'UNIT_PRICE', 'PROFIT', 'COST', 'REVENUE', 'CUSTOMER_AGE']:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

        return df_processed

    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return pd.DataFrame()


# ---------------------------
# Execute Complex Query
# ---------------------------
def execute_complex_query(df, query, llm):
    system_prompt = f"""
    You are an expert data analyst.
    Generate Python code to answer complex analytical queries.
    DataFrame name: df
    Columns: {df.columns.tolist()}
    RULES:
    - Use exact column names
    - Final output must be variable named 'result'
    - Use pandas only
    - Return ONLY Python code
    """

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ])

        code = re.sub(r"```.*?```", "", response.content, flags=re.DOTALL).strip()
        local_vars = {'df': df, 'pd': pd, 'np': np}
        exec(code, local_vars)

        return local_vars.get("result", "No result variable found"), code

    except Exception as e:
        return f"Error running query: {e}", ""


# ---------------------------
# Output Formatter
# ---------------------------
def format_result_output(result):
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.to_frame()
    if isinstance(result, dict):
        return pd.DataFrame([result])
    return str(result)


# ---------------------------
# MAIN APP
# ---------------------------
def main():

    # Sidebar
    with st.sidebar:

        st.header("Configuration")

        # AI Status
        if llm:
            st.success("Connected")
        else:
            st.error("LLM Not Available")

        # NEW — DATA PREPROCESSING OPTIONS
        st.subheader("Data Cleaning Options")

        preprocessing_choice = st.selectbox(
            "Choose missing value handling:",
            [
                "None",
                "Fill numeric with Mean",
                "Fill numeric with Median",
                "Fill categorical with Mode",
                "Fill all with NULL/Unknown",
                "Drop rows with missing values",
                "Drop columns with missing % > threshold",
            ]
        )

        missing_threshold = None
        if preprocessing_choice == "Drop columns with missing % > threshold":
            missing_threshold = st.slider("Missing % Threshold", 1, 100, 40)

        st.subheader("Example Queries")
        st.write("• Product with highest frequency in each region")
        st.write("• Average order value by gender")
        st.write("• Plot total revenue by category")


    # Upload File
    uploaded_file = st.file_uploader("Upload your sales data (CSV)", type=["csv"])

    if uploaded_file:

        # Safe load with multiple encodings
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                uploaded_file.seek(0)
                df_raw = pd.read_csv(uploaded_file, encoding=enc)
                break
            except:
                continue

        st.write("### Raw Data Preview")
        st.dataframe(df_raw.head())

        # Apply user-selected preprocessing BEFORE main processing
        if preprocessing_choice != "None":
            df_raw = apply_user_preprocessing(df_raw, preprocessing_choice, missing_threshold)

        # Main Preprocessing
        df = preprocess_dataframe(df_raw)

        st.write("### Cleaned Data")
        st.dataframe(df.head())

        # Ask Query
        st.subheader("Ask Your Question")
        query = st.text_input("Enter your query")

        col1, col2 = st.columns(2)
        mode = col1.selectbox("Mode", ["Smart Analysis", "Chart", "Simple Query"])
        show_code = col2.checkbox("Show Code")

        if query and llm:

            if mode == "Smart Analysis":
                result, code = execute_complex_query(df, query, llm)

                if show_code:
                    st.code(code, language="python")

                formatted = format_result_output(result)
                st.dataframe(formatted)

            elif mode == "Chart":
                st.write("Chart mode coming with LLM-based plot generation")

            elif mode == "Simple Query":
                agent = create_pandas_dataframe_agent(
                    llm, df, verbose=False, allow_dangerous_code=True
                )
                response = agent.invoke(query)
                st.markdown(response["output"])


if __name__ == "__main__":
    main()
