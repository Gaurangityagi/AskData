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
# PDF generator
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


# ---------------------------
# User preprocessing options
# ---------------------------
def apply_user_preprocessing(df, choice, threshold):
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    if choice == "Fill numeric with Mean":
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)

    elif choice == "Fill numeric with Median":
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)

    elif choice == "Fill categorical with Mode":
        for col in cat_cols:
            if df[col].mode().empty:
                df[col].fillna("Unknown", inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    elif choice == "Fill all with NULL/Unknown":
        df.fillna("Unknown", inplace=True)

    elif choice == "Drop rows with missing values":
        df.dropna(inplace=True)

    elif choice == "Drop columns with missing % > threshold":
        percent = df.isna().mean()
        cols_to_drop = percent[percent > (threshold / 100)].index
        df.drop(columns=cols_to_drop, inplace=True)

    return df


# ---------------------------
# Main preprocessing
# ---------------------------
def preprocess_dataframe(df):
    try:
        df_processed = df.copy()

        # Standardize column names
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

        for k, v in rename_map.items():
            if k in df_processed.columns and v not in df_processed.columns:
                df_processed.rename(columns={k: v}, inplace=True)

        if 'PRODUCT_CATEGORY' in df_processed.columns and 'PRODUCT' in df_processed.columns:
            df_processed.drop(columns=['PRODUCT'], inplace=True)
        elif 'PRODUCT' in df_processed.columns:
            df_processed.rename(columns={"PRODUCT": "PRODUCT_CATEGORY"}, inplace=True)

        if 'ORDER_DATE' in df_processed.columns:
            df_processed['ORDER_DATE'] = pd.to_datetime(df_processed['ORDER_DATE'], errors='coerce', dayfirst=True)
            df_processed.dropna(subset=['ORDER_DATE'], inplace=True)

        numeric_cols = [
            'AMOUNT', 'ORDERS', 'QUANTITY', 'UNIT_COST', 'UNIT_PRICE',
            'PROFIT', 'COST', 'REVENUE', 'CUSTOMER_AGE'
        ]
        for col in numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

        # Fill remaining object columns with 'Unknown' (optional)
        for col in df_processed.select_dtypes(include=['object']).columns:
            df_processed[col] = df_processed[col].fillna('Unknown')

        return df_processed

    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return pd.DataFrame()


# ---------------------------
# Execute complex (analysis) query via LLM
# ---------------------------
def execute_complex_query(df, query, llm):
    system_prompt = f"""
    You are an expert data analyst.
    Generate Python code using pandas only.
    DataFrame name is df.
    Columns: {df.columns.tolist()}
    Rules:
    - Use exact column names
    - Final output MUST be in variable named 'result'
    - Return ONLY Python code (no markdown, no explanations)
    """

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ])

        code = response.content.strip()
        code = re.sub(r'^\s*```(python)?\s*\n', '', code)
        code = re.sub(r'\n\s*```\s*$', '', code)
        code = code.strip()

        local_vars = {"df": df, "pd": pd, "np": np}
        exec(code, local_vars)

        if "result" in local_vars:
            return local_vars["result"], code
        return "No result variable found", code

    except Exception as e:
        return f"Error executing query: {e}", ""


# ---------------------------
# Execute chart-generation via LLM
# ---------------------------
def execute_chart_query(df, query, llm):
    # Instruction: create figure variable named 'fig' using px or go
    system_prompt = f"""
    You are an expert data visualization system.
    Use pandas + plotly.express or plotly.graph_objects.
    DataFrame name is df.
    Columns: {df.columns.tolist()}
    RULES:
    - Do NOT call fig.show()
    - Define the final figure as variable named 'fig'
    - Use only pandas, plotly.express (px) or plotly.graph_objects (go), and numpy if needed
    - Return ONLY Python code
    """

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ])

        chart_code = response.content.strip()
        chart_code = re.sub(r'^\s*```(python)?\s*\n', '', chart_code)
        chart_code = re.sub(r'\n\s*```\s*$', '', chart_code)
        chart_code = chart_code.strip()

        return chart_code
    except Exception as e:
        return ""


# ---------------------------
# Format result (safe)
# ---------------------------
def format_result_output(result):
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.to_frame()
    if isinstance(result, dict):
        return pd.DataFrame([result])
    return result  # string, int, float, or error message


# ---------------------------
# MAIN APP
# ---------------------------
def main():

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        if llm:
            st.success("LLM Connected")
        else:
            st.error("LLM Not Available")

        st.subheader("Data Cleaning Options")
        preprocessing_choice = st.selectbox(
            "Handle Missing Values:",
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

        threshold = None
        if preprocessing_choice == "Drop columns with missing % > threshold":
            threshold = st.slider("Missing % threshold:", 1, 100, 40)

        st.subheader("Example Queries")
        st.write("• Product with highest frequency in each region")
        st.write("• Top 5 customers by revenue")
        st.write("• Revenue by category")
        st.write("• Average order value by gender")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])

    if uploaded_file:
        df_raw = None
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                uploaded_file.seek(0)
                df_raw = pd.read_csv(uploaded_file, encoding=enc)
                break
            except Exception:
                continue

        if df_raw is None:
            st.error("Failed to read uploaded CSV (encoding issue).")
            return

        st.subheader("Raw Data Preview")
        st.dataframe(df_raw.head(10))

        # Apply user preprocessing before main preprocessing
        if preprocessing_choice != "None":
            df_raw = apply_user_preprocessing(df_raw, preprocessing_choice, threshold)

        # Main preprocessing
        df = preprocess_dataframe(df_raw)

        if df.empty:
            st.error("No valid data after preprocessing.")
            return

        st.subheader("Cleaned Data")
        st.dataframe(df.head(10))

        # Ask question
        st.subheader("Ask Your Question")
        query = st.text_input("Enter your query", placeholder="e.g. Top 5 customers by revenue")

        col1, col2, col3 = st.columns(3)
        mode = col1.selectbox("Mode", ["Smart Analysis", "Chart", "Simple Query"])
        use_ai = col2.checkbox("Use AI Analysis", value=llm is not None)
        show_code = col3.checkbox("Show Code", value=False)

        # If user wants AI but LLM unavailable
        if use_ai and llm is None:
            st.warning("AI Analysis requested but LLM is not connected. Disable 'Use AI Analysis' or configure secrets.")

        # SMART ANALYSIS
        if query and mode == "Smart Analysis":
            if not use_ai:
                st.warning("Enable 'Use AI Analysis' to use Smart Analysis mode.")
            else:
                with st.spinner("Analyzing your query..."):
                    result, code = execute_complex_query(df, query, llm)

                    if show_code and code:
                        st.subheader("Generated Code")
                        st.code(code, language="python")

                    formatted = format_result_output(result)

                    # Safe display
                    if isinstance(formatted, pd.DataFrame):
                        st.dataframe(formatted)
                        csv = formatted.to_csv(index=False)
                        st.download_button("Download CSV", csv, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                    else:
                        st.write(formatted)

                    # PDF export
                    if st.button("📄 Export to PDF"):
                        pdf_buffer = generate_pdf(str(formatted))
                        if pdf_buffer:
                            st.download_button(
                                label="Download PDF",
                                data=pdf_buffer,
                                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            )

        # CHART MODE (LLM generates chart code)
        elif query and mode == "Chart":
            if not use_ai:
                st.warning("Enable 'Use AI Analysis' to use Chart mode.")
            else:
                with st.spinner("Generating chart code..."):
                    chart_code = execute_chart_query(df, query, llm)

                    if not chart_code:
                        st.error("Chart-generation failed or returned empty code.")
                    else:
                        if show_code:
                            st.subheader("Chart Code")
                            st.code(chart_code, language="python")

                        # Execute the chart code safely in limited namespace
                        local_vars = {}
                        exec_globals = {"df": df, "pd": pd, "px": px, "go": go, "np": np}
                        try:
                            exec(chart_code, exec_globals, local_vars)
                            # The LLM must define 'fig'
                            if "fig" in local_vars:
                                st.plotly_chart(local_vars["fig"], use_container_width=True)
                            elif "fig" in exec_globals:
                                st.plotly_chart(exec_globals["fig"], use_container_width=True)
                            else:
                                st.warning("No 'fig' variable was created by the chart code.")
                        except Exception as e:
                            st.error(f"Error executing chart code: {e}")

        # SIMPLE QUERY (langchain agent)
        elif query and mode == "Simple Query":
            if not use_ai:
                st.warning("Enable 'Use AI Analysis' to use Simple Query mode.")
            else:
                try:
                    agent = create_pandas_dataframe_agent(
                        llm, df, verbose=False, allow_dangerous_code=True
                    )
                    with st.spinner("Processing simple query..."):
                        result = agent.invoke(query)
                        # agent.invoke often returns dict with 'output'
                        if isinstance(result, dict) and 'output' in result:
                            st.markdown(result['output'])
                        else:
                            st.write(result)
                except Exception as e:
                    st.error(f"Simple query failed: {e}")

if __name__ == "__main__":
    main()
