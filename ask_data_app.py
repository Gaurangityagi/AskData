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

st.set_page_config(page_title="📊 E-Commerce Assistant", layout="wide")
st.title("📊 AskData-Data answers made easy")


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
# Generate PDF
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
            try:
                clean_line = line.encode('latin-1', 'ignore').decode('latin-1')
                pdf.cell(0, 6, clean_line, ln=True)
            except Exception:
                pdf.cell(0, 6, "Special characters removed or line skipped due to encoding issue.", ln=True)

        pdf_buffer = io.BytesIO()
        pdf_string = pdf.output(dest='S').encode('latin-1')
        pdf_buffer.write(pdf_string)
        pdf_buffer.seek(0)
        return pdf_buffer

    except Exception as e:
        st.error(f"PDF generation error: {e}")
        return None


# ---------------------------
# Preprocess DataFrame
# ---------------------------
def preprocess_dataframe(df):
    try:
        df_processed = df.copy()

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

        cols_to_rename = {}
        for original_col, canonical_col in flexible_rename_map.items():
            if original_col in df_processed.columns and original_col != canonical_col:
                if canonical_col not in df_processed.columns:
                    cols_to_rename[original_col] = canonical_col

        if cols_to_rename:
            df_processed.rename(columns=cols_to_rename, inplace=True)

        if 'PRODUCT_CATEGORY' in df_processed.columns and 'PRODUCT' in df_processed.columns:
            st.warning("Both 'PRODUCT_CATEGORY' and 'PRODUCT' exist. Dropping 'PRODUCT'.")
            df_processed.drop(columns=['PRODUCT'], inplace=True, errors='ignore')

        elif 'PRODUCT' in df_processed.columns and 'PRODUCT_CATEGORY' not in df_processed.columns:
            df_processed.rename(columns={'PRODUCT': 'PRODUCT_CATEGORY'}, inplace=True)
            st.info("Renamed 'PRODUCT' to 'PRODUCT_CATEGORY'.")

        if 'ORDER_DATE' in df_processed.columns:
            try:
                original_rows = len(df_processed)
                df_processed['ORDER_DATE'] = pd.to_datetime(
                    df_processed['ORDER_DATE'], errors='coerce', dayfirst=True
                )
                df_processed.dropna(subset=['ORDER_DATE'], inplace=True)
                dropped = original_rows - len(df_processed)
                if dropped > 0:
                    st.warning(f"Dropped {dropped} rows with invalid 'ORDER_DATE'.")
            except Exception as e:
                st.warning(f"Date conversion issue: {e}")

        numeric_columns = [
            'AMOUNT', 'ORDERS', 'QUANTITY', 'UNIT_COST', 'UNIT_PRICE',
            'PROFIT', 'COST', 'REVENUE', 'CUSTOMER_AGE'
        ]
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

        if 'GENDER' in df_processed.columns:
            if not df_processed['GENDER'].dropna().empty:
                df_processed['GENDER'] = df_processed['GENDER'].astype(str).str.upper().replace({
                    'MALE': 'M', 'FEMALE': 'F', 'UNKNOWN': 'U', 'OTHER': 'O'
                })
            else:
                df_processed['GENDER'] = df_processed['GENDER'].fillna('Unknown')

        for col in df_processed.columns:
            if df_processed[col].dtype == 'object' and df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna('Unknown')

        return df_processed

    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return pd.DataFrame()


# ---------------------------
# Complex Query Execution
# ---------------------------
def execute_complex_query(df, query, llm):
    system_prompt = f"""
    You are an expert data analyst.
    Generate Python code to answer complex analytical queries.
    The dataframe is named 'df' and has columns: {df.columns.tolist()}
    Sample data: {df.head(3).to_dict('records')}

    RULES:
    1. Use exact column names
    2. Final output MUST be in variable 'result'
    3. Use pandas only
    4. Return ONLY Python code, no markdown
    """

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate Python code for: {query}")
        ]

        response = llm.invoke(messages)
        code = response.content.strip()

        code = re.sub(r'^\s*```(python)?\s*\n', '', code)
        code = re.sub(r'\n\s*```\s*$', '', code)
        code = code.strip()

        local_vars = {'df': df, 'pd': pd, 'np': np}
        exec(code, local_vars)

        if 'result' in local_vars:
            return local_vars['result'], code
        else:
            return "No result variable found.", code

    except Exception as e:
        return f"Error executing query: {str(e)}", code


# ---------------------------
# Summary Builder
# ---------------------------
def get_data_summary(df):
    summary_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": {},
        "categorical_summary": {}
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary_info["numeric_summary"][col] = {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max()
        }

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        top_values = df[col].value_counts().head(10)
        summary_info["categorical_summary"][col] = {
            "unique_count": df[col].nunique(),
            "top_values": top_values.to_dict(),
            "mode": df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
        }

    return summary_info


# ---------------------------
# Format Output
# ---------------------------
def format_result_output(result):
    if isinstance(result, pd.DataFrame):
        return result
    elif isinstance(result, pd.Series):
        return result.to_frame(name="Value")
    elif isinstance(result, dict):
        return pd.DataFrame([result])
    else:
        return str(result)


# ---------------------------
# MAIN APP
# ---------------------------
def main():

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        if llm is not None:
            st.success("✅ LLM Connected")
        else:
            st.warning("⚠️ LLM Not Available")

        st.subheader("Query Examples")
        st.write("• Product with highest frequency in each region")
        st.write("• Top 5 customers by total spending")
        st.write("• Monthly sales trend by category")
        st.write("• Average order value by gender")
        st.write("• Most popular product category by zone")

    uploaded_file = st.file_uploader("📁 Upload your sales data (CSV)", type=["csv"])

    if uploaded_file:
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df_raw = None
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df_raw = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if df_raw is None or df_raw.empty:
                st.error("❌ Failed to load file.")
                return

            df = preprocess_dataframe(df_raw)

            if df.empty:
                st.error("❌ No valid data after preprocessing.")
                return

            st.subheader(" Data Preview")
            st.dataframe(df.head(10))

            with st.expander(" Data Summary"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Rows:**", df.shape[0])
                    st.write("**Columns:**", df.shape[1])
                    st.write("**Column Names:**", list(df.columns))

                with col2:
                    missing = df.isnull().sum()
                    missing = pd.DataFrame({
                        "Column": missing.index,
                        "Missing Count": missing.values
                    })
                    missing = missing[missing["Missing Count"] > 0]
                    if not missing.empty:
                        st.dataframe(missing)
                    else:
                        st.write("No missing values.")

            st.subheader("💬 Ask Your Question")
            query = st.text_input("Enter your query", placeholder="e.g. Top 5 customers by revenue")

            col1, col2, col3 = st.columns(3)
            mode = col1.selectbox("Response Type", ["Smart Analysis", "Chart", "Simple Query"])
            use_ai = col2.checkbox("Use AI Analysis", value=llm is not None)
            show_code = col3.checkbox("Show Code", value=False)

            # ------------------------------------
            # SMART ANALYSIS MODE
            # ------------------------------------
            if query and use_ai and llm is not None:

                st.subheader("🔍 Results")

                if mode == "Smart Analysis":
                    with st.spinner("Analyzing your query..."):
                        try:
                            result, code = execute_complex_query(df, query, llm)

                            if show_code:
                                st.subheader("Generated Code")
                                st.code(code, language="python")

                            formatted = format_result_output(result)

                            if isinstance(formatted, pd.DataFrame):
                                st.dataframe(formatted)

                                csv = formatted.to_csv(index=False)
                                st.download_button(
                                    "Download CSV",
                                    csv,
                                    f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                )
                            else:
                                st.write(formatted)

                            if st.button("📄 Export to PDF"):
                                pdf_content = f"Query: {query}\n\nResult:\n{formatted}"
                                pdf_buffer = generate_pdf(pdf_content)

                                if pdf_buffer:
                                    st.download_button(
                                        "Download PDF",
                                        pdf_buffer,
                                        f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                    )

                        except Exception as e:
                            st.error(f"Error: {e}")

                            try:
                                agent = create_pandas_dataframe_agent(
                                    llm, df, verbose=True, allow_dangerous_code=True
                                )
                                result = agent.invoke(query)
                                st.markdown(result['output'])
                            except Exception as e2:
                                st.error(f"Fallback failed: {e2}")

                # ------------------------------------
                # CHART MODE
                # ------------------------------------
                elif mode == "Chart":
                    with st.spinner("Generating chart..."):

                        system_prompt = f"""
                        You are an expert data visualization system.
                        Use pandas + plotly.express.
                        DataFrame = df
                        Columns: {df.columns.tolist()}
                        Do NOT use fig.show().
                        Define figure as: fig
                        Return ONLY Python code.
                        """

                        messages = [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=f"Generate Python code for: '{query}'")
                        ]

                        try:
                            response = llm.invoke(messages)
                            chart_code = response.content

                            chart_code = re.sub(r'^\s*```(python)?\s*\n', '', chart_code)
                            chart_code = re.sub(r'\n\s*```\s*$', '', chart_code)
                            chart_code = chart_code.strip()

                            if show_code:
                                st.code(chart_code, language="python")

                            local_vars = {}
                            exec(chart_code, {"df": df, "px": px, "pd": pd, "go": go, "np": np}, local_vars)

                            if "fig" in local_vars:
                                st.plotly_chart(local_vars['fig'], use_container_width=True)
                            else:
                                st.warning("⚠️ No figure named 'fig' was created.")

                        except Exception as e:
                            st.error(f"Chart generation failed: {e}")

                # ------------------------------------
                # SIMPLE QUERY MODE
                # ------------------------------------
                elif mode == "Simple Query":
                    try:
                        agent = create_pandas_dataframe_agent(
                            llm, df, verbose=True, allow_dangerous_code=True
                        )
                        with st.spinner("Processing simple query..."):
                            result = agent.invoke(query)
                            st.markdown(result['output'])
                    except Exception as e:
                        st.error(f"Error in simple query: {e}")

            elif query and not use_ai:
                st.warning("Enable 'Use AI Analysis' to get results.")

        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.info("Check your CSV formatting and try again.")


if __name__ == "__main__":
    main()
