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

# ---------------------------
# Page config and styling
# ---------------------------
st.set_page_config(page_title="E-Commerce Assistant", layout="wide")

# Simple CSS for gradients and spacing (no emojis)
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7fbff 0%, #ffffff 40%);
    }
    .header {
        background: linear-gradient(90deg, #4b79a1 0%, #283e51 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .section-title {
        font-size:20px;
        font-weight:600;
        margin-bottom:8px;
    }
    .card {
        background: #ffffff;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .small-muted {
        color: #6b7280;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="header"><h1 style="margin:0">AskData — Data answers made easy</h1></div>', unsafe_allow_html=True)

# ---------------------------
# LLM initialization
# ---------------------------
@st.cache_resource
def initialize_llm():
    try:
        llm = AzureChatOpenAI(
            openai_api_version=st.secrets.get("OPENAI_API_VERSION"),
            azure_deployment=st.secrets.get("AZURE_DEPLOYMENT"),
            azure_endpoint=st.secrets.get("AZURE_ENDPOINT"),
            api_key=st.secrets.get("AZURE_API_KEY")
        )
        return llm
    except Exception as e:
        # show error but return None so UI still loads
        st.error(f"Failed to initialize LLM: {e}")
        return None

llm = initialize_llm()

# ---------------------------
# Utility functions
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

def preprocess_dataframe(df):
    try:
        df_processed = df.copy()
        # Uppercase columns and strip whitespace
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

        # If both PRODUCT and PRODUCT_CATEGORY exist, keep PRODUCT_CATEGORY
        if 'PRODUCT_CATEGORY' in df_processed.columns and 'PRODUCT' in df_processed.columns:
            st.warning("Both PRODUCT_CATEGORY and PRODUCT found. Dropping PRODUCT and keeping PRODUCT_CATEGORY.")
            df_processed.drop(columns=['PRODUCT'], inplace=True, errors='ignore')
        elif 'PRODUCT' in df_processed.columns and 'PRODUCT_CATEGORY' not in df_processed.columns:
            df_processed.rename(columns={'PRODUCT': 'PRODUCT_CATEGORY'}, inplace=True)

        # Date parsing
        if 'ORDER_DATE' in df_processed.columns:
            try:
                original_rows = len(df_processed)
                df_processed['ORDER_DATE'] = pd.to_datetime(df_processed['ORDER_DATE'], errors='coerce', dayfirst=True)
                df_processed.dropna(subset=['ORDER_DATE'], inplace=True)
                dropped = original_rows - len(df_processed)
                if dropped > 0:
                    st.warning(f"Dropped {dropped} rows with invalid ORDER_DATE.")
            except Exception as e:
                st.warning(f"Could not convert ORDER_DATE to datetime: {e}")
                if df_processed['ORDER_DATE'].isnull().any():
                    df_processed.dropna(subset=['ORDER_DATE'], inplace=True)

        # Numeric conversions
        numeric_columns = ['AMOUNT', 'ORDERS', 'QUANTITY', 'UNIT_COST', 'UNIT_PRICE', 'PROFIT', 'COST', 'REVENUE', 'CUSTOMER_AGE']
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                if df_processed[col].isnull().any():
                    df_processed[col] = df_processed[col].fillna(0)

        # Gender standardization
        if 'GENDER' in df_processed.columns:
            if not df_processed['GENDER'].dropna().empty:
                df_processed['GENDER'] = df_processed['GENDER'].astype(str).str.upper().replace({'MALE': 'M', 'FEMALE': 'F', 'UNKNOWN': 'U', 'OTHER': 'O'})
            else:
                df_processed['GENDER'] = df_processed['GENDER'].fillna('Unknown')

        # Fill remaining object nulls
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object' and df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna('Unknown')

        return df_processed

    except Exception as e:
        st.error(f"Error during dataframe preprocessing: {e}")
        return pd.DataFrame()

def execute_complex_query(df, query, llm):
    """
    Execute complex analytical queries using LLM-generated code.
    Ensures 'code' variable always exists and uses llm.invoke(...)
    """
    code = ""  # ensure defined for safe returns

    system_prompt = f"""
    You are an expert data analyst. Generate Python code to answer complex analytical queries.

    The dataframe is named 'df' and has the following columns: {df.columns.tolist()}
    Sample data (first 3 rows): {df.head(3).to_dict('records')}

    IMPORTANT RULES:
    1. Always use exact column names from the dataframe.
    2. Store the final result in a variable named 'result'.
    3. Include data processing steps with comments.
    4. Handle missing values and edge cases within the generated code if necessary.
    5. Use pandas operations efficiently.
    6. Return the result as a dataframe or series when possible.
    7. Generate ONLY the Python code without any explanations or markdown formatting.
    """

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate Python code for: {query}")
        ]

        response = llm.invoke(messages)
        code = getattr(response, "content", "") or ""
        code = code.strip()

        # Remove possible code fences
        code = re.sub(r'^\s*```(?:python)?\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n\s*```\s*$', '', code, flags=re.MULTILINE)
        code = code.strip()

        if not code:
            return "LLM returned no code for the query.", code

        local_vars = {'df': df, 'pd': pd, 'np': np}
        exec(code, local_vars)

        if 'result' in local_vars:
            return local_vars['result'], code
        else:
            return "No 'result' variable found in the generated code. Please ensure the code assigns the final output to a variable named 'result'.", code

    except Exception as e:
        return f"Error executing query: {str(e)}", code

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
# Main application
# ---------------------------
def main():
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Configuration")
        if llm is not None:
            st.success("LLM Connected")
        else:
            st.warning("LLM Not Available")

        st.markdown("**Query Examples**")
        st.write("- Product with highest frequency in each region")
        st.write("- Top 5 customers by total spending")
        st.write("- Monthly sales trend by category")
        st.write("- Average order value by gender")
        st.write("- Most popular product category by zone")
        st.markdown('</div>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Upload your sales data (CSV)", type=["csv"])

    if uploaded_file:
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df_raw = None
            for enc in encodings:
                try:
                    uploaded_file.seek(0)
                    df_raw = pd.read_csv(uploaded_file, encoding=enc)
                    break
                except Exception:
                    continue

            if df_raw is None or df_raw.empty:
                st.error("Failed to load file. Please ensure it's a valid CSV with data.")
                return

            df = preprocess_dataframe(df_raw)

            if not isinstance(df, pd.DataFrame) or df.empty:
                st.error("The uploaded file resulted in no valid data after preprocessing.")
                return

            # Layout: left preview, right actions
            left, right = st.columns([2, 1])
            with left:
                st.subheader("Data Preview")
                st.dataframe(df.head(10))

                with st.expander("Data Summary"):
                    info = get_data_summary(df)
                    st.write("Rows:", df.shape[0], "Columns:", df.shape[1])
                    st.write("Columns:", list(df.columns))
                    st.write("Missing values (post-processing):")
                    missing_counts = pd.Series(info["missing_values"])
                    missing_df = missing_counts[missing_counts > 0].rename_axis("Column").reset_index(name="Missing Count")
                    if not missing_df.empty:
                        st.dataframe(missing_df)
                    else:
                        st.write("No missing values detected.")

            with right:
                st.subheader("Ask a question")
                query = st.text_input("Enter your analytical question",
                                      placeholder="e.g., Find the product with highest frequency in each region")
                mode = st.selectbox("Response type", ["Smart Analysis", "Chart", "Simple Query"])
                use_ai = st.checkbox("Use AI Analysis", value=llm is not None)
                show_code = st.checkbox("Show Generated Code", value=False)

            if query and use_ai and llm is not None:
                st.subheader("Results")
                if mode == "Smart Analysis":
                    with st.spinner("Analyzing your query..."):
                        try:
                            result, code = execute_complex_query(df, query, llm)

                            if show_code and code:
                                st.subheader("Generated Code")
                                st.code(code, language="python")

                            st.subheader("Result")
                            formatted_result = format_result_output(result)

                            if isinstance(formatted_result, pd.DataFrame) and not formatted_result.empty:
                                st.dataframe(formatted_result)
                                csv = formatted_result.to_csv(index=False)
                                st.download_button(
                                    label="Download Results as CSV",
                                    data=csv,
                                    file_name=f"analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.write(formatted_result)

                            if st.button("Export to PDF"):
                                pdf_content = f"Query: {query}\n\nResult:\n{formatted_result}"
                                pdf_buffer = generate_pdf(pdf_content)
                                if pdf_buffer:
                                    st.download_button(
                                        label="Download PDF",
                                        data=pdf_buffer,
                                        file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                        mime="application/pdf"
                                    )
                        except Exception as e:
                            st.error(f"Error in smart analysis: {e}")
                            st.write("Trying fallback to simple agent...")
                            try:
                                if not df.empty:
                                    agent = create_pandas_dataframe_agent(
                                        llm,
                                        df,
                                        verbose=True,
                                        allow_dangerous_code=True
                                    )
                                    # agent.invoke or agent.run depending on agent object - using invoke as before
                                    result = agent.invoke(query)
                                    st.markdown(result.get('output', str(result)))
                                else:
                                    st.warning("Cannot use fallback agent: DataFrame is empty.")
                            except Exception as e2:
                                st.error(f"Fallback also failed: {e2}")

                elif mode == "Chart":
                    with st.spinner("Generating chart..."):
                        system_prompt = f"""
                        You are an expert in data visualization. Your task is to generate Python code using pandas and plotly.express to create a chart based on the user's query.
                        The dataframe is named 'df' and has the following columns: {df.columns.tolist()}.
                        Do NOT use 'fig.show()'. Define the figure as 'fig'.
                        Ensure all column names used match exactly those in df.
                        Generate ONLY the Python code without any explanations or markdown formatting.
                        """
                        user_message = f"Generate Python code for: '{query}'"
                        messages = [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=user_message)
                        ]
                        try:
                            response = llm.invoke(messages)
                            chart_code = getattr(response, "content", "") or ""
                            chart_code = re.sub(r'^\s*```(?:python)?\s*\n', '', chart_code, flags=re.MULTILINE)
                            chart_code = re.sub(r'\n\s*```\s*$', '', chart_code, flags=re.MULTILINE)
                            chart_code = chart_code.strip()

                            if show_code:
                                st.code(chart_code, language="python")

                            try:
                                local_vars = {}
                                exec(chart_code, {"df": df, "px": px, "pd": pd, "go": go, "np": np}, local_vars)
                                if 'fig' in local_vars:
                                    st.plotly_chart(local_vars['fig'], use_container_width=True)
                                else:
                                    st.warning("The generated code did not create a Plotly figure named 'fig'.")
                            except Exception as e:
                                st.error(f"Chart execution error: {e}")

                        except Exception as e:
                            st.error(f"Failed to generate chart: {e}")

                elif mode == "Simple Query":
                    try:
                        if not df.empty:
                            agent = create_pandas_dataframe_agent(
                                llm,
                                df,
                                verbose=True,
                                allow_dangerous_code=True
                            )

                            with st.spinner("Processing simple query..."):
                                result = agent.invoke(query)
                                st.markdown(result.get('output', str(result)))
                        else:
                            st.warning("Cannot process simple query: DataFrame is empty.")
                    except Exception as e:
                        st.error(f"Error in simple query: {e}")

            elif query and not use_ai:
                st.warning("Please enable 'Use AI Analysis' to get responses.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.info("Please ensure your CSV file is correctly formatted and try again.")

if __name__ == "__main__":
    main()
