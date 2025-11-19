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
# Page config
# ---------------------------
st.set_page_config(page_title="AskData — E-Commerce Assistant", layout="wide")

# ---------------------------
# Theme CSS
# ---------------------------
CSS = """
<style>
:root {
  --bg: #0f0f0f;
  --card: #1d1d1d;
  --text: #f5e6c8;
  --muted: #cfc1a8;
  --accent: #f5e6c8;
  --nav-bg: rgba(255,255,255,0.04);
}

/* Light Mode overrides */
.light :root {
  --bg: #f7f2e7;
  --card: #ffffff;
  --text: #111111;
  --muted: #6b6b6b;
  --accent: #2a2a2a;
  --nav-bg: rgba(0,0,0,0.06);
}

/* Apply base colors */
body, .stApp, .main {
  background-color: var(--bg) !important;
}

/* Navbar card look (non-fixed to avoid overlap issues) */
.navbar {
  background: var(--nav-bg);
  border-radius: 12px;
  padding: 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  margin-bottom: 18px;
}

/* Title */
.nav-title {
  font-size: 20px;
  font-weight: 700;
  color: var(--text);
  margin: 0;
}
.nav-sub {
  font-size: 13px;
  color: var(--muted);
  margin-top: 4px;
}

/* Center (example select) */
.center-control select, .center-control .stSelectbox>div {
  min-width: 380px;
}

/* Card */
.card {
  background-color: var(--card);
  padding: 18px;
  border-radius: 12px;
  color: var(--text);
  box-shadow: 0 6px 20px rgba(0,0,0,0.25);
  margin-bottom: 20px;
}

/* Section title */
.section-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 10px;
  color: var(--text);
}

/* Buttons */
.stButton>button {
  background-color: var(--accent) !important;
  color: var(--bg) !important;
  font-weight: 600;
  border: none;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------
# LLM initialization
# ---------------------------
@st.cache_resource
def initialize_llm():
    try:
        return AzureChatOpenAI(
            openai_api_version=st.secrets.get("OPENAI_API_VERSION"),
            azure_deployment=st.secrets.get("AZURE_DEPLOYMENT"),
            azure_endpoint=st.secrets.get("AZURE_ENDPOINT"),
            api_key=st.secrets.get("AZURE_API_KEY"),
        )
    except Exception:
        return None

llm = initialize_llm()

# ---------------------------
# Theme toggle state
# ---------------------------
if "light_mode" not in st.session_state:
    st.session_state["light_mode"] = False

# ---------------------------
# Top "navbar" using Streamlit components (split layout)
# ---------------------------
with st.container():
    st.markdown('<div class="navbar">', unsafe_allow_html=True)
    nav_cols = st.columns([3, 3, 1])
    with nav_cols[0]:
        st.markdown('<div class="nav-title">AskData — E-Commerce Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-sub">Interactive data queries & visualizations</div>', unsafe_allow_html=True)
    with nav_cols[1]:
        # Example queries selectbox (center)
        example_query = st.selectbox(
            label="Example queries",
            options=[
                "",
                "Find the product with highest frequency in each region",
                "Top 5 customers by total spending",
                "Monthly sales trend by category",
                "Average order value by gender",
                "Most popular product category by zone"
            ],
            index=0,
            key="nav_example"
        )
    with nav_cols[2]:
        # Light mode toggle (right)
        lm = st.checkbox("Light mode", value=st.session_state["light_mode"], key="nav_light")
    st.markdown('</div>', unsafe_allow_html=True)

# Apply theme class based on toggle (non-invasive)
if st.session_state.get("nav_light", False):
    st.session_state["light_mode"] = True
    st.markdown("<script>document.documentElement.classList.add('light');</script>", unsafe_allow_html=True)
else:
    st.session_state["light_mode"] = False
    st.markdown("<script>document.documentElement.classList.remove('light');</script>", unsafe_allow_html=True)

# ---------------------------
# Utility functions (unchanged logic)
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
            df_processed.drop(columns=['PRODUCT'], inplace=True, errors='ignore')
        elif 'PRODUCT' in df_processed.columns and 'PRODUCT_CATEGORY' not in df_processed.columns:
            df_processed.rename(columns={'PRODUCT': 'PRODUCT_CATEGORY'}, inplace=True)

        if 'ORDER_DATE' in df_processed.columns:
            try:
                original_rows = len(df_processed)
                df_processed['ORDER_DATE'] = pd.to_datetime(df_processed['ORDER_DATE'], errors='coerce', dayfirst=True)
                df_processed.dropna(subset=['ORDER_DATE'], inplace=True)
                dropped = original_rows - len(df_processed)
                if dropped > 0:
                    st.warning(f"Dropped {dropped} rows with invalid ORDER_DATE.")
            except Exception as e:
                if 'ORDER_DATE' in df_processed.columns and df_processed['ORDER_DATE'].isnull().any():
                    df_processed.dropna(subset=['ORDER_DATE'], inplace=True)

        numeric_columns = ['AMOUNT', 'ORDERS', 'QUANTITY', 'UNIT_COST', 'UNIT_PRICE', 'PROFIT', 'COST', 'REVENUE', 'CUSTOMER_AGE']
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                if df_processed[col].isnull().any():
                    df_processed[col] = df_processed[col].fillna(0)

        if 'GENDER' in df_processed.columns:
            if not df_processed['GENDER'].dropna().empty:
                df_processed['GENDER'] = df_processed['GENDER'].astype(str).str.upper().replace({'MALE': 'M', 'FEMALE': 'F', 'UNKNOWN': 'U', 'OTHER': 'O'})
            else:
                df_processed['GENDER'] = df_processed['GENDER'].fillna('Unknown')

        for col in df_processed.columns:
            if df_processed[col].dtype == 'object' and df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna('Unknown')

        return df_processed
    except Exception as e:
        st.error(f"Error during dataframe preprocessing: {e}")
        return pd.DataFrame()

def execute_complex_query(df, query, llm):
    code = ""
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
# Main app content
# ---------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload your CSV</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload sales data (CSV)", type=["csv"])
st.markdown('</div>', unsafe_allow_html=True)

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
            st.error("Failed to load file. Please ensure it's a valid CSV.")
        else:
            df = preprocess_dataframe(df_raw)
            if not isinstance(df, pd.DataFrame) or df.empty:
                st.error("Uploaded file produced no valid rows after preprocessing.")
            else:
                # Preview & controls
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Data Preview</div>', unsafe_allow_html=True)
                st.dataframe(df.head(10))
                st.markdown('</div>', unsafe_allow_html=True)

                # Query controls
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Ask a Question</div>', unsafe_allow_html=True)

                # If example was selected in nav, fill query by button
                query_default = ""
                if example_query:
                    query_default = example_query

                query = st.text_input("Enter your analytical question", value=query_default, key="main_query")
                mode = st.selectbox("Response type", ["Smart Analysis", "Chart", "Simple Query"])
                use_ai = st.checkbox("Use AI Analysis", value=(llm is not None))
                show_code = st.checkbox("Show Generated Code", value=False)
                st.markdown('</div>', unsafe_allow_html=True)

                # Run query
                if query and use_ai and llm is not None:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

                    if mode == "Smart Analysis":
                        with st.spinner("Analyzing your query..."):
                            try:
                                result, code = execute_complex_query(df, query, llm)
                                if show_code and code:
                                    st.subheader("Generated Code")
                                    st.code(code, language="python")
                                formatted_result = format_result_output(result)
                                if isinstance(formatted_result, pd.DataFrame) and not formatted_result.empty:
                                    st.dataframe(formatted_result)
                                    csv = formatted_result.to_csv(index=False)
                                    st.download_button("Download Results as CSV", csv, file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
                                else:
                                    st.write(formatted_result)
                                if st.button("Export to PDF"):
                                    pdf_buffer = generate_pdf(f"Query: {query}\n\nResult:\n{formatted_result}")
                                    if pdf_buffer:
                                        st.download_button("Download PDF", data=pdf_buffer, file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
                            except Exception as e:
                                st.error(f"Error in smart analysis: {e}")
                                st.write("Trying fallback to simple agent...")
                                try:
                                    if not df.empty:
                                        agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
                                        result = agent.invoke(query)
                                        out = result.get('output', str(result)) if isinstance(result, dict) else str(result)
                                        st.markdown(out)
                                    else:
                                        st.warning("Cannot use fallback agent: DataFrame is empty.")
                                except Exception as e2:
                                    st.error(f"Fallback also failed: {e2}")

                    elif mode == "Chart":
                        with st.spinner("Generating chart..."):
                            system_prompt = f"""
                            You are an expert in data visualization. Generate Python code using pandas + plotly.express.
                            The dataframe is 'df'. Put the final figure in a variable named 'fig'.
                            Use exact column names. Return only Python code.
                            Columns: {df.columns.tolist()}
                            """
                            try:
                                messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"Generate Python code for: {query}")]
                                response = llm.invoke(messages)
                                chart_code = getattr(response, "content", "") or ""
                                chart_code = re.sub(r'^\s*```(?:python)?\s*\n', '', chart_code, flags=re.MULTILINE)
                                chart_code = re.sub(r'\n\s*```\s*$', '', chart_code, flags=re.MULTILINE)
                                chart_code = chart_code.strip()
                                if show_code:
                                    st.code(chart_code, language="python")
                                local_vars = {}
                                exec(chart_code, {"df": df, "px": px, "pd": pd, "go": go, "np": np}, local_vars)
                                if 'fig' in local_vars:
                                    st.plotly_chart(local_vars['fig'], use_container_width=True)
                                else:
                                    st.warning("The generated code did not create a Plotly figure named 'fig'.")
                            except Exception as e:
                                st.error(f"Chart generation error: {e}")

                    elif mode == "Simple Query":
                        try:
                            agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
                            with st.spinner("Processing simple query..."):
                                result = agent.invoke(query)
                                out = result.get('output', str(result)) if isinstance(result, dict) else str(result)
                                st.markdown(out)
                        except Exception as e:
                            st.error(f"Error in simple query: {e}")

                    st.markdown('</div>', unsafe_allow_html=True)

                elif query and not use_ai:
                    st.warning("Please enable 'Use AI Analysis' to get responses.")

    except Exception as e:
        st.error(f"An unexpected error occurred while processing file: {e}")

# End of app
