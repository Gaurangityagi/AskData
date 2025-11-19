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
# Theme CSS (floating glass navbar + black/beige theme)
# ---------------------------
BASE_CSS = """
<style>
:root{
  --bg-color: #0f0f0f;
  --card-bg: #1a1a1a;
  --text-color: #f5e6c8;
  --muted: #cfc1a8;
  --accent: #f5e6c8;
  --nav-bg: rgba(255,255,255,0.04);
}

/* Light theme overrides */
.light :root{
  --bg-color: #f7f2e7;
  --card-bg: #ffffff;
  --text-color: #111111;
  --muted: #6b6b6b;
  --accent: #2a2a2a;
  --nav-bg: rgba(0,0,0,0.06);
}

/* App background */
.block-container {
  padding-top: 90px;
}

/* Floating glass navbar */
.top-nav {
  position: fixed;
  top: 14px;
  left: 50%;
  transform: translateX(-50%);
  width: calc(100% - 48px);
  max-width: 1200px;
  background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(0,0,0,0.03));
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  border-radius: 12px;
  padding: 12px 18px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 6px 20px rgba(0,0,0,0.35);
  border: 1px solid rgba(255,255,255,0.04);
}

/* Title */
.app-title {
  font-size: 20px;
  font-weight: 700;
  color: var(--text-color);
  margin: 0;
}

/* Right controls */
.nav-controls {
  display:flex;
  gap: 12px;
  align-items:center;
}

/* Query examples dropdown */
.select-examples select {
  padding: 8px 10px;
  border-radius: 8px;
  border: 1px solid rgba(0,0,0,0.06);
  background: var(--card-bg);
  color: var(--text-color);
  font-size: 14px;
}

/* Toggle wrapper */
.toggle-wrapper {
  display:flex;
  align-items:center;
  gap:8px;
  font-size:14px;
  color:var(--muted);
}

/* Card style for main sections */
.card {
  background: var(--card-bg);
  border-radius: 10px;
  padding: 16px;
  box-shadow: 0 4px 18px rgba(0,0,0,0.25);
  color: var(--text-color);
}

/* Section headings */
.section-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 8px;
  color: var(--text-color);
}

/* Muted text */
.small-muted {
  color: var(--muted);
  font-size: 13px;
}

/* Buttons */
.stButton>button {
  background: var(--accent);
  color: var(--text-color);
  border: none;
}
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)

# ---------------------------
# Theme state (dark/light)
# ---------------------------
if "theme_light" not in st.session_state:
    st.session_state["theme_light"] = False  # default to dark

# Top navbar container (floating)
nav_container = st.container()
with nav_container:
    st.markdown(
        """
        <div class="top-nav" id="topNav">
            <div style="display:flex;align-items:center;gap:18px;">
                <div style="display:flex;flex-direction:column;">
                    <div class="app-title">AskData — E-Commerce Assistant</div>
                    <div class="small-muted" style="margin-top:4px;">Interactive data queries & visualizations</div>
                </div>
            </div>
            <div class="nav-controls">
                <div class="select-examples">
                    <select id="exampleSelect" onchange="document.getElementById('exampleSelectValue').value=this.value">
                        <option value="">Select example query</option>
                        <option value="Find the product with highest frequency in each region">Product with highest frequency in each region</option>
                        <option value="Top 5 customers by total spending">Top 5 customers by total spending</option>
                        <option value="Monthly sales trend by category">Monthly sales trend by category</option>
                        <option value="Average order value by gender">Average order value by gender</option>
                        <option value="Most popular product category by zone">Most popular product category by zone</option>
                    </select>
                </div>
                <div class="toggle-wrapper">
                    <label for="themeToggle" class="small-muted">Light mode</label>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# hidden text input to capture example selection from JS
st.markdown(
    """
    <input type="hidden" id="exampleSelectValue" />
    <script>
    const exampleSelect = document.getElementById('exampleSelect');
    exampleSelect.addEventListener('change', () => {
        const value = exampleSelect.value || "";
        // set Streamlit session via custom event - Streamlit doesn't support direct JS->Py
        // We'll write value into a temporary text element that Streamlit can read via st.experimental_get_query_params is unavailable.
        // Use the clipboard as a fallback? Instead: write to a hidden element; Streamlit cannot read it directly.
        // So we will use an approach: when user picks example, prompt them to press the "Use Example" button below.
    });
    </script>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# LLM init (cached)
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
        # Do not spam error on load; return None to allow app to run
        return None

llm = initialize_llm()

# ---------------------------
# Utilities (same as before, stable)
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
                if df_processed['ORDER_DATE'].isnull().any():
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
# Main layout (no sidebar)
# ---------------------------
def main():
    # theme toggle controls inside top area - we provide a UI element below for actual toggle
    top_row = st.container()
    with top_row:
        cols = st.columns([1, 1, 1, 1])
        # example chooser (mirrors the nav dropdown; user can also use this)
        example = cols[0].selectbox("Example queries", ["", 
                                                       "Find the product with highest frequency in each region",
                                                       "Top 5 customers by total spending",
                                                       "Monthly sales trend by category",
                                                       "Average order value by gender",
                                                       "Most popular product category by zone"], index=0)
        # button to copy example into query
        use_example = cols[1].button("Use example")
        # theme toggle
        light_toggle = cols[2].checkbox("Light mode", value=st.session_state["theme_light"])
        # placeholder for spacing / future controls
        cols[3].markdown("")

    # Apply theme to body by setting a class on <body> via JS (Streamlit doesn't provide direct API)
    # We'll emulate by toggling a CSS class that affects :root variables usage above.
    if light_toggle != st.session_state["theme_light"]:
        st.session_state["theme_light"] = light_toggle

    if st.session_state["theme_light"]:
        st.markdown('<script>document.documentElement.classList.add("light");</script>', unsafe_allow_html=True)
    else:
        st.markdown('<script>document.documentElement.classList.remove("light");</script>', unsafe_allow_html=True)

    # File uploader and controls
    uploaded_file = st.file_uploader("Upload sales data (CSV)", type=["csv"])

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
                return

            df = preprocess_dataframe(df_raw)
            if not isinstance(df, pd.DataFrame) or df.empty:
                st.error("Uploaded file produced no valid rows after preprocessing.")
                return

            # Main content layout
            top, bottom = st.columns([2, 1])

            with top:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Data Preview</div>', unsafe_allow_html=True)
                st.dataframe(df.head(12))
                st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("Data Summary"):
                    info = get_data_summary(df)
                    st.write("Rows:", df.shape[0], "Columns:", df.shape[1])
                    st.write("Columns:", list(df.columns))
                    missing_counts = pd.Series(info["missing_values"])
                    missing_df = missing_counts[missing_counts > 0].rename_axis("Column").reset_index(name="Missing Count")
                    if not missing_df.empty:
                        st.dataframe(missing_df)
                    else:
                        st.write("No missing values detected.")

            with bottom:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Ask a Question</div>', unsafe_allow_html=True)

                # Query input area
                if "query_text" not in st.session_state:
                    st.session_state["query_text"] = ""

                if use_example and example:
                    st.session_state["query_text"] = example

                query = st.text_input("Enter your analytical question", value=st.session_state["query_text"], key="query_text_input")
                mode = st.selectbox("Response type", ["Smart Analysis", "Chart", "Simple Query"])
                use_ai = st.checkbox("Use AI Analysis", value=llm is not None)
                show_code = st.checkbox("Show Generated Code", value=False)

                st.markdown('</div>', unsafe_allow_html=True)

            # When user asks query
            if query and use_ai and llm is not None:
                st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

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
                                    result = agent.invoke(query)
                                    # result may be dict-like
                                    out = result.get('output', str(result)) if isinstance(result, dict) else str(result)
                                    st.markdown(out)
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
                                out = result.get('output', str(result)) if isinstance(result, dict) else str(result)
                                st.markdown(out)
                        else:
                            st.warning("Cannot process simple query: DataFrame is empty.")
                    except Exception as e:
                        st.error(f"Error in simple query: {e}")

                st.markdown('</div>', unsafe_allow_html=True)

            elif query and not use_ai:
                st.warning("Please enable 'Use AI Analysis' to get responses.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.info("Please ensure your CSV file is correctly formatted and try again.")

if __name__ == "__main__":
    main()
