import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.schema import SystemMessage, HumanMessage
import re
import io
import base64
from datetime import datetime
import numpy as np
import json
import ast

st.set_page_config(page_title="📊 E-Commerce Assistant", layout="wide")
st.title("📊 AskData-Data answers made easy")

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
        
        # 1. Standardize all current column names to uppercase initially
        df_processed.columns = df_processed.columns.str.strip().str.upper()
        
        # 2. Define a flexible rename map for common variations to canonical names
        # This will be applied carefully to avoid creating duplicate column names.
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
            'ORDERID': 'ORDER_ID' # Added for completeness based on your previous code
        }

        # Apply renames carefully: only if the original column exists AND
        # the canonical column name doesn't already exist or isn't the same as original
        cols_to_rename = {}
        for original_col, canonical_col in flexible_rename_map.items():
            if original_col in df_processed.columns and original_col != canonical_col:
                if canonical_col not in df_processed.columns: # Prevent overwriting existing canonical name
                    cols_to_rename[original_col] = canonical_col
        
        if cols_to_rename:
            df_processed.rename(columns=cols_to_rename, inplace=True)

        # 3. Handle specific column collision scenarios (e.g., 'PRODUCT' and 'PRODUCT_CATEGORY')
        # Based on your CSV, you have 'Product_Category' and 'Product'.
        # After uppercase, these are 'PRODUCT_CATEGORY' and 'PRODUCT'.
        if 'PRODUCT_CATEGORY' in df_processed.columns and 'PRODUCT' in df_processed.columns:
            st.warning("Both 'PRODUCT_CATEGORY' and 'PRODUCT' columns are present after standardization. Prioritizing 'PRODUCT_CATEGORY' and dropping 'PRODUCT' to ensure unique category definition.")
            # Drop 'PRODUCT' if 'PRODUCT_CATEGORY' is the preferred one and both exist.
            df_processed.drop(columns=['PRODUCT'], inplace=True, errors='ignore')
        elif 'PRODUCT' in df_processed.columns and 'PRODUCT_CATEGORY' not in df_processed.columns:
            # If 'PRODUCT_CATEGORY' is missing but 'PRODUCT' exists, rename 'PRODUCT' to 'PRODUCT_CATEGORY'
            df_processed.rename(columns={'PRODUCT': 'PRODUCT_CATEGORY'}, inplace=True)
            st.info("Renamed 'PRODUCT' column to 'PRODUCT_CATEGORY'.")


        # 4. Date processing for 'ORDER_DATE' (which maps from 'DATE')
        if 'ORDER_DATE' in df_processed.columns:
            try:
                original_rows = len(df_processed)
                # Attempt to convert to datetime, coercing errors to NaT (Not a Time)
                df_processed['ORDER_DATE'] = pd.to_datetime(df_processed['ORDER_DATE'], errors='coerce', dayfirst=True)
                # Drop rows where date conversion resulted in NaT
                df_processed.dropna(subset=['ORDER_DATE'], inplace=True)
                dropped = original_rows - len(df_processed)
                if dropped > 0:
                    st.warning(f"Dropped {dropped} rows with invalid 'ORDER_DATE'.")
            except Exception as e:
                st.warning(f"Could not convert 'ORDER_DATE' to datetime. Column might contain invalid dates or NaTs. Error: {e}")
                # Fallback: if critical error during conversion (unlikely after coerce), ensure NaTs are removed.
                if 'ORDER_DATE' in df_processed.columns:
                     if df_processed['ORDER_DATE'].isnull().any():
                         df_processed.dropna(subset=['ORDER_DATE'], inplace=True)

        # 5. Numeric processing for relevant columns
        numeric_columns = ['AMOUNT', 'ORDERS', 'QUANTITY', 'UNIT_COST', 'UNIT_PRICE', 'PROFIT', 'COST', 'REVENUE', 'CUSTOMER_AGE']
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                # Fill NaNs only if they exist after conversion
                if df_processed[col].isnull().any():
                    df_processed[col] = df_processed[col].fillna(0)
        
        # 6. Gender standardization
        if 'GENDER' in df_processed.columns:
            # Ensure the column has non-null values before applying .str accessor
            if not df_processed['GENDER'].dropna().empty:
                 df_processed['GENDER'] = df_processed['GENDER'].astype(str).str.upper().replace({'MALE': 'M', 'FEMALE': 'F', 'UNKNOWN': 'U', 'OTHER': 'O'})
            else: # If column is all NaN or empty after dropna, fill with 'Unknown'
                df_processed['GENDER'] = df_processed['GENDER'].fillna('Unknown')
        
        # 7. Fill missing values for any remaining object columns
        for col in df_processed.columns:
            # Check if the column is of 'object' dtype and if it contains any null values
            if df_processed[col].dtype == 'object' and df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna('Unknown')
        
        return df_processed
    except Exception as e:
        # Catch any unexpected errors during preprocessing and provide a message
        st.error(f"Error during dataframe preprocessing: {e}")
        return pd.DataFrame() # Always return an empty DataFrame on critical error

def execute_complex_query(df, query, llm):
    """Execute complex analytical queries using LLM-generated code"""
    
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
    7. Generate ONLY the Python code without any explanations, markdown formatting (no ```python), or leading/trailing comments.
    """
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate Python code for: {query}")
        ]
        
        response = llm(messages)
        code = response.content.strip()
        
        # Clean the code (robustly remove markdown fences if the LLM adds them)
        code = re.sub(r'^\s*```(python)?\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n\s*```\s*$', '', code, flags=re.MULTILINE)
        code = code.strip()
        
        # Execute the code
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
    
    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary_info["numeric_summary"][col] = {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max()
        }
    
    # Categorical summary
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
    """Format the result for better display"""
    if isinstance(result, pd.DataFrame):
        return result
    elif isinstance(result, pd.Series):
        return result.to_frame(name="Value") 
    elif isinstance(result, dict):
        return pd.DataFrame([result])
    else:
        return str(result)

def main():
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
            # Load data with multiple encoding attempts
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
                st.error("❌ Failed to load file. Please ensure it's a valid CSV with data.")
                return

            # Preprocess
            df = preprocess_dataframe(df_raw)

            # SAFETY CHECK: Ensure df is a DataFrame and not empty after preprocessing
            if not isinstance(df, pd.DataFrame) or df.empty:
                st.error("❌ The uploaded file resulted in no valid data after preprocessing. This might be due to critical errors during column conversions or date parsing. Please check your CSV data for integrity.")
                return

            # Show preview
            st.subheader(" Data Preview")
            st.dataframe(df.head(10))

            # Show summary
            with st.expander(" Data Summary"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Dataset Info:**")
                    st.write(f"Rows: {df.shape[0]}")
                    st.write(f"Columns: {df.shape[1]}")
                    st.write("**Columns:**", list(df.columns))
                
                with col2:
                    st.write("**Missing Values (after processing):**")
                    missing_counts = df.isnull().sum()
                    missing_df = pd.DataFrame({
                        'Column': missing_counts.index,
                        'Missing Count': missing_counts.values
                    })
                    missing_with_values = missing_df[missing_df['Missing Count'] > 0]
                    if not missing_with_values.empty:
                        st.dataframe(missing_with_values)
                    else:
                        st.write(" No missing values detected.")

            # User query
            st.subheader("💬 Ask Your Question")
            query = st.text_input(
                "Enter your analytical question", 
                placeholder="e.g., 'Find the product with highest frequency in each region' or 'Top 5 customers by total spending'"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                mode = st.selectbox("Response type", ["Smart Analysis", "Chart", "Simple Query"])
            with col2:
                use_ai = st.checkbox("Use AI Analysis", value=llm is not None)
            with col3:
                show_code = st.checkbox("Show Generated Code", value=False)

            if query and use_ai and llm is not None:
                st.subheader("🔍 Results")
                
                if mode == "Smart Analysis":
                    with st.spinner("Analyzing your query..."):
                        try:
                            result, code = execute_complex_query(df, query, llm)
                            
                            if show_code and code:
                                st.subheader(" Generated Code")
                                st.code(code, language="python")
                            
                            st.subheader(" Result")
                            formatted_result = format_result_output(result)
                            
                            if isinstance(formatted_result, pd.DataFrame) and not formatted_result.empty:
                                st.dataframe(formatted_result)
                                csv = formatted_result.to_csv(index=False)
                                st.download_button(
                                    label=" Download Results as CSV",
                                    data=csv,
                                    file_name=f"analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            elif isinstance(formatted_result, pd.DataFrame):
                                st.write(" Analysis returned an empty DataFrame.")
                            else: 
                                st.write(formatted_result)
                            
                            if st.button("📄 Export to PDF"):
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
                                    st.markdown(result['output'])
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
                        Generate ONLY the Python code without any explanations, markdown formatting (no ```python), or leading/trailing comments.
                        """
                        
                        user_message = f"Generate Python code for: '{query}'"
                        messages = [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=user_message)
                        ]
                        
                        try:
                            response = llm(messages)
                            chart_code = response.content
                            chart_code = re.sub(r'^\s*```(python)?\s*\n', '', chart_code, flags=re.MULTILINE)
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
                                    st.warning("⚠️ The generated code did not create a Plotly figure named 'fig'.")
                            except Exception as e:
                                st.error(f"Chart execution error: {e}")
                                
                        except Exception as e:
                            st.error(f"Failed to generate chart: {e}")
                
                elif mode == "Simple Query":
                    try:
                        if not df.empty:
                            agent = create_pandas_dataframe_agent(
                                llm=llm,
                                df=df,
                                verbose=True,
                                allow_dangerous_code=True,
                                agent_type=AgentType.OPENAI_FUNCTIONS
                            )
                            with st.spinner("Processing simple query..."):
                                result = agent.invoke(query)
                                st.markdown(result['output'])
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


