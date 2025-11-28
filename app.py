import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
import io

# =========================================================================
# 1. API KEY CONFIGURATION
# =========================================================================
# The API key is embedded directly into this file for simplification, as requested.
API_KEY = "AIzaSyAD5Yi_m8Ru7U9V0GKTegwIMngMPhm4nTA" 

# --- Configuration ---
LEADS_PER_PAGE = 30
SYSTEM_PROMPT = """
You are an expert lead generation data extractor. Use the Google Search tool to find relevant businesses. 
Your output MUST be a clean Markdown table containing ONLY the columns: 
| Business Name | Full Address | Phone Number | Operating Hours |. 
Do not include any introductory or concluding text, only the formatted table.
"""
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

# --- Streamlit Setup ---
st.set_page_config(
    page_title="Final AI Prospecting Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Final Local Business Prospector")
st.caption("Key is embedded directly in the code (for personal use only).")

# --- Session State Management ---
if 'leads' not in st.session_state:
    st.session_state.leads = []
if 'page_number' not in st.session_state:
    st.session_state.page_number = 0
if 'search_criteria' not in st.session_state:
    st.session_state.search_criteria = {}

# --- API Initialization ---
@st.cache_resource
def get_gemini_client():
    """Initializes and returns the Gemini client using the embedded key."""
    try:
        # Client initialized directly with the hardcoded key
        return genai.Client(api_key=API_KEY)
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        st.stop()

client = get_gemini_client()


# --- Core Logic: API Call ---
def run_prospecting_query(user_query):
    """Calls the Gemini API with search grounding."""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_query,
            config={
                "system_instruction": SYSTEM_PROMPT,
                # Enable Google Search grounding
                "tools": [{"google_search": {}}]
            }
        )
        return response.text
    except APIError as e:
        st.error(f"Gemini API Error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Data Processing (FIXED TO HANDLE INTRO TEXT) ---
def parse_markdown_table(markdown_text):
    """Converts the raw markdown table output into a pandas DataFrame, 
    made robust to handle pre-table text."""
    if not markdown_text:
        return pd.DataFrame()
        
    try:
        # 1. Clean up the text by removing introductory lines
        lines = markdown_text.strip().split('\n')
        
        # Find the line that marks the start of the table header (usually starts with '|')
        table_start_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('|'):
                # Check if the next line is the separator line (e.g., |---|---|)
                if i + 1 < len(lines) and all(c in '-| ' for c in lines[i+1].strip()):
                    table_start_index = i
                    break
        
        if table_start_index == -1:
            st.warning("Could not find a valid Markdown table structure in the response.")
            return pd.DataFrame()

        # Join the relevant lines (from header onwards) back into a string
        table_content = "\n".join(lines[table_start_index:])
        data_io = io.StringIO(table_content)
        
        # 2. Use the standard pandas parsing method
        # Skip the separator line (index 1 relative to table_start_index)
        df = pd.read_csv(data_io, sep='|', skiprows=[1], skipinitialspace=True)
        
        # 3. Final cleaning (similar to before)
        df = df.dropna(axis=1, how='all')
        df.columns = [col.strip() for col in df.columns]
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        
        # Remove empty columns at the start/end if they exist
        if df.columns[0] == df.columns[-1] and df.columns[0] == '':
             df = df.iloc[:, 1:-1]
        
        # Standardize column names
        df.columns = [col.replace(' ', '_').replace('.', '').replace('-', '') for col in df.columns]

        return df
    
    except Exception as e:
        st.warning(f"Failed during robust parsing. Received text started with: \n{markdown_text[:500]}...\nError: {e}")
        return pd.DataFrame()


def generate_query(keyword, location, radius, current_count):
    """Constructs the query string for the LLM."""
    if current_count == 0:
        return f"""
        Find the first {LEADS_PER_PAGE} businesses matching the keyword "{keyword}" within a {radius} km radius of "{location}". 
        Provide the response as a single, clean markdown table.
        """
    else:
        # Query for the next batch, explicitly avoiding duplicates
        return f"""
        Find the next batch of up to {LEADS_PER_PAGE} unique businesses matching the keyword "{keyword}" within a {radius} km radius of "{location}". 
        These results MUST NOT be duplicates of the first {current_count} leads already returned. 
        Provide the response as a single, clean markdown table.
        """

# --- Actions ---
def search_leads(is_initial=True):
    """Handles the main search and 'Load More' actions."""
    keyword = st.session_state.keyword_input
    location = st.session_state.location_input
    radius = st.session_state.radius_select

    if not keyword or not location:
        st.error("Please enter both a Keyword and a Target Location.")
        return

    if is_initial:
        st.session_state.leads = []
        st.session_state.page_number = 0
        st.session_state.search_criteria = {'keyword': keyword, 'location': location, 'radius': radius}
    
    current_count = len(pd.concat(st.session_state.leads)) if st.session_state.leads else 0
    
    query = generate_query(keyword, location, radius, current_count)
    
    # Show loading spinner
    status_message = "Searching leads..." if is_initial else f"Loading next page (starting from result #{current_count + 1})...."
    with st.spinner(status_message):
        markdown_text = run_prospecting_query(query)

    if markdown_text:
        new_df = parse_markdown_table(markdown_text)
        
        if not new_df.empty:
            st.session_state.leads.append(new_df)
            st.session_state.page_number += 1
            st.success(f"Successfully added {len(new_df)} leads. Total leads found: {len(pd.concat(st.session_state.leads))}.")
        else:
            if is_initial:
                 st.warning("No leads found for this query or the model output could not be parsed. Try a different query.")
            else:
                 st.info("No more unique leads were found for this query.")
    

# --- UI Layout ---

# Sidebar for controls
with st.sidebar:
    st.header("Search Parameters")
    
    st.text_input(
        "Keywords (e.g., Cabinet maker, restaurant)", 
        key="keyword_input",
        value="Coffee Shops" # Default for easy testing
    )
    
    st.text_input(
        "Target Location (e.g., New York City, NY)", 
        key="location_input",
        value="Vancouver, BC" # Default for easy testing
    )
    
    st.selectbox(
        "Search Radius (km)", 
        options=[5, 10, 20, 50, 100], 
        index=1, # 10 km default
        key="radius_select"
    )

    st.button("Find Leads (Start New Search)", on_click=lambda: search_leads(is_initial=True), type="primary")

    # Load More Button
    total_leads = len(pd.concat(st.session_state.leads)) if st.session_state.leads else 0
    
    st.markdown("---")
    st.metric("Total Leads Found", total_leads)

    can_load_more = (total_leads > 0) and (len(st.session_state.leads[-1]) == LEADS_PER_PAGE) if st.session_state.leads else False

    st.button(
        f"Load Next {LEADS_PER_PAGE} Leads", 
        on_click=lambda: search_leads(is_initial=False), 
        disabled=not can_load_more
    )


# Main Content Area
if not st.session_state.leads:
    st.info("Enter your search parameters in the sidebar and click 'Find Leads' to begin prospecting.")
else:
    # Concatenate all results into a single DataFrame for display/export
    full_df = pd.concat(st.session_state.leads, ignore_index=True)
    
    st.header(f"Prospecting Results ({len(full_df)} Total)")

    # Export Button
    csv_export = full_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Export All {len(full_df)} Leads to CSV",
        data=csv_export,
        file_name=f"leads_{st.session_state.search_criteria['keyword'].replace(' ', '_')}.csv",
        mime='text/csv',
    )
    
    # Display the table
    st.dataframe(full_df, use_container_width=True)
