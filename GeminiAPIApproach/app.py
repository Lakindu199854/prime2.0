import streamlit as st
import os
import json
import tempfile
# Assuming extractor.py has been updated to accept `api_key` as a second argument
from extractor import extract_pdf_to_json 

# ----------------------------------------------------
# 1. Configuration and Title
# ----------------------------------------------------

st.set_page_config(
    page_title="Structured PDF Metadata Extractor",
    layout="wide"
)

st.title("📄 Structured PDF Metadata Extractor (JSON Output)")
st.caption("Uses Gemini 2.5 Flash with Pydantic Schema for reliable data extraction.")

# ----------------------------------------------------
# 2. API Key Management (Centralized in app.py)
# ----------------------------------------------------

# 1. Attempt to load API Key from environment variable (Best practice)
api_key = os.getenv("GEMINI_API_KEY")

# 2. Fallback: If not found, ask the user for it
if not api_key:
    # Use a sidebar for key input to keep the main area clean
    with st.sidebar:
        st.warning("GEMINI_API_KEY not found in environment variables.")
        api_key_input = st.text_input(
            "Enter your Gemini API Key:",
            type="password",
            help="The key is used only for this session and is not stored."
        )
        if api_key_input:
            api_key = api_key_input

# Check if key is available after all attempts
if not api_key:
    st.error("Please provide a valid Gemini API Key in the sidebar or environment variables to proceed.")
# ----------------------------------------------------
# 3. File Uploader and Processing
# ----------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF Article File",
    type="pdf",
    accept_multiple_files=False,
    disabled=(not api_key), # Disable uploader if no key is present
    help="The article will be analyzed for structured metadata (titles, authors, citation details)."
)

# 4. Processing Logic
if uploaded_file and st.button("Process PDF & Generate JSON", type="primary", disabled=(not api_key)):
    
    # Use a spinner to show activity
    with st.spinner("⏳ Analyzing PDF and generating structured JSON... (This may take up to 30 seconds due to file processing and retries)"):
        temp_file_path = None
        try:
            # 4.1 Save the uploaded file to a temporary location
            # The extractor function requires a physical path to upload the file to Gemini
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_file_path = tmp.name
            
            # 4.2 Call the core extraction logic (PASSING THE API KEY)
            # The retry logic is built into this function
            json_output_string = extract_pdf_to_json(temp_file_path, api_key)

            if json_output_string:
                st.success("✅ Extraction complete! Valid structured JSON generated.")
                
                # Try to parse and pretty-print the JSON
                try:
                    parsed_json = json.loads(json_output_string)
                    
                    st.subheader("Extracted JSON Data")
                    # Display the structured data using Streamlit's JSON widget
                    st.json(parsed_json)
                    
                    # 5. Download Button
                    st.download_button(
                        label="⬇️ Download JSON File",
                        data=json_output_string,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_metadata.json",
                        mime="application/json",
                    )
                except json.JSONDecodeError:
                    st.error("❌ The API returned data that was not valid JSON.")
                    st.code(json_output_string) # Show raw output for debugging
                
            else:
                st.error("❌ Extraction failed. Check the logs/console for details.")

        except Exception as e:
            st.error(f"An unexpected error occurred during processing: {e}")
            # st.exception(e) # Optionally display the full stack trace for debugging
            
        finally:
            # 4.3 Cleanup the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)