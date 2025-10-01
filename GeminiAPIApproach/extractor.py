import os
import sys
import json
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError 
from dotenv import load_dotenv 
from schema import ArticleData  # Assuming schema.py is present

# Load environment variables (for command-line testing or default key)
load_dotenv() 

INPUT_DIR = "_files" # Not strictly used in the Streamlit context
OUTPUT_DIR = "."     # Used by the main() function for command-line output
MODEL_NAME = 'gemini-2.5-flash'

MAX_RETRIES = 5
INITIAL_DELAY = 2 # seconds

# ----------------------------------------------------
# CORE EXTRACTION FUNCTION (MODIFIED TO ACCEPT API_KEY)
# ----------------------------------------------------

def extract_pdf_to_json(pdf_path: str, api_key: str) -> str | None:
    """
    Uploads a PDF, extracts full metadata using the complex structured schema,
    and returns the validated JSON data as a string.
    
    Args:
        pdf_path: The local path to the PDF file.
        api_key: The Gemini API key provided by the frontend or env variable.
        
    Returns:
        The clean, validated JSON string, or None on failure.
    """
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found at path: {pdf_path}")
        return None

    client = None
    uploaded_file = None
    
    try:
        # Use the API key passed from the calling environment (Streamlit)
        if not api_key:
             # This check is mostly defensive, as app.py already checks
             raise ValueError("The GEMINI_API_KEY is not provided.")
        client = genai.Client(api_key=api_key)
        
    except Exception as e:
        print(f"❌ Error initializing client. Error: {e}")
        return None 

    print(f"Processing file: {pdf_path}")
    
    try:
        # 1. Upload the file
        uploaded_file = client.files.upload(file=pdf_path)
        print(f"File uploaded successfully to Gemini API: {uploaded_file.name}")

        # 2. Define prompt and configuration (using the complex schema)
        prompt_text = (
            "Analyze the uploaded PDF scientific article. Extract all requested bibliographic and structural "
            "metadata, including the full English and Chinese abstracts, all contributor names and their "
            "linked affiliation IDs, the full affiliation descriptions, and all citation details (DOI, year, volume, etc.). "
            "You MUST strictly adhere to the provided JSON schema for formatting. Do not skip any fields. "
            "Return only the resulting JSON object."
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ArticleData, 
        )

        # 3. Call the Gemini API with Retry Logic (Exponential Backoff)
        data_json = "" # Initialize data_json here for use in the catch blocks
        for attempt in range(MAX_RETRIES):
            try:
                print(f"\nAttempting structured API call (Attempt {attempt + 1}/{MAX_RETRIES})...")
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[uploaded_file, prompt_text],
                    config=config,
                )
                
                # If successful, process the JSON response
                data_json = response.text.strip()
                if data_json.startswith("```json"):
                    data_json = data_json.strip("```json").strip("```").strip()

                # Validate and parse the JSON into the Pydantic model
                article_data_model = ArticleData.model_validate_json(data_json)
                
                # Use indent=4 to make the output readable and saveable
                final_json_string = article_data_model.model_dump_json(indent=4)

                return final_json_string # Success, return the clean JSON string
            
            except APIError as e:
                error_message = str(e)
                # Check for transient API errors that warrant a retry
                should_retry = ("503 UNAVAILABLE" in error_message or 
                                "overloaded" in error_message or
                                "500 INTERNAL" in error_message)
                
                if should_retry:
                    if attempt < MAX_RETRIES - 1:
                        delay = INITIAL_DELAY * (2 ** attempt)
                        print(f"⚠️ API Error ({error_message.split(' ')[0]}): Retrying in {delay:.2f} seconds (Attempt {attempt + 1}/{MAX_RETRIES})...")
                        time.sleep(delay)
                    else:
                        print(f"❌ API Error: Model remains unavailable after {MAX_RETRIES} attempts. Giving up.")
                        raise # Re-raise to be caught by the outer except block
                else:
                    raise # Re-raise non-transient API errors
            
            except Exception as e:
                # Catch JSON decoding or Pydantic validation errors
                print(f"❌ Error during JSON processing or Pydantic validation: {e}")
                print(f"Raw response text (first 500 chars): {data_json[:500]}...")
                # Exit gracefully, as this is typically not fixed by retrying the API call
                return None


    except Exception as e:
        print(f"\n❌ An error occurred during the API call or processing: {e}")
        return None
    finally:
        # Clean up the uploaded file from the server, regardless of success/failure
        if client and uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
                print(f"Cleanup: File {uploaded_file.name} deleted from server.")
            except Exception as e:
                print(f"Warning: Could not delete uploaded file {uploaded_file.name}. Error: {e}")

# ----------------------------------------------------
# COMMAND LINE ENTRY POINT (Modified for key consistency)
# ----------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python extractor.py <PDF_FILE_NAME>")
        sys.exit(1)

    pdf_file_path = sys.argv[1]
    
    # In command-line mode, pull the key directly from the environment
    cli_api_key = os.getenv("GEMINI_API_KEY")
    if not cli_api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)
        
    final_json = extract_pdf_to_json(pdf_file_path, cli_api_key)

    if final_json:
        output_filename = os.path.splitext(os.path.basename(pdf_file_path))[0] + ".json"
        
        # Save the JSON output to a file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(final_json)
            
        print("\n=======================================================")
        print("✅ Structured JSON Extraction Complete.")
        print(f"Output saved to: {output_filename}")
        print("=======================================================")
        
        # Print the final JSON for inspection
        print("\n--- GENERATED JSON (Preview) ---\n")
        print(final_json[:2000]) 
        print("\n------------------------------------\n")

if __name__ == "__main__":
    main()