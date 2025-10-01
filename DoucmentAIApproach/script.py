import os
from dotenv import load_dotenv
from google.cloud import documentai_v1 as documentai

# Load .env file
load_dotenv()

project_id = os.getenv("PROJECT_ID")
processor_id = os.getenv("PROCESSOR_ID")
location = "us"  

client = documentai.DocumentProcessorServiceClient()

name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

file_path = "A6.pdf"
with open(file_path, "rb") as f:
    pdf_content = f.read()

raw_document = documentai.RawDocument(content=pdf_content, mime_type="application/pdf")
request = documentai.ProcessRequest(name=name, raw_document=raw_document)

result = client.process_document(request=request)

print(result.document.text)
