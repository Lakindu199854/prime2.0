from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import os

def extract_text_from_pdf(pdf_path, output_dir="extracted_images"):
    poppler_path = r'C:\Users\beecom\Downloads\Compressed\Release-25.07.0-0\poppler-25.07.0\Library\bin'

    """
    Extracts text from an image-based PDF using PaddleOCR.
    """
    # 1. Convert PDF pages to images
    print("Converting PDF pages to images...")
    images = convert_from_path(pdf_path, poppler_path=poppler_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save images to a temporary directory
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i+1}.jpg")
        image.save(image_path, "JPEG")
        image_paths.append(image_path)
    
    # 2. Initialize PaddleOCR
    # The 'en' and 'latin' flags are for English text recognition. 
    # 'latin' is used for a broader range of Latin-based scripts.
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang=["en", "latin"])

    # 3. Process each image and extract text
    full_text = []
    for i, image_path in enumerate(image_paths):
        print(f"Processing page {i+1}...")
        result = ocr.ocr(image_path, cls=True)

        # Parse and append the text
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                full_text.append(text)
    
    # 4. Clean up temporary image files
    for image_path in image_paths:
        os.remove(image_path)
    os.rmdir(output_dir)
    
    return "\n".join(full_text)

# --- Main part of the script ---
if __name__ == "__main__":
    pdf_file = "A6.pdf"  # 👈 Change this to your PDF file name!

    if os.path.exists(pdf_file):
        extracted_content = extract_text_from_pdf(pdf_file)
        print("\n--- Extracted Text ---")
        print(extracted_content)
    else:
        print(f"Error: The file '{pdf_file}' was not found.")
        print("Please make sure the PDF file is in the same directory as the script.")