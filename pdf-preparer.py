# Author: Winston Yan
# Date: Summer 2026
# Desc:


# Base Imports
import json
from pathlib import Path

# Third-party Imports
from PIL import Image

import easyocr
import numpy as np
import ollama
import pymupdf
import warnings # suppress torch warnings


#############
#  Globals  #
#############

# Initialize reader once (downloads models on first run)
warnings.filterwarnings("ignore", category=UserWarning)  # suppress torch warnings
reader = easyocr.Reader(['en'], gpu=False, )  # set gpu=False if no CUDA

#############
#  Helpers  #
#############
def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def ocr_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    full_text = ""
    
    for page in doc:
        # Render page to image
        pix = page.get_pixmap(dpi=360)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # EasyOCR expects numpy array
        img_array = np.array(img)
        
        # Run OCR
        results = reader.readtext(img_array)
        
        # Extract text (each result is [bbox, text, confidence])
        page_text = "\n".join([detection[1] for detection in results])
        full_text += page_text + "\n\n"
    
    return full_text

def parse_exam_with_llm(raw_text):
    """Send text to local LLM for structured parsing."""
    
    prompt = f"""You are an exam parser. Given the following exam solution text, 
extract each question into a structured JSON format.

For each question, provide:
- "question_number": the question number/identifier
- "question_body": the full question text
- "answer_format": either {{"type": "multiple_choice", "options": ["A) ...", "B) ...", ...]}} 
  or {{"type": "free_response"}}
- "correct_answer": for multiple choice, the correct option letter and text. 
  For free response, the rubric/marking scheme/model answer.

Return a JSON array of question objects. Only return valid JSON, nothing else.

--- EXAM TEXT ---
{raw_text}
--- END ---"""

    response = ollama.chat(
        model='llama3.1:8b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.05, 'num_ctx': 8192}
    )
    
    response_text = response['message']['content']
    
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]
    
    return json.loads(response_text)

def smart_chunk(text, max_chars=6400):
    """Split on question boundaries instead of arbitrary positions."""
    import re
    
    # Common question boundary patterns
    # Adjust regex to match your exam format
    pattern = r'\n(?=\s*(?:Question\s+\d|Q\d|\d+[\.\)]\s|Part\s+[A-Z]))'
    
    questions = re.split(pattern, text)
    
    chunks = []
    current_chunk = ""
    
    for q in questions:
        if len(current_chunk) + len(q) < max_chars:
            current_chunk += q
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = q
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


########################
#  Primary Functions   #
########################
def process_exam(file_path, output_path="exam_output.json"):
    """Complete pipeline: file → OCR → LLM → JSON"""
    
    ext = Path(file_path).suffix.lower()
    
    # Step 1: Extract text
    if ext == '.pdf':
        # Try selectable text first
        text = extract_text_from_pdf(file_path)
        if len(text.strip()) < 50:  # Probably scanned
            print("Selectable text too short, falling back to OCR...")
            text = ocr_pdf(file_path)
    elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        img = Image.open(file_path)
        img_array = np.array(img)
        results = reader.readtext(img_array)
        text = "\n".join([detection[1] for detection in results])
    elif ext in ['.txt', '.md']:
        with open(file_path, 'r') as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    print(f"Extracted {len(text)} characters")
    
    # Step 2: Chunk and parse with LLM
    chunks = smart_chunk(text, 4800)
    
    all_questions = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        try:
            questions = parse_exam_with_llm(chunk)
            if isinstance(questions, list):
                all_questions.extend(questions)
            else:
                all_questions.append(questions)
        except json.JSONDecodeError as e:
            print(f"Failed to parse chunk {i+1}: {e}")
            continue
    
    # Step 3: Validate and save
    validated = validate_questions(all_questions)
    
    return validated

def validate_questions(questions):
    """Ensure all required fields are present."""
    required_fields = ["question_number", "question_body", "answer_format", "correct_answer"]
    validated = []
    
    for q in questions:
        if all(field in q for field in required_fields):
            validated.append(q)
        else:
            # Try to salvage partial data
            fixed = {field: q.get(field, "MISSING") for field in required_fields}
            validated.append(fixed)
    
    return validated

###############
#  Execution  #
###############
def prepare_multiple(fns):
    for fn in fns:
        print("Parsing exam file #" + str(fns.index(fn)))
        if Path(fn).exists():
            exam = process_exam(fn)
        else:
            print(f"File {fn} does not exist.")

        with open("./cleaned/" + fn.split("/")[1].split(".")[0] + "-cleaned.json", 'w') as f:
            json.dump(exam, f, indent=2)
            print("Written to ./cleaned/" + fn.split("/")[1].split(".")[0] + "-cleaned.json")

if __name__ == "__main__":
    print("\nEnter path to filename of exam to process (PDF, image, or text): ", end="")
    fn = input().strip()
    if Path(fn).exists():
        exam = process_exam(fn)
    else:
        print(f"File {fn} does not exist.")

    with open("./cleaned/" + fn.split("/")[1].split(".")[0] + "-cleaned.json", 'w') as f:
        json.dump(exam, f, indent=2)