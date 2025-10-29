'''os → Work with files and folders.

pdfplumber → Extract text from PDFs.

fitz (PyMuPDF) → Extract images and metadata from PDFs.

json → Save and load structured data (like dictionaries/lists).

numpy (np) → For mathematical operations and handling embeddings.

re → Regular expressions (for cleaning text).

PIL (Image, ImageStat) → Work with images (brightness, convert, etc.).

faiss → Facebook’s library for fast vector search (used for similarity search).

SentenceTransformer → Turns text into embeddings (numerical vectors).

genai → Google Gemini API (AI model for summarizing images/text).
'''
import os
import pdfplumber
import fitz  # PyMuPDF
import json
import numpy as np
import re
from PIL import Image, ImageStat
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Loads a sentence embedding model (MiniLM).
# Converts text into 384-dimensional vectors that capture meaning.
# Initialize SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# Stores your Google Gemini API key.
# configures the Gemini library to use it.
# Set Gemini API key here
GEMINI_API_KEY = 'AIzaSyDaoMJXZsSmHzgmXAfTctQU5u_UzmABG3g'  
genai.configure(api_key=GEMINI_API_KEY)

# Load the Gemini model
try:
    gemini_model = genai.GenerativeModel('models/gemini-1.5-flash')
except Exception as e:
    print(f"[Gemini ERROR] Failed to load Gemini model: {e}")


# Converts text to lowercase.
# Removes special characters (.,?!@#$ etc.), keeps only letters, numbers, spaces.
# Strips extra spaces.
# 👉 Purpose: clean text before embedding.
# ✅ Helper: clean + normalize query or text
def preprocess_text(text):
    """Removes extra spaces, converts to lowercase, and removes special characters."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Keep only alphanumeric and spaces
    return text.strip()



#  Loops through all files in folder_path.
#  If file is PDF, extract text page by page.
#  Stores extracted text in a dictionary:
# { "filename.pdf": "all extracted text" }.
# ✅ Extract text from PDFs (same as before)
def extract_text_from_pdfs(folder_path):
    extracted_texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            text = ""
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + '\n'
                extracted_texts[filename] = text
                print(f"Extracted text from {filename}")
            except Exception as e:
                print(f"Error extracting text from {filename}: {e}")
    return extracted_texts



# Opens image, converts it to grayscale.
# Calculates average brightness (0=black, 255=white).
# If brightness ≤ 5 → treat as black image.
# ✅ Check if image is too dark
def is_black_image(image_path, brightness_threshold=5):
    try:
        with Image.open(image_path) as img:
            img = img.convert('L')  # Convert to grayscale
            stat = ImageStat.Stat(img)
            brightness = stat.mean[0]
            print(f"Brightness for {image_path}: {brightness}")
            
            # Check if brightness is below the threshold (likely a black image)
            return brightness <= brightness_threshold
    except Exception as e:
        print(f"Error checking image brightness: {e}")
        return False



# Converts image to RGB format and saves it back.
# Fixes weird orientation issues from PDFs.
# ✅ Fix orientation issues
def correct_image_orientation(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img.save(image_path)
    except Exception as e:
        print(f"Error correcting image orientation: {e}")
        

# ✅ Extract images and surrounding text with fallback for full-page image
def extract_images_and_surrounded_text_from_pdfs(folder_path, black_image_folder, real_image_folder):
    extracted_data = []
    os.makedirs(black_image_folder, exist_ok=True)
    os.makedirs(real_image_folder, exist_ok=True)

    for pdf_file in os.listdir(folder_path):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, pdf_file)

            try:
                pdf_document = fitz.open(pdf_path)

                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    image_list = page.get_images(full=True)
                    surrounding_text = page.get_text()

                    # ✅ Full-page fallback using pixmap
                    if not image_list:
                        print(f"No images found on page {page_num+1}, extracting full-page image.")
                        try:
                            pix = page.get_pixmap(dpi=300, alpha=True)
                            img_path = os.path.join(real_image_folder, f"{pdf_file}_page_{page_num+1}_full.png")
                            pix.save(img_path)

                            if is_black_image(img_path):
                                new_black_path = os.path.join(black_image_folder, os.path.basename(img_path))
                                os.rename(img_path, new_black_path)
                                print(f"Black Full-Page Image Detected and Moved: {new_black_path}")
                                extracted_data.append({
                                    "pdf_file": pdf_file,
                                    "image_path": None,
                                    "surrounding_text": surrounding_text,
                                    "image_summary": None
                                })
                            else:
                                print(f"Full-Page Image Saved: {img_path}")
                                summary = generate_image_summary_with_gemini(img_path, surrounding_text)
                                extracted_data.append({
                                    "pdf_file": pdf_file,
                                    "image_path": img_path,
                                    "surrounding_text": surrounding_text,
                                    "image_summary": summary
                                })
                        except Exception as e:
                            print(f"Error extracting full-page image on page {page_num+1} of {pdf_file}: {e}")
                        continue

                    # ✅ Embedded image extraction
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = pdf_document.extract_image(xref)
                            image_bytes = base_image["image"]
                            img_extension = base_image["ext"]
                            img_path = os.path.join(real_image_folder, f"{pdf_file}_page_{page_num+1}_img_{img_index+1}.{img_extension}")

                            with open(img_path, "wb") as img_file:
                                img_file.write(image_bytes)

                            correct_image_orientation(img_path)

                            if is_black_image(img_path):
                                new_black_path = os.path.join(black_image_folder, os.path.basename(img_path))
                                os.rename(img_path, new_black_path)
                                print(f"Black Image Detected and Moved: {new_black_path}")
                                extracted_data.append({
                                    "pdf_file": pdf_file,
                                    "image_path": None,
                                    "surrounding_text": surrounding_text,
                                    "image_summary": None
                                })
                            else:
                                print(f"Real Image Saved: {img_path}")
                                summary = generate_image_summary_with_gemini(img_path, surrounding_text)
                                extracted_data.append({
                                    "pdf_file": pdf_file,
                                    "image_path": img_path,
                                    "surrounding_text": surrounding_text,
                                    "image_summary": summary
                                })
                        except Exception as e:
                            print(f"Error extracting image on page {page_num+1} of {pdf_file}: {e}")

            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")

    return extracted_data


# ✅ Store vectors in FAISS
def store_vectors_in_faiss(image_data, pdf_text_data):
    vectors, metadata = [], []

    for item in image_data:
        if item['surrounding_text']:
            summary = item.get('image_summary') or ''
            combined_text = summary + " " + item.get('surrounding_text', '')
            embedding = model.encode(preprocess_text(combined_text))
            vectors.append(embedding)
            metadata.append(item)

    for pdf_file, pdf_text in pdf_text_data.items():
        if pdf_text:
            embedding = model.encode(preprocess_text(pdf_text))
            vectors.append(embedding)
            metadata.append({
                "pdf_file": pdf_file,
                "image_path": None,
                "surrounding_text": pdf_text
            })

    vector_dim = len(vectors[0])
    index = faiss.IndexFlatIP(vector_dim)  # Using Cosine Similarity
    vectors = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)  # Normalize vectors for cosine similarity
    index.add(vectors)
    
    faiss.write_index(index, 'vector_database.index')
    with open('metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    print("Vectors and metadata stored in FAISS.")

# ✅ FAISS Search
def search_faiss(query, top_k=5):
    index = faiss.read_index('vector_database.index')
    with open('metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    query_embedding = model.encode([preprocess_text(query)])
    faiss.normalize_L2(query_embedding)  # Normalize query for cosine similarity
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    
    print("\n🔍 Search Results:")
    found_results = []
    
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            print(f"{i+1}. No result found")
        else:
            result = metadata[idx]
            # # Filtering for Chapter 1 relevance if mentioned
            # if "chapter 1" in preprocess_text(result['surrounding_text']):
            #     found_results.append(result)
            
    if not found_results:
        # print("No relevant Chapter 1 results found. Showing general results:")
        found_results = [metadata[idx] for idx in indices[0] if idx != -1]
    
    for i, result in enumerate(found_results[:top_k]):
        print(f"{i+1}. PDF: {result['pdf_file']}")
        print(f"   Image Path: {result.get('image_path', 'No Image')}")
        print(f"   Surrounding Text: {result['surrounding_text'][:500]}...\n")


# ✅ Gemini Image Summary
def generate_image_summary_with_gemini(image_path, surrounding_text=""):
    try:
        print(f"[Gemini DEBUG] Starting summary for image: {image_path}")
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()

        response = gemini_model.generate_content(
            [
                {"text": "You are a helpful educational assistant. Based on this image and its surrounding text, summarize what the image is about."},
                {"mime_type": "image/png", "data": img_bytes},
                {"text": surrounding_text[:500]}
            ]
        )

        print(f"[Gemini DEBUG] Response: {response.text}")
        return response.text
    except Exception as e:
        print(f"[Gemini ERROR] Failed to summarize image {image_path}: {e}")
        return None
  

# Execution Steps
folder_path = r"D:\360DigitMG\Project - 2\ChatBot Code\NCRT DATA"
black_image_folder = r"D:\360DigitMG\Project - 2\ChatBot Code\Extracted Images\Black Images"
real_image_folder = r"D:\360DigitMG\Project - 2\ChatBot Code\Extracted Images\Real Images"

image_data = extract_images_and_surrounded_text_from_pdfs(folder_path, black_image_folder, real_image_folder)
pdf_text_data = extract_text_from_pdfs(folder_path)

store_vectors_in_faiss(image_data, pdf_text_data)
print("Process Completed!")

# Example Search
search_query = "What is Friction?"     
search_faiss(search_query)



# # Debugging Step: Check if Chapter 3 is indexed
# for i, text in enumerate(pdf_text_data):
#     if "Chapter 3" in text or "Metals and Non-metals" in text:
#         print(f"Found Chapter 3 at index {i}: {text[:200]}")









