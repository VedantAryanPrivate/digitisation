import subprocess
import requests
import streamlit as st
import os
import json
import cv2
import numpy as np
import fitz  # PyMuPDF, used to split PDF into pages
from PIL import Image
from pathlib import Path
from streamlit_drawable_canvas import st_canvas  # type: ignore
import glob
from natsort import natsorted
import time
import re
from openai import OpenAI
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import asyncio
from typing import Dict, List, Optional
import queue
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

# Set Streamlit to wide layout
st.set_page_config(layout="wide")
st.title("📄 PDF Image Annotation Viewer & JSON Renderer")

# OpenAI API Configuration
try:
    client = OpenAI(api_key="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")
except Exception as e:
    st.error(f"Error initializing OpenAI client: {str(e)}")
    client = None

# GPT-4 Processing Prompt
prompt = r"""
Process the following markdown text carefully according to the rules below, ensuring precise and consistent formatting:
0. Do not add any extra content, explanations, or AI-generated text beyond the specified modifications.
1. Convert Tables to Pipe-Separated Format:
   - Identify all tables in the markdown and convert them to a pipe-separated format, using the pipe symbol (`|`) to separate each column.
   - Ensure each row and column is aligned correctly with the pipe symbol for readability.
2. Format Inline Equations:
   - Locate all inline equations marked by `\(` and `\)`.
   - Replace `\(` and `\)` with `$` symbols at the beginning and end of each equation, using LaTeX syntax for inline math.
3. Format Display Equations:
   - Identify all display equations marked by `\[` and `\]`.
   - Replace `\[` and `\]` with `$$` symbols at the beginning and end of each equation, using LaTeX syntax for display math.
   - Place each display equation on a new line, starting and ending with `$$`.
4. Separate Multiple Equations on a Single Line:
   - If two or more equations are on the same line, format each one individually according to whether they are inline or display equations.
   - Place each equation on a separate line if there are multiple equations, maintaining clarity and uniformity.
5. Process Every Line Thoroughly:
   - Go through each line of the markdown text without skipping any content.
   - Do not add any extra content, explanations, or AI-generated text beyond the specified modifications.
   - Apply each change exactly as directed across the entire document.
Here is the markdown text to process:
            """

# Function to process text with GPT-4
def process_with_gpt4(text):
    if client is None:
        st.error("OpenAI client is not initialized. Cannot process text.")
        return text
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise text processing assistant. Only return the processed text without any additional commentary or explanations."},
                {"role": "user", "content": prompt + "\n\nHere is the markdown text to process:\n" + text}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error processing with GPT-4: {str(e)}")
        return text

# Mathpix API Credentials
MATHPIX_APP_ID = "webtech_allen_ac_in_b6eda4_55dc4b"
MATHPIX_APP_KEY = "a869e65df7d85c35385bcc8ca72f8c83a5423865f702df7d56e52cf8366d1113"

# Initialize session state variables
if "current_json_idx" not in st.session_state:
    st.session_state.current_json_idx = 0
if "adding_new_box" not in st.session_state:
    st.session_state.adding_new_box = False
if "new_box_params" not in st.session_state:
    st.session_state.new_box_params = {"x": 50, "y": 50, "width": 100, "height": 100, "text": ""}
if "json_files" not in st.session_state:
    st.session_state.json_files = []
if "selected_json" not in st.session_state:
    st.session_state.selected_json = None

def extract_page_number(filename):
    parts = filename.split("_page_")
    return int(parts[1].split(".")[0]) if len(parts) > 1 and parts[1].split(".")[0].isdigit() else None

# ------------------ PDF Processing Function ------------------
def process_pdf(uploaded_pdf):
    folder_name = os.path.splitext(uploaded_pdf.name)[0]
    target_folder = os.path.join("pdf_uploads", folder_name)
    os.makedirs(target_folder, exist_ok=True)
    
    pdf_path = os.path.join(target_folder, uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.success(f"PDF saved to: {pdf_path}")

    upload_url = "https://api.mathpix.com/v3/pdf"
    headers = {"app_id": MATHPIX_APP_ID, "app_key": MATHPIX_APP_KEY}
    options = {
        "rm_spaces": True,
        "metadata": {"improve_mathpix": False},
        "auto_number_sections": False,
        "remove_section_numbering": False,
        "preserve_section_numbering": True,
        "enable_tables_fallback ": True
    }
    data = {"options_json": json.dumps(options)}
    with open(pdf_path, "rb") as pdf_file:
        files = {"file": (uploaded_pdf.name, pdf_file, "application/pdf")}
        response = requests.post(upload_url, headers=headers, files=files, data=data)
    
    if response.status_code == 200:
        resp_json = response.json()
        pdf_job_id = resp_json.get("pdf_id") or resp_json.get("id")
        if not pdf_job_id:
            st.error("Failed to retrieve PDF job ID from Mathpix response.")
            return
        st.success(f"PDF uploaded to Mathpix. Job ID: {pdf_job_id}")
    else:
        st.error(f"Error uploading PDF: {response.status_code} - {response.text}")
        return

    status_url = f"https://api.mathpix.com/v3/pdf/{pdf_job_id}"
    with st.spinner("Processing PDF with Mathpix..."):
        while True:
            status_response = requests.get(status_url, headers=headers)
            if status_response.status_code == 200:
                status_json = status_response.json()
                status = status_json.get("status")
                st.write(f"Current status: {status}")
                if (isinstance(status, str) and status.lower() in ("100%", "completed")) or \
                   (isinstance(status, (int, float)) and status >= 100):
                    st.success("PDF processing complete!")
                    break
                else:
                    time.sleep(5)
            else:
                st.error("Error checking PDF status: " + status_response.text)
                return

    lines_url = f"https://api.mathpix.com/v3/pdf/{pdf_job_id}.lines.mmd.json"
    lines_response = requests.get(lines_url, headers=headers)
    if lines_response.status_code == 200:
        json_content = lines_response.text
        json_filename = f"{folder_name}.lines.mmd.json"
        json_path = os.path.join(target_folder, json_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_content)
        st.success(f"JSON file saved to: {json_path}")
    else:
        st.error(f"Error retrieving JSON: {lines_response.status_code} - {lines_response.text}")
        return

    images_output_dir = os.path.join(target_folder, "images")
    os.makedirs(images_output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)
        output_image_path = os.path.join(images_output_dir, f"{folder_name}_page_{page_num+1}.png")
        pix.save(output_image_path)
    st.success(f"PDF split into {len(doc)} images and saved in {images_output_dir}")

    json_output_dir = os.path.join(target_folder, "json")
    os.makedirs(json_output_dir, exist_ok=True)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "pages" in data:
        for entry in data["pages"]:
            page_number = entry.get("page")
            page_data = {
                "image_id": entry.get("image_id"),
                "page": page_number,
                "lines": entry.get("lines", []),
                "page_height": entry.get("page_height"),
                "page_width": entry.get("page_width"),
                "languages_detected": entry.get("languages_detected", [])
            }
            output_json_path = os.path.join(json_output_dir, f"{folder_name}_page_{page_number}.json")
            with open(output_json_path, "w", encoding="utf-8") as f_out:
                json.dump(page_data, f_out, indent=4, ensure_ascii=False)
        st.success(f"JSON split into {len(data['pages'])} pages and saved in {json_output_dir}")
    else:
        st.error("JSON structure does not contain a 'pages' key. Skipping JSON split.")

    st.session_state.processed_folder = target_folder
    st.session_state.processed_json_dir = json_output_dir
    st.session_state.processed_image_dir = os.path.join(target_folder, "images")

# --------------- Helper Functions ---------------
def send_to_mathpix(image_path):
    url = "https://api.mathpix.com/v3/text"
    headers = {"app_id": MATHPIX_APP_ID, "app_key": MATHPIX_APP_KEY}
    files = {"file": open(image_path, "rb")}
    options = {"rm_spaces": True}
    response = requests.post(url, headers=headers, files=files, data={"options_json": json.dumps(options)})
    if response.status_code == 200:
        return response.json().get("text", "No text extracted.")
    else:
        return f"Error: {response.status_code} - {response.text}"
    
def get_latest_downloaded_image():
    downloads_folder = os.path.expanduser("~/Downloads")
    image_files = sorted(glob.glob(os.path.join(downloads_folder, "*.[pj][np]g")),
                         key=os.path.getctime, reverse=True)
    return image_files[0] if image_files else None

# --------------- Sidebar Controls ---------------
st.sidebar.header("📁 PDF Processing")
uploaded_pdf = st.sidebar.file_uploader("Select a PDF file", type=["pdf"], key="pdf_uploader_1")

if uploaded_pdf is not None:
    st.sidebar.write(f"Selected PDF: {uploaded_pdf.name}")
    if st.sidebar.button("Process PDF with Mathpix", key="process_pdf_button_1"):
        process_pdf(uploaded_pdf)

# --------------- Data Directories ---------------
st.sidebar.markdown("---")
st.sidebar.header("🗂 Data Directories")

# Initialize json_dir and image_dir
json_dir = None
image_dir = None

if "processed_json_dir" in st.session_state and "processed_image_dir" in st.session_state:
    json_dir = st.session_state.processed_json_dir
    image_dir = st.session_state.processed_image_dir
    st.sidebar.info(f"Using processed JSON Directory: {json_dir}")
    st.sidebar.info(f"Using processed Image Directory: {image_dir}")
else:
    json_dir = st.sidebar.text_input("📂 JSON Directory")
    image_dir = st.sidebar.text_input("🖼 Image Directory")

    if json_dir and image_dir:
        if not Path(json_dir).is_dir():
            st.sidebar.error("🚨 JSON Directory does not exist.")
        elif not Path(image_dir).is_dir():
            st.sidebar.error("🚨 Image Directory does not exist.")
        else:
            st.session_state.processed_json_dir = json_dir
            st.session_state.processed_image_dir = image_dir
            st.session_state.processed_folder = str(Path(json_dir).parent)
            st.sidebar.info(f"Processed folder set to: {st.session_state.processed_folder}")

# Only proceed if we have valid directories
if json_dir and image_dir:
    # --------------- Function for URL Conversion -------------------
    def convert_url_to_html(json_data):
        # This function converts markdown image URLs to HTML figures.
        # If a caption is present in the annotation, it is inserted into alt and figcaption.
        if isinstance(json_data, list):
            for page in json_data:
                if "lines" in page:
                    for line in page["lines"]:
                        if "text" in line:
                            text = line["text"]
                            urls = re.findall(r'!\[\]\((https?://[^\)]+)\)', text)
                            for url in urls:
                                caption_text = line.get("caption", "")
                                html_format = f'<figure><img src="{url}" alt="{caption_text}"><figcaption>{caption_text}</figcaption></figure>'
                                text = text.replace(f"![]({url})", html_format)
                            line["text"] = text
        elif isinstance(json_data, dict):
            if "lines" in json_data:
                for line in json_data["lines"]:
                    if "text" in line:
                        text = line["text"]
                        urls = re.findall(r'!\[\]\((https?://[^\)]+)\)', text)
                        for url in urls:
                            caption_text = line.get("caption", "")
                            html_format = f'<figure><img src="{url}" alt="{caption_text}"><figcaption>{caption_text}</figcaption></figure>'
                            text = text.replace(f"![]({url})", html_format)
                        line["text"] = text
        else:
            raise ValueError("Expected data format is either a list or dictionary containing 'lines'.")
        return json_data

    with st.sidebar:
        convert_urls_to_html = st.checkbox("Change URLs to HTML format", value=False)

    if "processed_json_dir" in st.session_state:
        raw_json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
        sorted_json_files = natsorted(raw_json_files)
        # Only one selectbox is used here in the sidebar
        selected_json = st.selectbox("📜 Select JSON File", options=sorted_json_files, index=st.session_state.current_json_idx)
        st.session_state.selected_json = selected_json
        if convert_urls_to_html and st.session_state.selected_json:
            json_path = os.path.join(json_dir, st.session_state.selected_json)
            with open(json_path, 'r', encoding="utf-8") as f:
                page_data = json.load(f)
            updated_page_data = convert_url_to_html(page_data)
            with open(json_path, 'w', encoding="utf-8") as f:
                json.dump(updated_page_data, f, indent=4, ensure_ascii=False)
            st.success("URLs in JSON file have been converted to HTML format.")
    else:
        st.error("Kindly Click on change urls to https format , if the images need")

    # --------------- MMD File Merging Logic ---------------
    def merge_mmd_files():
        pdf_folder = st.session_state.get("processed_folder")
        if not pdf_folder:
            st.error("Processed folder not found in session state.")
            return
        if not os.path.exists(pdf_folder):
            st.error(f"The processed folder {pdf_folder} does not exist.")
            return
        mmd_dir = os.path.join(pdf_folder, "temp_mmd")
        if not os.path.exists(mmd_dir):
            st.error(f"temp_mmd folder not found in {pdf_folder}. Please check your folder structure.")
            return
        mmd_files = [f for f in os.listdir(mmd_dir) if f.endswith(".mmd")]
        if not mmd_files:
            st.error("No MMD files found in the temp_mmd folder.")
            return
        try:
            mmd_files_sorted = sorted(mmd_files, key=lambda x: extract_page_number(x))
        except Exception as e:
            st.error(f"Error sorting MMD files: {str(e)}")
            return
        merged_text = ""
        for mmd_file in mmd_files_sorted:
            file_path = os.path.join(mmd_dir, mmd_file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                merged_text += content + "\n"
            except Exception as e:
                st.error(f"Error reading file {file_path}: {str(e)}")
                return
        final_name = os.path.basename(pdf_folder)
        merged_mmd_path = os.path.join(pdf_folder, f"{final_name}.mmd")
        try:
            with open(merged_mmd_path, "w", encoding="utf-8") as f:
                f.write(merged_text)
        except Exception as e:
            st.error(f"Error writing merged file: {str(e)}")
            return
        st.success(f"Merged MMD saved as: {merged_mmd_path}")
  
    def merge_lexical_jsons():
        try:
            # Get the lexical JSON directory
            lexical_json_dir = os.path.join(os.path.dirname(json_dir), "lexical_json")
            if not os.path.exists(lexical_json_dir):
                st.error("Lexical JSON directory not found!")
                return
            
            # Get all lexical JSON files
            lexical_json_files = [f for f in os.listdir(lexical_json_dir) if f.endswith(".json")]
            if not lexical_json_files:
                st.error("No lexical JSON files found!")
                return
            
            # Sort files by page number
            lexical_json_files = natsorted(lexical_json_files)
            
            # Merge all lexical JSON content
            merged_text = ""
            for json_file in lexical_json_files:
                json_path = os.path.join(lexical_json_dir, json_file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Extract text from annotations (excluding soft-deleted)
                for annotation in json_data.get("lines", []):
                    if annotation.get("soft_delete", False):
                        continue
                    merged_text += annotation["text"] + "\n"
            
            # Create lexical MMD file
            pdf_folder = st.session_state.get("processed_folder")
            if not pdf_folder:
                st.error("Processed folder not found!")
                return
            
            final_name = os.path.basename(pdf_folder)
            lexical_mmd_path = os.path.join(pdf_folder, f"{final_name}_lexical.mmd")
            
            with open(lexical_mmd_path, "w", encoding="utf-8") as f:
                f.write(merged_text)
            
            st.success(f"✅ Merged lexical MMD saved as: {lexical_mmd_path}")
            
            # Convert to HTML
            lexical_html_path = os.path.join(pdf_folder, f"{final_name}_lexical.html")
            result = subprocess.run(["mpx", "convert", lexical_mmd_path, lexical_html_path], capture_output=True, text=True)
            if result.returncode == 0:
                st.success(f"✅ Converted to HTML: {lexical_html_path}")
            else:
                st.error("🚨 Lexical MMD conversion failed!")
                st.text(result.stderr)
            
        except Exception as e:
            st.error(f"❌ Error merging lexical JSONs: {str(e)}")

    # --------------- Switch LMM button---------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Switch LMM")

    # Disable the checkbox if OpenAI is not available
    switch_lmm = st.sidebar.checkbox("Enable LMM Processing", value=False, disabled=client is None)

    if switch_lmm and client is not None:
        st.sidebar.info("LMM Processing is enabled")
        lexical_json_dir = os.path.join(os.path.dirname(json_dir), "lexical_json")
        os.makedirs(lexical_json_dir, exist_ok=True)
        
        # Add worker configuration with increased max value
        converter = st.session_state.lexical_converter
        new_max_workers = st.sidebar.slider(
            "Number of Parallel Workers",
            min_value=1,
            max_value=40,  # Increased to 40
            value=converter.max_workers,
            help="Increase for faster processing, but be mindful of API rate limits and system resources. Recommended: 4-8 workers for stable operation."
        )
        
        if new_max_workers != converter.max_workers:
            converter.max_workers = new_max_workers
            st.sidebar.info(f"Worker count updated to {new_max_workers}")
            if new_max_workers > 8:
                st.sidebar.warning("⚠️ High worker count may cause API rate limit issues!")
        
        # Get all JSON files that need processing
        json_files_to_process = [
            f for f in sorted_json_files 
            if not os.path.exists(os.path.join(lexical_json_dir, f))
        ]
        
        if json_files_to_process:
            # Start conversion if not already running
            if not converter.is_running:
                # Start the conversion in a separate thread
                import threading
                conversion_thread = threading.Thread(
                    target=converter.start_conversion,
                    args=(json_files_to_process, json_dir, lexical_json_dir),
                    daemon=True
                )
                conversion_thread.start()
            
            # Show progress
            progress = converter.get_progress()
            if progress["total"] > 0:
                # Create a progress bar
                progress_bar = st.sidebar.progress(progress["processed"] / progress["total"])
                st.sidebar.write(f"Processed: {progress['processed']}/{progress['total']}")
                
                # Show ready pages
                if progress["ready"]:
                    st.sidebar.subheader("Ready Pages")
                    for page in progress["ready"]:
                        st.sidebar.success(f"✓ {page}")
                
                # Show processing pages
                if progress["processing"]:
                    st.sidebar.subheader("Processing Pages")
                    for page in progress["processing"]:
                        st.sidebar.info(f"⏳ {page}")
                
                # Show errors
                if progress["errors"]:
                    st.sidebar.subheader("Errors")
                    for page, error in progress["errors"].items():
                        st.sidebar.error(f"❌ {page}: {error}")
                
                # Add stop button
                if st.sidebar.button("Stop Conversion"):
                    converter.stop_conversion()
                    st.rerun()
            
            # Use Streamlit's native progress tracking
            time.sleep(0.1)  # Small delay to prevent UI blocking
        
        # Load current page's lexical JSON if available
        if selected_json:
            lexical_json_path = os.path.join(lexical_json_dir, selected_json)
            if os.path.exists(lexical_json_path):
                try:
                    with open(lexical_json_path, 'r', encoding='utf-8') as f:
                        lexical_json_data = json.load(f)
                except Exception as e:
                    st.error(f"Error loading lexical JSON: {str(e)}")
    elif switch_lmm and client is None:
        st.sidebar.error("Cannot enable LMM Processing: OpenAI API key not found")
    else:
        st.sidebar.info("LMM Processing is disabled")

    # --------------- Navigation Controls for pages ---------------
    if json_dir:
        raw_json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
        sorted_json_files = natsorted(raw_json_files)
        st.session_state.json_files = sorted_json_files
        st.sidebar.subheader("📑 Navigation Controls")
        st.session_state.current_json_idx = st.session_state.json_files.index(selected_json)
        col_nav_prev, col_nav_next = st.sidebar.columns(2)
        with col_nav_prev:
            if st.button("⏮️ Previous") and st.session_state.current_json_idx > 0:
                st.session_state.current_json_idx -= 1
                st.session_state.canvas_data = None
                st.rerun()
        with col_nav_next:
            if st.session_state.current_json_idx < len(st.session_state.json_files) - 1:
                if st.button("⏭️ Next"):
                    st.session_state.current_json_idx += 1
                    st.session_state.canvas_data = None
                    st.rerun()
            else:
                st.sidebar.info("PDF has ENDED CONGRATULATIONS !!!!!😊")
                if switch_lmm:
                    if st.button("Merge all Lexical JSONs", key="merge_lexical"):
                        merge_lexical_jsons()
                else:
                    if st.button("Merge all MMD'S", key="merge_pdf"):
                        merge_mmd_files()

    # --------------- Main App: Annotation & Rendering ---------------
    raw_json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    sorted_json_files = natsorted(raw_json_files)
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    if selected_json:
        selected_page = extract_page_number(selected_json)
        matching_image = next((img for img in image_files if extract_page_number(img) == selected_page), None)
    else:
        matching_image = None

    # Initialize page_data and json_data
    page_data = None
    json_data = None
    lexical_json_data = None

    if selected_json:
        json_file_path = os.path.join(json_dir, selected_json)
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            if isinstance(json_data, list):
                page_data = json_data[0]
            elif isinstance(json_data, dict):
                page_data = json_data
            else:
                st.error("Invalid JSON format")
                st.stop()
        except Exception as e:
            st.error(f"Error loading JSON file: {str(e)}")
            st.stop()
    else:
        st.error("No JSON file selected.")
        st.stop()

    # Load lexical JSON if LMM is enabled
    if switch_lmm and client is not None:
        lexical_json_dir = os.path.join(os.path.dirname(json_dir), "lexical_json")
        lexical_json_path = os.path.join(lexical_json_dir, selected_json)
        if os.path.exists(lexical_json_path):
            try:
                with open(lexical_json_path, 'r', encoding='utf-8') as f:
                    lexical_json_data = json.load(f)
            except Exception as e:
                st.error(f"Error loading lexical JSON: {str(e)}")

    if isinstance(json_data, list):
        page_info = next((item for item in json_data if isinstance(item, dict) and "page_width" in item and "page_height" in item), {})
    else:
        page_info = json_data

    col1, col2 = st.columns(2)
    base_dir = str(Path(json_dir).parent)

    with st.sidebar:
        st.subheader("🖼 Image Display Options")
        show_raw_image = st.checkbox("Show Image Without Bounding Boxes", value=False)

    with col1:
        st.subheader("🖼 Image with Bounding Boxes")
        if matching_image:
            image_path = os.path.join(image_dir, matching_image)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as f:
                    json_data = json.load(f)
                if isinstance(json_data, dict):
                    json_data = [json_data]
                first_entry = json_data[0] if json_data else None
                if first_entry and 'lines' in first_entry:
                    json_width = first_entry.get("page_width", image.shape[1])
                    json_height = first_entry.get("page_height", image.shape[0])

                    def draw_boxes(img, annotations):
                        img_copy = img.copy()
                        scale_x = img_copy.shape[1] / json_width
                        scale_y = img_copy.shape[0] / json_height
                        for idx, ann in enumerate(annotations):
                             if ann.get("soft_delete", False):
                                color = (0, 0, 255)  # Blue for deleted annotations
                                # Strikethrough the text for soft-deleted annotations
                                text = f"~~{ann['text']}~~"  # Adding strikethrough in the text itself
                             else:
                                color = (255, 0, 0)  # Red for active annotations
                                text = ann['text']

                             if ann.get("deleted", False):
                                continue  # Skip this annotation if it is marked as deleted
                        
                             if "cnt" in ann:
                                pts = np.array(ann["cnt"], dtype=np.float32)
                                pts[:, 0] *= scale_x
                                pts[:, 1] *= scale_y
                                pts = pts.astype(np.int32)
                                x, y, w, h = cv2.boundingRect(pts)
                                cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
                                cv2.putText(img_copy, str(idx),
                                            (x - cv2.getTextSize(str(idx), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0][0] - 5, y+h-5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 4)
                        return img_copy
                    
                    if show_raw_image:
                        display_image = image.copy()
                    else:
                        display_image = draw_boxes(image, first_entry["lines"])
                else:
                    display_image = image.copy()
            else:
                display_image = image.copy()
            
            if st.session_state.adding_new_box:
                params = st.session_state.new_box_params
                preview_cnt = np.array([
                    [params["x"], params["y"]],
                    [params["x"], params["y"] + params["height"]],
                    [params["x"] + params["width"], params["y"] + params["height"]],
                    [params["x"] + params["width"], params["y"]]
                ], dtype=np.float32)
                scale_x = display_image.shape[1] / json_width
                scale_y = display_image.shape[0] / json_height
                preview_cnt[:, 0] *= scale_x
                preview_cnt[:, 1] *= scale_y
                preview_cnt = preview_cnt.astype(np.int32)
                x_new, y_new, w_new, h_new = cv2.boundingRect(preview_cnt)
                cv2.rectangle(display_image, (x_new, y_new), (x_new+w_new, y_new+h_new), (0, 255, 0), 2)
                cv2.putText(display_image, "New",
                            (x_new - cv2.getTextSize("New", cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0][0] - 5, y_new+h_new-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
            
            st.image(Image.fromarray(display_image), caption=f"Image {matching_image}", use_column_width=True)

    # Column 2: Render Output (either regular JSON or lexical JSON based on LMM state)
    with col2:
        if switch_lmm and lexical_json_data:
            st.subheader("📄 Lexical JSON Rendered Output")
            try:
                # Extract text from lexical JSON annotations (excluding soft-deleted)
                extracted_text = ""
                for annotation in lexical_json_data.get("lines", []):
                    if annotation.get("soft_delete", False):
                        continue  # Skip soft-deleted annotations
                    extracted_text += annotation["text"] + "\n"

                # Create lexical_mmd directory if it doesn't exist
                lexical_mmd_dir = os.path.join(base_dir, "lexical_mmd")
                os.makedirs(lexical_mmd_dir, exist_ok=True)
                
                # Prepare MMD file for rendering
                temp_html_dir = os.path.join(base_dir, "temp_html")
                os.makedirs(temp_html_dir, exist_ok=True)
                base_name = Path(selected_json).stem
                lexical_mmd_path = os.path.join(lexical_mmd_dir, f"{base_name}_lexical.mmd")
                temp_html_path = os.path.join(temp_html_dir, f"{base_name}_lexical.html")

                with open(lexical_mmd_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)

                result = subprocess.run(["mpx", "convert", lexical_mmd_path, temp_html_path], capture_output=True, text=True)
                if result.returncode == 0:
                    with open(temp_html_path, "r", encoding="utf-8") as html_file:
                        html_content = html_file.read()
                    st.components.v1.html(html_content, height=900, scrolling=True)
                else:
                    st.error("🚨 Lexical MMD conversion failed!")
                    st.text(result.stderr)
            except Exception as e:
                st.error(f"❌ Error rendering lexical JSON: {str(e)}")
        else:
            st.subheader("📄 Rendered Output")
            if selected_json:
                try:
                    with open(json_file_path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)
                    if isinstance(json_data, list):
                        first_entry = json_data[0]
                    elif isinstance(json_data, dict):
                        first_entry = json_data
                    else:
                        st.error("❌ Invalid JSON format.")
                        first_entry = {}

                    # Extract text from the first page's annotations (excluding soft-deleted)
                    extracted_text = ""
                    for annotation in first_entry.get("lines", []):
                        if annotation.get("soft_delete", False):
                            continue  # Skip soft-deleted annotations
                        extracted_text += annotation["text"] + "\n"

                    # Prepare MMD file for rendering
                    temp_mmd_dir = os.path.join(base_dir, "temp_mmd")
                    temp_html_dir = os.path.join(base_dir, "temp_html")
                    os.makedirs(temp_mmd_dir, exist_ok=True)
                    os.makedirs(temp_html_dir, exist_ok=True)
                    base_name = Path(selected_json).stem
                    temp_mmd_path = os.path.join(temp_mmd_dir, f"{base_name}.mmd")
                    temp_html_path = os.path.join(temp_html_dir, f"{base_name}.html")

                    with open(temp_mmd_path, "w", encoding="utf-8") as f:
                        f.write(extracted_text)

                    result = subprocess.run(["mpx", "convert", temp_mmd_path, temp_html_path], capture_output=True, text=True)
                    if result.returncode == 0:
                        with open(temp_html_path, "r", encoding="utf-8") as html_file:
                            html_content = html_file.read()
                        st.components.v1.html(html_content, height=900, scrolling=True)
                    else:
                        st.error("🚨 MMD conversion failed!")
                        st.text(result.stderr)
                except json.JSONDecodeError:
                    st.error("❌ Failed to load JSON file.")

    # Add a checkbox to toggle canvas visibility
    with st.sidebar:
        open_canvas = st.checkbox("Open Canvas", value=False)

    # Show the canvas only when 'Open Canvas' checkbox is checked
    if open_canvas:
        st.subheader("🎨 Draw Bounding Box")
        if "canvas_data" not in st.session_state:
            st.session_state.canvas_data = None
        canvas_key = f"canvas_{st.session_state.current_json_idx}"
        col_reset, col_space = st.columns([1, 3])
        with col_reset:
            if st.button("🔄 Reset Canvas"):
                st.session_state.canvas_data = None
                st.session_state[f"reset_key_{st.session_state.current_json_idx}"] = str(os.urandom(8))
                st.rerun()
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=Image.open(image_path),
            update_streamlit=True,
            height=500,
            width=500,
            drawing_mode="rect",
            key=st.session_state.get(f"reset_key_{st.session_state.current_json_idx}", canvas_key),
        )
        if canvas_result.json_data is not None:
            st.session_state.canvas_data = canvas_result.json_data
            if st.session_state.canvas_data and "objects" in st.session_state.canvas_data:
                objects = st.session_state.canvas_data["objects"]
                if objects:
                    img_width = 500
                    img_height = 500
                    for obj in objects:
                        left, top, width, height = obj["left"], obj["top"], obj["width"], obj["height"]
                        scaled_x = int((left / img_width) * page_info.get("page_width", img_width))
                        scaled_y = int((top / img_height) * page_info.get("page_height", img_height))
                        scaled_w = int((width / img_width) * page_info.get("page_width", img_width))
                        scaled_h = int((height / img_height) * page_info.get("page_height", img_height))
                        image_id = page_data.get("image_id", "unknown_image_id")
                        url = f"https://cdn.mathpix.com/cropped/{image_id}.jpg?height={scaled_h}&width={scaled_w}&top_left_y={scaled_y}&top_left_x={scaled_x}"
                        st.write(f"Bounding Box URL: {url}")

    full_image = cv2.imread(image_path)
    temp_mmd_dir = os.path.join(os.path.expanduser("~"), "Desktop/automated/temp_mmd")
    os.makedirs(temp_mmd_dir, exist_ok=True)

    with st.sidebar:
        st.subheader("📦 Manage Annotations")
        if "lines" in page_data:
            for idx, annotation in enumerate(page_data["lines"]):
                with st.expander(f"Box {idx}", expanded=False):
                    pts = np.array(annotation["cnt"], dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(pts)
                    #Check if annotation is soft deleted
                    if annotation.get("soft_delete", False):
                        st.markdown(f"~~Box {idx}~~: {annotation['text']}")  # Strike-through the text
                        st.write(f"❌ This annotation is deleted. Recover it below.")
                        if st.button(f"♻️ Recover {idx}", key=f"recover_{idx}"):
                            annotation["soft_delete"] = False
                            # Only update the appropriate JSON file based on LMM state
                            if switch_lmm and lexical_json_data:
                                lexical_json_data["lines"][idx]["soft_delete"] = False
                                with open(lexical_json_path, "w", encoding="utf-8") as f:
                                    json.dump(lexical_json_data, f, indent=4)
                            else:
                                with open(json_file_path, "w", encoding="utf-8") as f:
                                    json.dump(page_data, f, indent=4)
                            st.rerun()
                        continue  # Skip further UI for deleted annotations
                    
                    # Use lexical JSON text if LMM is enabled, otherwise use original text
                    current_text = lexical_json_data["lines"][idx]["text"] if (switch_lmm and lexical_json_data and idx < len(lexical_json_data.get("lines", []))) else annotation["text"]
                    
                    new_x = st.slider(f"X Pos {idx}", 0, page_info.get("page_width", 2068), x, key=f"x_{idx}")
                    new_y = st.slider(f"Y Pos {idx}", 0, page_info.get("page_height", 2924), y, key=f"y_{idx}")
                    new_w = st.slider(f"Width {idx}", 1, page_info.get("page_width", 2068) - new_x, w, key=f"w_{idx}")
                    new_h = st.slider(f"Height {idx}", 1, page_info.get("page_height", 2924) - new_y, h, key=f"h_{idx}")
                    new_text = st.text_area(f"Text {idx}", value=current_text, key=f"text_{idx}")

                    if (new_x != x or new_y != y or new_w != w or new_h != h or new_text != current_text):
                        # Update only the appropriate JSON file based on LMM state
                        if switch_lmm and lexical_json_data:
                            # Update lexical JSON
                            lexical_json_data["lines"][idx]["cnt"] = [[new_x, new_y],
                                                                     [new_x, new_y + new_h],
                                                                     [new_x + new_w, new_y + new_h],
                                                                     [new_x + new_w, new_y]]
                            lexical_json_data["lines"][idx]["text"] = new_text
                            with open(lexical_json_path, "w", encoding="utf-8") as f:
                                json.dump(lexical_json_data, f, indent=4)
                        else:
                            # Update original JSON
                            annotation["cnt"] = [[new_x, new_y],
                                               [new_x, new_y + new_h],
                                               [new_x + new_w, new_y + new_h],
                                               [new_x + new_w, new_y]]
                            annotation["text"] = new_text
                            with open(json_file_path, "w", encoding="utf-8") as f:
                                json.dump(page_data, f, indent=4)
                        st.rerun()

                    # --- New Feature: Update Figure HTML with Caption ---
                    if re.search(r'<img\s+src="https?://[^"]+"', new_text):
                        existing_caption = annotation.get("caption", "")
                        caption = st.text_input("Add captions here", value=existing_caption, key=f"caption_{idx}")
                        if caption != existing_caption:
                            # Update only the appropriate JSON file based on LMM state
                            if switch_lmm and lexical_json_data:
                                lexical_json_data["lines"][idx]["caption"] = caption
                                pattern = r'<figure>\s*<img\s+src="([^"]+)"\s+alt="[^"]*"\s*>\s*<figcaption>[^<]*</figcaption>\s*</figure>'
                                new_text_updated = re.sub(
                                    pattern,
                                    lambda m: f'<figure><img src="{m.group(1)}" alt="{caption}"><figcaption>{caption}</figcaption></figure>',
                                    lexical_json_data["lines"][idx]["text"]
                                )
                                lexical_json_data["lines"][idx]["text"] = new_text_updated
                                with open(lexical_json_path, "w", encoding="utf-8") as f:
                                    json.dump(lexical_json_data, f, indent=4)
                            else:
                                annotation["caption"] = caption
                                pattern = r'<figure>\s*<img\s+src="([^"]+)"\s+alt="[^"]*"\s*>\s*<figcaption>[^<]*</figcaption>\s*</figure>'
                                new_text_updated = re.sub(
                                    pattern,
                                    lambda m: f'<figure><img src="{m.group(1)}" alt="{caption}"><figcaption>{caption}</figcaption></figure>',
                                    annotation["text"]
                                )
                                annotation["text"] = new_text_updated
                                with open(json_file_path, "w", encoding="utf-8") as f:
                                    json.dump(page_data, f, indent=4)
                            st.rerun()

                    # Soft Delete Button
                    if st.button(f"🗑️ Soft Delete {idx}", key=f"soft_delete_{idx}"):
                        # Update only the appropriate JSON file based on LMM state
                        if switch_lmm and lexical_json_data:
                            lexical_json_data["lines"][idx]["soft_delete"] = True
                            with open(lexical_json_path, "w", encoding="utf-8") as f:
                                json.dump(lexical_json_data, f, indent=4)
                        else:
                            annotation["soft_delete"] = True
                            with open(json_file_path, "w", encoding="utf-8") as f:
                                json.dump(page_data, f, indent=4)
                        st.rerun()

                    col_up, col_down= st.columns(2)
                    with col_up:
                        if st.button(f"⬆️ Move Up {idx}", key=f"up_{idx}") and idx > 0:
                            # Update only the appropriate JSON file based on LMM state
                            if switch_lmm and lexical_json_data:
                                lexical_json_data["lines"][idx - 1], lexical_json_data["lines"][idx] = lexical_json_data["lines"][idx], lexical_json_data["lines"][idx - 1]
                                with open(lexical_json_path, "w", encoding="utf-8") as f:
                                    json.dump(lexical_json_data, f, indent=4)
                            else:
                                page_data["lines"][idx - 1], page_data["lines"][idx] = page_data["lines"][idx], page_data["lines"][idx - 1]
                                with open(json_file_path, "w", encoding="utf-8") as f:
                                    json.dump(page_data, f, indent=4)
                            st.rerun()
                    with col_down:
                        if st.button(f"⬇️ Move Down {idx}", key=f"down_{idx}") and idx < len(page_data["lines"]) - 1:
                            # Update only the appropriate JSON file based on LMM state
                            if switch_lmm and lexical_json_data:
                                lexical_json_data["lines"][idx + 1], lexical_json_data["lines"][idx] = lexical_json_data["lines"][idx], lexical_json_data["lines"][idx + 1]
                                with open(lexical_json_path, "w", encoding="utf-8") as f:
                                    json.dump(lexical_json_data, f, indent=4)
                            else:
                                page_data["lines"][idx + 1], page_data["lines"][idx] = page_data["lines"][idx], page_data["lines"][idx + 1]
                                with open(json_file_path, "w", encoding="utf-8") as f:
                                    json.dump(page_data, f, indent=4)
                            st.rerun()
                    send_to_mathpix_flag = st.checkbox(f"📤 Send to Mathpix {idx}", key=f"mathpix_{idx}")
                    send_from_downloads_flag = st.checkbox(f"📥 Send Latest Downloaded Image {idx} to Mathpix", key=f"downloads_{idx}")
                    if send_to_mathpix_flag:
                        cropped_path = os.path.join(temp_mmd_dir, f"cropped_{idx}.png")
                        scale_x = full_image.shape[1] / page_info.get("page_width", full_image.shape[1])
                        scale_y = full_image.shape[0] / page_info.get("page_height", full_image.shape[0])
                        scaled_x = int(new_x * scale_x)
                        scaled_y = int(new_y * scale_y)
                        scaled_w = int(new_w * scale_x)
                        scaled_h = int(new_h * scale_y)
                        cropped_img = full_image[scaled_y:scaled_y + scaled_h, scaled_x:scaled_x + scaled_w]
                        if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                            cv2.imwrite(cropped_path, cropped_img)
                            st.write(f"✅ Saved cropped image: {cropped_path}")
                            st.image(cropped_img, caption=f"📸 Cropped Image for Box {idx}", use_column_width=True)
                        else:
                            st.error(f"🚨 Error: Cropped image is empty for Box {idx}!")
                        if st.button(f"🚀 Process Box {idx} with Mathpix", key=f"mathpix_button_{idx}"):
                            with st.spinner("Processing with Mathpix..."):
                                extracted_text = send_to_mathpix(cropped_path)
                            if "Error" in extracted_text:
                                st.error(f"⚠️ Mathpix API Error: {extracted_text}")
                            else:
                                st.text_area(f"Mathpix Extracted Text {idx}", value=extracted_text, height=100, key=f"mathpix_text_{idx}")
                    if send_from_downloads_flag:
                        latest_image = get_latest_downloaded_image()
                        if latest_image:
                            st.write(f"📥 Using latest downloaded image: {latest_image}")
                            st.image(latest_image, caption=f"🖼️ Downloaded Image for Box {idx}", use_column_width=True)
                            if st.button(f"🚀 Process Downloaded Image {idx} with Mathpix", key=f"download_mathpix_button_{idx}"):
                                with st.spinner("Processing downloaded image with Mathpix..."):
                                    extracted_text = send_to_mathpix(latest_image)
                                if "Error" in extracted_text:
                                    st.error(f"⚠️ Mathpix API Error: {extracted_text}")
                                else:
                                    st.text_area(f"Mathpix Extracted Text {idx}", value=extracted_text, height=100, key=f"download_mathpix_text_{idx}")
                        else:
                            st.error(f"🚨 No recent image found in Downloads!")

class LexicalConverter:
    def __init__(self):
        self.ready_pages = {}
        self.processing_pages = {}
        self.error_pages = {}
        self.total_pages = 0
        self.processed_pages = 0
        self.is_running = False
        self.lock = threading.Lock()
        self.thread_pool = None
        # Increase max_workers to 40 and adjust default
        self.max_workers = min(40, os.cpu_count() * 4 or 8)  # Default to min of 40 or 4x CPU count
        
        # Add rate limiting with shorter interval for more workers
        self.last_api_call = 0
        self.min_api_call_interval = 0.5  # Reduced to 0.5 seconds to handle more workers

    def convert_single_page(self, json_file: str, json_dir: str, lexical_json_dir: str) -> Dict:
        """Convert a single JSON page to lexical JSON"""
        try:
            json_path = os.path.join(json_dir, json_file)
            lexical_json_path = os.path.join(lexical_json_dir, json_file)
            
            # Skip if already converted
            if os.path.exists(lexical_json_path):
                return {"status": "already_exists", "file": json_file}
            
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Deep copy to avoid modifying original
            lexical_json_data = json.loads(json.dumps(json_data))
            
            # Process each line's text field
            if isinstance(lexical_json_data, dict) and "lines" in lexical_json_data:
                for line in lexical_json_data["lines"]:
                    if "text" in line:
                        original_text = line["text"]
                        # Add rate limiting for API calls
                        current_time = time.time()
                        time_since_last_call = current_time - self.last_api_call
                        if time_since_last_call < self.min_api_call_interval:
                            time.sleep(self.min_api_call_interval - time_since_last_call)
                        processed_text = process_with_gpt4(original_text)
                        self.last_api_call = time.time()
                        if processed_text != original_text:
                            line["text"] = processed_text
            
            # Save the processed JSON
            os.makedirs(lexical_json_dir, exist_ok=True)
            with open(lexical_json_path, 'w', encoding='utf-8') as f:
                json.dump(lexical_json_data, f, indent=4, ensure_ascii=False)
            
            return {"status": "success", "file": json_file}
            
        except Exception as e:
            return {"status": "error", "file": json_file, "error": str(e)}

    def process_page(self, json_file: str, json_dir: str, lexical_json_dir: str):
        """Process a single page"""
        with self.lock:
            self.processing_pages[json_file] = True
        
        result = self.convert_single_page(json_file, json_dir, lexical_json_dir)
        
        with self.lock:
            self.processed_pages += 1
            if result["status"] == "success":
                self.ready_pages[result["file"]] = True
            elif result["status"] == "error":
                self.error_pages[result["file"]] = result["error"]
            self.processing_pages.pop(result["file"], None)

    def start_conversion(self, json_files: List[str], json_dir: str, lexical_json_dir: str):
        """Start parallel conversion process for all JSON files"""
        with self.lock:
            self.is_running = True
            self.total_pages = len(json_files)
            self.processed_pages = 0
        
        # Create a thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Submit all files to the thread pool
        futures = []
        for json_file in json_files:
            if not self.is_running:
                break
            future = self.thread_pool.submit(
                self.process_page,
                json_file,
                json_dir,
                lexical_json_dir
            )
            futures.append(future)
        
        # Wait for all futures to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        # Shutdown the thread pool
        self.thread_pool.shutdown(wait=True)
        self.thread_pool = None
        
        with self.lock:
            self.is_running = False

    def stop_conversion(self):
        """Stop the conversion process"""
        with self.lock:
            self.is_running = False
            if self.thread_pool:
                self.thread_pool.shutdown(wait=False)
                self.thread_pool = None

    def get_progress(self) -> Dict:
        """Get current conversion progress"""
        with self.lock:
            return {
                "total": self.total_pages,
                "processed": self.processed_pages,
                "ready": self.ready_pages.copy(),
                "processing": self.processing_pages.copy(),
                "errors": self.error_pages.copy()
            }

# Initialize converter in session state
if 'lexical_converter' not in st.session_state:
    st.session_state.lexical_converter = LexicalConverter()
# Updated version
