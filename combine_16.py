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


# Set Streamlit to wide layout
st.set_page_config(layout="wide")
st.title("📄 PDF Image Annotation Viewer & JSON Renderer")

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
uploaded_pdf = st.sidebar.file_uploader("Select a PDF file", type=["pdf"])
if uploaded_pdf is not None:
    st.sidebar.write(f"Selected PDF: {uploaded_pdf.name}")
    if st.sidebar.button("Process PDF with Mathpix"):
        process_pdf(uploaded_pdf)

st.sidebar.markdown("---")
st.sidebar.header("🗂 Data Directories")
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

def merge_mmd_files():
    # Get the processed folder from session state.
    pdf_folder = st.session_state.get("processed_folder")
    if not pdf_folder:
        st.error("Processed folder not found in session state.")
        return

    # Define the directory where per-page MMD files are stored.
    mmd_dir = os.path.join(pdf_folder, "temp_mmd")
    if not os.path.exists(mmd_dir):
        st.error(f"temp_mmd folder not found in {pdf_folder}.")
        return

    # List all .mmd files in mmd_dir.
    mmd_files = [f for f in os.listdir(mmd_dir) if f.endswith(".mmd")]
    if not mmd_files:
        st.error("No MMD files found in the temp_mmd folder.")
        return

    # Sort the files by page number.
    try:
        mmd_files_sorted = sorted(mmd_files, key=lambda x: extract_page_number(x))
    except Exception as e:
        st.error("Error sorting MMD files: " + str(e))
        return

    # Merge the contents.
    merged_text = ""
    for mmd_file in mmd_files_sorted:
        file_path = os.path.join(mmd_dir, mmd_file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            merged_text += content + "\n"
        except Exception as e:
            st.error("Error reading file " + file_path + ": " + str(e))
            return

    # Save the final merged file in the processed folder using the folder name.
    final_name = os.path.basename(pdf_folder)  # e.g., "hesc104"
    merged_mmd_path = os.path.join(pdf_folder, f"{final_name}.mmd")
    try:
        with open(merged_mmd_path, "w", encoding="utf-8") as f:
            f.write(merged_text)
    except Exception as e:
        st.error("Error writing merged file: " + str(e))
        return

    st.success(f"Merged MMD saved as: {merged_mmd_path}")


# Navigation Controls for pages
if json_dir:
    raw_json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    sorted_json_files = natsorted(raw_json_files)
    st.session_state.json_files = sorted_json_files
    st.sidebar.subheader("📑 Navigation Controls")
    # Add a selectbox for JSON file selection.
    selected_json = st.selectbox(
        "📜 Select JSON File", 
        options=st.session_state.json_files, 
        index=st.session_state.current_json_idx
    )
    # Update the current JSON index to match the selection.
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
                if st.button("Merge all MMD'S", key="merge_pdf"):
                    merge_mmd_files()
    selected_json = st.session_state.json_files[st.session_state.current_json_idx] if st.session_state.json_files else None
else:
    selected_json = None
    

# --------------- Prepare Current Page Data ---------------
if selected_json:
    json_path = os.path.join(json_dir, selected_json)
    with open(json_path, 'r', encoding="utf-8") as f:
        page_data = json.load(f)
    if isinstance(page_data, list):
        page_data = page_data[0]
    if "lines" not in page_data or not isinstance(page_data["lines"], list):
        page_data["lines"] = []
else:
    st.error("No JSON file selected.")
    st.stop()

PAGE_WIDTH = page_data.get("page_width", 2068)
PAGE_HEIGHT = page_data.get("page_height", 2924)

# --------------- Add New Box Mode in Sidebar ---------------
with st.sidebar:
    st.subheader("➕ Add New Box")
    if not st.session_state.adding_new_box:
        if st.button("Start Add New Box"):
            st.session_state.adding_new_box = True
            st.session_state.new_box_params = {"x": 50, "y": 50, "width": 100, "height": 100, "text": ""}
            st.rerun()
    else:
        new_box_x = st.slider("New Box: Top-left X", 0, PAGE_WIDTH, st.session_state.new_box_params["x"], key="new_box_x")
        new_box_y = st.slider("New Box: Top-left Y", 0, PAGE_HEIGHT, st.session_state.new_box_params["y"], key="new_box_y")
        new_box_width = st.slider("New Box: Width", 1, PAGE_WIDTH, st.session_state.new_box_params["width"], key="new_box_width")
        new_box_height = st.slider("New Box: Height", 1, PAGE_HEIGHT, st.session_state.new_box_params["height"], key="new_box_height")
        new_box_text = st.text_input("New Box: Text", value=st.session_state.new_box_params["text"], key="new_box_text")
        st.session_state.new_box_params.update({
            "x": new_box_x,
            "y": new_box_y,
            "width": new_box_width,
            "height": new_box_height,
            "text": new_box_text
        })
        insert_after = st.number_input("Insert new box after index (-1 for beginning)", value=-1, step=1, key="insert_after")
        
        # Process new box with Mathpix (live preview via cropping)
        if st.button("Process New Box with Mathpix", key="process_new_box"):
            image_path = os.path.join(
                image_dir,
                next((img for img in sorted(os.listdir(image_dir))
                      if extract_page_number(img) == extract_page_number(selected_json)), None)
            )
            full_image = cv2.imread(image_path)
            scale_x = full_image.shape[1] / PAGE_WIDTH
            scale_y = full_image.shape[0] / PAGE_HEIGHT
            preview_cnt = np.array([
                [new_box_x, new_box_y],
                [new_box_x, new_box_y + new_box_height],
                [new_box_x + new_box_width, new_box_y + new_box_height],
                [new_box_x + new_box_width, new_box_y]
            ], dtype=np.float32)
            preview_cnt[:, 0] *= scale_x
            preview_cnt[:, 1] *= scale_y
            preview_cnt = preview_cnt.astype(np.int32)
            x, y, w, h = cv2.boundingRect(preview_cnt)
            temp_dir = os.path.join(os.path.expanduser("~"), "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            cropped_path = os.path.join(temp_dir, "temp_new_box.png")
            cropped_img = full_image[y:y+h, x:x+w]
            if cropped_img.size == 0:
                st.error("Cropped image is empty.")
            else:
                cv2.imwrite(cropped_path, cropped_img)
                extracted_text = send_to_mathpix(cropped_path)
                st.text_area("New Box Mathpix Extracted Text", value=extracted_text, height=100, key="new_box_mpx")
        
        # Save new box.
        if st.button("Save New Box", key="save_new_box"):
            new_box_cnt = [
                [new_box_x, new_box_y],
                [new_box_x, new_box_y + new_box_height],
                [new_box_x + new_box_width, new_box_y + new_box_height],
                [new_box_x + new_box_width, new_box_y]
            ]
            new_box = {"cnt": new_box_cnt, "text": new_box_text}
            if insert_after == -1:
                page_data["lines"].insert(0, new_box)
            else:
                page_data["lines"].insert(int(insert_after) + 1, new_box)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(page_data, f, indent=4, ensure_ascii=False)
            st.success(f"New box added at index {0 if insert_after == -1 else int(insert_after)+1}.")
            st.session_state.adding_new_box = False
            st.rerun()

# --------------- Main App: Annotation & Rendering ---------------
def extract_page_number(filename):
    parts = filename.split("_page_")
    return int(parts[1].split(".")[0]) if len(parts) > 1 and parts[1].split(".")[0].isdigit() else None

raw_json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
sorted_json_files = natsorted(raw_json_files)
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

if selected_json:
    selected_page = extract_page_number(selected_json)
    matching_image = next((img for img in image_files if extract_page_number(img) == selected_page), None)
else:
    matching_image = None

if selected_json:
    json_file_path = os.path.join(json_dir, selected_json)
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
else:
    st.error("No JSON file selected.")
    st.stop()

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
        
        # Load existing annotation data if available.
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
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
                        if "cnt" in ann:
                            pts = np.array(ann["cnt"], dtype=np.float32)
                            pts[:, 0] *= scale_x
                            pts[:, 1] *= scale_y
                            pts = pts.astype(np.int32)
                            x, y, w, h = cv2.boundingRect(pts)
                            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            cv2.putText(img_copy, str(idx),
                                        (x - cv2.getTextSize(str(idx), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0][0] - 5, y+h-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 4)
                    return img_copy
                
                # Use raw image if requested, else overlay existing annotations.
                if show_raw_image:
                    display_image = image.copy()
                else:
                    display_image = draw_boxes(image, first_entry["lines"])
            else:
                display_image = image.copy()
        else:
            display_image = image.copy()
        
        # Overlay live preview of new box if in add-new-box mode.
        if st.session_state.adding_new_box:
            params = st.session_state.new_box_params
            preview_cnt = np.array([
                [params["x"], params["y"]],
                [params["x"], params["y"] + params["height"]],
                [params["x"] + params["width"], params["y"] + params["height"]],
                [params["x"] + params["width"], params["y"]]
            ], dtype=np.float32)
            # Use the same scaling as for existing boxes.
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
        
        st.image(Image.fromarray(display_image), caption=f"📌 {matching_image}", use_column_width=True)


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
                new_x = st.slider(f"X Pos {idx}", 0, page_info.get("page_width", 2068), x, key=f"x_{idx}")
                new_y = st.slider(f"Y Pos {idx}", 0, page_info.get("page_height", 2924), y, key=f"y_{idx}")
                new_w = st.slider(f"Width {idx}", 1, page_info.get("page_width", 2068) - new_x, w, key=f"w_{idx}")
                new_h = st.slider(f"Height {idx}", 1, page_info.get("page_height", 2924) - new_y, h, key=f"h_{idx}")
                new_text = st.text_area(f"Text {idx}", value=annotation["text"], key=f"text_{idx}")
                if (new_x != x or new_y != y or new_w != w or new_h != h or new_text != annotation["text"]):
                    annotation["cnt"] = [[new_x, new_y],
                                         [new_x, new_y + new_h],
                                         [new_x + new_w, new_y + new_h],
                                         [new_x + new_w, new_y]]
                    annotation["text"] = new_text
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(page_data, f, indent=4)
                    st.rerun()
                col_up, col_down, col_delete = st.columns(3)
                with col_up:
                    if st.button(f"⬆️ Move Up {idx}", key=f"up_{idx}") and idx > 0:
                        page_data["lines"][idx - 1], page_data["lines"][idx] = page_data["lines"][idx], page_data["lines"][idx - 1]
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(page_data, f, indent=4)
                        st.rerun()
                with col_down:
                    if st.button(f"⬇️ Move Down {idx}", key=f"down_{idx}") and idx < len(page_data["lines"]) - 1:
                        page_data["lines"][idx + 1], page_data["lines"][idx] = page_data["lines"][idx], page_data["lines"][idx + 1]
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(page_data, f, indent=4)
                        st.rerun()
                with col_delete:
                    if st.button(f"🗑️ Delete {idx}", key=f"delete_{idx}"):
                        del page_data["lines"][idx]
                        with open(json_path, "w", encoding="utf-8") as f:
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

with col2:
    st.subheader("📄 Rendered Output")
    if selected_json:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            if isinstance(json_data, list):
                if len(json_data) > 0 and isinstance(json_data[0], dict):
                    first_entry = json_data[0]
                else:
                    st.error("❌ Invalid JSON format: Expected a non-empty list of dictionaries.")
                    first_entry = {}
            elif isinstance(json_data, dict):
                first_entry = json_data
            else:
                st.error("❌ Invalid JSON format: Expected a dictionary or a list of dictionaries.")
                first_entry = {}
            extracted_text = "\n".join(line["text"] for line in first_entry.get("lines", []))
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
                st.error("🚨 Conversion failed! Full error message below:")
                st.text(result.stderr)
        except json.JSONDecodeError:
            st.error("❌ Failed to load JSON file.")
