import subprocess
import requests
import streamlit as st
import os
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from streamlit_drawable_canvas import st_canvas # type: ignore
import glob

# Set Streamlit to wide layout
st.set_page_config(layout="wide")
st.title("📄 PDF Image Annotation Viewer & JSON Renderer")

# Constants for reference page dimensions
PAGE_WIDTH = 2068
PAGE_HEIGHT = 2924
# Mathpix API Credentials
MATHPIX_APP_ID = "webtech_allen_ac_in_b6eda4_55dc4b"
MATHPIX_APP_KEY = "a869e65df7d85c35385bcc8ca72f8c83a5423865f702df7d56e52cf8366d1113"

# Function to send image to Mathpix API
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
    
# Function to get the most recent image from Downloads
def get_latest_downloaded_image():
    downloads_folder = os.path.expanduser("~/Downloads")
    image_files = sorted(glob.glob(os.path.join(downloads_folder, "*.[pj][np]g")), key=os.path.getctime, reverse=True)

    if image_files:
        return image_files[0]  # Return the most recent image
    else:
        return None  # No image found

# ------------------- Sidebar Inputs -------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    json_dir = st.text_input("📂 JSON Directory", value="/Users/simrannaik/Desktop/automated/json")
    image_dir = st.text_input("🖼 Image Directory", value="/Users/simrannaik/Desktop/automated/images")

    if json_dir and image_dir:
        json_dir_path = Path(json_dir)
        image_dir_path = Path(image_dir)

        if not json_dir_path.exists() or not json_dir_path.is_dir():
            st.error("🚨 JSON Directory does not exist.")
        elif not image_dir_path.exists() or not image_dir_path.is_dir():
            st.error("🚨 Image Directory does not exist.")
        else:
            json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

            if not json_files:
                st.error("⚠️ No JSON files found.")
            else:
                if 'json_files' not in st.session_state:
                    st.session_state.json_files = json_files
                    st.session_state.current_json_idx = 0

                st.subheader("📑 Navigation Controls")
                col_nav_prev, col_nav_next = st.columns(2)

                with col_nav_prev:
                    if st.button("⏮️ Previous") and st.session_state.current_json_idx > 0:
                        st.session_state.current_json_idx -= 1
                        st.session_state.canvas_data = None  # Reset canvas when switching pages
                        st.experimental_rerun()

                with col_nav_next:
                    if st.button("⏭️ Next") and st.session_state.current_json_idx < len(st.session_state.json_files) - 1:
                        st.session_state.current_json_idx += 1
                        st.session_state.canvas_data = None  # Reset canvas when switching pages
                        st.experimental_rerun()

                selected_json = st.selectbox("📜 Select JSON File", options=st.session_state.json_files, index=st.session_state.current_json_idx)
                st.session_state.current_json_idx = st.session_state.json_files.index(selected_json)

# Ensure valid directory inputs before proceeding
if not json_dir or not image_dir:
    st.warning("⚠️ Please enter valid JSON and image directory paths.")
    st.stop()

# Extract page number from filename
def extract_page_number(filename):
    parts = filename.split("_page_")
    return int(parts[1].split(".")[0]) if len(parts) > 1 and parts[1].split(".")[0].isdigit() else None

# Get available images
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

# Automatically select corresponding image based on JSON
image_page = extract_page_number(selected_json)
matching_image = next((img for img in image_files if extract_page_number(img) == image_page), None)

# Create two-column layout
col1, col2 = st.columns(2)
# ----------------------------------------------
# SIDEBAR: Additional Controls
# ----------------------------------------------
with st.sidebar:
    st.subheader("🖼 Image Display Options")
    
    # Checkbox to toggle bounding boxes
    show_raw_image = st.checkbox("Show Image Without Bounding Boxes", value=False)

# ------------------------------
# COLUMN 1: Image with Bounding Boxes
# ------------------------------
with col1:
    st.subheader("🖼 Image with Bounding Boxes")

    if matching_image:
        json_path = os.path.join(json_dir, selected_json)
        image_path = os.path.join(image_dir, matching_image)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                json_data = json.load(f)

            if isinstance(json_data, dict):
                json_data = [json_data]

            first_entry = json_data[0] if json_data else None

            if first_entry and 'lines' in first_entry:
                json_width = first_entry.get("page_width", image.shape[1])
                json_height = first_entry.get("page_height", image.shape[0])

                def draw_boxes(image, annotations):
                    img_copy = image.copy()
                    scale_x = img_copy.shape[1] / json_width
                    scale_y = img_copy.shape[0] / json_height

                    for idx, annotation in enumerate(annotations):
                        if 'cnt' in annotation:
                            points = np.array(annotation['cnt'], dtype=np.float32)
                            points[:, 0] *= scale_x
                            points[:, 1] *= scale_y
                            points = points.astype(np.int32)

                            x, y, w, h = cv2.boundingRect(points)
                            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.putText(img_copy, str(idx), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    return img_copy

                if show_raw_image:
                    st.image(Image.fromarray(image), caption=f"📌 Raw {matching_image}", use_column_width=True)
                else:
                    image = draw_boxes(image, first_entry['lines'])
                    st.image(Image.fromarray(image), caption=f"📌 Annotated {matching_image}", use_column_width=True)

# ------------------------
# 🎨 DRAWING CANVAS
# ------------------------
st.subheader("🎨 Draw Bounding Box")

# Initialize canvas data if not set
if 'canvas_data' not in st.session_state:
    st.session_state.canvas_data = None

# Create a dynamic key for the canvas to ensure it resets properly
canvas_key = f"canvas_{st.session_state.current_json_idx}"

# Button to reset the canvas
col_reset, col_space = st.columns([1, 3])
with col_reset:
    if st.button("🔄 Reset Canvas"):
        st.session_state.canvas_data = None  # Clear stored canvas data
        st.session_state[f"reset_key_{st.session_state.current_json_idx}"] = str(os.urandom(8))  # Generate new unique key
        st.experimental_rerun()  # Refresh UI to clear bounding boxes

# Draw on the canvas
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=2,
    stroke_color="#FF0000",
    background_image=Image.open(image_path),
    update_streamlit=True,
    height=500,
    width=500,
    drawing_mode="rect",
    key=st.session_state.get(f"reset_key_{st.session_state.current_json_idx}", canvas_key),  # Use dynamic reset key
)

# Store drawn objects in session state
if canvas_result.json_data is not None:
    st.session_state.canvas_data = canvas_result.json_data

# Extract bounding box coordinates
if st.session_state.canvas_data and "objects" in st.session_state.canvas_data:
    objects = st.session_state.canvas_data["objects"]
    if objects:
        img_width = 500
        img_height = 500
        for obj in objects:
            left, top, width, height = obj["left"], obj["top"], obj["width"], obj["height"]

            scaled_x = int((left / img_width) * json_width)
            scaled_y = int((top / img_height) * json_height)
            scaled_w = int((width / img_width) * json_width)
            scaled_h = int((height / img_height) * json_height)

            # Format the URL based on extracted image_id
            image_id = first_entry.get("image_id", "unknown_image_id")
            url = f"https://cdn.mathpix.com/cropped/{image_id}.jpg?height={scaled_h}&width={scaled_w}&top_left_y={scaled_y}&top_left_x={scaled_x}"

            st.write(f"Bounding Box URL: {url}")

# ----------------------------------------------
# SIDEBAR: Manage Bounding Boxes
# ----------------------------------------------
full_image = cv2.imread(image_path)
# Ensure temp directory exists before cropping images
temp_mmd_dir = os.path.join(os.path.expanduser("~"), "Desktop/automated/temp_mmd")
os.makedirs(temp_mmd_dir, exist_ok=True)

with st.sidebar:
    st.subheader("📦 Manage Annotations")
    if first_entry and "lines" in first_entry:
        for idx, annotation in enumerate(first_entry["lines"]):
            with st.expander(f"Box {idx}", expanded=False):
                points = np.array(annotation["cnt"], dtype=np.int32)
                x, y, w, h = cv2.boundingRect(points)

                # Sliders for adjusting bounding box properties
                new_x = st.slider(f"X Pos {idx}", 0, json_width, x, key=f"x_{idx}")
                new_y = st.slider(f"Y Pos {idx}", 0, json_height, y, key=f"y_{idx}")
                new_w = st.slider(f"Width {idx}", 1, json_width - new_x, w, key=f"w_{idx}")
                new_h = st.slider(f"Height {idx}", 1, json_height - new_y, h, key=f"h_{idx}")
                new_text = st.text_area(f"Text {idx}", value=annotation["text"], key=f"text_{idx}")

                # Update bounding box when sliders change
                if (new_x != x or new_y != y or new_w != w or new_h != h or new_text != annotation["text"]):
                    annotation["cnt"] = [[new_x, new_y], [new_x, new_y + new_h], [new_x + new_w, new_y + new_h], [new_x + new_w, new_y]]
                    annotation["text"] = new_text
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(json_data, f, indent=4)
                    st.experimental_rerun()

                col_up, col_down, col_delete = st.columns(3)

                with col_up:
                    if st.button(f"⬆️ Move Up {idx}", key=f"up_{idx}") and idx > 0:
                        first_entry['lines'][idx - 1], first_entry['lines'][idx] = first_entry['lines'][idx], first_entry['lines'][idx - 1]
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, indent=4)
                        st.experimental_rerun()

                with col_down:
                    if st.button(f"⬇️ Move Down {idx}", key=f"down_{idx}") and idx < len(first_entry['lines']) - 1:
                        first_entry['lines'][idx + 1], first_entry['lines'][idx] = first_entry['lines'][idx], first_entry['lines'][idx + 1]
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, indent=4)
                        st.experimental_rerun()

                with col_delete:
                    if st.button(f"🗑️ Delete {idx}", key=f"delete_{idx}"):
                        del first_entry["lines"][idx]
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, indent=4)
                        st.experimental_rerun()

                # Mathpix Checkbox
                send_to_mathpix_flag = st.checkbox(f"📤 Send to Mathpix {idx}", key=f"mathpix_{idx}")

                # New Checkbox for sending downloaded image
                send_from_downloads_flag = st.checkbox(f"📥 Send Latest Downloaded Image {idx} to Mathpix", key=f"downloads_{idx}")

                if send_to_mathpix_flag:
                    cropped_path = os.path.join(temp_mmd_dir, f"cropped_{idx}.png")

                    # Ensure bounding box is valid
                    scale_x = full_image.shape[1] / json_width
                    scale_y = full_image.shape[0] / json_height
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

                    

                    # Button to Process with Mathpix
                    if st.button(f"🚀 Process Box {idx} with Mathpix", key=f"mathpix_button_{idx}"):
                        with st.spinner("Processing with Mathpix..."):
                            extracted_text = send_to_mathpix(cropped_path)

                        if "Error" in extracted_text:
                            st.error(f"⚠️ Mathpix API Error: {extracted_text}")
                        else:
                            st.text_area(f"Mathpix Extracted Text {idx}", value=extracted_text, height=100, key=f"mathpix_text_{idx}")
                            # Process the Downloaded Image
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
# ----------------------------------------------
# COLUMN 2: Rendered JSON as HTML
# ----------------------------------------------
with col2:
    st.subheader("📄 Rendered Output")

    if selected_json:
        json_path = os.path.join(json_dir, selected_json)

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Handle both list and dictionary formats
            if isinstance(json_data, list):
                if len(json_data) > 0 and isinstance(json_data[0], dict):
                    first_entry = json_data[0]
                else:
                    st.error("❌ Invalid JSON format: Expected a non-empty list of dictionaries.")
                    first_entry = {}
            elif isinstance(json_data, dict):
                first_entry = json_data  # ✅ Handle single dictionary case
            else:
                st.error("❌ Invalid JSON format: Expected a dictionary or a list of dictionaries.")
                first_entry = {}

            # Extract text from JSON `lines` key
            extracted_text = "\n".join(line["text"] for line in first_entry.get("lines", []))

            # Ensure the temp directories exist
            temp_mmd_dir = "/Users/simrannaik/Desktop/automated/temp_mmd"
            temp_html_dir = "/Users/simrannaik/Desktop/automated/temp_html"
            os.makedirs(temp_mmd_dir, exist_ok=True)
            os.makedirs(temp_html_dir, exist_ok=True)
            

            # Generate unique temp filenames based on selected JSON
            base_name = Path(selected_json).stem  # Extracts 'FJN_Decimals_DPP_page_1' from filename
            temp_mmd_path = os.path.join(temp_mmd_dir, f"{base_name}.mmd")  # Store in temporary directory
            temp_html_path = os.path.join(temp_html_dir, f"{base_name}.html")

            # Save extracted text to `.mmd`
            with open(temp_mmd_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)

            # Convert Markdown to HTML using `mpx`
            result = subprocess.run(["mpx", "convert", temp_mmd_path, temp_html_path], capture_output=True, text=True)

            if result.returncode == 0:
                # Read and display the converted HTML
                with open(temp_html_path, "r", encoding="utf-8") as html_file:
                    html_content = html_file.read()
                st.components.v1.html(html_content, height=700, scrolling=True)
            else:
                st.error("🚨 Conversion failed! Full error message below:")
                st.text(result.stderr)

        except json.JSONDecodeError:
            st.error("❌ Failed to load JSON file.")
