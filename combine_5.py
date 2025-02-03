import streamlit as st
import os
import json
import cv2
import numpy as np
from PIL import Image
import subprocess
from pathlib import Path
from streamlit_drawable_canvas import st_canvas

# Set Streamlit to wide layout
st.set_page_config(layout="wide")
st.title("📄 PDF Image Annotation Viewer & JSON Renderer")

# Constants for reference page dimensions
PAGE_WIDTH = 2068
PAGE_HEIGHT = 2924
MAX_DISPLAY_HEIGHT = 800  # Limit to prevent UI cutoff

# ------------------- Sidebar Inputs -------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Input for JSON directory
    json_dir = st.text_input(
        "📂 JSON Directory",
        value="/Users/simrannaik/Desktop/automated/json",
        help="Enter the directory path containing JSON annotation files."
    )

    # Input for Image directory
    image_dir = st.text_input(
        "🖼 Image Directory",
        value="/Users/simrannaik/Desktop/automated/images",
        help="Enter the directory path containing image files."
    )

    # Checkbox for showing URLs
    require_url = st.checkbox("Require URL for Bounding Box", value=False)

    # Validate directories
    if json_dir and image_dir:
        json_dir_path = Path(json_dir)
        image_dir_path = Path(image_dir)

        if not json_dir_path.exists() or not json_dir_path.is_dir():
            st.error("🚨 JSON Directory does not exist or is not a directory.")
        elif not image_dir_path.exists() or not image_dir_path.is_dir():
            st.error("🚨 Image Directory does not exist or is not a directory.")
        else:
            # Get list of available JSON files
            json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

            if not json_files:
                st.error("⚠️ No JSON files found in the specified directory.")
            else:
                # Store JSON files in session state
                if 'json_files' not in st.session_state:
                    st.session_state.json_files = json_files
                    st.session_state.current_json_idx = 0

                # Navigation Controls
                st.subheader("📑 Navigation Controls")
                col_nav_prev, col_nav_next = st.columns(2)

                with col_nav_prev:
                    if st.button("⏮️ Previous"):
                        if st.session_state.current_json_idx > 0:
                            st.session_state.current_json_idx -= 1
                            st.experimental_rerun()
                        else:
                            st.warning("🚨 This is the first file.")

                with col_nav_next:
                    if st.button("⏭️ Next"):
                        if st.session_state.current_json_idx < len(st.session_state.json_files) - 1:
                            st.session_state.current_json_idx += 1
                            st.experimental_rerun()
                        else:
                            st.warning("🚨 This is the last file.")

                # Dropdown to select a specific JSON file
                selected_json = st.selectbox(
                    "📜 Select JSON File",
                    options=st.session_state.json_files,
                    index=st.session_state.current_json_idx,
                )
                st.session_state.current_json_idx = st.session_state.json_files.index(selected_json)

# Ensure valid directory inputs before proceeding
if not json_dir or not image_dir:
    st.warning("⚠️ Please enter valid JSON and image directory paths.")
    st.stop()

# Extract page number from filename
def extract_page_number(filename):
    parts = filename.split("_page_")
    if len(parts) > 1 and parts[1].split(".")[0].isdigit():
        return int(parts[1].split(".")[0])
    return None

# Get available images
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

# Automatically select the corresponding image based on JSON
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

# ----------------------------------------------
# COLUMN 1: Image with Bounding Boxes
# ----------------------------------------------
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

                # Extract image ID from JSON (assuming image_id is available in the JSON)
                image_id = first_entry["image_id"] if "image_id" in first_entry else "unknown_image_id"

                # Canvas for drawing bounding boxes
                st.subheader("🎨 Draw Bounding Box")
                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 0, 0)",  # Transparent background
                    stroke_width=2,
                    stroke_color="#FF0000",  # Red stroke for bounding box
                    background_image=Image.open(image_path),  # Use selected image as background
                    update_streamlit=True,
                    height=500,  # Height of canvas
                    width=700,  # Width of canvas
                    drawing_mode="rect",  # Rectangle mode for bounding boxes
                    key="canvas",
                )

                # Extract drawn bounding box coordinates and scale to page dimensions
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    if objects:
                        img_width = 700  # Canvas width
                        img_height = 500  # Canvas height
                        for obj in objects:
                            left, top, width, height = obj["left"], obj["top"], obj["width"], obj["height"]
                            
                            # Convert coordinates relative to page width and height
                            scaled_x = int((left / img_width) * json_width)
                            scaled_y = int((top / img_height) * json_height)
                            scaled_w = int((width / img_width) * json_width)
                            scaled_h = int((height / img_height) * json_height)

                            # Format the URL based on the extracted image_id
                            url = f"https://cdn.mathpix.com/cropped/{image_id}.jpg?height={scaled_h}&width={scaled_w}&top_left_y={scaled_y}&top_left_x={scaled_x}"
                            
                            # Display the URL without rendering the image
                            st.write(f"Bounding Box URL: {url}")

# ----------------------------------------------
# SIDEBAR: Manage Bounding Boxes
# ----------------------------------------------
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

            # Generate unique temp filenames based on selected JSON
            base_name = Path(selected_json).stem  # Extracts 'FJN_Decimals_DPP_page_1' from filename
            temp_mmd_path = f"/tmp/{base_name}.mmd"  # Store in temporary directory
            temp_html_path = f"/tmp/{base_name}.html"

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
