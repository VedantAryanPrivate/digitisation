#breaking pdf pages into images
import fitz
import os

# Paths
pdf_path = "/Users/simrannaik/Desktop/automated/FJN Decimals DPP.pdf"
output_dir = "/Users/simrannaik/Desktop/automated/images"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Open the PDF file
doc = fitz.open(pdf_path)

# Loop through each page and save it as a PNG image
for page_num in range(len(doc)):
    page = doc[page_num]
    pix = page.get_pixmap(dpi=300)  # Increase DPI for better quality
    output_path = os.path.join(output_dir, f"FJN Decimals DPP_page_{page_num + 1}.png")
    pix.save(output_path)

print(f"PDF split into {len(doc)} images and saved in {output_dir}")




#breaking json into json pages
import json
import os

# Input and output paths
json_path = "/Users/simrannaik/Desktop/automated/FJN Decimals DPP.lines.mmd.json"
output_dir = "/Users/simrannaik/Desktop/automated/json"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the JSON file
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Process and split pages
for entry in data["pages"]:
    page_number = entry["page"]
    
    # Structure the output JSON for each page
    page_data = {
        "image_id": entry["image_id"],
        "page": page_number,
        "lines": entry["lines"],
        "page_height": entry.get("page_height", None),  # Keep original value if available
        "page_width": entry.get("page_width", None),
        "languages_detected": entry.get("languages_detected", [])
    }

    # Save the JSON for the page
    output_path = os.path.join(output_dir, f"FJN Decimals DPP_page_{page_number}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(page_data, f, indent=4, ensure_ascii=False)

print(f"JSON split into {len(data['pages'])} pages and saved in {output_dir}")
