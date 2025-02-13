🚀 Streamlit Interface for PDF & JSON Correction 📑

Welcome to the Streamlit Interface that automates and enhances the process of reviewing, correcting, and refining OCR data for Mathpix-generated PDFs and JSON files. This interface provides powerful tools to improve the efficiency and accuracy of OCR corrections.

🌟 Features
The Streamlit interface seamlessly integrates PDF and JSON files from Mathpix, providing the following capabilities:

📦 View PDF with Bounding Boxes:
Display PDFs with bounding boxes based on JSON coordinates. Toggle between the PDF with or without bounding boxes for better clarity.

👨‍💻 Rendered Version Side-by-Side:
Compare the original PDF with the rendered version side by side, enabling real-time updates and visual comparisons.

✍️ Text Correction:
Easily correct text associated with bounding boxes on the PDF pages via the sidebar. Modify the text in both the MMD and JSON files. You can also send text directly to Mathpix for reprocessing.

🖼️ OCR for Images:
Take screenshots of images within the PDF and directly send them to Mathpix for OCR extraction.

🔄 Text Reordering:
Correct any text order issues by simply moving text boxes up or down with the "Move Up" / "Move Down" buttons.

🎨 Canvas for Bounding Box Creation:
Below the PDF, draw bounding boxes on a canvas and extract a URL from the selected region. Reset the canvas at any time.

📄 Page Navigation:
Seamlessly navigate through PDF pages using the Next and Previous buttons for a smooth user experience.

🔧 Key Changes
🖼️ Automated Image OCR:
Previously, images not processed by Mathpix required manual adjustments. Now, users can draw bounding boxes directly on images to retrieve OCR URLs without hassle.

🔢 Text Reordering:
Instead of manually correcting text order in MMD/JSON, simply click "Move Up" and "Move Down" to adjust the order instantly.

📸 Reprocess Text via Screenshot:
If text in the bounding box is not accurate, take a screenshot (automatically saved to Downloads) and send it directly to Mathpix for re-OCR.

⚙️ Seamless Bounding Box Adjustments:
Easily adjust bounding box dimensions (width, height, top-left X/Y) and automatically resend the updated data to Mathpix for re-OCR.

⏱️ Real-time Updates:
All changes are reflected live in both the JSON and MMD files, ensuring everything stays synchronized.

                                                                                                                                 📥 How to Use
Clone the repository
git clone https://github.com/SimNaik/digitisation.git
Install dependencies
Run:
pip install -r requirements.txt
Launch the Streamlit app
Run:
streamlit run streamlit_interface.py

                                                                                                                               🚀 Why Use This Interface?
This interface automates and simplifies the process of:

Reviewing and correcting OCR data,
Adjusting bounding boxes,
Reordering text seamlessly,
Sending images and text back to Mathpix for reprocessing.
All these features work in real time, boosting both accuracy and efficiency in dataset correction!

✨ Feel free to contribute or open issues if you have feedback or improvements! ✨
