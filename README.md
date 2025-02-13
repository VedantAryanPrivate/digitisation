Overview of the Streamlit Version

To automate parts of the correction process, here the  Streamlit interface  integrates PDF and JSON files from Mathpix. The interface allows users to:

View PDF with Bounding Boxes: Display PDFs with bounding boxes calculated from JSON coordinates provided by Mathpix. Users can also toggle between viewing the PDF with or without bounding boxes.

Rendered Version Side-by-Side: A side-by-side view shows the PDF along with its rendered version, allowing real-time updates and visual comparisons.

Text Correction: The sidebar displays text associated with the bounding boxes on the PDF pages, enabling easy corrections of text both in the MMD and the JSON file. Users can also send text directly to Mathpix to reprocess the OCR.

OCR for Images: Users can take screenshots of any image on the PDF and send it to Mathpix for OCR extraction.

Text Reordering: If the order of the text is incorrect, users can adjust the position of text boxes (moving them up and down).

Canvas for Bounding Box Creation: Below the PDF, a canvas allows users to draw bounding boxes on a page and extract a URL from the selected region. The canvas can be reset as needed.

Page Navigation: Users can easily navigate between PDF pages using "Next" and "Previous" buttons.

This interface significantly streamlines the process of reviewing, correcting, and refining OCR and bounding box accuracy for the dataset, enhancing both productivity and accuracy.

Key Changes

Automated Image OCR: Previously, manual adjustments were needed for images not processed by Mathpix. Now, users can draw bounding boxes on images to directly retrieve OCR URLs through the Streamlit interface.

Text Reordering: Instead of manually correcting text order in the MMD/JSON, users can now quickly adjust the order with "Move Up" and "Move Down" buttons.

Reprocess Text via Screenshot: Text in bounding boxes can be reprocessed by taking a screenshot (automatically saved to Downloads) and sending it directly to Mathpix for OCR.

Seamless Bounding Box Adjustments: Bounding box dimensions and coordinates (width, height, top-left X/Y) can be easily modified and sent back to Mathpix for re-OCR.

Real-time Updates: Changes are automatically reflected in both the JSON and MMD files, ensuring synchronization.
