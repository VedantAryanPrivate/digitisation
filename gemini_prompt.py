import base64
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
from google.cloud import vision_v1 as vision
import io

# Initialize Vertex AI and Google Cloud Vision API
def generate(prompt):
    vertexai.init(project="llm-sandbox-426711", location="us-central1")
    model = GenerativeModel("gemini-1.5-flash-002")

    # Call the generate_content method with the text prompt
    responses = model.generate_content(
        [prompt],  # Pass the text as input
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    # Output the model response
    for response in responses:
        print(response.text, end="")

# Perform OCR on an image using Google Cloud Vision API
def perform_ocr(image_path):
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    # Use vision_v1.types to create the Image object
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}')

    # Return the extracted text from the image
    if texts:
        return texts[0].description
    else:
        return "No text detected in the image."

# Configuration for the generative model
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

# Safety settings for the model
safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF  # Turn off safety block
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

# Get user input for prompt and optional image
user_prompt = input("Enter your prompt: ")
image_path = input("Enter the image file path (or leave blank if none): ")

# If an image path is provided, perform OCR and use the extracted text as the prompt
if image_path.strip():
    extracted_text = perform_ocr(image_path)
    print(f"Extracted text from image: {extracted_text}")
    generate(extracted_text)
else:
    generate(user_prompt)
