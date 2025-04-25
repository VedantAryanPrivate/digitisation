import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Import tqdm for progress bar

# Set the root folder path here
FOLDER_PATH = "/Users/mehaknawab/Desktop/ADPL/ADPL_PNCF_NCERT/Class 10"  
# Replace this with the actual path

APP_ID = 'webtech_allen_ac_in_b6eda4_55dc4b'
APP_KEY = 'a869e65df7d85c35385bcc8ca72f8c83a5423865f702df7d56e52cf8366d1113'


def process_pdf(args):
    pdf_file, pbar = args
    url = 'https://api.mathpix.com/v3/pdf'
    headers = {
        'app_id': APP_ID,
        'app_key': APP_KEY
    }
    files = {'file': open(pdf_file, 'rb')}
    data = {
        'options_json': '{"rm_spaces": true, "metadata": {"improve_mathpix": false}, "auto_number_sections": false, "remove_section_numbering": false, "preserve_section_numbering": true}'
    }
    response = requests.post(url, headers=headers, files=files, data=data)
    response_data = response.json()
    pdf_id = response_data.get('pdf_id')
    
    # Polling for completion
    url = f'https://api.mathpix.com/v3/pdf/{pdf_id}'
    status = None
    while status != "completed":
        response = requests.get(url, headers=headers)
        response_data = response.json()
        status = response_data.get('status')
    
    # Save .mmd file
    output_location = os.path.join(os.path.dirname(pdf_file), f"{os.path.splitext(os.path.basename(pdf_file))[0]}.mmd")
    url = f'https://api.mathpix.com/v3/pdf/{pdf_id}.mmd'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(output_location, 'wb') as f:
            f.write(response.content)
        print("Data saved successfully to:", output_location)
    else:
        print("Error:", response.text)
    
    # Fetch and save .lines.mmd.json file
    lines_url = f'https://api.mathpix.com/v3/pdf/{pdf_id}.lines.mmd.json'
    response = requests.get(lines_url, headers=headers)
    lines_output_location = os.path.join(os.path.dirname(pdf_file), f"{os.path.splitext(os.path.basename(pdf_file))[0]}.lines.mmd.json")
    if response.status_code == 200:
        with open(lines_output_location, 'wb') as f:
            f.write(response.content)
        print("Lines data saved successfully to:", lines_output_location)
    else:
        print("Error fetching lines data:", response.text)
    
    pbar.update(1)  # Update progress bar


def get_all_pdfs(root_folder):
    pdf_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(dirpath, filename))
    return pdf_files


if __name__ == "__main__":
    pdf_files = get_all_pdfs(FOLDER_PATH)
    
    # Initialize tqdm progress bar
    with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
        # Use ThreadPoolExecutor with tqdm progress bar
        with ThreadPoolExecutor(max_workers=20) as executor:  
            executor.map(process_pdf, ((pdf_file, pbar) for pdf_file in pdf_files))
