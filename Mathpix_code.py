#to get the id from mathpix
!curl --location --request POST 'https://api.mathpix.com/v3/pdf' \
--header 'app_id: webtech_allen_ac_in_b6eda4_55dc4b' \
--header 'app_key: a869e65df7d85c35385bcc8ca72f8c83a5423865f702df7d56e52cf8366d1113' \
--form 'file=@"/Users/simrannaik/Desktop/11bio/bio1/bio1.pdf"' \
--form 'options_json="{\"rm_spaces\": true, \"metadata\": {\"improve_mathpix\": false}, \"auto_number_sections\": false, \"remove_section_numbering\": false, \"preserve_section_numbering\": true}"'


#to get the status of the id from mathpix
!curl --location --request GET 'https://api.mathpix.com/v3/pdf/2023_12_05_309596025b04d4d734dbg' \
--header 'app_key: a869e65df7d85c35385bcc8ca72f8c83a5423865f702df7d56e52cf8366d1113' \
--header 'app_id: webtech_allen_ac_in_b6eda4_55dc4b'

#2.To get line by line data containing geometric and contextual information from a PDF- lines.mmd.json:
!curl --location --request GET 'https://api.mathpix.com/v3/pdf/2023_12_05_309596025b04d4d734dbg.lines.mmd.json' \
--header 'app_key: a869e65df7d85c35385bcc8ca72f8c83a5423865f702df7d56e52cf8366d1113' \
--header 'app_id: webtech_allen_ac_in_b6eda4_55dc4b' >'2023_12_05_af6e20b46ce05795be4dg'.lines.mmd.json

#to get only mmd 
import requests

pdf_id = "2023_12_05_309596025b04d4d734dbg"
url = "https://api.mathpix.com/v3/pdf/" + pdf_id + ".mmd"
headers = {
    # Your headers here (if required by the API)
    'app_id': 'webtech_allen_ac_in_b6eda4_55dc4b',
    'app_key': 'a869e65df7d85c35385bcc8ca72f8c83a5423865f702df7d56e52cf8366d1113'
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    with open(pdf_id + ".mmd", "w") as f:
        f.write(response.text)
    print("File downloaded successfully as", pdf_id + ".mmd")
else:
    print("Failed to download the file. Status code:", response.status_code)

#to get only md 
import requests

pdf_id = "2023_12_05_309596025b04d4d734dbg"
url = "https://api.mathpix.com/v3/pdf/" + pdf_id + ".md"
headers = {
    # Your headers here (if required by the API)
    'app_id': 'webtech_allen_ac_in_b6eda4_55dc4b',
    'app_key': 'a869e65df7d85c35385bcc8ca72f8c83a5423865f702df7d56e52cf8366d1113'
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    with open(pdf_id + ".md", "w") as f:
        f.write(response.text)
    print("File downloaded successfully as", pdf_id + ".md")
else:
    print("Failed to download the file. Status code:", response.status_code)
