import streamlit as st
import json
import easyocr
import os
import tempfile
from PIL import Image
import base64
from openai import OpenAI
import requests
import os

# Set page config
st.set_page_config(page_title="OCR + GPT Data Extractor", layout="wide")

# Create a temp directory for uploaded files
temp_dir = tempfile.mkdtemp()

# Hardcoded Airtable credentials
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY") or st.secrets.get("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID") or st.secrets.get("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME") or st.secrets.get("AIRTABLE_TABLE_NAME")

# Headers for Airtable API
HEADERS_AIRTABLE = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json"
}

def test_openai_key(api_key):
    "Tests if the provided OpenAI API key is valid."
    if not api_key or api_key == "GPT Key":
        return False
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "This is a test message."},
                {"role": "user", "content": "Test"}
            ],
            max_tokens=5
        )
        return True
    except Exception as e:
        st.error(f"Error testing OpenAI API key: {str(e)}")
        return False

def extract_text_from_images(image_paths):
    "Extracts text from multiple images using EasyOCR."
    with st.spinner("Initializing OCR engine..."):
        reader = easyocr.Reader(['en'])
    
    extracted_texts = []
    for i, image_path in enumerate(image_paths):
        with st.spinner(f"Extracting text from image {i+1}/{len(image_paths)}..."):
            try:
                result = reader.readtext(image_path, detail=0, paragraph=True)
                extracted_texts.append("\n".join(result))
            except Exception as e:
                st.error(f"Error processing image {i+1}: {str(e)}")
                extracted_texts.append("")
    return extracted_texts

def structure_data_with_gpt(extracted_texts, api_key):
    "Uses GPT to convert extracted texts into structured JSON formats."
    client = OpenAI(api_key=api_key)
    structured_data_list = []
    
    for i, text in enumerate(extracted_texts):
        if not text.strip():
            structured_data_list.append({"error": "No text extracted from this image"})
            continue
            
        with st.spinner(f"Structuring data for image {i+1}/{len(extracted_texts)}..."):
            prompt = (
                "I want you to extract structured information from the OCR text of a document (likely a business card, form, or receipt).\n"
                "Here's the extracted text:\n"
                "```\n"
                f"{text}\n"
                "```\n"
                "Instructions:\n"
                "1. Identify what type of document this is (business card, invoice, receipt, etc.)\n"
                "2. Extract all relevant fields and values based on the document type\n"
                "3. Organize them into a structured dictionary format\n"
                "4. Use appropriate keys that describe the data (name, email, phone, address, company, job_title, etc.)\n"
                "5. For unclear or missing information, use null values\n"
                "6. Format phone numbers, addresses, and dates consistently\n"
                "7. Return ONLY the JSON data as a properly formatted Python dictionary object\n"
                "For business cards, I need the following fields (use null if not available):\n"
                "- Name\n"
                "- Company\n"
                "- Primary Email\n"
                "- Secondary Email\n"
                "- Primary Number\n"
                "- Secondary Number\n"
                "Return only valid, parseable JSON with proper quotes and escaped characters."
            )
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert in extracting structured data from unstructured text. You will return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                structured_text = response.choices[0].message.content.strip()
                structured_text = structured_text.replace("```json", "").replace("```", "")
                structured_data = json.loads(structured_text)
                structured_data_list.append(structured_data)
            except Exception as e:
                structured_data_list.append({
                    "error": f"Error processing with GPT: {str(e)}",
                    "extracted_text": text
                })
    
    return structured_data_list

def fetch_all_airtable_records():
    "Fetch all existing Airtable records for comparison."
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
    all_records = []
    params = {"pageSize": 100}
    
    while True:
        response = requests.get(url, headers=HEADERS_AIRTABLE, params=params)
        if response.status_code != 200:
            st.error(f"Failed to fetch Airtable records: {response.text}")
            return []
        data = response.json()
        all_records.extend(data.get("records", []))
        if "offset" not in data:
            break
        params["offset"] = data["offset"]
    return all_records

def add_to_airtable(structured_data_list):
    "Optimized Airtable update logic with upsert functionality."
    if not all([AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME]):
        return [{"status": "error", "message": "Airtable credentials not configured"}]
    
    existing_records = fetch_all_airtable_records()
    existing_records_dict = {
        record["fields"].get("Primary Email", "").lower(): {"id": record["id"], "fields": record["fields"]}
        for record in existing_records
    }
    
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
    results = []
    new_records = []
    update_records = []

    for i, data in enumerate(structured_data_list):
        with st.spinner(f"Processing data from image {i+1}/{len(structured_data_list)} for Airtable..."):
            if "error" in data:
                results.append({"status": "error", "message": data["error"]})
                continue
            
            email = data.get("Primary Email", "").lower()
            airtable_data = {
                "Name": data.get("Name"),
                "Company": data.get("Company"),
                "Primary Email": email,
                "Secondary Email": data.get("Secondary Email"),
                "Primary Number": data.get("Primary Number"),
                "Secondary Number": data.get("Secondary Number")
            }
            airtable_data = {k: v for k, v in airtable_data.items() if v is not None}
            
            existing_record = existing_records_dict.get(email)
            if existing_record:
                needs_update = any(
                    existing_record["fields"].get(key) != value
                    for key, value in airtable_data.items()
                )
                if needs_update:
                    update_records.append({
                        "id": existing_record["id"],
                        "fields": airtable_data
                    })
                    results.append({"status": "updated", "message": f"Updated record for {email}"})
                else:
                    results.append({"status": "skipped", "message": f"No changes needed for {email}"})
            else:
                new_records.append({"fields": airtable_data})
                results.append({"status": "new", "message": f"Added new record for {email}"})
    
    # Batch process new records
    for i in range(0, len(new_records), 10):
        batch = {"records": new_records[i:i + 10]}
        response = requests.post(url, headers=HEADERS_AIRTABLE, json=batch)
        if response.status_code not in (200, 201):
            st.error(f"Failed to add batch: {response.text}")

    # Batch process updates
    for i in range(0, len(update_records), 10):
        batch = {"records": update_records[i:i + 10]}
        response = requests.patch(url, headers=HEADERS_AIRTABLE, json=batch)
        if response.status_code != 200:
            st.error(f"Failed to update batch: {response.text}")
    
    return results

def main():
    st.title("OCR + GPT Data Extraction Tool")
    st.markdown("Upload images to extract text, structure the data using GPT, and add to Airtable.")
    
    with st.sidebar:
        st.header("OpenAI API Configuration")
        openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password", 
                                     help="Your API key will not be stored.")
        if st.button("Test API Key"):
            if test_openai_key(openai_api_key):
                st.success("✅ API key is working!")
            else:
                st.error("❌ Invalid API key or API request failed.")
    
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png", "bmp"], 
                                    accept_multiple_files=True)
    
    if uploaded_files:
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        
        image_paths = []
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            image_paths.append(file_path)
        
        st.subheader("Image Previews")
        cols = st.columns(min(4, len(uploaded_files)))
        for i, file_path in enumerate(image_paths):
            with cols[i % len(cols)]:
                img = Image.open(file_path)
                st.image(img, caption=f"Image {i+1}", width=150)
        
        processing_container = st.container()
        results_container = st.container()
        
        if processing_container.button("Process Images"):
            if not openai_api_key:
                processing_container.error("Please enter an OpenAI API key.")
            else:
                try:
                    extracted_texts = extract_text_from_images(image_paths)
                    st.session_state.extracted_texts = extracted_texts
                    
                    with processing_container.expander("Show extracted OCR text", expanded=False):
                        for i, text in enumerate(extracted_texts):
                            st.subheader(f"Image {i+1}")
                            st.text_area(f"OCR Text {i+1}", text, height=150, key=f"text_{i}")
                    
                    structured_data = structure_data_with_gpt(extracted_texts, openai_api_key)
                    st.session_state.structured_data = structured_data
                    
                    st.session_state.show_json = True
                    
                except Exception as e:
                    processing_container.error(f"An error occurred: {str(e)}")
        
        if 'structured_data' in st.session_state and st.session_state.structured_data:
            with processing_container:
                show_json = st.checkbox("Show Structured Data", value=st.session_state.get('show_json', True))
                if show_json:
                    st.subheader("Structured Data")
                    st.json(st.session_state.structured_data)
                
                json_str = json.dumps(st.session_state.structured_data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="extracted_data.json",
                    mime="application/json"
                )
            
            if results_container.button("Add to Airtable", key="airtable_button"):
                results = add_to_airtable(st.session_state.structured_data)
                results_container.subheader("Airtable Results")
                for i, result in enumerate(results):
                    if result["status"] == "new":
                        results_container.success(f"✅ Added new data from image {i+1}: {result['message']}")
                    elif result["status"] == "updated":
                        results_container.info(f"🔄 Updated data from image {i+1}: {result['message']}")
                    elif result["status"] == "skipped":
                        results_container.info(f"⏭️ Skipped image {i+1}: {result['message']}")
                    else:
                        results_container.error(f"❌ Failed for image {i+1}: {result['message']}")

if __name__ == "__main__":
    main()