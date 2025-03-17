import streamlit as st
import json
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
AIRTABLE_API_KEY = "patnYEjjaOa6wLijy.2b24acbfb63058d86bacf5d832778a838988d2030971096a6ed251d15066e33c"
AIRTABLE_BASE_ID = "appSYBM6d3WM2c9ht"
AIRTABLE_TABLE_NAME = "Business Card Data"

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

def encode_image_to_base64(image_path):
    """Convert an image to base64 encoding."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_and_structure_data_with_vision(image_paths, api_key):
    """Uses OpenAI Vision API to extract and structure data from images directly."""
    client = OpenAI(api_key=api_key)
    structured_data_list = []
    
    for i, image_path in enumerate(image_paths):
        with st.spinner(f"Processing image {i+1}/{len(image_paths)}..."):
            try:
                # Encode image to base64
                b64_image = encode_image_to_base64(image_path)
                
                # Create the prompt for Vision API
                prompt = (
                    "Extract all text from this image (likely a business card, form, or receipt) and structure it into JSON format.\n"
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
                
                # Call the Vision API with the correct format
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert in extracting structured data from images. You will return only valid JSON."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url", 
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.3
                )
                
                structured_text = response.choices[0].message.content.strip()
                structured_text = structured_text.replace("```json", "").replace("```", "")
                structured_data = json.loads(structured_text)
                structured_data_list.append(structured_data)
                
            except Exception as e:
                structured_data_list.append({
                    "error": f"Error processing with Vision API: {str(e)}",
                    "image_path": image_path
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
            
            email = data.get("Primary Email", "")
            if email:
                email = email.lower()
            else:
                results.append({"status": "failed", "message": f"failed to add record for {email}"})
                continue    
                
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
    st.title("Vision API + GPT Data Extraction Tool")
    st.markdown("Upload images to extract and structure data using OpenAI's Vision API, and add to Airtable.")
    
    with st.sidebar:
        st.header("OpenAI API Configuration")
        openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password", 
                                     help="Your API key will not be stored.")
        if st.button("Test API Key"):
            if test_openai_key(openai_api_key):
                st.success("✅ API key is working!")
            else:
                st.error("❌ Invalid API key or API request failed.")
    
    # Initialize file uploader key in session state if not present
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    
    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )
    
    if uploaded_files:
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))
            except OSError:
                pass
        
        image_paths = []
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            image_paths.append(file_path)
        
        processing_container = st.container()
        results_container = st.container()
        
        if processing_container.button("Clear All", key="clear_button"):
            # Clear temporary directory
            for file in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except OSError:
                    pass
            # Clear session state
            keys_to_clear = ['structured_data', 'show_json', 'image_paths']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            # Increment uploader key to reset file uploader widget
            st.session_state.uploader_key += 1
            st.rerun()
        
        if processing_container.button("Process Images"):
            if not openai_api_key:
                processing_container.error("Please enter an OpenAI API key.")
            else:
                try:
                    # Process images directly with Vision API
                    structured_data = extract_and_structure_data_with_vision(image_paths, openai_api_key)
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
