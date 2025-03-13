import streamlit as st
import requests
import pandas as pd
import time
import os  # For handling file extensions

# URL of your FastAPI server
INFERENCE_URL = "http://172.24.16.73:8000/infer"

# Configure Streamlit layout
st.set_page_config(page_title="Artwork Descriptions Generator", layout="wide")
st.title("🎨 Step 1: Generating Artwork Descriptions")

st.write("""
Upload multiple images to generate a **raw description** of each of them.
We will store only the **Artwork ID** (image name without extension) and **Description**.
""")

# File uploader for multiple images
uploaded_files = st.file_uploader(
    "📁 Upload multiple images (Ctrl/Cmd + click to select multiple)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    # Prepare an empty DataFrame with only required columns
    df = pd.DataFrame(columns=["Artwork ID", "Description"])
    total_files = len(uploaded_files)

    # Create a progress bar
    progress_bar = st.progress(0)

    for i, file in enumerate(uploaded_files):
        st.write(f"**Processing image**: {file.name}")

        # Remove the file extension from the name
        filename_no_ext = os.path.splitext(file.name)[0]

        # Convert file for HTTP POST
        files = {"file": (file.name, file.read(), file.type)}

        try:
            response = requests.post(INFERENCE_URL, files=files, timeout=300)
            if response.status_code == 200:
                description = response.json().get("description", "")
            else:
                description = f"Error: {response.text}"
        except Exception as e:
            description = f"Exception: {str(e)}"

        # Append data to DataFrame
        df.loc[len(df)] = [filename_no_ext, description]

        # Update progress
        progress_bar.progress((i + 1) / total_files)
        time.sleep(0.2)

    st.success("All images processed!")
    st.dataframe(df)

    # Save DataFrame to CSV
    csv_filename = "art_data.csv"
    df.to_csv(csv_filename, index=False)
    
    # Provide a download button for the CSV
    st.download_button(
        label="📥 Download CSV",
        data=df.to_csv(index=False),
        file_name=csv_filename,
        mime="text/csv"
    )