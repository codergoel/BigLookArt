import streamlit as st
import requests
import pandas as pd
import time
import os  # <-- Import for splitting file extensions

# URL of your FastAPI server (change <SERVER_IP> to the actual IP or domain)
INFERENCE_URL = "http://172.24.16.73:8000/infer"

# Configure Streamlit layout
st.set_page_config(page_title="GalleryGPT Descriptions", layout="wide")
st.title("🎨 GalleryGPT Bulk Description Generator")

st.write("""
Upload multiple images to generate a **raw description** from GalleryGPT. 
We will store only the Artwork ID (image name without extension) and Description for now, 
leaving other columns empty for future parsing.
""")

# Columns in our final CSV
columns = [
    "Artwork ID",
    "Description",
    "Art Style(s)",
    "Medium",
    "Keywords/Tags",
    "Dominant Colors",
    "Mood/Tone"
]

# File uploader for multiple images
uploaded_files = st.file_uploader(
    "📁 Upload multiple images (Ctrl/Cmd + click to select multiple)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    # Prepare an empty DataFrame
    df = pd.DataFrame(columns=columns)
    total_files = len(uploaded_files)

    # Create a progress bar
    progress_bar = st.progress(0)

    for i, file in enumerate(uploaded_files):
        st.write(f"**Processing image**: {file.name}")

        # Remove the file extension from the name
        filename_no_ext = os.path.splitext(file.name)[0]

        # Convert file for HTTP POST
        files = {
            "file": (file.name, file.read(), file.type)
        }

        try:
            response = requests.post(INFERENCE_URL, files=files, timeout=300)
            if response.status_code == 200:
                description = response.json().get("description", "")
                # Add a row to the DataFrame
                df.loc[len(df)] = [
                    filename_no_ext,  # Artwork ID (no extension)
                    description,      # Description
                    "",               # Art Style(s)
                    "",               # Medium
                    "",               # Keywords/Tags
                    "",               # Dominant Colors
                    ""                # Mood/Tone
                ]
            else:
                df.loc[len(df)] = [
                    filename_no_ext,
                    f"Error: {response.text}",
                    "", "", "", "", ""
                ]
        except Exception as e:
            df.loc[len(df)] = [
                filename_no_ext,
                f"Exception: {str(e)}",
                "", "", "", "", ""
            ]

        # Update progress
        progress_bar.progress((i + 1) / total_files)
        time.sleep(0.2)

    st.success("All images processed!")
    st.dataframe(df)

    # Provide a download button for the CSV
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name="gallerygpt_descriptions.csv",
        mime="text/csv"
    )
