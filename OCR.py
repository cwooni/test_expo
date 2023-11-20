# import easyocr as ocr  #OCR
import streamlit as st  #Web App
from PIL import Image #Image Processing
import numpy as np #Image Processing
from numpy import asarray
import os
from google.cloud import vision_v1
import torch
torch.cuda.empty_cache()
st.set_page_config(page_title="OCR web", layout="wide")

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#title
st.title("OCR web")

#subtitle
st.markdown("## Optical Character Recognition - Extract `Text` from  `Images`")

st.markdown("")

def extract_text_from_image(image):
    # Instantiates a client
    client = vision_v1.ImageAnnotatorClient()

    # Perform OCR (Optical Character Recognition) on the image
    response = client.text_detection(image=image)

    # Process the response and extract the text
    text_annotations = response.text_annotations
    if text_annotations:
        return text_annotations[0].description
    else:
        return "No text found in the image."

def main():
    # st.title("Text Extraction from Image using Google Cloud Vision API")
    st.write("Upload an image")

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Read the image file
        content = uploaded_file.read()

        # Perform text extraction
        image = vision_v1.Image(content=content)
        extracted_text = extract_text_from_image(image)
        print(extracted_text)
        # Display the extracted text
        st.subheader("Extracted Text:")
        st.write(extracted_text)

if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"openocr-394310-95d8b763df38.json"
    main()
