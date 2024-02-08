
import numpy as np
import cv2
from tqdm.notebook import tqdm
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
import base64
import io

animalmodel = load_model('mobileanimalmodel.h5')

animal_mapping_hrv = {
    0: 'Krava',
    1: 'Pauk',
    2: 'Vjeverica',
    3: 'Konj',
    4: 'Leptir',
    5: 'Slon',
    6: 'Kokos',
    7: 'Ovca',
    8: 'Pas',
    9: 'Macka'
}

animal_mapping_eng = {
    0: 'Cow',
    1: 'Butterfly', 
    2: 'Dog', 
    3: 'Cat', 
    4: 'Horse', 
    5: 'Squirrel',
    6: 'Elephant',
    7: 'Sheep',
    8: 'Spider',
    9: 'Chicken'
}

def extract_features_single_image(image, height, width):
    var_img = image.convert('L').resize((height, width), Image.Resampling.LANCZOS)
    var_img = np.array(var_img)
    var_img = var_img / 255.0
    var_img = var_img.reshape((1, height, width, 1)).astype(np.float32)
    return var_img

def predict_animal(image, animalmodel, height=128, width=128):
    preprocessed_image = extract_features_single_image(image, height, width)
    if preprocessed_image is not None:
        animal_pred = animalmodel.predict(preprocessed_image)
        return animal_pred
    else:
        print("Could not extract features from the image.")
        return None

st.markdown(
    f"""
    <style>
    
    h1{{
        color: #000000;
        padding-bottom: 3rem;
        font-size: 60px;
    }}
    [data-testid="{"stApp"}"]{{
        flex: 1;
        padding: 20px;
        background: linear-gradient(rgba(255,255,255,.5), rgba(255,255,255,.2)), url('https://cdn.pixabay.com/photo/2024/01/08/15/54/defile-8495836_1280.jpg');
        background-size: cover;
        background-repeat: no-repeat;
    }}
    .st-emotion-cache-1y4p8pa{{
        padding-top: 3rem;
    }}
    [data-testid="{"stFileUploader"}"] {{
        background-color: #000000;
        color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.1);
    }}
    
    .uploadedFileName {{
        font-weight: bold;  
        color: #ffffff;
    }}
    .uploadedFileData{{
        width: 22.5rem;
    }}
    .st-emotion-cache-1lp7pgu {{
        justify-content: flex-end;  
    }}
    .row-widget{{
        align-items: center !important;
    }}
    .stButton {{
        position: fixed;
        top: 3.5rem;
        right: 3rem;
        margin-right: -38.5rem;
    }}
    [kind="{"secondary"}"]{{
        width: 8rem !important;
        background-color: #000000;
    }}
    p {{
        font-size: 1rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Animal Prediction App")

uploaded_files = st.file_uploader("Choose multiple photos", type=["jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    col_count = 2
    image_columns = st.columns(col_count)
    predictions = []
    
    st.markdown('<div style="display: flex; flex-wrap: wrap; justify-content: center;">', unsafe_allow_html=True)
    
    for i, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        resized_image = image.resize((180, 180))
        image_bytes = io.BytesIO()
        resized_image.save(image_bytes, format='JPEG')
        image_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
        

        image_columns[i % col_count].markdown(
            f"""
            <style>
                .st-emotion-cache-13itxba{{
                    width: 25rem;
                }}
                .custom-figure {{
                    background: #ffffff;
                    display: inline-block;
                    margin: 2% auto 25%;
                    margin-left: 4rem;
                    padding: 3% 3% 3%;
                    text-align: center;
                    text-decoration: none;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, .3);
                    transition: all .20s linear;
                    cursor: pointer;
                    border: 3px solid #000000;
                    border-radius: 10px;
                }}

                .custom-figure-caption {{
                    color: #333;
                    font-size: 100%;
                    padding-top: 12px;
                }}

                .custom-figure img {{
                    display: block;
                    width: 180px;
                }}

                .custom-figure:hover {{
                    transform: scale(1.5);
                    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.7);
                }}
            </style>
            <div class="custom-figure">
                <img src="data:image/png;base64,{image_base64}" alt="Uploaded Image"/>
                <figcaption class="custom-figure-caption">Prediction: <span id="prediction_{i}"></span></figcaption>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Predict"):
        for i, uploaded_file in enumerate(uploaded_files):
            uploaded_image = Image.open(uploaded_file)
            
            animal_pred = predict_animal(uploaded_image, animalmodel)
            animal = animal_mapping_eng.get(np.argmax(animal_pred), 'Unknown')

            if animal_pred is not None:
                st.markdown(f"<style>#prediction_{i}::after{{content:' {animal}';}}</style>", unsafe_allow_html=True)
            else:
                st.markdown(f"<style>#prediction_{i}::after{{content:' Greska.';}}</style>", unsafe_allow_html=True)
                
    if predictions:
        st.success("\n".join(predictions))
        
        st.write("\n---\n")