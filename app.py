import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# App settings
st.set_page_config(page_title="💊 Pill Identifier", layout="centered")
st.markdown("<h1 style='text-align: center; color: #E5C1CD;'>💊 Pharmaceutical Pill Identifier 💊</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-style:italic;'>Somita Chaudhari, Tiffany Brown.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-style:italic;'> Mentor: Dr. Michael Brown</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a pill image to get top-3 predictions and full medication information.</p>", unsafe_allow_html=True)
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pill_classifier_stage4_retrain.keras")

model = load_model()

# Labels
class_names = ['Decolgen', 'Fish Oil', 'Bactidol', 'Kremil S', 'Biogesic', 
               'DayZinc', 'Bioflu', 'Neozep', 'Alaxan', 'Medicol']

# Preprocess image
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = ImageOps.fit(image, (128, 128), method=Image.Resampling.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    return img_array / 255.0

# Load pill metadata
@st.cache_data
def load_pill_data():
    df = pd.read_csv("cleaned_pills_info.csv")
    df['model_label'] = df['model_label'].str.strip().str.lower()
    return df

pill_info_df = load_pill_data()

# Upload image
uploaded_file = st.file_uploader("📤 Upload pill image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.markdown("---")
    image = Image.open(uploaded_file)

    # Columns for image preview
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, caption='📷 Pill Image Uploaded Successfully', use_container_width =True)

    # Preprocess and predict
    processed = preprocess_image(image)
    predictions = model.predict(processed)[0]
    top_3_idx = predictions.argsort()[-3:][::-1]
    top_3_probs = predictions[top_3_idx]
    top_3_labels = [class_names[i] for i in top_3_idx]
    top_3_labels_cleaned = [label.lower().replace(" ", "").replace("-", "") for label in top_3_labels]

    # Top 3 predictions block
    st.markdown("### 🔍 Top 3 Predictions")
    for i in range(3):
        st.success(f"**{top_3_labels[i]}** — `{top_3_probs[i]*100:.2f}%`")

    # Fetch info for top prediction
    top_label = top_3_labels_cleaned[0]
    matched_row = pill_info_df[pill_info_df['model_label'].str.replace("-", "").str.replace(" ", "") == top_label]

    st.markdown("---")
    st.markdown("### 📋 Pill Information")

    if not matched_row.empty:
        pill_data = matched_row.iloc[0]
        info_fields = {
            "Original Pill Name": "💊 ",
            "Generic Name": "🧪 ",
            "Formulation Type": "📦 ",
            "Strength/Dosage": "💉 ",
            "Manufacturer": "🏭 ",
            "Indications / Uses": "📖 ",
            "Dosage Instructions": "📌 ",
            "Side Effects": "⚠️ ",
            "Contraindications": "🚫 ",
            "Imprint Code": "🔤 ",
            "Color and Shape": "🔵 ",
            "Warnings and Precautions": "❗"
        }

        for field, emoji in info_fields.items():
            value = pill_data.get(field, "N/A")
            st.markdown(f"**{emoji} {field}:** {value}")
    else:
        st.error("⚠️ No matching pill information found.")
