# 💊 Deep Learning-Based Recognition of Pharmaceutical Pills for Pharmacovigilance using CNN 
--
*Somita Chaudhari, Tiffany Brown*

*Mentor: Dr. Michael Brown*

## 📖 Introduction & Importance  
Medication safety is a global challenge, with dispensing and misidentification errors causing significant patient harm and financial strain on healthcare systems. Traditional safeguards such as **barcoding** and **electronic prescribing** reduce—but do not eliminate—the risk of administering the wrong medication.  

This project introduces a **deep learning–based pill recognition framework** using **Convolutional Neural Networks (CNNs)**. By directly analyzing pill images instead of packaging, it adds a **critical layer of pharmacovigilance**, strengthening patient trust, supply chain security, and pharmacist efficiency.  

---

## 📊 Data Overview  
- **Source**: Kaggle dataset of 10 popular pharmaceutical pills & vitamins.  
- **Size**: ~10,020 images (1,002 images per class after augmentation).  
- **Image Setup**: Authentic pills on **synthetic backgrounds** to simulate real-world conditions.  
- **Preprocessing**:  
  - Resized to 128×128 pixels  
  - Pixel normalization  
  - Augmentation (rotation, flipping, shifting, zooming, brightness adjustment)  
- **Supplementary Curated Metadata**: Drug name, active ingredient, therapeutic use, linked with recognition output for richer context.  

---

## ⚙️ Methodology  
1. **Data Preparation**  
   - Image resizing, normalization, augmentation  
   - Metadata integration  

2. **Model Development**  
   - Backbone: **EfficientNetB0** (transfer learning from ImageNet)  
   - Training: 4 staged fine-tuning phases (freeze → partial unfreeze → full fine-tune → stabilization)  
   - Optimizer: Adam  
   - Loss: Categorical Cross-Entropy  
   - Callbacks: Early Stopping, Learning Rate Scheduler  

3. **Evaluation**  
   - Train/validation/test split  
   - Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix  

4. **Deployment**  
   - Streamlit web app  
   - Upload pill image → Top-3 predictions with confidence scores → Metadata retrieval  

---

## 📈 Analysis and Findings  
- **Top-1 Accuracy**: **91.2%** (correct pill ranked #1)  
- **Top-3 Accuracy**: **98.2%** (correct pill within top-3 predictions)  
- **Performance Observations**:  
  - Misclassifications occurred mainly among visually similar analgesics.  
  - Top-3 output significantly reduced risk of critical error.  
- **Visualization**: Classification reports, confusion matrix, and precision-recall metrics confirmed strong generalization.  

---

## 💡 Insights & Recommendations  
- **Operational Impact**: The model is robust enough to support **real-time pharmacist decision-making**.  
- **Scalability**: With expanded datasets, it can be adapted for **multi-country pharmaceutical markets**.  
- **Integration Potential**:  
  - EHR/EMR systems for **hospital-grade safety checks**  
  - **Mobile deployment** for patient-side verification in low-resource settings  
- **Next Steps**:  
  - Increase dataset diversity across pill shapes, imprints, and lighting conditions.  
  - Explore **multi-modal fusion** (image + text imprint recognition).  
  - Build APIs for integration with **supply chain monitoring systems**.  

👉 [Watch demo video](./app%20demo.mov)
