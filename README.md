
# 💖 CystoBae: The Pink Scan

CystoBae is a deep learning-powered diagnostic assistant designed to detect Polycystic Ovary Syndrome (PCOS) from ultrasound images. With an aesthetic hot pink theme and a sleek Streamlit interface, CystoBae makes PCOS prediction user-friendly and accessible for clinicians, researchers, and curious minds alike.

## 🌟 Features

- 📸 Upload ultrasound images for real-time PCOS detection.
- 🧠 Choose between two trained models: `SimpleCNN` and `VGG16`.
- 🎯 Display of predictions with confidence and clear interpretation.
- 📝 Automatically saves results with timestamps in the `logs/` folder.
- 💅 Clean and themed UI built using Streamlit with a bold black & hot pink aesthetic.

## 🧠 Models Used

- **SimpleCNN**: A lightweight custom-built Convolutional Neural Network.
- **VGG16**: Transfer learning model fine-tuned for PCOS image classification.

## 🩺 Dataset

Ultrasound image dataset of ovarian scans (PCOS and Non-PCOS).  
Preprocessing includes:
- Image resizing to 100x100
- Normalization
- Data augmentation
- Gaussian noise injection for model robustness


## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/DivyanshiJ93/CystoBae.git
   cd CystoBae
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit app:
   ```bash
   streamlit run app2.py
   ```

4. Upload an ultrasound image and view the prediction results!

## 📂 Project Structure

```
CystoBae/
├── app2.py                 # Streamlit app interface
├── SimpleCNN_model.h5      # Saved SimpleCNN model
├── VGG16_model.h5          # Saved VGG16 model
├── logs/                   # Automatically saved predictions
├── requirements.txt        # List of Python dependencies
└── README.md               # This file
```

## 🔬 Related Work

- Used **VGG16 Transfer Learning** for PCOS image classification.
- Inspired by CNN-based diagnostic imaging in medical AI.
- Real-time prediction powered by **Streamlit** for interactivity.

## 📸 UI Preview

![CystoBae UI Screenshot](screenshot.png)

## ❤️ Acknowledgements

- Developed as part of an academic project focused on PCOS detection.


---

<p align="center">
  Made with 💖 by your AI BFF – <strong>CystoBae</strong>  
  <br>
  🔗 <a href="https://github.com/DivyanshiJ93/CystoBae">github.com/DivyanshiJ93/CystoBae</a>
</p>
```



