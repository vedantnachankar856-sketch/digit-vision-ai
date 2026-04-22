# 🔢 Digit Vision AI — CNN Image Prediction App

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

A beautifully animated Streamlit app for handwritten digit classification using a CNN model trained with Keras.

</div>

---

## 🚀 Live Demo

Deploy on Streamlit Community Cloud — free and instant!

---

## 🧠 Model Architecture

```
Input (28×28×1)
    ↓
Conv2D → MaxPooling2D
    ↓
Conv2D → MaxPooling2D
    ↓
Flatten
    ↓
Dense(64, relu)
    ↓
Dense(10, softmax)  ← 10 digit classes (0–9)
```

---

## 📁 Project Structure

```
digit-vision-ai/
├── app.py                  # Main Streamlit app
├── prediction.pkl          # Trained Keras CNN model
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/vedantnachankar856-sketch/digit-vision-ai
cd digit-vision-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → Select this repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** ✅

---

## 🎯 Features

- 🌟 Animated dark UI with glowing effects
- ⚡ Real-time CNN prediction
- 📊 Top-5 confidence breakdown with animated bars
- 🔍 Shows preprocessed 28×28 input sent to model
- 💡 Confidence level indicator (High / Moderate / Low)
- 📱 Responsive layout

---

## 👤 Author

**Vedant Nachankar**

[![GitHub](https://img.shields.io/badge/GitHub-vedantnachankar856--sketch-181717?style=flat-square&logo=github)](https://github.com/vedantnachankar856-sketch)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Vedant_Nachankar-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/vedant-nachankar-6396783b1)
