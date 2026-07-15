<div align="center">
  <h1>🐶 AI Dog Breed Identification System</h1>
  <p><strong>A Deep Learning-powered web app for real-time dog breed identification.</strong></p>
</div>

---

# 📝 Description
A Flask-based web application that uses TensorFlow/Keras models to accurately identify dog breeds from user-uploaded images, featuring automated background retraining.

---

# 📖 Overview
Upload a dog image to instantly receive AI-based breed predictions. The system uses independently trained models for accurate classification to maximize accuracy and supports continuous learning.

---

# ✨ Features
- 🐕 Real-time dog breed prediction
- 🔄 Automated background model retraining
- 🛡️ Smart image validation (blur, blank, and duplicate detection)
- 👥 Secure User and Admin authentication
- 📊 Admin Dashboard for dataset and training management

---

# 🐾 Supported Breeds
- 🦮 Golden Retriever
- 🐕‍🦺 German Shepherd
- 🐩 Poodle
- And many more...

---

# 🛠️ Technology Stack
- **Backend:** Python, Flask
- **AI/ML:** TensorFlow, Keras (MobileNetV2), OpenCV, ImageHash
- **Frontend:** HTML5, CSS3, JavaScript

---

# 📁 Project Structure
```text
dog breed identification/
│── app.py                 # Main Flask application
│── train.py               # Background training script
│── requirements.txt       # Dependencies
│── app_data/              # Local JSON database
│── dog_datasets/          # Categorized training images
│── models/                # Trained .keras models
│── user_uploads/          # Validated user image uploads
└── templates/             # UI templates
```

---

# 📂 Dataset
Models are trained on publicly available datasets (e.g., Stanford Dogs Dataset, Kaggle). 
Structure: `Breed > Images`.

---

# 🚀 Installation
```bash
git clone https://github.com/nvnpalani/dog-breed-identification.git
cd dog-breed-identification
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
# source venv/bin/activate

pip install -r requirements.txt
python app.py
```

---

# 💻 Usage
- **User:** Register, upload a dog image, and view the breed prediction.
- **Admin:** Monitor dataset volume, manage users, and trigger model retraining.

---

# 🧠 AI Workflow
`Upload Image` ➔ `Validation` ➔ `Prediction` ➔ `Store Valid Image` ➔ `Background Retraining` ➔ `Updated Model`

---

# 🚀 Future Enhancements
- Expand support for more dog breeds and cross-breeds.
- Cloud deployment and mobile application integration.

---

# 📌 Note
> Large trained AI models (`.keras`) and original datasets are not included in this repository.

---

# ✍️ Author
**N. V. N. Palani** | [GitHub](https://github.com/nvnpalani)
