# PromptClass — Text-Guided Image Classification using Vision-Language Models (CLIP)

A web-based image classification system that uses **Zero-Shot Learning** with **OpenAI CLIP**.  
Users can upload an image, type a natural-language prompt such as:

This is an image of a {class_name}.


Add multiple class names and the system predicts the best-matching class using **cosine similarity** between image and text embeddings.

---

## 🚀 Features

### 🔍 Zero-Shot Image Classification
No training required — CLIP understands relationships between text and images out of the box.

### 🖼️ Interactive Web UI
- Upload any image  
- Add class names dynamically  
- Customize the prompt template  
- Classify instantly  
- View similarity scores for each class  

### ⚙️ FastAPI Backend
Handles:
- Image upload  
- Prompt construction  
- CLIP inference  
- Scoring & prediction  

### 🎨 Fully Customizable
Change prompts, model, or UI easily.

---

## 🧠 How It Works

Frontend (HTML/CSS/JS)
│ (image + class names + prompt)
▼
FastAPI Backend
│
▼
CLIP Model (HuggingFace)
│
▼
Cosine Similarity Scores
│
▼
Predicted Class

---

## 📂 Folder Structure

promptclass-app/
├── backend/
│ ├── main.py
│ └── model.py
├── frontend/
│ ├── index.html
│ ├── styles.css
│ └── app.js
├── requirements.txt
└── README.md

---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository

git clone https://github.com/anoynomous00/ImageClassification.git

cd ImageClassification/promptclass-app

2️⃣ Create Virtual Environment

python -m venv .venv

3️⃣ Activate It

Windows

.\.venv\Scripts\Activate.ps1

4️⃣ Install Requirements

pip install -r requirements.txt

▶️ Running the Project

Start Backend

uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

🌐Open UI in Browser

http://localhost:8000

#### 📸 Example

Prompt template:
This is an image of a {class_name}.

Classes: cat, dog

Output:

cat → 0.92
dog → 0.13

Predicted: cat


### 🧪 API Documentation
#### POST /api/classify

###### Field | Type	| Description

file | image | Upload an image file

classes	| JSON | e.g. ["cat","dog","car"]

template | string	| Must contain {class_name}

Response:

{

  "classes": {
  
    "cat": { "cosine": 0.92, "scaled": 0.96 },
    
    "dog": { "cosine": 0.13, "scaled": 0.56 }
    
  },
  
  "best_class": "cat",
  
  "best_prompt": "This is an image of a cat.",
  
  "best_score": { "cosine": 0.92, "scaled": 0.96 }
  
}

### 🛠️ Technologies Used

Tech	Purpose

FastAPI	Backend API

HTML/CSS/JS	Frontend

HuggingFace Transformers	CLIP model

PyTorch	Inference

Uvicorn	ASGI Server

### 📦 Deployment

You can deploy using:

Docker

Railway

Render

AWS EC2

Azure App Service

Heroku

(Dockerfile available on request)

### 🧑‍💻 Author
Amith Banakar - Developer

OpenAI CLIP Team — Vision-Language Model

### 📜 License
MIT License — free to use and modify.

---
