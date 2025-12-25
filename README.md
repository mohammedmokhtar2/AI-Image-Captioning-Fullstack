# 📸 AI Image Captioning (SOTA + React)

A modern **full-stack AI application** that generates high-quality, human-like captions for images using **State-of-the-Art (SOTA) deep learning models**.

The system follows a **decoupled architecture**:

- **FastAPI backend** for inference (Dockerized & deployed on Hugging Face Spaces)
- **Next.js / React frontend** for user interaction (deployed on Vercel)

---

## 🚀 Live Demo

- **Frontend:** *[https://captioning-hnu.vercel.app/](https://captioning-hnu.vercel.app/)
- **Backend API:** [https://mokh2x-captioning.hf.space/predict](https://mokh2x-captioning.hf.space/predict)

---

## 🏗️ Architecture

The project is divided into two independent components:

### 🔹 Backend — Inference Engine

- **Framework:** FastAPI (Python)
- **Models:**
  - BLIP (default)
  - ViT-GPT2
  - Custom ResNet50 + GPT-2
- **Deployment:** Docker container on Hugging Face Spaces (CPU Basic)

### 🔹 Frontend — User Interface

- **Framework:** Next.js (React) + TypeScript
- **Styling:** Tailwind CSS with animations
- **Deployment:** Vercel

---

## 🧠 Supported AI Models

### 1️⃣ BLIP (Bootstrapping Language-Image Pre-training)

- **Status:** ✅ Default (Best Performance)
- **Description:** Produces highly accurate, detailed, and natural image captions.

### 2️⃣ ViT-GPT2

- **Status:** ✅ Available
- **Description:** Combines a Vision Transformer (ViT) encoder with a GPT-2 decoder.

### 3️⃣ ResNet50 + GPT-2 (Custom)

- **Status:** 🧪 Experimental / Legacy
- **Description:** Custom implementation trained from scratch on the Flickr30k dataset.

---

## 🛠️ Installation & Local Setup

### 1️⃣ Backend Setup (Python / Docker)

#### Option A: Run Locally

```bash
# Clone the repository
git clone -b sota https://github.com/Tu2525/MLProject.git
cd MLProject

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Backend will be available at: 👉 [http://localhost:8000](http://localhost:8000)

---

#### Option B: Run with Docker

```bash
# Build Docker image
docker build -t caption-api .

# Run container
docker run -p 7860:7860 caption-api
```

Backend will be available at: 👉 [http://localhost:7860](http://localhost:7860)

---

### 2️⃣ Frontend Setup (Next.js)

```bash
# Create a Next.js app
npx create-next-app@latest my-portfolio
cd my-portfolio

# Install dependencies
npm install axios

# Run development server
npm run dev
```

Frontend will be available at: 👉 [http://localhost:3000](http://localhost:3000)

---

## ⚙️ Configuration

To change the active model, edit `config/config.py`:

```python
class Config:
    # Options: "blip", "vit_gpt2", "resnet_gpt2"
    MODEL_TYPE = "blip"

    # Force CPU (required for Hugging Face free tier)
    DEVICE = "cpu"
```

---

## 👥 Team

Developed by **Intelligent Systems Engineering students** under the supervision of **Dr. Hadeer Ahmed** at **Helwan National University**:

- Mohammed Mokhtar
- Amr Khaled
- Eyad Ahmed
- Tarek Shereen

---

## 📄 License

This project is licensed under the **MIT License**.

