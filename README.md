# Deepfake Detection System

## Project Overview

This project is a **Deepfake Detection System** that identifies whether an image or video is real or AI-generated (deepfake).

The system uses a deep learning model based on **Xception architecture** trained on the **FaceForensics++ dataset**.  
It consists of:

-  Frontend (React + Vite)
-  Backend (Python + PyTorch)
-  Trained Deep Learning Model (.pth)

---

## Project Structure

```
deepfake-detection/
│
├── public/                 # Static frontend files
├── src/                    # React frontend source code
│
├── backend/
│   ├── detection/          # Training and detection scripts
│   ├── models/             # Trained model (.pth files)
│   ├── main.py             # Backend entry point
│   └── requirements.txt    # Python dependencies
│
├── package.json
├── vite.config.ts
└── README.md
```

---

## Features

- Upload image/video for analysis
- Deepfake vs Real classification
- Confidence score output
- Trained Xception-based deep learning model
- Clean and responsive UI
- Modular backend structure

---

## Model Information

- Architecture: Xception
- Framework: PyTorch
- Dataset: FaceForensics++
- Model Format: `.pth`

⚠ Dataset is not included in this repository due to size limitations.

---

## Frontend Setup

Make sure Node.js is installed.

```bash
npm install
npm run dev
```

The frontend will run on:
```
http://localhost:5173
```

---

## Backend Setup

Navigate to backend folder:

```bash
cd backend
```

Create virtual environment (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run backend:

```bash
python main.py
```

---

## Requirements

### Frontend
- Node.js
- npm

### Backend
- Python 3.8+
- PyTorch
- OpenCV
- NumPy

---

## Dataset

This project uses the **FaceForensics++ dataset**.

Due to large size, the dataset is not included in this repository.

You can download it from:
https://github.com/ondyari/FaceForensics


## 👩‍💻 Author

Developed as a Deepfake Detection System using React and PyTorch.

## 📄 License

This project is for academic use.
