# Intelligent Resume Screener AI

A high-precision, production-ready AI pipeline for resume entity extraction, professional timeline evaluation, and job category prediction.

## 🚀 How to Run

### 1. Setup
Install all requirements and download the spaCy core:
```bash
pip install torch transformers spacy pymupdf fastapi uvicorn scikit-learn matplotlib joblib pandas numpy
python -m spacy download en_core_web_sm
```

### 2. Training (Deep Learning Objective)
Run the training script to generate the MLP model and the Adam vs RMSProp comparison plot:
```bash
python train_dl.py
```

### 3. Running the App (Frontend + Backend)
1. **Start the API:** 
   ```bash
   python app.py
   ```
2. **Open the UI:**
   Open `index.html` in your browser.

### 4. Evaluation (NLP Objective)
Generate Precision, Recall, and F1-Score metrics:
```bash
python evaluate_ner.py
```

### 5. Visualizations
- **Entities:** Check `entities.html` after scanning a resume.
- **DL Performance:** Check `optimizer_comparison.png` after training.

## 🛠️ Tech Stack
- **NLP:** BERT (Transformer), spaCy (Dependency Parsing), Regex.
- **DL:** PyTorch (MLP), Scikit-Learn (TF-IDF, Feature Selection).
- **Backend:** FastAPI.
- **Frontend:** Vanilla JS + CSS (Cyber-Glass Design).
