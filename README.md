# SmartResist — AI Antibiotic Resistance Prediction System

A **hybrid ANN-based architecture** that uses a Feedforward Neural Network (MLP) along with rule-based filtering to predict antibiotic susceptibility from genomic markers.

---

## System Architecture

The system follows a **22-step architecture**:

| Step | Component | Description |
|------|-----------|-------------|
| 1 | User Input Layer | gene_symbol, region_start, region_end, allergic_medication |
| 2 | Genotype Cleaning | Shape, nulls, dropna, unique analysis |
| 3 | Phenotype Cleaning | Shape, nulls, dropna, unique analysis |
| 4 | Safe Integration | Aggregation → Safe Merge → Controlled Pair Generation |
| 5 | Preprocessing | Label encoding, feature scaling, region_length |
| 6 | Splitting | BioSample_ID-based train/test split (no leakage) |
| 7 | Internal Linking | BioSample_ID for grouping only, not as feature |
| 8 | ANN Model (MLP) | Feedforward dense layers, embeddings |
| 9 | Training Control | Early stopping (2 epoch patience) |
| 10 | Loss Function | Binary Cross Entropy |
| 11 | Imbalance Handling | Weighted loss, balanced sampling, min support |
| 12-16 | Hybrid System | ANN prediction + rule-based filtering + ranking |
| 17 | Unseen Genes | Returns "Talk to a professional" |
| 18 | Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC |
| 19 | Training Setup | Batch 64, Epochs ≤100, BCE |
| 20 | Model Saving | Pickle storage (encoders, scaler) + Keras model |
| 21 | Constraint | TensorFlow/Keras only — NO PyTorch |
| 22 | Testing | Multiple test cases with edge cases |

---

## Project Structure

```
AMM/
├── main.py                     # Entry point — train or serve
├── train_model.py              # Training orchestrator
├── app.py                      # Flask API server
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── Project_Backend.py          # Original monolithic script (reference)
│
├── pipeline/                   # Modular ML pipeline
│   ├── __init__.py
│   ├── step1_data_loading.py       # Data acquisition
│   ├── step2_clean_genotype.py     # Genotype cleaning
│   ├── step3_clean_phenotype.py    # Phenotype cleaning
│   ├── step4_integration.py        # Safe merge + controlled pairing
│   ├── step5_preprocessing.py      # Encoding + scaling + splitting
│   ├── step8_model_training.py     # ANN build + training
│   ├── step12_recommendation.py    # Hybrid recommendation system
│   ├── step18_evaluation.py        # Metrics + overfitting check
│   └── step20_save_model.py        # Artifact saving
│
├── ANN_Project/                # Saved model artifacts
│   ├── model.pkl               # Trained ANN model (Keras)
│   ├── encoders.pkl            # Label encoders
│   └── scaler.pkl              # StandardScaler
│
├── templates/
│   └── index.html              # Frontend UI
├── static/
│   ├── style.css               # Design system
│   ├── script.js               # Frontend logic
│   └── training_curves.png     # Training visualization
└── metadata.json               # Dataset stats + gene-drug mappings
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python main.py --train
```
This runs the full pipeline: data loading → cleaning → integration → preprocessing → ANN training → evaluation → saving.

### 3. Start the Prediction Server
```bash
python main.py --serve
```
Opens at `http://localhost:5000`

### 4. Run Both (Train + Serve)
```bash
python main.py --train --serve
```

---

## Target Variable

| Phenotype | Label | Value |
|-----------|-------|-------|
| Susceptible | S | **1** |
| Resistant | R | 0 |
| Intermediate | I | 0 |
| Non-susceptible | NS | 0 |
| Susceptible-dose dependent | SDD | 0 |

---

## Model Specifications

- **Type**: Feedforward Artificial Neural Network (MLP)
- **Layers**: 512 → 256 → 128 → 64 → 1 (sigmoid)
- **Features**: Label-encoded categoricals + scaled numericals
- **Loss**: Binary Cross Entropy (weighted)
- **Batch Size**: 64
- **Max Epochs**: 100 (with early stopping, patience=2)
- **No embeddings** — label encoding only
- **No PyTorch** — TensorFlow/Keras only

---

## Data Source

[NCBI AMR Portal](https://huggingface.co/datasets/ayates/amr_portal) — genotype.parquet + phenotype.parquet

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend UI |
| `/api/status` | GET | Backend health + dataset stats |
| `/api/autocomplete?q=` | GET | Gene name autocomplete |
| `/api/predict` | POST | Antibiotic prediction |
| `/api/test` | GET | Run all test cases |

---

## License

For research and clinical decision support only.

© 2026 SmartResist AI
