"""
SmartResist ML Pipeline — Modular Package
==========================================
Each module corresponds to a stage in the 22-step system architecture.

Modules:
    step1_data_loading      — Data Acquisition (HuggingFace AMR Portal)
    step2_clean_genotype    — Genotype Cleaning (shape, nulls, unique)
    step3_clean_phenotype   — Phenotype Cleaning (shape, nulls, mapping)
    step4_integration       — Safe Integration (aggregate → merge → pairs)
    step5_preprocessing     — Encoding + Scaling + BioSample Splitting
    step8_model_training    — ANN (MLP) Build + Training + Early Stopping
    step12_recommendation   — Hybrid Recommendation + Test Cases
    step18_evaluation       — Metrics (Acc, Prec, Recall, F1, ROC-AUC)
    step20_save_model       — Pickle Storage to /ANN_Project/

Usage:
    from pipeline.step1_data_loading import load_datasets
    from pipeline.step2_clean_genotype import clean_genotype
    ...
"""

from pipeline.step1_data_loading import load_datasets
from pipeline.step2_clean_genotype import clean_genotype
from pipeline.step3_clean_phenotype import clean_phenotype
from pipeline.step4_integration import integrate_datasets
from pipeline.step5_preprocessing import preprocess
from pipeline.step8_model_training import train_model
from pipeline.step12_recommendation import recommend_best_drug, run_test_cases
from pipeline.step18_evaluation import evaluate, overfitting_check
from pipeline.step20_save_model import save_model
