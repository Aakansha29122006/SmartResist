#!/usr/bin/env python
# coding: utf-8
"""
SmartResist — Training Orchestrator
=====================================
Imports each pipeline step as a module and runs them in order.
Follows the 22-step system architecture.

Usage:
    python train_model.py
"""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from pipeline.step1_data_loading import load_datasets
from pipeline.step2_clean_genotype import clean_genotype
from pipeline.step3_clean_phenotype import clean_phenotype
from pipeline.step4_integration import integrate_datasets
from pipeline.step5_preprocessing import preprocess
from pipeline.step8_model_training import train_model
from pipeline.step18_evaluation import evaluate, overfitting_check
from pipeline.step20_save_model import save_model
from pipeline.step12_recommendation import run_test_cases


def main():
    print("============================================================")
    print("|                       SmartResist                        |")
    print("============================================================\n")

    # ── Step 1: Data Loading ──
    geno_df, pheno_df = load_datasets()

    # ── Step 2: Data Cleaning (Genotype) ──
    geno_clean = clean_genotype(geno_df)

    # ── Step 3: Data Cleaning (Phenotype) ──
    pheno_clean = clean_phenotype(pheno_df)

    # ── Step 4: Dataset Integration (Safe Strategy) ──
    #    4.1 Aggregation → 4.2 Safe Merge → 4.3 Controlled Pair Generation
    df_integrated = integrate_datasets(geno_clean, pheno_clean)

    # ── Step 5: Preprocessing (Encode + Scale) ──
    # ── Step 6: Dataset Splitting (BioSample_ID-based) ──
    # ── Step 7: Internal Linking (BioSample_ID not used as feature) ──
    train_df, test_df, encoders, scaler, df_model = preprocess(
        df_integrated, save_dir=PROJECT_DIR
    )

    # ── Step 8:  ANN Model (MLP) ──
    # ── Step 9:  Training Control (Early Stopping) ──
    # ── Step 10: Loss Function (BCE) ──
    # ── Step 11: Imbalance Handling (Weighted Loss) ──
    # ── Step 19: Training Setup (batch=1024, epochs≤100) ──
    # ── Step 21: No PyTorch — TensorFlow/Keras only ──
    model, history, X_test, y_test = train_model(
        train_df, test_df, encoders, scaler, save_dir=PROJECT_DIR
    )

    # ── Step 18: Evaluation Metrics ──
    metrics = evaluate(model, X_test, y_test)

    # ── Overfitting Diagnostic ──
    overfitting_check(history, save_dir=PROJECT_DIR)

    # ── Step 20: Model Saving (only if trained this session) ──
    if history is not None:
        save_model(model, encoders, scaler, df_model, metrics, save_dir=PROJECT_DIR)

    # ── Step 22: Test Cases ──
    run_test_cases(model, encoders, scaler, df_model)

    # ── Complete ──
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Final Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"  Model saved to: ANN_Project/")
    print(f"  Run 'python main.py --serve' to start the prediction server.")


if __name__ == '__main__':
    main()
