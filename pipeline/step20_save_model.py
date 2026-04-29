#!/usr/bin/env python
# coding: utf-8
"""
STEP 20 — MODEL SAVING (PICKLE STORAGE)
=========================================
Save after successful training:
  /ANN_Project/model.pkl      — Trained ANN model (Keras saved format)
  /ANN_Project/encoders.pkl   — Label encoders
  /ANN_Project/scaler.pkl     — StandardScaler

Save only if training completes successfully.
"""

import os
import json
import pickle
import pandas as pd


def save_model(model, encoders, scaler, df_model, metrics, save_dir=None):
    """
    Step 20: Save all trained artifacts to /ANN_Project/ folder.

    Only called if training completes successfully.

    Args:
        model: Trained Keras model.
        encoders: Dict of LabelEncoders.
        scaler: StandardScaler for numerical features.
        df_model: Full model dataframe (for building mappings).
        metrics: Evaluation metrics dict.
        save_dir: Project root directory.
    """
    print("\n" + "=" * 60)
    print("STEP 20: MODEL SAVING (PICKLE STORAGE)")
    print("=" * 60)

    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.dirname(save_dir)

    # Create /ANN_Project/ directory
    ann_dir = os.path.join(save_dir, "ANN_Project")
    os.makedirs(ann_dir, exist_ok=True)

    model_path = os.path.join(ann_dir, "model.keras")
    encoder_path = os.path.join(ann_dir, "encoders.pkl")
    scaler_path = os.path.join(ann_dir, "scaler.pkl")

    # Save model (Keras .keras format inside model.pkl path)
    model.save(model_path)
    print(f"  ✓ Model saved: {model_path}")

    # Save encoders as pickle
    with open(encoder_path, "wb") as f:
        pickle.dump(encoders, f)
    print(f"  ✓ Encoders saved: {encoder_path}")

    # Save scaler as pickle
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler saved: {scaler_path}")

    # ── Build gene-drug mapping (for rule-based filtering — Step 13) ──
    gene_drug_map = {}
    for gene in df_model['gene_symbol'].unique():
        subset = df_model[df_model['gene_symbol'] == gene]
        gene_drug_map[gene] = subset['antibiotic_name'].unique().tolist()

    # ── Build gene-species mapping ──
    gene_species_map = {}
    for gene in df_model['gene_symbol'].unique():
        subset = df_model[df_model['gene_symbol'] == gene]
        gene_species_map[gene] = subset['species'].value_counts().index[0]

    # ── Drug support counts (Step 11: frequency normalization) ──
    drug_support = df_model.groupby(
        ['gene_symbol', 'antibiotic_name']
    ).size().reset_index(name='support')

    support_path = os.path.join(ann_dir, "drug_support.pkl")
    with open(support_path, 'wb') as f:
        pickle.dump(drug_support, f)
    print(f"  ✓ Drug support saved: {support_path}")

    # ── Save metadata.json in project root ──
    gene_encoder = encoders['gene_encoder']
    species_encoder = encoders['species_encoder']
    drug_encoder = encoders['drug_encoder']

    metadata = {
        "total_samples": int(df_model['BioSample_ID'].nunique()),
        "total_genes": int(len(gene_encoder.classes_)),
        "total_drugs": int(len(drug_encoder.classes_)),
        "total_species": int(len(species_encoder.classes_)),
        "model_accuracy": f"{metrics['accuracy'] * 100:.2f}%",
        "precision": f"{metrics['precision'] * 100:.2f}%",
        "recall": f"{metrics['recall'] * 100:.2f}%",
        "f1_score": f"{metrics['f1_score'] * 100:.2f}%",
        "roc_auc": f"{metrics['roc_auc'] * 100:.2f}%",
        "data_source": "NCBI AMR Portal (ayates/amr_portal)",
        "known_genes": list(df_model['gene_symbol'].unique()),
        "gene_drug_map": gene_drug_map,
        "gene_species_map": gene_species_map,
    }

    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"  ✓ Metadata saved: {metadata_path}")

    print(f"\n  ✅ ALL ARTIFACTS SAVED SUCCESSFULLY")
    print(f"     Location: {ann_dir}")
    print(f"     Accuracy: {metadata['model_accuracy']}")
