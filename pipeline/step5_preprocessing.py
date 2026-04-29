#!/usr/bin/env python
# coding: utf-8
"""
STEP 5 — PREPROCESSING LAYER
==============================
- Label encode gene_symbol
- Scale numerical features: region_start, region_end, region_length
- Compute region_length
- Target Variable: Susceptible = 1, Resistant / Intermediate / NS / SDD = 0

STEP 6 — DATASET SPLITTING STRATEGY
=====================================
- Train/test split based on BioSample_ID
- Prevent data leakage
- Log train and test dataset sizes

STEP 7 — INTERNAL LINKING LAYER
=================================
- BioSample_ID used only for grouping
- Not used as model feature
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def build_target(df):
    """
    Build binary target variable per specification:
      Susceptible = 1
      Resistant / Intermediate / Non-susceptible / Susceptible-dose dependent = 0
    """
    df2 = df.copy()
    df2['resistance_normalized'] = (
        df2['resistance_phenotype'].astype(str).str.strip().str.lower()
    )

    # Map all phenotypes: S=1, everything else=0
    target_map = {
        'susceptible': 1,
        's': 1,
        'resistant': 0,
        'r': 0,
        'intermediate': 0,
        'i': 0,
        'non-susceptible': 0,
        'susceptible-dose dependent': 0,
    }
    df2['target'] = df2['resistance_normalized'].map(target_map)

    # Drop rows with unmapped phenotypes
    before = len(df2)
    df2 = df2.dropna(subset=['target'])
    df2['target'] = df2['target'].astype(int)
    print(f"\n  Target built: Susceptible=1, Resistant/Intermediate/NS/SDD=0")
    print(f"  Rows with valid target: {before} → {len(df2)}")
    print(f"  Target distribution:")
    print(f"    {df2['target'].value_counts().to_string()}")

    # Compute region_length
    df2['region_length'] = df2['region_end'] - df2['region_start']

    # Drop rows missing critical fields
    df2 = df2.dropna(subset=[
        'gene_symbol', 'species', 'antibiotic_name',
        'region_start', 'region_end', 'region_length', 'target'
    ])
    print(f"  After dropna: {df2.shape}")

    return df2


def encode_and_scale(df_model, save_dir=None):
    """
    Label encode gene_symbol.
    Scale numerical features: region_start, region_end, region_length.

    Returns:
        df_model: DataFrame with encoded + scaled columns.
        encoders: Dict of LabelEncoders.
        scaler: Fitted StandardScaler for numerical features.
    """
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.dirname(save_dir)

    ann_dir = os.path.join(save_dir, "ANN_Project")
    encoder_path = os.path.join(ann_dir, "encoders.pkl")
    model_path = os.path.join(ann_dir, "model.keras")
    model_exists = os.path.exists(model_path) and os.path.exists(encoder_path)

    if model_exists:
        print(f"\n  Loading saved encoders from {encoder_path}...")
        with open(encoder_path, "rb") as f:
            encoders = pickle.load(f)
        with open(os.path.join(ann_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

        # Transform using saved encoders
        df_model['gene_encoded'] = encoders["gene_encoder"].transform(
            df_model['gene_symbol'].astype(str))
        df_model['species_encoded'] = encoders["species_encoder"].transform(
            df_model['species'].astype(str))
        df_model['drug_encoded'] = encoders["drug_encoder"].transform(
            df_model['antibiotic_name'].astype(str))

        # Scale numerical features using saved scaler
        df_model[['region_start_scaled', 'region_end_scaled', 'region_length_scaled']] = (
            scaler.transform(df_model[['region_start', 'region_end', 'region_length']])
        )
    else:
        print(f"\n  Fitting new encoders...")

        # Label encode gene_symbol
        gene_encoder = LabelEncoder()
        df_model['gene_encoded'] = gene_encoder.fit_transform(
            df_model['gene_symbol'].astype(str))

        # Also encode species and drug for model features
        species_encoder = LabelEncoder()
        df_model['species_encoded'] = species_encoder.fit_transform(
            df_model['species'].astype(str))

        drug_encoder = LabelEncoder()
        df_model['drug_encoded'] = drug_encoder.fit_transform(
            df_model['antibiotic_name'].astype(str))

        encoders = {
            "gene_encoder": gene_encoder,
            "species_encoder": species_encoder,
            "drug_encoder": drug_encoder,
        }

        # Scale numerical features: region_start, region_end, region_length
        scaler = StandardScaler()
        df_model[['region_start_scaled', 'region_end_scaled', 'region_length_scaled']] = (
            scaler.fit_transform(df_model[['region_start', 'region_end', 'region_length']])
        )

        # Save initial metadata
        metadata = {
            "total_samples": int(df_model['BioSample_ID'].nunique()),
            "total_genes": int(len(gene_encoder.classes_)),
            "total_drugs": int(len(drug_encoder.classes_)),
            "total_species": int(len(species_encoder.classes_)),
            "model_accuracy": "Awaiting Training...",
            "data_source": "NCBI AMR Portal (ayates/amr_portal)"
        }
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"  📊 Metadata saved to {metadata_path}")

    print(f"\n  Encoded: gene({len(encoders['gene_encoder'].classes_)}), "
          f"species({len(encoders['species_encoder'].classes_)}), "
          f"drug({len(encoders['drug_encoder'].classes_)})")
    print(f"  Scaled: region_start, region_end, region_length")
    print(f"  ✓ Preprocessing complete: {df_model.shape}")

    return df_model, encoders, scaler


def split_by_biosample(df_model, test_size=0.2, random_state=42):
    """
    Step 6: Split based on BioSample_ID to prevent data leakage.
    Step 7: BioSample_ID used only for grouping, NOT as model feature.
    """
    print("\n" + "=" * 60)
    print("STEP 6: DATASET SPLITTING (BioSample_ID-based)")
    print("=" * 60)

    unique_samples = df_model['BioSample_ID'].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_samples)

    split_idx = int(len(unique_samples) * (1 - test_size))
    train_samples = set(unique_samples[:split_idx])
    test_samples = set(unique_samples[split_idx:])

    train_df = df_model[df_model['BioSample_ID'].isin(train_samples)]
    test_df = df_model[df_model['BioSample_ID'].isin(test_samples)]

    print(f"  Unique BioSamples: {len(unique_samples)}")
    print(f"  Train samples: {len(train_samples)} → {len(train_df)} rows")
    print(f"  Test samples:  {len(test_samples)} → {len(test_df)} rows")
    print(f"  No data leakage: {len(train_samples & test_samples) == 0}")

    return train_df, test_df


def preprocess(df, save_dir=None):
    """
    Full preprocessing pipeline (Steps 5-7).

    Args:
        df: Merged DataFrame from step 4.
        save_dir: Project root directory.

    Returns:
        train_df, test_df, encoders, scaler, df_model
    """
    print("\n" + "=" * 60)
    print("STEP 5: PREPROCESSING LAYER")
    print("=" * 60)

    df_model = build_target(df)
    df_model, encoders, scaler = encode_and_scale(df_model, save_dir)
    train_df, test_df = split_by_biosample(df_model)

    return train_df, test_df, encoders, scaler, df_model
