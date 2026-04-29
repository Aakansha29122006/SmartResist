#!/usr/bin/env python
# coding: utf-8
"""
STEP 3 — DATA CLEANING (PHENOTYPE DATASET)
============================================
3.1 Shape check
3.2 Null value check
3.3 Null handling (dropna)
3.4 Unique value analysis
3.5 Unique value inspection (antibiotic_name, resistance_phenotype)
"""

import pandas as pd


def check_shape(df, label="Phenotype"):
    """3.1 — Check dataset dimensions."""
    print(f"\n  3.1 Shape of {label} Dataset")
    print(f"      No of rows: {df.shape[0]}")
    print(f"      No of columns: {df.shape[1]}")
    return df.shape


def check_nulls(df):
    """3.2 — Identify missing values."""
    null_counts = df.isnull().sum()
    null_percentages = (df.isnull().sum() / df.shape[0]) * 100

    null_info_df = pd.DataFrame({
        'Column_name': null_counts.index,
        'No of Null Values': null_counts.values,
        '% of column': null_percentages.values
    })

    print(f"\n  3.2 Null Values Information:")
    print(null_info_df.to_string(index=False))
    return null_info_df


def handle_nulls(df):
    """3.3 — Remove rows with null resistance_phenotype."""
    before = len(df)
    df_clean = df.dropna(subset=["resistance_phenotype"])
    after = len(df_clean)
    print(f"\n  3.3 Null Handling: {before} → {after} rows (removed {before - after})")
    return df_clean


def unique_analysis(df):
    """3.4 — Unique value counts."""
    unique_values = df.nunique()
    unique_df = pd.DataFrame({
        'Column_name': unique_values.index,
        'No of Unique Values': unique_values.values
    })
    print(f"\n  3.4 Unique Value Analysis:")
    print(unique_df.to_string(index=False))
    return unique_df


def unique_inspection(df):
    """3.5 — Inspect antibiotic_name and resistance_phenotype."""
    print(f"\n  3.5 Unique Value Inspection:")
    for col in ['antibiotic_name', 'resistance_phenotype']:
        if col in df.columns:
            vals = df[col].unique()
            print(f"      {col}: {len(vals)} unique values")
            if len(vals) <= 30:
                print(f"        → {list(vals)}")


def select_columns(pheno_df):
    """Select usable columns from raw phenotype dataframe."""
    pheno_type = pheno_df[[
        "BioSample_ID",
        "antibiotic_name",
        "resistance_phenotype"
    ]].copy()
    return pheno_type


def map_resistance_phenotype(pheno_type_df):
    """Map resistance_phenotype to S/R/I codes."""
    pheno_new = pheno_type_df.copy()

    pheno_new["resistance_phenotype"] = (
        pheno_new["resistance_phenotype"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    mapping = {
        "susceptible": "S",
        "resistant": "R",
        "non-susceptible": "R",
        "intermediate": "I",
        "susceptible-dose dependent": "I"
    }

    pheno_new["resistance_phenotype"] = pheno_new["resistance_phenotype"].map(mapping)
    print(f"\n  Mapped resistance_phenotype to S/R/I codes")
    print(f"  Distribution:\n{pheno_new['resistance_phenotype'].value_counts().to_string()}")
    return pheno_new


def clean_phenotype(pheno_df):
    """
    Full phenotype cleaning pipeline (Steps 3.1 — 3.5).

    Args:
        pheno_df: Raw phenotype DataFrame from step 1.

    Returns:
        pheno_clean: Cleaned phenotype DataFrame ready for merging.
    """
    print("\n" + "=" * 60)
    print("STEP 3: DATA CLEANING — PHENOTYPE")
    print("=" * 60)

    # Raw dataset inspection
    check_shape(pheno_df, "Raw Phenotype")
    check_nulls(pheno_df)
    unique_analysis(pheno_df)

    # Select usable columns
    pheno_type = select_columns(pheno_df)
    check_shape(pheno_type, "Filtered Phenotype")
    check_nulls(pheno_type)

    # Drop rows where resistance_phenotype is NaN
    pheno_dropped = handle_nulls(pheno_type)
    check_shape(pheno_dropped, "After dropna")

    # Map resistance_phenotype
    pheno_clean = map_resistance_phenotype(pheno_dropped)
    check_shape(pheno_clean, "After mapping")

    # Final inspection
    unique_analysis(pheno_clean)
    unique_inspection(pheno_clean)

    print(f"\n  ✓ Phenotype cleaned: {pheno_clean.shape}")
    return pheno_clean


if __name__ == '__main__':
    from step1_data_loading import load_datasets
    _, pheno_df = load_datasets()
    pheno_clean = clean_phenotype(pheno_df)
    print(f"\nFinal: {pheno_clean.shape}")
