#!/usr/bin/env python
# coding: utf-8
"""
STEP 2 — DATA CLEANING (GENOTYPE DATASET)
==========================================
2.1 Shape check
2.2 Null value check
2.3 Null handling (dropna)
2.4 Unique value analysis
2.5 Unique value inspection (gene_symbol, species, genus)
"""

import numpy as np
import pandas as pd


def check_shape(df, label="Genotype"):
    """2.1 — Check dataset dimensions using .shape"""
    print(f"\n  2.1 Shape of {label} Dataset")
    print(f"      No of rows: {df.shape[0]}")
    print(f"      No of columns: {df.shape[1]}")
    return df.shape


def check_nulls(df):
    """2.2 — Identify missing values using .isnull().sum()"""
    null_counts = df.isnull().sum()
    null_percentages = (df.isnull().sum() / df.shape[0]) * 100

    null_info_df = pd.DataFrame({
        'Column_name': null_counts.index,
        'No of Null Values': null_counts.values,
        '% of column': null_percentages.values
    })

    print(f"\n  2.2 Null Values Information:")
    print(null_info_df.to_string(index=False))
    return null_info_df


def handle_nulls(df):
    """2.3 — Remove rows with null values using .dropna()"""
    before = len(df)
    df_clean = df.dropna()
    after = len(df_clean)
    print(f"\n  2.3 Null Handling: {before} → {after} rows (removed {before - after})")
    return df_clean


def unique_analysis(df):
    """2.4 — Check number of unique values using .nunique()"""
    unique_values = df.nunique()
    unique_df = pd.DataFrame({
        'Column_name': unique_values.index,
        'No of Unique Values': unique_values.values
    })
    print(f"\n  2.4 Unique Value Analysis:")
    print(unique_df.to_string(index=False))
    return unique_df


def unique_inspection(df):
    """2.5 — Inspect unique values for gene_symbol, species, genus"""
    print(f"\n  2.5 Unique Value Inspection:")
    for col in ['gene_symbol', 'species', 'genus']:
        if col in df.columns:
            vals = df[col].unique()
            print(f"      {col}: {len(vals)} unique values")
            if len(vals) <= 20:
                print(f"        → {list(vals)}")


def select_columns(geno_df):
    """Select usable columns from raw genotype dataframe."""
    geno_type = geno_df[[
        'BioSample_ID',
        'gene_symbol',
        'region_start',
        'region_end',
        'genus',
        'species',
        'class',
        'subclass'
    ]].copy()
    return geno_type


def clean_class_subclass(geno_type):
    """Clean 'class' & 'subclass' columns — explode list-like values."""
    def to_list(cell):
        if isinstance(cell, list):
            return cell
        elif isinstance(cell, np.ndarray):
            return cell.tolist()
        else:
            return [i.strip() for i in str(cell).strip().split('/') if i.strip()]

    geno_type['class'] = geno_type['class'].apply(to_list)
    geno_type['subclass'] = geno_type['subclass'].apply(to_list)

    # Explode both columns
    geno_type_df = geno_type.explode('class').explode('subclass')

    # Flatten lists to strings
    geno_type_df['class'] = geno_type_df['class'].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )
    geno_type_df['subclass'] = geno_type_df['subclass'].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )

    geno_type_df = geno_type_df.reset_index(drop=True)
    print(f"\n  Class/Subclass cleaned. New shape: {geno_type_df.shape}")
    return geno_type_df


def clean_genotype(geno_df):
    """
    Full genotype cleaning pipeline (Steps 2.1 — 2.5).

    Args:
        geno_df: Raw genotype DataFrame from step 1.

    Returns:
        geno_clean: Cleaned genotype DataFrame ready for merging.
    """
    print("\n" + "=" * 60)
    print("STEP 2: DATA CLEANING — GENOTYPE")
    print("=" * 60)

    # Raw dataset inspection
    check_shape(geno_df, "Raw Genotype")
    check_nulls(geno_df)
    unique_analysis(geno_df)

    # Select usable columns
    geno_type = select_columns(geno_df)
    print(f"\n  Selected columns: {list(geno_type.columns)}")
    check_shape(geno_type, "Filtered Genotype")

    # Clean class/subclass
    geno_type_df = clean_class_subclass(geno_type)
    check_shape(geno_type_df, "After class/subclass cleaning")
    check_nulls(geno_type_df)

    # Drop nulls (specifically gene_symbol)
    geno_clean = geno_type_df.dropna(subset=["gene_symbol"])
    print(f"\n  2.3 Dropped gene_symbol nulls: {len(geno_type_df)} → {len(geno_clean)} rows")

    # Final inspection
    check_nulls(geno_clean)
    unique_analysis(geno_clean)
    unique_inspection(geno_clean)

    print(f"\n  ✓ Genotype cleaned: {geno_clean.shape}")
    return geno_clean


if __name__ == '__main__':
    from step1_data_loading import load_datasets
    geno_df, _ = load_datasets()
    geno_clean = clean_genotype(geno_df)
    print(f"\nFinal: {geno_clean.shape}")
