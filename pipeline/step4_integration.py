#!/usr/bin/env python
# coding: utf-8
"""
STEP 4 — DATASET INTEGRATION (SAFE STRATEGY)
==============================================
4.1 Stage 1: Aggregation (genotype + phenotype by BioSample_ID)
4.2 Stage 2: Safe Merge (merge aggregated tables on BioSample_ID)
4.3 Stage 3: Controlled Pair Generation (gene–drug pairs, NO Cartesian explosion)

Also includes:
- Duplicate removal
- Class distribution analysis
- Rare gene/species removal (noise reduction)
- Dataset size logged after every major step
"""

import random
import pandas as pd
import numpy as np


def aggregate_genotype(geno_clean):
    """
    4.1a — Genotype Aggregation.
    Group by BioSample_ID, aggregate:
      gene_symbol → list(set())
      species → first
      region_start → mean
      region_end → mean
    """
    geno_agg = geno_clean.groupby('BioSample_ID').agg(
        gene_symbols=('gene_symbol', lambda x: list(set(x))),
        species=('species', 'first'),
        region_start_mean=('region_start', 'mean'),
        region_end_mean=('region_end', 'mean')
    ).reset_index()
    print(f"  4.1a Genotype aggregated: {geno_agg.shape}")
    return geno_agg


def aggregate_phenotype(pheno_clean):
    """
    4.1b — Phenotype Aggregation.
    Group by BioSample_ID, aggregate:
      antibiotic_name → list(set())
      resistance_phenotype → list
    """
    pheno_agg = pheno_clean.groupby('BioSample_ID').agg(
        antibiotic_names=('antibiotic_name', lambda x: list(set(x))),
        resistance_phenotypes=('resistance_phenotype', list)
    ).reset_index()
    print(f"  4.1b Phenotype aggregated: {pheno_agg.shape}")
    return pheno_agg


def safe_merge(geno_agg, pheno_agg):
    """
    4.2 — Safe Merge.
    Merge aggregated genotype and phenotype tables on BioSample_ID.
    """
    merged = geno_agg.merge(pheno_agg, on='BioSample_ID', how='inner')
    print(f"  4.2 Safe merge complete: {merged.shape}")
    return merged


def controlled_pair_generation(merged_agg, geno_clean, pheno_clean):
    """
    4.3 — Controlled Pair Generation.
    Uses vectorized pandas inner merge on BioSample_ID instead of iterrows 
    to quickly and safely generate gene–drug pairs without Cartesian explosion.
    Removes duplicates automatically.
    """
    # Clean BioSample_ID for lookups
    geno_clean = geno_clean.copy()
    pheno_clean = pheno_clean.copy()
    geno_clean['BioSample_ID'] = geno_clean['BioSample_ID'].astype(str).str.strip()
    pheno_clean['BioSample_ID'] = pheno_clean['BioSample_ID'].astype(str).str.strip()

    # Filter to only the samples present in merged_agg (the safe overlap)
    overlap_samples = set(merged_agg['BioSample_ID'])
    g_filt = geno_clean[geno_clean['BioSample_ID'].isin(overlap_samples)]
    p_filt = pheno_clean[pheno_clean['BioSample_ID'].isin(overlap_samples)]

    # Deduplicate before merge to strictly control pair generation
    # Keep gene info unique per sample
    g_dedup = g_filt.drop_duplicates(subset=['BioSample_ID', 'gene_symbol'])
    
    # Keep drug info unique per sample
    p_dedup = p_filt.drop_duplicates(subset=['BioSample_ID', 'antibiotic_name', 'resistance_phenotype'])

    # Vectorized inner merge -> automatically pairs genes & drugs per sample
    training_df = g_dedup.merge(p_dedup, on='BioSample_ID', how='inner')

    before_dedup = len(training_df)

    # Remove any final duplicates
    training_df = training_df.drop_duplicates()
    after_dedup = len(training_df)

    print(f"  4.3 Controlled pairs generated: {before_dedup}")
    print(f"      After final dedup: {after_dedup}")

    return training_df


def class_distribution(df):
    """Check class imbalance in resistance_phenotype."""
    class_dist = df['resistance_phenotype'].value_counts()
    class_pct = df['resistance_phenotype'].value_counts(normalize=True) * 100

    class_info = pd.DataFrame({
        'Class': class_dist.index,
        'Count': class_dist.values,
        'Percentage': class_pct.values
    })

    print(f"\n  Class Distribution of 'resistance_phenotype':")
    print(class_info.to_string(index=False))
    return class_info


def remove_rare(df, min_gene_count=10, min_species_count=50):
    """Remove rare genes (<10) and rare species (<50) to reduce noise."""
    before = len(df)

    gene_counts = df['gene_symbol'].value_counts()
    valid_genes = gene_counts[gene_counts >= min_gene_count].index
    df = df[df['gene_symbol'].isin(valid_genes)]

    species_counts = df['species'].value_counts()
    valid_species = species_counts[species_counts >= min_species_count].index
    df = df[df['species'].isin(valid_species)]

    after = len(df)
    print(f"\n  Rare Gene/Species Removal:")
    print(f"      Rows before: {before}")
    print(f"      Rows after: {after}")
    print(f"      Data retained: {round((after / before) * 100, 2)}%")
    return df


def integrate_datasets(geno_clean, pheno_clean):
    """
    Full integration pipeline (Steps 4.1 — 4.3).

    Uses the 3-stage safe strategy:
      Stage 1: Aggregate both datasets by BioSample_ID
      Stage 2: Merge aggregated tables
      Stage 3: Generate controlled gene–drug pairs

    Args:
        geno_clean: Cleaned genotype DataFrame from step 2.
        pheno_clean: Cleaned phenotype DataFrame from step 3.

    Returns:
        df: Training-ready DataFrame with gene–drug pairs.
    """
    print("\n" + "=" * 60)
    print("STEP 4: DATASET INTEGRATION (SAFE STRATEGY)")
    print("=" * 60)

    # Clean BioSample_ID
    geno_clean = geno_clean.copy()
    pheno_clean = pheno_clean.copy()
    geno_clean['BioSample_ID'] = geno_clean['BioSample_ID'].astype(str).str.strip()
    pheno_clean['BioSample_ID'] = pheno_clean['BioSample_ID'].astype(str).str.strip()

    # 4.1 Stage 1: Aggregation
    print("\n  Stage 1: Aggregation")
    geno_agg = aggregate_genotype(geno_clean)
    pheno_agg = aggregate_phenotype(pheno_clean)

    # 4.2 Stage 2: Safe Merge
    print("\n  Stage 2: Safe Merge")
    merged_agg = safe_merge(geno_agg, pheno_agg)

    # 4.3 Stage 3: Controlled Pair Generation
    print("\n  Stage 3: Controlled Pair Generation")
    training_df = controlled_pair_generation(merged_agg, geno_clean, pheno_clean)
    print(f"      Final training dataset: {training_df.shape}")

    # Class distribution
    class_distribution(training_df)

    # Remove rare genes/species
    df = remove_rare(training_df)

    print(f"\n  ✓ Integration complete: {df.shape}")
    return df
