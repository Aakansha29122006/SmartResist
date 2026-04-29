#!/usr/bin/env python
# coding: utf-8
"""
STEP 1 — DATA LOADING
=====================
Load genotype and phenotype datasets from HuggingFace (NCBI AMR Portal).
Convert to Pandas DataFrames.
"""

import pandas as pd
from datasets import load_dataset


def load_datasets():
    """
    Load genotype and phenotype datasets from HuggingFace.

    Returns:
        geno_df (pd.DataFrame): Raw genotype dataframe.
        pheno_df (pd.DataFrame): Raw phenotype dataframe.
    """
    print("=" * 60)
    print("STEP 1: DATA LOADING")
    print("=" * 60)

    #@title Importing Datasets
    genotype = load_dataset("ayates/amr_portal", data_files="genotype.parquet", split="train")
    phenotype = load_dataset("ayates/amr_portal", data_files="phenotype.parquet", split="train")

    #@title Transforming to usable Dataframe
    geno_df = genotype.to_pandas()
    pheno_df = phenotype.to_pandas()

    print(f"  Genotype loaded: {geno_df.shape[0]} rows × {geno_df.shape[1]} columns")
    print(f"  Phenotype loaded: {pheno_df.shape[0]} rows × {pheno_df.shape[1]} columns")

    return geno_df, pheno_df


if __name__ == '__main__':
    geno_df, pheno_df = load_datasets()
    print("\nGenotype columns:", list(geno_df.columns))
    print("Phenotype columns:", list(pheno_df.columns))
