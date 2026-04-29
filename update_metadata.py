import os
import sys
import pickle
import json

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from pipeline.step1_data_loading import load_datasets
from pipeline.step2_clean_genotype import clean_genotype
from pipeline.step3_clean_phenotype import clean_phenotype
from pipeline.step4_integration import integrate_datasets
from pipeline.step5_preprocessing import preprocess

print("Loading data to compute missing metadata...")
geno_df, pheno_df = load_datasets()
geno_clean = clean_genotype(geno_df)
pheno_clean = clean_phenotype(pheno_df)
df_integrated = integrate_datasets(geno_clean, pheno_clean)
train_df, test_df, encoders, scaler, df_model = preprocess(df_integrated, save_dir=PROJECT_DIR)

print("Updating metadata.json...")

gene_drug_map = {}
for gene in df_model['gene_symbol'].unique():
    subset = df_model[df_model['gene_symbol'] == gene]
    gene_drug_map[gene] = subset['antibiotic_name'].unique().tolist()

gene_species_map = {}
for gene in df_model['gene_symbol'].unique():
    subset = df_model[df_model['gene_symbol'] == gene]
    gene_species_map[gene] = subset['species'].value_counts().index[0]

metadata_path = os.path.join(PROJECT_DIR, "metadata.json")

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

metadata['known_genes'] = list(df_model['gene_symbol'].unique())
metadata['known_drugs'] = list(df_model['antibiotic_name'].unique())
metadata['gene_drug_map'] = gene_drug_map
metadata['gene_species_map'] = gene_species_map

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)

print("Successfully updated metadata.json!")
