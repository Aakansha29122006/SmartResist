import os
import sys
import pickle

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from pipeline.step1_data_loading import load_datasets
from pipeline.step2_clean_genotype import clean_genotype
from pipeline.step3_clean_phenotype import clean_phenotype
from pipeline.step4_integration import integrate_datasets
from pipeline.step5_preprocessing import preprocess

print("Loading data to compute drug_support...")
geno_df, pheno_df = load_datasets()
geno_clean = clean_genotype(geno_df)
pheno_clean = clean_phenotype(pheno_df)
df_integrated = integrate_datasets(geno_clean, pheno_clean)
train_df, test_df, encoders, scaler, df_model = preprocess(df_integrated, save_dir=PROJECT_DIR)

print("Computing drug support...")
drug_support = df_model.groupby(['gene_symbol', 'antibiotic_name']).size().reset_index(name='support')

ann_dir = os.path.join(PROJECT_DIR, "ANN_Project")
os.makedirs(ann_dir, exist_ok=True)
support_path = os.path.join(ann_dir, "drug_support.pkl")
with open(support_path, 'wb') as f:
    pickle.dump(drug_support, f)

print(f"Successfully created and saved {support_path}!")
