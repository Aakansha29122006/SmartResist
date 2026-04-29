#!/usr/bin/env python
# coding: utf-8

# In[1]:


#@title Importing Libraries for loading Dataset
import pandas as pd
from datasets import load_dataset


# In[2]:


#@title Importing Datasets
genotype = load_dataset("ayates/amr_portal", data_files="genotype.parquet", split="train")
phenotype = load_dataset("ayates/amr_portal", data_files="phenotype.parquet", split="train")


# In[3]:


#@title Transfroming to usable Dataframe
geno_df = genotype.to_pandas()
pheno_df = phenotype.to_pandas()


# In[4]:


# Working of Genotype


# In[5]:


#@title Shape of Dataset
print("Shape of the Dataset")
print(f"No of rows: {geno_df.shape[0]}")
print(f"No of columns: {geno_df.shape[1]}")


# In[6]:


#@title Columns of Dataset
print("Columns of the Dataset")
geno_df.columns


# In[7]:


#@title Overview of Dataset
print("First 5 rows of the Dataset")
geno_df.head()


# In[8]:


#@title Column Data type
geno_df.dtypes


# In[9]:


#@title No of Null Values in each column
null_counts = geno_df.isnull().sum()
null_percentages = (geno_df.isnull().sum() / geno_df.shape[0]) * 100

null_info_df = pd.DataFrame({
    'Column_name': null_counts.index,
    'No of Null Values': null_counts.values,
    '% of column': null_percentages.values
})

print("Null Values Information:")
print(null_info_df)


# In[10]:


#@title No of Unique Values in each column
print("No of Unique Values in Each column")
unique_values = geno_df.nunique()
unique_values_df = pd.DataFrame({
    'Column_name': unique_values.index,
    'No of Unique Values': unique_values})
print(unique_values_df)


# In[11]:


#@title Unique Values in each column
for column in geno_df.columns:
    unique_values = geno_df[column].unique()
    print(f"Unique values in column '{column}':")
    print(unique_values)
    print()


# In[12]:


#@title Usable Columns
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


# In[13]:


#About New Dataset


# In[14]:


#@title Shape of Dataset
print("Shape of the Dataset")
print(f"No of rows: {geno_type.shape[0]}")
print(f"No of columns: {geno_type.shape[1]}")


# In[15]:


#@title Columns of Dataset
print("Columns of the Dataset")
geno_type.columns


# In[16]:


#@title Overview of Dataset
print("First 5 rows of the Dataset")
geno_type.head()


# In[17]:


#@title Column Data type
geno_type.dtypes


# In[18]:


#@title No of Null Values in each column
null_counts = geno_type.isnull().sum()
null_percentages = (geno_type.isnull().sum() / geno_type.shape[0]) * 100

null_info_df = pd.DataFrame({
    'Column_name': null_counts.index,
    'No of Null Values': null_counts.values,
    '% of column': null_percentages.values
})

print("Null Values Information:")
print(null_info_df)


# In[19]:


#@title No of Unique Values in each column
print("No of Unique Values in Each column")
unique_values = geno_type.nunique()
unique_values_df = pd.DataFrame({
    'Column_name': unique_values.index,
    'No of Unique Values': unique_values})
print(unique_values_df)


# In[20]:


#@title Unique Values in each column
for column in geno_type.columns:
    unique_values = geno_type[column].unique()
    print(f"Unique values in column '{column}':")
    print(unique_values)
    print()


# In[21]:


#Cleaning the Dataset


# In[22]:


#@title Cleaning 'Class' & 'Subclass'
import numpy as np

def to_list(cell):
    if isinstance(cell, list):
        return cell
    elif isinstance(cell, np.ndarray):
        return cell.tolist()
    else:
        return [i.strip() for i in str(cell).strip().split('/') if i.strip()]

# Convert to lists
geno_type['class'] = geno_type['class'].apply(to_list)
geno_type['subclass'] = geno_type['subclass'].apply(to_list)

# Explode both columns
geno_type_df = geno_type.explode('class').explode('subclass')

# Flatten lists to strings
geno_type_df['class'] = geno_type_df['class'].apply(lambda x: x[0] if isinstance(x, list) else x)
geno_type_df['subclass'] = geno_type_df['subclass'].apply(lambda x: x[0] if isinstance(x, list) else x)

# Reset index
geno_type_df = geno_type_df.reset_index(drop=True)


# In[23]:


#@title New Dataset
print("New Dataset")
geno_type_df.head()


# In[24]:


#@title Shape of the new Dataset
print("Shape of the Dataset")
print(f"No of rows: {geno_type_df.shape[0]}")
print(f"No of columns: {geno_type_df.shape[1]}")


# In[25]:


#@title Dropping rows with null values
geno_type_drop = geno_type_df.dropna(subset=["gene_symbol"])
print("Cleaning successfull")
print()


# In[26]:


#@title Shape of the new Dataset
print("Shape of the Dataset")
print(f"No of rows: {geno_type_drop.shape[0]}")
print(f"No of columns: {geno_type_drop.shape[1]}")


# In[27]:


#@title No of Null Values in each column
null_counts = geno_type_drop.isnull().sum()
null_percentages = (geno_type_drop.isnull().sum() / geno_type_drop.shape[0]) * 100

null_info_df = pd.DataFrame({
    'Column_name': null_counts.index,
    'No of Null Values': null_counts.values,
    '% of column': null_percentages.values
})

print("Null Values Information:")
print(null_info_df)


# In[28]:


#Working on Phenotype


# In[29]:


#@title Shape of Dataset
print("Shape of the Dataset")
print(f"No of rows: {pheno_df.shape[0]}")
print(f"No of columns: {pheno_df.shape[1]}")


# In[30]:


#@title Columns of Dataset
print("Columns of the Dataset")
pheno_df.columns


# In[31]:


#@title Overview of Dataset
print("First 5 rows of the Dataset")
pheno_df.head()


# In[32]:


#@title Column Data type
pheno_df.dtypes


# In[33]:


#@title No of Null Values in each column
null_counts = pheno_df.isnull().sum()
null_percentages = (pheno_df.isnull().sum() / pheno_df.shape[0]) * 100

null_info_df = pd.DataFrame({
    'Column_name': null_counts.index,
    'No of Null Values': null_counts.values,
    '% of column': null_percentages.values
})

print("Null Values Information:")
print(null_info_df)


# In[34]:


#@title No of Unique Values in each column
print("No of Unique Values in Each column")
unique_values = pheno_df.nunique()
unique_values_df = pd.DataFrame({
    'Column_name': unique_values.index,
    'No of Unique Values': unique_values})
print(unique_values_df)


# In[35]:


#@title Unique Values in each column
for column in pheno_df.columns:
    unique_values = pheno_df[column].unique()
    print(f"Unique values in column '{column}':")
    print(unique_values)
    print()


# In[36]:


#@title Usable Columns
pheno_type = pheno_df[[
    "BioSample_ID",
    "antibiotic_name",
    "resistance_phenotype"
]].copy()


# In[37]:


#@title Shape of Dataset
print("Shape of the Dataset")
print(f"No of rows: {pheno_type.shape[0]}")
print(f"No of columns: {pheno_type.shape[1]}")


# In[38]:


#@title Columns of Dataset
print("Columns of the Dataset")
pheno_type.columns


# In[39]:


#@title Overview of Dataset
print("First 5 rows of the Dataset")
pheno_type.head()


# In[40]:


#@title Column Data type
pheno_type.dtypes


# In[41]:


#@title No of Null Values in each column
null_counts = pheno_type.isnull().sum()
null_percentages = (pheno_type.isnull().sum() / pheno_type.shape[0]) * 100

null_info_df = pd.DataFrame({
    'Column_name': null_counts.index,
    'No of Null Values': null_counts.values,
    '% of column': null_percentages.values
})

print("Null Values Information:")
print(null_info_df)


# In[42]:


#@title No of Unique Values in each column
print("No of Unique Values in Each column")
unique_values = pheno_type.nunique()
unique_values_df = pd.DataFrame({
    'Column_name': unique_values.index,
    'No of Unique Values': unique_values})
print(unique_values_df)


# In[43]:


#@title Unique Values in each column
for column in pheno_type.columns:
    unique_values = pheno_type[column].unique()
    print(f"Unique values in column '{column}':")
    print(unique_values)
    print()


# In[44]:


#Cleaning the Dataset


# In[45]:


#@title Dropping rows where "resistance_phenotype" is NaN
pheno_type_df = pheno_type.dropna(subset=["resistance_phenotype"])


# In[46]:


#@title Shape of new dataset
print("Shape of the Dataset")
print(f"No of rows: {pheno_type_df.shape[0]}")
print(f"No of columns: {pheno_type_df.shape[1]}")


# In[47]:


#@title Mapping "resistance_phenotype"

pheno_type_new = pheno_type_df.copy()

pheno_type_new["resistance_phenotype"] = (
    pheno_type_new["resistance_phenotype"]
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

pheno_type_new["resistance_phenotype"] = pheno_type_new["resistance_phenotype"].map(mapping)


# In[48]:


#@title New Dataset
print("New Dataset")
pheno_type_new.head()


# In[49]:


#@title Shape of the new Dataset
print("Shape of the Dataset")
print(f"No of rows: {pheno_type_new.shape[0]}")
print(f"No of columns: {pheno_type_new.shape[1]}")


# In[50]:


#Merging The Dataset


# In[51]:


#@title Merging to Final Dataset

#@markdown CLEAN BioSample_ID
geno_type_drop['BioSample_ID'] = geno_type_drop['BioSample_ID'].astype(str).str.strip()
pheno_type_new['BioSample_ID'] = pheno_type_new['BioSample_ID'].astype(str).str.strip()

#@markdown CHECK SAMPLE OVERLAP
geno_samples = set(geno_type_drop['BioSample_ID'])
pheno_samples = set(pheno_type_new['BioSample_ID'])

overlap_samples = geno_samples & pheno_samples

print("Total Genotype Samples:", len(geno_samples))
print("Total Phenotype Samples:", len(pheno_samples))
print("Overlapping Samples:", len(overlap_samples))

#@markdown Filter only overlapping samples
geno_filtered = geno_type_drop[geno_type_drop['BioSample_ID'].isin(overlap_samples)]
pheno_filtered = pheno_type_new[pheno_type_new['BioSample_ID'].isin(overlap_samples)]

#@markdown PERFORM MERGE
merged = geno_filtered.merge(
    pheno_filtered,
    on="BioSample_ID",
    how="inner"
)


# In[52]:


#@title Shape of new Dataset
print("Shape of new Dataset")
print(f"No of rows: {merged.shape[0]}")
print(f"No of columns: {merged.shape[1]}")


# In[53]:


#@title No of Null Values in each column
null_counts = merged.isnull().sum()
null_percentages = (merged.isnull().sum() / merged.shape[0]) * 100

null_info_df = pd.DataFrame({
    'Column_name': null_counts.index,
    'No of Null Values': null_counts.values,
    '% of column': null_percentages.values
})

print("Null Values Information:")
print(null_info_df)


# In[54]:


#@title Check for duplicates
duplicate_count = merged.duplicated().sum()
print("Duplicate Rows:", duplicate_count)


# In[55]:


#@title Remove Duplicates
df_cleaned = merged.drop_duplicates()


# In[56]:


#@title Shape of final dataset
print("Shape of final Dataset")
print(f"No of rows: {df_cleaned.shape[0]}")
print(f"No of columns: {df_cleaned.shape[1]}")


# In[57]:


#@title Checking the Integrity of the data
#@markdown Structural Integrity

print("STRUCTURAL INTEGRITY CHECK")

#@markdown Verify unique sample count
print("Merged unique samples:", df_cleaned['BioSample_ID'].nunique())
print("Should equal overlap:", len(overlap_samples))

#@markdown Check duplicates
duplicate_count = df_cleaned.duplicated().sum()
print("Duplicate Rows:", duplicate_count)

#@markdown Row explosion verification (5 random samples)
import random

print("\nSample-level row verification:\n")

random_samples = random.sample(list(overlap_samples), 5)

for sid in random_samples:
    g_count = geno_filtered[geno_filtered['BioSample_ID'] == sid].shape[0]
    p_count = pheno_filtered[pheno_filtered['BioSample_ID'] == sid].shape[0]
    m_count = merged[merged['BioSample_ID'] == sid].shape[0]

    print(f"Sample {sid}")
    print("Genes:", g_count)
    print("Drugs:", p_count)
    print("Merged rows:", m_count)
    print("Expected:", g_count * p_count)
    print("-----------------------------")


# In[58]:


#@title Final Dataset
df_cleaned.head(10)


# In[59]:


#@title No of Unique Values in each column
print("No of Unique Values in Each column")
unique_values =df_cleaned.nunique()
unique_values_df = pd.DataFrame({
    'Column_name': unique_values.index,
    'No of Unique Values': unique_values})
print(unique_values_df)


# In[60]:


#@title Unique values in column
for column in df_cleaned.columns:
    unique_values = df_cleaned[column].unique()
    print(f"Unique values in column '{column}':")
    print(unique_values)
    print()


# In[61]:


#@title Shape of final dataset
print("Shape of final Dataset")
print(f"No of rows: {df_cleaned.shape[0]}")
print(f"No of columns: {df_cleaned.shape[1]}")


# In[62]:


#@title Check for Class Distribution (Class Imbalance)

class_distribution = df_cleaned['resistance_phenotype'].value_counts()
class_percentage = df_cleaned['resistance_phenotype'].value_counts(normalize=True) * 100

class_info_df = pd.DataFrame({
    'Class': class_distribution.index,
    'Count': class_distribution.values,
    'Percentage': class_percentage.values
})

print("Class Distribution of 'resistance_phenotype':")
print(class_info_df)


# In[63]:


##Check for Distrubution preservance


# In[64]:


#@title Using stats
print(geno_type.describe())
print("\n\n\n\n")
print(geno_type_df.describe())

#@title Value count table of categorical columns
cat_cols = ['gene_symbol', 'genus', 'species', 'class', 'subclass']

for col in cat_cols:
    if col in ['class', 'subclass']:
        # Explode the list-like column for value_counts
        original_counts = geno_type[col].explode().value_counts(normalize=True).rename("Original %")
    else:
        original_counts = geno_type[col].value_counts(normalize=True).rename("Original %")

    processed_counts = geno_type_df[col].value_counts(normalize=True).rename("Processed %")

    combined_df = pd.concat([original_counts, processed_counts], axis=1).fillna(0) * 100
    combined_df = combined_df.round(2)

    print(f"\nColumn: {col}")
    print(combined_df)


# In[65]:


#@title Visual Distribution Check (Genotype Data)

# import matplotlib.pyplot as plt
# import seaborn as sns

# # Create derived feature
# geno_type['region_length'] = geno_type['region_end'] - geno_type['region_start']
# geno_type_df['region_length'] = geno_type_df['region_end'] - geno_type_df['region_start']

# # Columns to compare (numeric only)
# cols = ['region_start', 'region_end', 'region_length']

# for col in cols:
#     plt.figure(figsize=(8,4))

#     sns.kdeplot(geno_type[col], label='Original', fill=True)
#     sns.kdeplot(geno_type_df[col], label='Processed', fill=True, alpha=0.5)

#     plt.title(f'Distribution Check: {col}')
#     plt.xlabel(col)
#     plt.ylabel('Density')
#     plt.legend()
#     plt.show()


# In[66]:


# import pandas as pd
# from scipy.stats import ks_2samp

# def check_distribution_preservation(df_orig, df_proc, column_name):
#     # 1. Summary Statistics
#     stats = pd.DataFrame({
#         'Original': df_orig[column_name].describe(),
#         'Processed': df_proc[column_name].describe()
#     })

#     # % change
#     stats['% Change'] = ((stats['Processed'] - stats['Original']) / stats['Original']) * 100

#     # 2. K-S Test
#     statistic, p_value = ks_2samp(
#         df_orig[column_name].dropna(),
#         df_proc[column_name].dropna()
#     )

#     print(f"\n--- Distribution Check: {column_name} ---")
#     print(stats.round(2))
#     print(f"\nK-S Statistic: {statistic:.4f}")
#     print(f"P-Value: {p_value:.4f}")

#     if p_value > 0.05:
#         print("Result: Distribution is statistically PRESERVED (p > 0.05)")
#     else:
#         print("Result: Distribution is statistically DIFFERENT (p < 0.05)")
#     print("-" * 50)


# # ðŸ”¹ Create derived feature (IMPORTANT)
# geno_type['region_length'] = geno_type['region_end'] - geno_type['region_start']
# geno_type_df['region_length'] = geno_type_df['region_end'] - geno_type_df['region_start']

# # ðŸ”¹ Run checks
# cols = ['region_start', 'region_end', 'region_length']

# for col in cols:
#     check_distribution_preservation(geno_type, geno_type_df, col)

# def check_relative_ranking(df_orig, df_proc, column, top_n=10):
#     print(f"\n--- Relative Ranking Check: {column} ---")

#     # Get top categories
#     orig_counts = df_orig[column].value_counts(normalize=True)
#     proc_counts = df_proc[column].value_counts(normalize=True)

#     # Get top N
#     orig_top = orig_counts.head(top_n)
#     proc_top = proc_counts.head(top_n)

#     # Combine
#     combined = pd.DataFrame({
#         'Original %': orig_top,
#         'Processed %': proc_top
#     }).fillna(0) * 100

#     combined = combined.round(2)

#     print("\nTop categories comparison:")
#     print(combined)

#     # Check overlap
#     orig_set = set(orig_top.index)
#     proc_set = set(proc_top.index)

#     common = orig_set & proc_set

#     print(f"\nCommon categories in Top {top_n}: {len(common)}/{top_n}")
#     print(f"Overlap: {len(common)/top_n * 100:.2f}%")

#     print("-" * 50)


# # Run for your dataset
# check_relative_ranking(geno_type, geno_type_df, 'gene_symbol')
# check_relative_ranking(geno_type, geno_type_df, 'genus')
# check_relative_ranking(geno_type, geno_type_df, 'species')


# In[67]:


#@title Remove Rare Gene & Species (Reduce Noise + Improve Model)

# Store initial row count
before_rows = len(df_cleaned)

# ðŸ”¹ Remove rare gene_symbol (<10)
gene_counts = df_cleaned['gene_symbol'].value_counts()
valid_genes = gene_counts[gene_counts >= 10].index
df = df_cleaned[df_cleaned['gene_symbol'].isin(valid_genes)]

# ðŸ”¹ Remove rare species (<50)
species_counts = df['species'].value_counts()
valid_species = species_counts[species_counts >= 50].index
df = df[df['species'].isin(valid_species)]

# Store final row count
after_rows = len(df)

# ðŸ”¹ Print summary
print("Rows before:", before_rows)
print("Rows after:", after_rows)
print("Data retained (%):", round((after_rows / before_rows) * 100, 2))


# In[ ]:


#@title Data Preparation for ANN

import numpy as np
from sklearn.model_selection import train_test_split

def build_resistance_data(df):
    df2 = df.copy()
    df2['resistance_normalized'] = df2['resistance_phenotype'].astype(str).str.strip().str.upper()
    df2 = df2[df2['resistance_normalized'].isin(['S', 'R'])].copy()
    df2['target'] = (df2['resistance_normalized'] == 'S').astype(int)
    df2['region_length'] = df2['region_end'] - df2['region_start']
    df2['class'] = df2['class'].fillna("Unknown")
    df2['subclass'] = df2['subclass'].fillna("Unknown")
    df2 = df2.dropna(subset=['gene_symbol', 'species', 'antibiotic_name', 'class', 'subclass', 'region_length', 'target'])
    
    return df2

df_model = build_resistance_data(df)

print("Model data shape:", df_model.shape)
print("Target distribution:")
print(df_model['target'].value_counts(normalize=True).round(4) * 100)


# In[ ]:


#@title Encode Categorical Features
from sklearn.preprocessing import LabelEncoder
import pickle
import os

save_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
model_path = os.path.join(save_dir, "ann_model.keras")
encoder_path = os.path.join(save_dir, "encoders.pkl")
model_exists = os.path.exists(model_path) and os.path.exists(encoder_path)

if model_exists:
    print("Loading saved encoders...")
    with open(encoder_path, "rb") as f:
        encoders = pickle.load(f)
    gene_encoder = encoders["gene_encoder"]
    species_encoder = encoders["species_encoder"]
    drug_encoder = encoders["drug_encoder"]
    class_encoder = encoders["class_encoder"]
    subclass_encoder = encoders["subclass_encoder"]
    
    df_model['gene_encoded'] = gene_encoder.transform(df_model['gene_symbol'].astype(str))
    df_model['species_encoded'] = species_encoder.transform(df_model['species'].astype(str))
    df_model['drug_encoded'] = drug_encoder.transform(df_model['antibiotic_name'].astype(str))
    df_model['class_encoded'] = class_encoder.transform(df_model['class'].astype(str))
    df_model['subclass_encoded'] = subclass_encoder.transform(df_model['subclass'].astype(str))
else:
    gene_encoder = LabelEncoder()
    species_encoder = LabelEncoder()
    drug_encoder = LabelEncoder()
    class_encoder = LabelEncoder()
    subclass_encoder = LabelEncoder()

    df_model['gene_encoded'] = gene_encoder.fit_transform(df_model['gene_symbol'].astype(str))
    df_model['species_encoded'] = species_encoder.fit_transform(df_model['species'].astype(str))
    df_model['drug_encoded'] = drug_encoder.fit_transform(df_model['antibiotic_name'].astype(str))
    df_model['class_encoded'] = class_encoder.fit_transform(df_model['class'].astype(str))
    df_model['subclass_encoded'] = subclass_encoder.fit_transform(df_model['subclass'].astype(str))

    encoders = {
        "gene_encoder": gene_encoder,
        "species_encoder": species_encoder,
        "drug_encoder": drug_encoder,
        "class_encoder": class_encoder,
        "subclass_encoder": subclass_encoder
    }

    # Save initial metadata (No need to wait for training)
    import json
    metadata = {
        "total_samples": int(df_model.shape[0]),
        "total_genes": int(len(gene_encoder.classes_)),
        "total_drugs": int(len(drug_encoder.classes_)),
        "total_species": int(len(species_encoder.classes_)),
        "model_accuracy": "Awaiting Training...",
        "data_source": "NCBI AMR Portal (ayates/amr_portal)"
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"📊 Project stats saved to metadata.json. Dashboard ready!")


# In[ ]:


#@title Build Feedforward ANN Model
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

if model_exists:
    print("Loading pre-trained model...")
    model = tf.keras.models.load_model(model_path)
    model.summary()
else:
    tf.keras.utils.set_random_seed(42)

    # Inputs
    input_gene = Input(shape=(1,), name="gene_input")
    input_species = Input(shape=(1,), name="species_input")
    input_drug = Input(shape=(1,), name="drug_input")
    input_class = Input(shape=(1,), name="class_input")
    input_subclass = Input(shape=(1,), name="subclass_input")
    input_region = Input(shape=(1,), name="region_length_input")

    # Embeddings
    emb_gene = Embedding(input_dim=len(gene_encoder.classes_), output_dim=128)(input_gene)
    emb_gene_flat = Flatten()(emb_gene)

    emb_species = Embedding(input_dim=len(species_encoder.classes_), output_dim=64)(input_species)
    emb_species_flat = Flatten()(emb_species)

    emb_drug = Embedding(input_dim=len(drug_encoder.classes_), output_dim=32)(input_drug)
    emb_drug_flat = Flatten()(emb_drug)

    emb_class = Embedding(input_dim=len(class_encoder.classes_), output_dim=32)(input_class)
    emb_class_flat = Flatten()(emb_class)

    emb_subclass = Embedding(input_dim=len(subclass_encoder.classes_), output_dim=32)(input_subclass)
    emb_subclass_flat = Flatten()(emb_subclass)

    # Concatenate all features
    concat = Concatenate()([emb_gene_flat, emb_species_flat, emb_drug_flat, emb_class_flat, emb_subclass_flat, input_region])

    # Dense Layers
    x = Dense(512)(concat)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.1)(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Output
    output = Dense(1, activation='sigmoid', name="output")(x)

    model = Model(inputs=[input_gene, input_species, input_drug, input_class, input_subclass, input_region], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()


# In[ ]:


#@title Train the Model
from sklearn.preprocessing import StandardScaler

X_gene = df_model['gene_encoded'].values
X_species = df_model['species_encoded'].values
X_drug = df_model['drug_encoded'].values
X_class = df_model['class_encoded'].values
X_subclass = df_model['subclass_encoded'].values
X_region = df_model['region_length'].values
y = df_model['target'].values

(Xg_train, Xg_test, 
 Xs_train, Xs_test, 
 Xd_train, Xd_test,
 Xc_train, Xc_test,
 Xsc_train, Xsc_test,
 Xr_train, Xr_test, 
 y_train, y_test) = train_test_split(
    X_gene, X_species, X_drug, X_class, X_subclass, X_region, y,
    test_size=0.2, random_state=42
)

if model_exists:
    scaler = encoders["scaler"]
    Xr_test = scaler.transform(Xr_test.reshape(-1, 1)).flatten()
    print("Skipping training as pre-trained model was loaded.")
else:
    # Standard Scaling for Region Length
    scaler = StandardScaler()
    Xr_train = scaler.fit_transform(Xr_train.reshape(-1, 1)).flatten()
    Xr_test = scaler.transform(Xr_test.reshape(-1, 1)).flatten()

    # Save scaler
    encoders["scaler"] = scaler

    # Compute Class Weights (Re-added to prevent majority inflation)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {
        0: float(weights[0]) if 0 in classes else 1.0,
        1: float(weights[1]) if 1 in classes else 1.0
    }

    # Let the model train freely, waiting for val_accuracy plateau of 2 epochs
    early_stop = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True, mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6, verbose=1)

    history = model.fit(
        x=[Xg_train, Xs_train, Xd_train, Xc_train, Xsc_train, Xr_train],
        y=y_train,
        validation_data=([Xg_test, Xs_test, Xd_test, Xc_test, Xsc_test, Xr_test], y_test),
        epochs=100,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )


# In[ ]:


#@title Evaluate and Save Model
from sklearn.metrics import accuracy_score, classification_report

y_pred_probs = model.predict([Xg_test, Xs_test, Xd_test, Xc_test, Xsc_test, Xr_test])
y_pred_class = (y_pred_probs >= 0.5).astype(int).flatten()

test_acc = accuracy_score(y_test, y_pred_class)
print(f"\nTest Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_class, target_names=['Resistant (0)', 'Susceptible (1)']))

if not model_exists:
    print("\nSaving the trained model and encoders unconditionally in the AMM folder...")
    import os
    save_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    model_path = os.path.join(save_dir, "ann_model.keras")
    encoder_path = os.path.join(save_dir, "encoders.pkl")

    model.save(model_path)
    with open(encoder_path, "wb") as f:
        pickle.dump(encoders, f)
    
    # Update Metadata with Final Accuracy
    try:
        with open(os.path.join(save_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        metadata["model_accuracy"] = f"{test_acc*100:.2f}%"
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        print("✅ Accuracy updated in metadata.json")
    except:
        pass

    print(f"Model successfully saved in {model_path}!")
    print(f"Encoders successfully saved in {encoder_path}!")
    print(f"Metadata saved in metadata.json")
else:
    print("\nModel already exists and was loaded. Skipping save.")



# In[ ]:


#@title Overfitting Check
import matplotlib.pyplot as plt

if not model_exists:
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    acc_gap = final_train_acc - final_val_acc

    print("=" * 50)
    print("OVERFITTING DIAGNOSTIC REPORT")
    print("=" * 50)
    print(f"Final Training Accuracy:   {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Gap (Train - Val):         {acc_gap:.4f}")
    print(f"Final Training Loss:       {train_loss[-1]:.4f}")
    print(f"Final Validation Loss:     {val_loss[-1]:.4f}")
    print("-" * 50)

    if acc_gap < 0.02:
        print("VERDICT: No overfitting detected. Model generalizes well.")
    elif acc_gap < 0.05:
        print("VERDICT: Slight overfitting. Acceptable for this task.")
    elif acc_gap < 0.10:
        print("VERDICT: Moderate overfitting. Consider more dropout or less epochs.")
    else:
        print("VERDICT: Significant overfitting! Train acc >> Val acc.")
    print("=" * 50)

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_acc, label='Train Accuracy', linewidth=2)
    ax1.plot(val_acc, label='Val Accuracy', linewidth=2)
    ax1.set_title('Accuracy Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_loss, label='Train Loss', linewidth=2)
    ax2.plot(val_loss, label='Val Loss', linewidth=2)
    ax2.set_title('Loss Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(save_dir, 'static', 'training_curves.png')
    plt.savefig(curves_path, dpi=150)
    plt.show()
    print(f"Training curves saved to '{curves_path}'")
else:
    print("\nSkipping overfitting diagnostic as model was not trained in this session.")


# In[ ]:


#@title Recommendation System Implementation

def recommend_best_drug(gene_symbol, region_length, species, allergic_drugs=None):
    if allergic_drugs is None:
        allergic_drugs = []
    
    allergic_drugs_lower = [d.lower() for d in allergic_drugs]
    valid_drugs = df_model[df_model['gene_symbol'] == gene_symbol]['antibiotic_name'].unique()
    candidate_drugs = [d for d in valid_drugs if d.lower() not in allergic_drugs_lower]
    
    if len(candidate_drugs) == 0:
        return "Talk to a professional"

    # Get class and subclass for this gene from the dataset
    gene_data = df_model[df_model['gene_symbol'] == gene_symbol].iloc[0]
    gene_class = gene_data['class']
    gene_subclass = gene_data['subclass']

    try:
        g_encoded = gene_encoder.transform([gene_symbol] * len(candidate_drugs))
        s_encoded = species_encoder.transform([species] * len(candidate_drugs))
        d_encoded = drug_encoder.transform(candidate_drugs)
        c_encoded = class_encoder.transform([gene_class] * len(candidate_drugs))
        sc_encoded = subclass_encoder.transform([gene_subclass] * len(candidate_drugs))
        
        # Scale region length before inference
        raw_r = np.array([region_length] * len(candidate_drugs)).reshape(-1, 1)
        r_encoded = encoders["scaler"].transform(raw_r).flatten()
    except ValueError:
        return "Talk to a professional (Unknown input values)"
    
    pred_probs = model.predict([g_encoded, s_encoded, d_encoded, c_encoded, sc_encoded, r_encoded], verbose=0).flatten()
    
    results = []
    for drug, prob in zip(candidate_drugs, pred_probs):
        susceptibility_pct = prob * 100
        label = "S" if prob >= 0.5 else "R"
        results.append({
            "drug": drug,
            "label": label,
            "susceptibility_pct": susceptibility_pct
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['label', 'susceptibility_pct'], ascending=[False, False])
    return results_df.head(5)

# Example Usage
print("Recommendation Example:")
example_gene = df_model['gene_symbol'].iloc[0]
example_species = df_model['species'].iloc[0]
example_region_len = df_model['region_length'].iloc[0]

results = recommend_best_drug(example_gene, example_region_len, example_species, allergic_drugs=['Medicine XYZ'])
if isinstance(results, str):
    print(results)
else:
    for _, row in results.iterrows():
        print(f"Drug {row['drug']} -> {row['label']} ({row['susceptibility_pct']:.2f}% susceptible)")


