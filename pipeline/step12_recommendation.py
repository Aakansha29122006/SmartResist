#!/usr/bin/env python
# coding: utf-8
"""
STEPS 12-17 — RECOMMENDATION SYSTEM (HYBRID DESIGN)
=====================================================
12. Candidate Antibiotic Space
    - All antibiotics seen in training OR filtered by gene association

13. Hybrid System Design
    - ANN (MLP) for prediction
    - Rule-based system for: gene–drug filtering, allergy removal, candidate restriction

14. Constraint and Filtering
    - Only valid gene–drug pairs allowed
    - Remove allergic medications
    - Remove invalid predictions

15. Ranking Layer
    - Sort by confidence score
    - Attach support count

16. Output Layer
    - Top 5 antibiotics
    - Confidence score
    - Support count

17. Unseen Gene Handling
    - If gene not found: Output "Talk to a professional"

STEP 22 — Testing Requirement
================================
- Multiple test cases: different gene inputs, edge cases, unseen gene scenario
- Validate system output consistency
"""

import numpy as np
import pandas as pd


def recommend_best_drug(gene_symbol, region_start, region_end, species,
                        model, encoders, scaler, df_model,
                        allergic_drugs=None):
    """
    Hybrid recommendation: ANN prediction + rule-based filtering.

    Args:
        gene_symbol: Target resistance gene.
        region_start: Genomic region start.
        region_end: Genomic region end.
        species: Bacterial species.
        model: Trained Keras model.
        encoders: Dict of LabelEncoders.
        scaler: StandardScaler for numerical features.
        df_model: Training dataframe (for candidate lookup).
        allergic_drugs: List of drug names to exclude (optional).

    Returns:
        pd.DataFrame with top 5 results, or str fallback message.
    """
    if allergic_drugs is None:
        allergic_drugs = []

    region_length = region_end - region_start

    # ── Step 17: Unseen Gene Handling ──
    known_genes = df_model['gene_symbol'].unique()
    if gene_symbol not in known_genes:
        return "Talk to a professional"

    # ── Step 12: Candidate Antibiotic Space (filtered by gene association) ──
    valid_drugs = df_model[
        df_model['gene_symbol'] == gene_symbol
    ]['antibiotic_name'].unique()

    # ── Step 14: Remove allergic medications ──
    allergic_lower = [d.lower().strip() for d in allergic_drugs if d.strip()]
    candidate_drugs = [d for d in valid_drugs if d.lower() not in allergic_lower]

    if len(candidate_drugs) == 0:
        return "Talk to a professional"

    # ── Step 14: Only valid gene–drug pairs allowed ──
    gene_encoder = encoders['gene_encoder']
    species_encoder = encoders['species_encoder']
    drug_encoder = encoders['drug_encoder']

    # Filter to drugs the encoder knows
    known_drug_set = set(drug_encoder.classes_)
    candidate_drugs = [d for d in candidate_drugs if d in known_drug_set]

    if len(candidate_drugs) == 0:
        return "Talk to a professional"

    try:
        n = len(candidate_drugs)
        g_enc = gene_encoder.transform([gene_symbol] * n)
        s_enc = species_encoder.transform([species] * n)
        d_enc = drug_encoder.transform(candidate_drugs)

        # Scale numerical features: region_start, region_end, region_length
        num_raw = np.array([[region_start, region_end, region_length]] * n)
        num_scaled = scaler.transform(num_raw)

        # Build feature matrix matching training format:
        # [gene_encoded, species_encoded, drug_encoded,
        #  region_start_scaled, region_end_scaled, region_length_scaled]
        X = np.column_stack([
            g_enc, s_enc, d_enc,
            num_scaled[:, 0], num_scaled[:, 1], num_scaled[:, 2]
        ])

    except ValueError:
        return "Talk to a professional (Unknown input values)"

    # ── Step 13: ANN (MLP) prediction ──
    pred_probs = model.predict(X, verbose=0).flatten()

    # ── Step 15: Ranking — sort by confidence + attach support count ──
    results = []
    for drug, prob in zip(candidate_drugs, pred_probs):
        label = "S" if prob >= 0.5 else "R"

        # Support count from training data
        support = len(df_model[
            (df_model['gene_symbol'] == gene_symbol) &
            (df_model['antibiotic_name'] == drug)
        ])

        # Step 11: Minimum support filtering (n >= 5)
        if support < 5:
            continue

        results.append({
            "drug": drug,
            "label": label,
            "susceptibility_pct": round(float(prob * 100), 2),
            "support": support
        })

    # Step 14: Remove invalid predictions (no results after filtering)
    if len(results) == 0:
        return "Talk to a professional"

    results_df = pd.DataFrame(results)

    # Sort: S first, then by confidence descending
    results_df = results_df.sort_values(
        by=['label', 'susceptibility_pct'],
        ascending=[False, False]
    )

    # ── Step 16: Top 5 antibiotics with confidence score + support count ──
    return results_df.head(5)


def run_test_cases(model, encoders, scaler, df_model):
    """
    Step 22: Multiple test cases.

    Tests:
      1. Common gene input
      2. Different gene input
      3. Rare gene (edge case)
      4. Unseen gene scenario
      5. With allergy filter
    """
    print("\n" + "=" * 60)
    print("STEP 22: TEST CASES")
    print("=" * 60)

    genes = df_model['gene_symbol'].unique()

    def get_gene_info(gene):
        subset = df_model[df_model['gene_symbol'] == gene]
        return (
            subset['region_start'].iloc[0],
            subset['region_end'].iloc[0],
            subset['species'].iloc[0]
        )

    test_cases = []

    # Test 1: Common gene
    g1 = genes[0]
    rs1, re1, sp1 = get_gene_info(g1)
    test_cases.append(("Common gene", g1, rs1, re1, sp1, []))

    # Test 2: Different gene
    idx2 = min(5, len(genes) - 1)
    g2 = genes[idx2]
    rs2, re2, sp2 = get_gene_info(g2)
    test_cases.append(("Different gene", g2, rs2, re2, sp2, []))

    # Test 3: Rare gene (last in list, edge case)
    g3 = genes[-1]
    rs3, re3, sp3 = get_gene_info(g3)
    test_cases.append(("Rare gene (edge)", g3, rs3, re3, sp3, []))

    # Test 4: Unseen gene scenario
    test_cases.append(("Unseen gene", "FAKE_GENE_XYZ_999", 0, 100, "Unknown", []))

    # Test 5: With allergy filter
    test_cases.append(("With allergy", g1, rs1, re1, sp1, ['Medicine XYZ']))

    for name, gene, rs, re, species, allergies in test_cases:
        print(f"\n  Test: {name}")
        print(f"  Gene: {gene}, Start: {rs}, End: {re}, Species: {species}")
        if allergies:
            print(f"  Allergies: {allergies}")

        result = recommend_best_drug(
            gene, rs, re, species,
            model, encoders, scaler, df_model, allergies
        )

        if isinstance(result, str):
            print(f"  Result: {result}")
        else:
            for _, row in result.iterrows():
                print(f"    Drug {row['drug']} → {row['label']} "
                      f"({row['susceptibility_pct']:.2f}%, support={row['support']})")
        print("  " + "-" * 40)
