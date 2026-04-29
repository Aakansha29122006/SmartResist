#!/usr/bin/env python
# coding: utf-8
"""
SmartResist — Flask API Server
================================
Loads trained model + encoders from /ANN_Project/, serves predictions via REST API.
Implements: Hybrid System (ANN + Rule-based filtering), Allergy removal,
            Ranking layer, Unseen gene handling, Support counts.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANN_DIR = os.path.join(BASE_DIR, "ANN_Project")
MODEL_PATH = os.path.join(ANN_DIR, "model.keras")
ENCODER_PATH = os.path.join(ANN_DIR, "encoders.pkl")
SCALER_PATH = os.path.join(ANN_DIR, "scaler.pkl")
SUPPORT_PATH = os.path.join(ANN_DIR, "drug_support.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.json")

model = None
encoders = None
scaler = None
metadata = None
drug_support = None
model_ready = False
tf_error = None


def load_model_artifacts():
    """Load all saved artifacts from /ANN_Project/ on startup."""
    global model, encoders, scaler, metadata, drug_support, model_ready, tf_error

    try:
        required = [MODEL_PATH, ENCODER_PATH, SCALER_PATH, METADATA_PATH]
        missing = [f for f in required if not os.path.exists(f)]
        if missing:
            tf_error = f"Missing: {', '.join(os.path.basename(f) for f in missing)}"
            print(f"[WARN] {tf_error}")
            print("  Run 'python main.py --train' first.")
            return

        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[OK] Model loaded from {MODEL_PATH}")

        with open(ENCODER_PATH, 'rb') as f:
            encoders = pickle.load(f)
        print(f"[OK] Encoders loaded")

        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"[OK] Scaler loaded")

        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        print(f"[OK] Metadata loaded")

        if os.path.exists(SUPPORT_PATH):
            with open(SUPPORT_PATH, 'rb') as f:
                drug_support = pickle.load(f)
            print(f"[OK] Drug support loaded")

        model_ready = True
        print(f"\n>> SmartResist AI ready! Accuracy: {metadata.get('model_accuracy', 'N/A')}")

    except Exception as e:
        tf_error = str(e)
        print(f"[FAIL] Loading failed: {tf_error}")


# ══════════════════════════════════════════════════════════════════
# HYBRID PREDICTION (Steps 12-17)
# ══════════════════════════════════════════════════════════════════

def recommend_antibiotics(gene_symbol, region_start, region_end, allergic_drugs=None):
    """ANN prediction + rule-based filtering + ranking."""
    if allergic_drugs is None:
        allergic_drugs = []

    known_genes = metadata.get('known_genes', [])
    gene_drug_map = metadata.get('gene_drug_map', {})
    gene_species_map = metadata.get('gene_species_map', {})

    # Step 17: Unseen gene (Case Insensitive)
    matched_gene = next((g for g in known_genes if g.lower() == gene_symbol.lower()), None)
    if not matched_gene:
        return {"error": "Talk to a professional", "reason": "Gene not found in training data"}
    
    gene_symbol = matched_gene

    # Step 12: Candidate space
    candidate_drugs = gene_drug_map.get(gene_symbol, [])

    # Step 14: Remove allergies
    allergic_lower = [d.lower().strip() for d in allergic_drugs if d.strip()]
    candidate_drugs = [d for d in candidate_drugs if d.lower() not in allergic_lower]

    if not candidate_drugs:
        return {"error": "Talk to a professional",
                "reason": "No candidates after allergy filtering"}

    species = gene_species_map.get(gene_symbol, 'Unknown')
    region_length = region_end - region_start

    gene_enc = encoders['gene_encoder']
    species_enc = encoders['species_encoder']
    drug_enc = encoders['drug_encoder']

    # Filter to known drugs
    known_drugs = set(drug_enc.classes_)
    valid_drugs = [d for d in candidate_drugs if d in known_drugs]
    if not valid_drugs:
        return {"error": "Talk to a professional", "reason": "No encodable candidates"}

    try:
        n = len(valid_drugs)
        g = gene_enc.transform([gene_symbol] * n)
        s = species_enc.transform([species] * n)
        d = drug_enc.transform(valid_drugs)

        # Scale numerical features
        num_raw = np.array([[region_start, region_end, region_length]] * n)
        num_scaled = scaler.transform(num_raw)

        X = np.column_stack([g, s, d, num_scaled[:, 0], num_scaled[:, 1], num_scaled[:, 2]])
    except ValueError as e:
        return {"error": "Talk to a professional", "reason": f"Encoding error: {e}"}

    pred_probs = model.predict(X, verbose=0).flatten()

    results = []
    for drug_name, prob in zip(valid_drugs, pred_probs):
        support = 0
        if drug_support is not None:
            match = drug_support[
                (drug_support['gene_symbol'] == gene_symbol) &
                (drug_support['antibiotic_name'] == drug_name)
            ]
            if len(match) > 0:
                support = int(match['support'].values[0])

        if support < 5:
            continue

        results.append({
            'drug': drug_name,
            'label': "S" if prob >= 0.5 else "R",
            'susceptibility': round(float(prob * 100), 2),
            'support': support
        })

    results.sort(key=lambda x: (-1 if x['label'] == 'S' else 0, -x['susceptibility']))
    top = results[:5]

    if not top:
        return {"error": "Talk to a professional",
                "reason": "No drugs with sufficient evidence (support >= 5)"}

    return {
        "gene": gene_symbol, "species": species,
        "region_start": region_start, "region_end": region_end,
        "recommendations": top, "total_candidates": len(results)
    }


# ══════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    stats = {}
    if metadata:
        stats = {
            'total_samples': metadata.get('total_samples', 0),
            'total_genes': metadata.get('total_genes', 0),
            'total_drugs': metadata.get('total_drugs', 0),
            'total_species': metadata.get('total_species', 0),
            'model_accuracy': metadata.get('model_accuracy', 'N/A'),
        }
    return jsonify({'model_ready': model_ready, 'tf_error': tf_error, 'dataset_stats': stats})


@app.route('/api/autocomplete')
def api_autocomplete():
    q = request.args.get('q', '').strip().lower()
    type_ = request.args.get('type', 'gene')
    if not q or not metadata:
        return jsonify([])
    
    if type_ == 'drug':
        known_drugs = metadata.get('known_drugs', []) # This might not exist directly, let's extract from encoders or something if not
        # Let's extract known_drugs if not in metadata, or we can use gene_drug_map keys
        # Wait, app.py has encoders available as a global, let's use it or just the keys of the gene_drug_map?
        # A simpler way if known_drugs is not in metadata is to collect all unique drugs from gene_drug_map
        if 'known_drugs' in metadata:
            known = metadata['known_drugs']
        else:
            known = list(set([d for d_list in metadata.get('gene_drug_map', {}).values() for d in d_list]))
    else:
        known = metadata.get('known_genes', [])
        
    return jsonify([g for g in known if q in g.lower()][:15])


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not model_ready:
        return jsonify({'error': 'Model not loaded. Run main.py --train first.'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    genes = data.get('genes', [])
    allergies = data.get('allergies', [])
    if not genes:
        return jsonify({'error': 'No genes provided'}), 400

    results = []
    for gi in genes:
        gene = gi.get('gene', '').strip()
        rs = float(gi.get('region_start', 0))
        re = float(gi.get('region_end', 0))
        if not gene:
            results.append({'gene': gene, 'error': 'Gene name required'})
            continue
        results.append(recommend_antibiotics(gene, rs, re, allergies))

    final = {'drug': 'Talk to a professional', 'susceptibility': '0%'}
    if results and 'recommendations' in results[0] and results[0]['recommendations']:
        t = results[0]['recommendations'][0]
        final = {'drug': t['drug'], 'susceptibility': f"{t['susceptibility']:.1f}%",
                 'support': t['support']}

    return jsonify({
        'genes': results, 'final_recommendation': final, 'allergies_applied': allergies
    })


@app.route('/api/test')
def api_test():
    """Step 22: Run test cases via API."""
    if not model_ready:
        return jsonify({'error': 'Model not loaded'}), 503

    known = metadata.get('known_genes', [])
    tests = []

    if known:
        tests.append({'case': 'Common gene', 'result': recommend_antibiotics(known[0], 125000, 126000)})
    if len(known) > 5:
        tests.append({'case': 'Different gene', 'result': recommend_antibiotics(known[5], 50000, 51000)})
    if len(known) > 1:
        tests.append({'case': 'Rare gene', 'result': recommend_antibiotics(known[-1], 100000, 101000)})

    tests.append({'case': 'Unseen gene', 'result': recommend_antibiotics("FAKE_XYZ", 0, 100)})

    if known:
        tests.append({'case': 'With allergy',
                      'result': recommend_antibiotics(known[0], 125000, 126000, ["amikacin"])})

    return jsonify({'test_cases': tests, 'total': len(tests)})


# ══════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("============================================================")
    print("|   SmartResist -- Flask Prediction Server                 |")
    print("============================================================\n")
    load_model_artifacts()
    app.run(debug=False, port=5000, host='0.0.0.0')
