#!/usr/bin/env python
# coding: utf-8
"""
STEP 18 — EVALUATION METRICS
==============================
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification Report
- Overfitting Diagnostic
"""

import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)


def evaluate(model, X_test, y_test):
    """
    Step 18: Full evaluation metrics.

    Returns:
        metrics: Dict with accuracy, precision, recall, f1, roc_auc.
    """
    print("\n" + "=" * 60)
    print("STEP 18: EVALUATION METRICS")
    print("=" * 60)

    y_pred_probs = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        roc = roc_auc_score(y_test, y_pred_probs)
    except ValueError:
        roc = 0.0

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Resistant (0)', 'Susceptible (1)']
    ))

    metrics = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc, 4)
    }
    return metrics


def overfitting_check(history, save_dir=None):
    """
    Overfitting diagnostic report + training curves plot.

    Args:
        history: Keras training history object.
        save_dir: Directory to save training_curves.png.
    """
    if history is None:
        print("\n  Skipping overfitting diagnostic (model was not trained this session).")
        return

    print("\n" + "=" * 60)
    print("OVERFITTING DIAGNOSTIC REPORT")
    print("=" * 60)

    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    acc_gap = final_train_acc - final_val_acc

    print(f"  Final Training Accuracy:   {final_train_acc:.4f}")
    print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"  Gap (Train - Val):         {acc_gap:.4f}")
    print(f"  Final Training Loss:       {train_loss[-1]:.4f}")
    print(f"  Final Validation Loss:     {val_loss[-1]:.4f}")

    if acc_gap < 0.02:
        print("  VERDICT: No overfitting detected. Model generalizes well.")
    elif acc_gap < 0.05:
        print("  VERDICT: Slight overfitting. Acceptable for this task.")
    elif acc_gap < 0.10:
        print("  VERDICT: Moderate overfitting. Consider more dropout or less epochs.")
    else:
        print("  VERDICT: Significant overfitting! Train acc >> Val acc.")

    # Plot training curves
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if save_dir is None:
            save_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.dirname(save_dir)

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
        os.makedirs(os.path.dirname(curves_path), exist_ok=True)
        plt.savefig(curves_path, dpi=150)
        plt.close()
        print(f"\n  Training curves saved to '{curves_path}'")
    except ImportError:
        print("\n  matplotlib not available — skipping plot generation.")
