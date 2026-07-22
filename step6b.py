
"""# **6b — CNN Branch Evaluation**"""

# @title
# -*- coding: utf-8 -*-
"""STEP 6b — CNN Branch Evaluation
====================================
Evaluates the quality of the CNN embeddings produced by Step 6
by training classical classifiers on top of them — exactly mirroring
Step 5b (radiomic branch evaluation) so results are directly comparable.

Input  : deep_embeddings_train/val/test.npy   (from step6_outputs)
         y_train/val/test.npy                 (from step2_outputs)

Output (saved to step6b_outputs):
  results_val.csv / results_test.csv
  comparison_val.png / comparison_test.png
  brier_val.png / brier_test.png
  roc_curves.png
  pr_curves.png
  confusion_matrices.png
  calibration.png
  best_model.pkl
  all_models.pkl
  best_hparams.pkl

Why this step exists
--------------------
The Step 6 training loop only measures the CNN's own linear head (one
Linear layer on top of the 128-D embedding).  That is a weak classifier
deliberately kept simple so the embedding head learns general features
rather than overfitting to the classification task.

Step 6b asks: how much PDAC signal do the embeddings contain when you
give them a proper classifier?  The answer becomes the CNN-only baseline
in your thesis ablation table:

  Radiomic only  (Step 5b best model)  → AUC ~ 0.57
  CNN only       (Step 6b best model)  → AUC ~ ?
  Fusion         (Step 7)              → AUC ~ ?

Without Step 6b you cannot prove the fusion is better than either
branch alone.

Metrics reported
----------------
  ROC-AUC, Average Precision, Brier Score,
  Accuracy, Sensitivity, Specificity, F1, MCC
"""

# @title
# ── Imports ───────────────────────────────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.calibration     import calibration_curve
from sklearn.model_selection import ParameterGrid
from sklearn.metrics         import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    roc_curve, precision_recall_curve,
)
from xgboost import XGBClassifier

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Update these paths to match your environment
STEP2_DIR   = "/home/student1/ftzina_thesis/step6/step2_outputs"
STEP6_DIR   = "/home/student1/ftzina_thesis/step6/step6_outputs"
OUTPUT_DIR  = "/home/student1/ftzina_thesis/step6/step6b_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
DPI          = 150

PALETTE = {
    "LogReg" : "#2196F3",
    "SVM"    : "#FF9800",
    "RF"     : "#4CAF50",
    "XGB"    : "#E91E63",
}

# ── Load embeddings and labels ────────────────────────────────────────────────
X_train = np.load(os.path.join(STEP6_DIR, "deep_embeddings_train.npy"))
X_val = np.load(os.path.join(STEP6_DIR, "deep_embeddings_val.npy"))
X_test = np.load(os.path.join(STEP6_DIR, "deep_embeddings_test.npy"))

y_train = np.load(os.path.join(STEP2_DIR, "y_train.npy"))
y_val = np.load(os.path.join(STEP2_DIR, "y_val.npy"))
y_test = np.load(os.path.join(STEP2_DIR, "y_test.npy"))

# Feature names — embeddings have no semantic names, just indices
feat_names = [f"emb_{i}" for i in range(X_train.shape[1])]

n_pos = int(y_train.sum())
n_neg = int((y_train == 0).sum())
spw   = n_neg / n_pos

print(f"Embeddings — train:{X_train.shape}  val:{X_val.shape}  test:{X_test.shape}")
print(f"PDAC       — train:{n_pos}  non-PDAC:{n_neg}")
print(f"Embedding dimension: {X_train.shape[1]}\n")

"""## Helpers  (identical to Step 5b)"""

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "ROC-AUC"    : round(roc_auc_score(y_true, y_prob), 4),
        "Avg-Prec"   : round(average_precision_score(y_true, y_prob), 4),
        "Brier"      : round(brier_score_loss(y_true, y_prob), 4),
        "Accuracy"   : round(accuracy_score(y_true, y_pred), 4),
        "Sensitivity": round(sens, 4),
        "Specificity": round(spec, 4),
        "F1"         : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC"        : round(matthews_corrcoef(y_true, y_pred), 4),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "threshold"  : threshold,
    }

def find_best_threshold(y_true, y_prob):
    """Youden's J: maximises sensitivity + specificity on the given split."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])

"""## Model grids  (identical to Step 5b)"""

MODEL_GRIDS = {
    "LogReg": {
        "class": LogisticRegression,
        "fixed": {"solver": "lbfgs", "max_iter": 1000,
                  "class_weight": "balanced", "random_state": RANDOM_STATE},
        "grid" : {"C": [0.01, 0.1, 1.0, 10.0]},
    },
    "SVM": {
        "class": SVC,
        "fixed": {"kernel": "rbf", "probability": True,
                  "class_weight": "balanced", "random_state": RANDOM_STATE},
        "grid" : {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
    },
    "RF": {
        "class": RandomForestClassifier,
        "fixed": {"n_estimators": 500, "class_weight": "balanced_subsample",
                  "random_state": RANDOM_STATE, "n_jobs": -1},
        "grid" : {"max_depth": [None, 5, 10],
                  "min_samples_leaf": [1, 2, 4]},
    },
    "XGB": {
        "class": XGBClassifier,
        "fixed": {"n_estimators": 300, "eval_metric": "logloss",
                  "scale_pos_weight": spw,
                  "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1},
        "grid" : {"max_depth": [3, 5, 7],
                  "learning_rate": [0.01, 0.05, 0.1],
                  "subsample": [0.8, 1.0]},
    },
}

"""## Training — grid search on val AUC, test set never touched"""

#CNN Branch — classifier grid search on val AUC
best_models     = {}
best_hparams    = {}
val_probs       = {}
test_probs      = {}
best_thresholds = {}

for name, cfg in MODEL_GRIDS.items():
    print(f"\n── {name} ──────────────────────────────────────")
    best_auc, best_model, best_param = -1, None, None

    for params in ParameterGrid(cfg["grid"]):
        clf = cfg["class"](**{**cfg["fixed"], **params})
        clf.fit(X_train, y_train)
        prob_v = clf.predict_proba(X_val)[:, 1]
        auc_v  = roc_auc_score(y_val, prob_v)
        if auc_v > best_auc:
            best_auc, best_model, best_param = auc_v, clf, params

    print(f"  Best params : {best_param}")
    print(f"  Val AUC     : {best_auc:.4f}")

    best_models[name]     = best_model
    best_hparams[name]    = best_param
    val_probs[name]       = best_model.predict_proba(X_val)[:, 1]
    test_probs[name]      = best_model.predict_proba(X_test)[:, 1]
    best_thresholds[name] = find_best_threshold(y_val, val_probs[name])

print()

"""## Metrics"""

records_val, records_test = [], []
for name in best_models:
    thr = best_thresholds[name]
    records_val.append( {"Model": name,
                          **compute_metrics(y_val,  val_probs[name],  thr)})
    records_test.append({"Model": name,
                          **compute_metrics(y_test, test_probs[name], thr)})

df_val  = pd.DataFrame(records_val).set_index("Model")
df_test = pd.DataFrame(records_test).set_index("Model")
best_name = df_val["ROC-AUC"].idxmax()

METRIC_DISPLAY = ["ROC-AUC","Avg-Prec","Brier","Accuracy",
                  "Sensitivity","Specificity","F1","MCC"]

print("=" * 65)
print("VALIDATION METRICS  (CNN embeddings)")
print("=" * 65)
print(df_val[METRIC_DISPLAY].to_string())
print(f"\n  ★ Best model by val AUC: {best_name}\n")

print("=" * 65)
print("TEST METRICS  (CNN embeddings)")
print("=" * 65)
print(df_test[METRIC_DISPLAY].to_string())

df_val.to_csv( os.path.join(OUTPUT_DIR, "results_val.csv"))
df_test.to_csv(os.path.join(OUTPUT_DIR, "results_test.csv"))

"""## Plots"""

model_names = list(best_models.keys())

# ── ROC curves ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (probs, y_true, title) in zip(axes, [
    (val_probs,  y_val,  "Validation"),
    (test_probs, y_test, "Test"),
]):
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        ax.plot(fpr, tpr, label=f"{name}  AUC={roc_auc_score(y_true, prob):.3f}",
                color=PALETTE[name], lw=2)
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {title}  (CNN embeddings)",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=DPI)
plt.close()

"""### ── Metric bar charts"""

METRIC_COLS = ["ROC-AUC","Avg-Prec","Accuracy","Sensitivity",
               "Specificity","F1","MCC"]
for df, split in [(df_val,"val"),(df_test,"test")]:
    fig, axes = plt.subplots(1, len(METRIC_COLS), figsize=(18, 4))
    for ax, metric in zip(axes, METRIC_COLS):
        vals   = df[metric]
        colors = [PALETTE[m] for m in vals.index]
        bars   = ax.bar(vals.index, vals.values, color=colors, edgecolor="white")
        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.08); ax.grid(axis="y", alpha=0.3)
        ax.set_xticklabels(vals.index, fontsize=8, rotation=20)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle(f"CNN Branch — {split.capitalize()} Set",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"comparison_{split}.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close()

    # Brier separately (lower = better, opposite direction)
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    bv = df["Brier"]
    bars2 = ax2.bar(bv.index, bv.values,
                    color=[PALETTE[m] for m in bv.index], edgecolor="white")
    ax2.set_title("Brier Score (↓ better)", fontweight="bold")
    ax2.set_ylim(0, max(bv.values)*1.3+0.05); ax2.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars2, bv.values):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.002,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, f"brier_{split}.png"), dpi=DPI)
    plt.close(fig2)

"""### ── Confusion matrices"""

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for col, name in enumerate(model_names):
    for row, (probs, y_true, split) in enumerate([
        (val_probs,  y_val,  "Val"),
        (test_probs, y_test, "Test"),
    ]):
        thr  = best_thresholds[name]
        pred = (probs[name] >= thr).astype(int)
        cm   = confusion_matrix(y_true, pred, labels=[0, 1])
        ax   = axes[row][col]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i,j] > cm.max()/2 else "black")
        ax.set_xticks([0,1]); ax.set_xticklabels(["non-PDAC","PDAC"])
        ax.set_yticks([0,1]); ax.set_yticklabels(["non-PDAC","PDAC"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{name} — {split}", fontweight="bold",
                     color=PALETTE[name])
fig.suptitle("Confusion Matrices — CNN Branch  "
             "(top: Validation | bottom: Test)", fontweight="bold")
plt.tight_layout()
plt.show()

plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"),
            dpi=DPI, bbox_inches="tight")
plt.close()

"""### ── Calibration"""

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (probs, y_true, title) in zip(axes, [
    (val_probs,  y_val,  "Validation"),
    (test_probs, y_test, "Test"),
]):
    ax.plot([0,1],[0,1],"k--",lw=1,label="Perfect calibration")
    for name, prob in probs.items():
        frac_pos, mean_pred = calibration_curve(y_true, prob,
                                                n_bins=8, strategy="uniform")
        brier = brier_score_loss(y_true, prob)
        ax.plot(mean_pred, frac_pos, "o-", color=PALETTE[name], lw=2,
                label=f"{name}  Brier={brier:.3f}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (PDAC)")
    ax.set_title(f"Calibration — {title}  (CNN embeddings)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "calibration.png"), dpi=DPI)
plt.close()

# ── Precision-Recall curves ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (probs, y_true, title) in zip(axes, [
    (val_probs,  y_val,  "Validation"),
    (test_probs, y_test, "Test"),
]):
    ax.axhline(y_true.mean(), color="gray", linestyle="--", lw=1,
               label=f"Baseline (prevalence={y_true.mean():.2f})")
    for name, prob in probs.items():
        prec, rec, _ = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        ax.plot(rec, prec, color=PALETTE[name], lw=2,
                label=f"{name}  AP={ap:.3f}")
    ax.set_xlabel("Recall (Sensitivity)"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {title}  (CNN embeddings)",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pr_curves.png"), dpi=DPI)
plt.close()

print("Saved: roc_curves.png  comparison_val/test.png  brier_val/test.png")
print("       confusion_matrices.png  calibration.png  pr_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Save models
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(best_models[best_name], os.path.join(OUTPUT_DIR, "best_model.pkl"))
joblib.dump(best_models,            os.path.join(OUTPUT_DIR, "all_models.pkl"))
joblib.dump(best_hparams,           os.path.join(OUTPUT_DIR, "best_hparams.pkl"))

"""## Final comparison table"""

print(f"\n{'='*65}")
print("FINAL COMPARISON SUMMARY — CNN Branch")
print(f"{'='*65}")
print(f"{'Model':<10} {'Val AUC':>8} {'Test AUC':>9} {'Val Sens':>9} "
      f"{'Test Sens':>10} {'Val Spec':>9} {'Test Spec':>10} "
      f"{'Val F1':>8} {'Test F1':>8}")
print("-" * 85)
for name in model_names:
    v = df_val.loc[name]
    t = df_test.loc[name]
    star = " ★" if name == best_name else ""
    print(f"{name:<10} {v['ROC-AUC']:>8.4f} {t['ROC-AUC']:>9.4f} "
          f"{v['Sensitivity']:>9.4f} {t['Sensitivity']:>10.4f} "
          f"{v['Specificity']:>9.4f} {t['Specificity']:>10.4f} "
          f"{v['F1']:>8.4f} {t['F1']:>8.4f}{star}")

print(f"\n  ★ Best CNN model : {best_name}  "
      f"(val AUC={df_val.loc[best_name,'ROC-AUC']:.4f}, "
      f"test AUC={df_test.loc[best_name,'ROC-AUC']:.4f})")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("Run step7_fusion.py next.")
