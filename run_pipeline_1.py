
# =====================================================================
# IMPORTS & REPRODUCIBILITY
# =====================================================================

# ---- Standard library
import os
import sys
import math
import re
import random
import warnings
from math import sqrt
from typing import Any, Optional, Sequence, Dict, Union, Callable, Tuple
from collections import defaultdict, Counter
from pathlib import Path

# ---- Core numerics & utils
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from scipy.stats import (
    ttest_ind,
    ttest_ind_from_stats,
    mannwhitneyu,
    pearsonr,
    spearmanr,
    truncnorm,
    f_oneway,
)
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm

# ---- Plotting
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib as mpl
from matplotlib.lines import Line2D

# ---- Excel I/O
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage  # Pillow-backed

# ---- Scikit-learn
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

# ---- PyTorch / skorch (neural nets)
import torch
import torch.nn as nn
from skorch import NeuralNetRegressor
from skorch.dataset import ValidSplit, Dataset
from skorch.callbacks import LRScheduler, GradientNormClipping, EarlyStopping
from skorch.helper import predefined_split

# ---- Optuna (HPO)
import optuna
from optuna.samplers import TPESampler

# ---- Project helpers
from functions_repo import *

# ============================================
# Reproducibility & GPU-determinism hints
# ============================================
RANDOM_STATE = 42
# Ensure deterministic CUDA math (where applicable)
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
# Local seed utility (kept to preserve original behavior; later import of set_seed
# from functions_repo intentionally shadows this definition, as in the source code.)
def set_seed(seed=42):
    # RNG seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic kernels (raises if a non-deterministic op is used)
    torch.use_deterministic_algorithms(True)
    # cuDNN: no autotune; fixed choices
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Turn TF32 OFF everywhere (important on A100/H100; harmless on V100)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Prefer stable FP32 matmul route (avoid "highest" which may pick TF32-like paths)
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    # (Optional) stabilize CPU math if you ever eval on CPU
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ.setdefault("OMP_NUM_THREADS", "32")
    os.environ.setdefault("MKL_NUM_THREADS", "32")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "32")))
    except Exception:
        pass
    # Guardrails: hard-assert nothing flipped TF32 back on
    assert not torch.backends.cuda.matmul.allow_tf32, "TF32 still ON for cuBLAS!"
    assert not torch.backends.cudnn.allow_tf32, "TF32 still ON for cuDNN!"
# Quick manifest to verify settings actually stuck in this process
def repro_manifest():
    import platform
    man = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_toolkit": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "allow_tf32_cuda": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "deterministic_flag": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }
    print(man)
    return man

set_seed(42)
repro_manifest()

# Expect:
# 'allow_tf32_cuda': False
# 'allow_tf32_cudnn': False
# 'cublas_workspace_config': ':4096:8'

TO_RUN = True

# Main Input Files
ProteomicsDataFile= "data/proteomics_data.xlsx"                  
Kegg2UniprotFile = "data/Kegg2Accession_pathway.xlsx"     
   
os.makedirs("results", exist_ok=True)
os.makedirs("results/data", exist_ok=True)


# Output Files PAth
pan_condition_outputFile = "results/output.xlsx"   # Main output File
save_SHAP_file = "results/data/SHAP_importance_mc_proteomics_Anchor_1.xlsx"
save_SHAP_dict = "results/data/shap_resultsWithAnchorPR_1a.pkl.gz"
cluster_SHAP_AP_file = "results/data/AP_SHAP_clusters_and_importance_proteomics_1.xlsx"
Feature_level_Others_conditionalFile= "results/data/per_feature_rankings_by_cluster_proteomics_allOthers_1.xlsx"
Feature_level_outside_conditionalFile= "results/data/per_feature_rankings_by_cluster_proteomics_Outside_1.xlsx"
feature_QuadrantsAndRanks_file = "results/data/per_feature_quadrants_ranks_AND_significance_1.xlsx"
cluster_membersFiles = "results/data/cluster_members_PR_1.xlsx"
FIGURE_DIR = "results/data/figures_1"

KAUG_global=0

cond_cols = [
     "Ac+CA_ae","Ac+CA_an","Ac+KL_ae","ANCHOR:Ac_ae","ANCHOR:Ac_an","Ac+KL_an",
     "Ac+SA_ae","Ac+SA_an","Ac+SF_ae","Ac+SF_an","Ac+pCA_ae","Ac+pCA_an",
     "CA_an","SF_ae","pC_ae","pC_an"
    ]

# Exact condition order used for columns in the heatmap
COND_ORDER = [
    # Aerobic
    "pC_ae", "Ac+pCA_ae", "Ac+CA_ae", "Ac+SF_ae",
    "Ac+SA_ae", "Ac+KL_ae", "SF_ae", 
    "ANCHOR:Ac_ae",	

    # Anaerobic
    "pC_an", "Ac+pCA_an", "Ac+CA_an", "Ac+SF_an",
    "Ac+SA_an", "Ac+KL_an", "CA_an",
    "ANCHOR:Ac_an",
]


# Mapping of condition short labels and O2 group (for top/bottom headers)
COND_META = {
    "pC_ae":     ("pC",    "Aerobic"),
    "Ac+pCA_ae": ("Ac+pCA","Aerobic"),
    "Ac+CA_ae":  ("Ac+CA", "Aerobic"),
    "Ac+SF_ae":  ("Ac+SF", "Aerobic"),
    "Ac+SA_ae":  ("Ac+SA", "Aerobic"),
    "Ac+KL_ae":  ("Ac+KL", "Aerobic"),
    "SF_ae":     ("SF",    "Aerobic"),
    "ANCHOR:Ac_ae":     ("Control",    "Aerobic"),
    "pC_an":     ("pC",    "Anaerobic"),
    "Ac+pCA_an": ("Ac+pCA","Anaerobic"),
    "Ac+CA_an":  ("Ac+CA", "Anaerobic"),
    "Ac+SF_an":  ("Ac+SF", "Anaerobic"),
    "Ac+SA_an":  ("Ac+SA", "Anaerobic"),
    "Ac+KL_an":  ("Ac+KL", "Anaerobic"),
    "CA_an":     ("CA",    "Anaerobic"),
    "ANCHOR:Ac_an":     ("Control",    "Anaerobic"),
}
   
 
SHAP_COLS = [
        # Aerobic
        "pC_ae", "Ac+pCA_ae", "Ac+CA_ae", "Ac+SF_ae", "Ac+SA_ae", "Ac+KL_ae", "SF_ae", "ANCHOR:Ac_ae",
        # Anaerobic
        "pC_an", "Ac+pCA_an", "Ac+CA_an", "Ac+SF_an", "Ac+SA_an", "Ac+KL_an", "CA_an", "ANCHOR:Ac_an",
    ]
 
aero = ["pC_ae","Ac+pCA_ae","Ac+CA_ae","Ac+SF_ae","Ac+SA_ae","Ac+KL_ae","SF_ae","ANCHOR:Ac_ae"]
ana  = ["pC_an","Ac+pCA_an","Ac+CA_an","Ac+SF_an","Ac+SA_an","Ac+KL_an","CA_an","ANCHOR:Ac_an"]
fc_cols = ["Ac+CA_ae","Ac+CA_an","Ac+KL_ae","Ac+KL_an","Ac+SA_ae","Ac+SA_an",
            "Ac+SF_ae","Ac+SF_an","Ac+pCA_ae","Ac+pCA_an","CA_an","SF_ae","pC_ae","pC_an"
        ]

# =====================================================================
#  DATA LOADING
# =====================================================================
# ============ Load your data ============
df = pd.read_excel(ProteomicsDataFile, sheet_name='Sheet1')

# Features (assuming columns 5-1861 are omics features)
X = df.iloc[:, 5:1862].to_numpy(dtype=np.float32)
# Targets (both GR_mean and GR_sd)
y = df[["GR_mean", "GR_sd"]].to_numpy(dtype=np.float32)
y_meta = df[['Condition', 'Oxygen', 'Replicate', 'Condition_key','Condition_key_rep']]

n_samples, n_features = X.shape
print(f"Data: X.shape={X.shape}, y.shape={y.shape}, y_meta.shape={y_meta.shape}")

# --- Anchored condition-level LOOCV (test = whole condition of 5 replicates) ---
always_train_keys = {"Ac_ae", "Ac_an"}  # anchors that must stay in train
cond_keys = y_meta["Condition_key"].to_numpy()   # shape (N,), e.g., 'Ac+CA_ae', ...
N = len(cond_keys)
# All unique condition groups present in the data
all_groups = pd.unique(cond_keys)
# Candidate test groups = all groups except anchors (anchors must always remain in train)
candidate_test_groups = [g for g in all_groups if g not in always_train_keys]


# =====================================================================
#  MODEL DEFINITION
# =====================================================================
set_seed(42)
print("Torch device available:", "cuda" if torch.cuda.is_available() else "cpu")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Torch device:", device)

class GaussianNLLWithObservedSigma(nn.Module):
    """Expects y_true of shape (N, 2): [:,0]=target mean, [:,1]=target sd."""
    def __init__(self, min_var=1e-8):
        super().__init__()
        self.min_var = min_var
    def forward(self, y_pred, y_true):
        target = y_true[:, 0]
        sd     = y_true[:, 1].clamp_min(0.0)
        var    = (sd ** 2).clamp_min(self.min_var)
        diff2  = (y_pred - target) ** 2
        nll    = 0.5 * (torch.log(var) + diff2 / var)
        return nll.mean()

class DeclineMLP(nn.Module):
    def __init__(self, in_dim, n_layers=1, hidden_start=64, decay=0.5, pdrop=0.0):
        super().__init__()
        n_layers = int(max(1, n_layers))
        layers = []
        last = in_dim
        width = int(hidden_start)
        for _ in range(n_layers):
            layers += [nn.Linear(last, width), nn.ReLU()]
            if pdrop > 0:
                layers += [nn.Dropout(pdrop)]
            last = width
            width = max(2, int(width * decay))
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, X):
        return self.net(X).squeeze(-1)


# =====================================================================
#  BUILDERS & FACTORIES
# =====================================================================
# (Optional) Optuna trial  pipeline (CPU target to keep picklable)
def build_pipe_from_trial_decline(trial, n_features: int):
    #set_seed(42)
    n_layers     = trial.suggest_int("n_layers", 1, 8)
    hidden_start = trial.suggest_int("hidden_start", 16, max(16, int(n_features // 2)))
    decay        = trial.suggest_float("decay", 0.25, 0.75)
    pdrop        = trial.suggest_float("dropout", 0.0, 0.5)
    lr           = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    max_epochs   = trial.suggest_int("max_epochs", 1000, 1000)

    net = NeuralNetRegressor(
        DeclineMLP,
        module__in_dim=int(n_features),
        module__n_layers=int(n_layers),
        module__hidden_start=int(hidden_start),
        module__decay=float(decay),
        module__pdrop=float(pdrop),
        max_epochs=int(max_epochs),
        lr=float(lr),
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=float(weight_decay),
        batch_size=-1,
        device=device,                         # build on CPU - picklable
        train_split=None,
        callbacks=[],
        criterion=GaussianNLLWithObservedSigma,
        verbose=0,
    )
    return Pipeline([('sc', StandardScaler()), ('net', net)])

# Fixed-parameter, CPU factory (picklable for joblib 'loky')
def pipe_factory_cpu():
    #set_seed(42)
    net = NeuralNetRegressor(
        DeclineMLP,
        module__in_dim=n_features,
        module__n_layers=3,
        module__hidden_start=311,
        module__decay=0.38770789337563566,
        module__pdrop=0.11860258004613565,
        max_epochs=1000,
        lr=2.342687593995353e-04,
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=1.4772395543233135e-06,
        batch_size=-1,
        device='cpu',
        train_split=None,
        callbacks=[],
        criterion=GaussianNLLWithObservedSigma,
        verbose=0,
    )
    return Pipeline([('sc', StandardScaler()), ('net', net)])

# Params  pipeline (deferred device binding)
def build_pipe_from_params_decline(p):
    # set_seed(42)
    net = NeuralNetRegressor(
        DeclineMLP,
        module__in_dim=n_features,
        module__n_layers=int(p["n_layers"]),
        module__hidden_start=int(p["hidden_start"]),
        module__decay=float(p["decay"]),
        module__pdrop=float(p["dropout"]),
        max_epochs=int(p["max_epochs"]),
        lr=float(p["lr"]),
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=float(p["weight_decay"]),
        batch_size=-1,
        device=device,  # set below before use
        train_split=None,
        callbacks=[],
        criterion=GaussianNLLWithObservedSigma,
        verbose=0,
    )
    return Pipeline([('sc', StandardScaler()), ('net', net)])

# Top-level picklable factory (uses global `params`)
def pipe_factory():
    # Top-level, no lambdas, returns unfitted estimator; picklable for 'loky'
    return build_pipe_from_params_decline(params)


# =====================================================================
#  LOOCV EVALUATION (anchored)
# =====================================================================
params = {
    "n_layers": 3,
    "hidden_start": 311,
    "decay": 0.38770789337563566,
    "dropout": 0.11860258004613565,
    "lr": 2.342687593995353e-04,
    "weight_decay": 1.4772395543233135e-06,
    "max_epochs": 1000,
}

best_trial = optuna.trial.FixedTrial(params)
# ---- evaluate best with LOOCV and print summary ----
best_pipe = build_pipe_from_trial_decline(best_trial, n_features)
set_seed(42)
# best_sum = loocv_scores(best_pipe, X, y,  cond_keys, always_train_keys, K_AUG = 0)
# best_sum = loocv_scores(best_pipe, X, y,  cond_keys, always_train_keys, K_AUG = 1)
#best_sum = loocv_scores(best_pipe, X, y,  cond_keys, always_train_keys, K_AUG = 100)
best_sum = loocv_scores(best_pipe, X, y,  cond_keys, always_train_keys, K_AUG = KAUG_global)

print_summary("Best FlexMLP (Optuna, LOOCV, reproducible)", best_sum, y)
for info in best_sum['fold_info']:
    print(
        f"Fold {info['fold']}: "
        f"train={info['n_train_internal']}, valid={info['n_valid_internal']}, "
        f"epochs_run={info['epochs_run']} (best@{info['best_epoch']}), "
        f"early_stopped={info['early_stopped']}"
    )

import gc as _gc
_del = best_pipe
_del = None
_gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if TO_RUN:
    # =====================================================================
    #  SHAP MONTE-CARLO IMPORTANCE
    # =====================================================================
    set_seed(42)
    feature_names = list(df.columns[5:5 + n_features])
    shap_df, shap_raw = shap_importance_mc_deterministic(
        pipe_factory_name=pipe_factory_cpu,
        X=X, y=y,
        cond_keys=cond_keys, always_train_keys=always_train_keys,
        feature_names=feature_names,
        test_k=3, exhaustive=True, shuffle_combos=False,
        k_valid_groups=3, K_AUG=KAUG_global,
        # nsamples="auto",
        nsamples=5460,
        n_jobs=5,
        backend="loky", prefer="processes",
        rng_seed=42,
        use_cuda=True,
        device_ids= list(range(torch.cuda.device_count())), #[0],
        use_cuda_deterministic=True,
    )

    print(shap_df.head(10)[["feature","mean_abs_shap","mean_abs_shap_over_baseRMSE","ci95_lo","ci95_hi"]])

    by_group_df_out = shap_raw.get("by_group_df", None)
    with pd.ExcelWriter(save_SHAP_file, engine="openpyxl") as xl:
        shap_df.to_excel(xl, sheet_name="aggregated_shaps", index=False)
        if by_group_df_out is not None:
            by_group_df_out.to_excel(xl, sheet_name="shap_by_group")

    import pickle, gzip
    bundle = {"shap_df": shap_df, "shap_raw": shap_raw}
    with gzip.open(save_SHAP_dict, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    # # Load later
    # import gzip, pickle
    # with gzip.open("results/data/shap_resultsWithAnchorPR_1a.gz", "rb") as f:
    #     data = pickle.load(f)
    # shap_df_loaded = data["shap_df"]
    # shap_raw_loaded = data["shap_raw"]

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================================================================
    #  AP CLUSTERING & CLUSTER-LEVEL DELTA-RMSE (necessity tests)
    # =====================================================================
    # Each Cluster represents condition-specific genes that are likely co-regulated.
    # Delta-RMSE > 0 Ò cluster carries predictive/mechanistic signal.
    # Delta-RMSE < 0 Ò potential noise/confounding.

    xls = pd.ExcelFile(save_SHAP_file, engine="openpyxl")
    outputworkbook = cluster_SHAP_AP_file
    
    shap_df_loaded = pd.read_excel(xls, sheet_name="aggregated_shaps", index_col=0)
    by_group_df_out_loaded = None
    if "shap_by_group" in xls.sheet_names:
        by_group_df = pd.read_excel(xls, sheet_name="shap_by_group", index_col=0)

    # --- Global feature clustering with AP ---
    set_seed(42)
    ap_assign, ap_exemplars, ap_model = ap_cluster_features_global(
        by_group_df, scale=True, similarity="cosine",
        pref_quantile=0.50, damping=0.80, random_state=42
    )

    ap_assign_df = ap_assign.reset_index().rename(columns={"index":"feature", "cluster":"cluster"})
    ap_exemplars_df = (
        ap_exemplars.reset_index().rename(columns={"index":"cluster", 0:"exemplar"})
        if hasattr(ap_exemplars, "reset_index") else ap_exemplars
    )

    # --- Optional: refine locally within each global cluster ---
    # (call kept as-is from source; helper expected to exist in environment)
    set_seed(42)
    ap_local_df = ap_refine_local_subclusters(
        by_group_df, ap_assign, scale=True, similarity="cosine",
        pref_quantile=0.50, damping=0.80, random_state=42,
        min_cluster_size=6
    )

    # --- Cluster conditions (local condition modules) ---
    set_seed(42)
    assign_cond, exemplars_cond, _ = ap_cluster_conditions(
        by_group_df, scale=True, similarity="cosine",
        pref_quantile=0.50, damping=0.80, random_state=42
    )
    assign_cond_df = assign_cond.reset_index().rename(columns={"index":"condition", "cond_cluster":"cond_cluster"})
    exemplars_cond_df = (
        exemplars_cond.reset_index().rename(columns={"index":"cond_cluster", 0:"cond_exemplar"})
        if hasattr(exemplars_cond, "reset_index") else exemplars_cond
    )

    # --- Use AP clusters for necessity testing (DELTA-RMSE by cluster) ---
    set_seed(42)
    cluster_imp_df_cond, base_rmse_list = cluster_permutation_importance(
        pipe_template=pipe_factory_cpu,
        X=X, y=y, cond_keys=cond_keys, always_train_keys=always_train_keys,
        cluster_assign=ap_assign,                 # <--- AP feature clusters here
        test_k=3, k_valid_groups=3, K_AUG=KAUG_global,
        rng=42, verbose=True, exhaustive=True, shuffle_combos=True,
        conditional=True, glm_alpha=1.0
    )

    with pd.ExcelWriter(outputworkbook, engine="openpyxl") as xl:
        by_group_df.to_excel(xl, sheet_name="shap_by_group")
        ap_assign_df.to_excel(xl, sheet_name="feat_clusters", index=False)
        ap_exemplars.to_excel(xl, sheet_name="feat_exemplars")
        ap_local_df.to_excel(xl, sheet_name="feat_local_subclusters", index=False)
        assign_cond_df.to_excel(xl, sheet_name="cond_clusters", index=False)
        exemplars_cond.to_excel(xl, sheet_name="cond_exemplars")
        cluster_imp_df_cond.to_excel(xl, sheet_name="cluster_dRMSE_cond", index=False)
    
    
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # =====================================================================
    #  FEATURE-LEVEL CONDITIONAL DELTA-RMSE
    # =====================================================================
    # Two modes on residualized permutations to address collinearity/redundancy:
    # (1) outside_cluster  condition on non-cluster features (credits shared signal to feature)
    # (2) all_others      condition on all other features (penalizes redundancy; unique signal)

    # Reload artifacts for ranking exports
    file_path = cluster_SHAP_AP_file

    by_group_df = pd.read_excel(file_path, sheet_name="shap_by_group", engine="openpyxl")
    ap_assign_df = pd.read_excel(file_path, sheet_name="feat_clusters", engine="openpyxl")
    ap_exemplars = pd.read_excel(file_path, sheet_name="feat_exemplars", engine="openpyxl")
    ap_local_df = pd.read_excel(file_path, sheet_name="feat_local_subclusters", engine="openpyxl")
    assign_cond_df = pd.read_excel(file_path, sheet_name="cond_clusters", engine="openpyxl")
    exemplars_cond = pd.read_excel(file_path, sheet_name="cond_exemplars", engine="openpyxl")
    cluster_imp_df_cond = pd.read_excel(file_path, sheet_name="cluster_dRMSE_cond", engine="openpyxl")

    ap_assign = ap_assign_df["cluster"]

    # Feature names (adjust slices to match how X was built)
    feature_names = list(df.columns[5:5 + n_features])
    cluster_assign = ap_assign  # length d, cluster id per feature

    # Optional protein mapping (to UniProt):
    acc2kegg = pd.read_excel(Kegg2UniprotFile, sheet_name="MasterDict")
    feature_to_protein_df = ap_assign_df.copy()
    feature_to_protein_df["protein_id"] = feature_to_protein_df["feature"]
       

    # ---- (1) outside_cluster ----
    set_seed(42)
    out_file = export_cluster_rankings_to_xlsx(
        pipe_factory=pipe_factory,     # <- use this for loky
        X=X, y=y,
        cond_keys=cond_keys, always_train_keys=always_train_keys,
        cluster_assign=cluster_assign, feature_names=feature_names,
        out_path=Feature_level_outside_conditionalFile,
        test_k=3, k_valid_groups=3, exhaustive=True, K_AUG=KAUG_global,
        conditional_mode="outside_cluster",
        glm_alpha=1.0,
        n_repeats=10,
        feature_to_protein_df=feature_to_protein_df,
        acc2kegg_df=acc2kegg,
        n_jobs=-1,           # parallel
        backend="loky",      # process-based
        use_cuda=True, device_ids=[0]
    )
    
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # ---- (2) all_others ----
    set_seed(42)
    out_file = export_cluster_rankings_to_xlsx(
        pipe_factory=pipe_factory,     # <- use this for loky
        X=X, y=y,
        cond_keys=cond_keys, always_train_keys=always_train_keys,
        cluster_assign=cluster_assign, feature_names=feature_names,
        out_path=Feature_level_Others_conditionalFile,
        test_k=3, k_valid_groups=3, exhaustive=True, K_AUG=KAUG_global,
        conditional_mode="all_others",
        glm_alpha=1.0,
        n_repeats=10,
        feature_to_protein_df=feature_to_protein_df,
        acc2kegg_df=acc2kegg,
        n_jobs=-1,           # parallel
        backend="loky",      # process-based
        use_cuda=True, device_ids=[0]
    )
    
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Heatmap using SHAP values from the "shap_by_group" sheet of cluster_members_PR.xlsx
# Creates a SEPARATE heatmap per family (parent cluster).

if TO_RUN:
    
    # ============================================
    #  cluster membership workbook
    # ============================================
    set_seed(42)
    outs = build_outputs_single_sheet(
        xlsx_path=cluster_SHAP_AP_file,
        assign_sheet="feat_local_subclusters",
        cluster_drmse_sheet="cluster_dRMSE_cond",
        feat_exemplars_sheet="feat_exemplars",
        shap_by_group_sheet="shap_by_group",
        out_xlsx=cluster_membersFiles,
        acc2k_xlsx_path=Kegg2UniprotFile,
        acc2k_sheet="MasterDict",
    )
    
    
    OUT_DIR = FIGURE_DIR
    os.makedirs(OUT_DIR, exist_ok=True)
    # =========================
    # --- User configuration ---
    # =========================
    XLSX_PATH     = cluster_membersFiles
    SHEET_TC      = "shap_by_group"   # sheet containing SHAP values
    SHEET_TC_CODE = "all_clusters"    # label for output subfolder

    # Output directory (use current dir if OUT_DIR is not pre-defined)
    OUT_DIR = globals().get("OUT_DIR", ".")
    TMP_OUT_DIR = os.path.join(OUT_DIR, f"{SHEET_TC_CODE}_PR")
    os.makedirs(TMP_OUT_DIR, exist_ok=True)

    # # Exact condition order used for columns in the heatmap
    # COND_ORDER = [
        # # Aerobic
        # "pC_ae", "Ac+pCA_ae", "Ac+CA_ae", "Ac+SF_ae",
        # "Ac+SA_ae", "Ac+KL_ae", "SF_ae", 
        # "ANCHOR:Ac_ae",	

        # # Anaerobic
        # "pC_an", "Ac+pCA_an", "Ac+CA_an", "Ac+SF_an",
        # "Ac+SA_an", "Ac+KL_an", "CA_an",
        # "ANCHOR:Ac_an",
    # ]


    # # Mapping of condition short labels and O2 group (for top/bottom headers)
    # COND_META = {
        # "pC_ae":     ("pC",    "Aerobic"),
        # "Ac+pCA_ae": ("Ac+pCA","Aerobic"),
        # "Ac+CA_ae":  ("Ac+CA", "Aerobic"),
        # "Ac+SF_ae":  ("Ac+SF", "Aerobic"),
        # "Ac+SA_ae":  ("Ac+SA", "Aerobic"),
        # "Ac+KL_ae":  ("Ac+KL", "Aerobic"),
        # "SF_ae":     ("SF",    "Aerobic"),
        # "ANCHOR:Ac_ae":     ("Control",    "Aerobic"),
        # "pC_an":     ("pC",    "Anaerobic"),
        # "Ac+pCA_an": ("Ac+pCA","Anaerobic"),
        # "Ac+CA_an":  ("Ac+CA", "Anaerobic"),
        # "Ac+SF_an":  ("Ac+SF", "Anaerobic"),
        # "Ac+SA_an":  ("Ac+SA", "Anaerobic"),
        # "Ac+KL_an":  ("Ac+KL", "Anaerobic"),
        # "CA_an":     ("CA",    "Anaerobic"),
        # "ANCHOR:Ac_an":     ("Control",    "Anaerobic"),
    # }


    # =========================
    # --- Helpers ---
    # =========================
    def pick(colnames_lower, options):
        """Return the first option present in a list of lowercase column names."""
        for opt in options:
            if opt in colnames_lower:
                return opt
        return None


    # =========================
    # --- Load & prepare data ---
    # =========================
    tc = pd.read_excel(XLSX_PATH, sheet_name=SHEET_TC)
    tc.columns = [c.strip() for c in tc.columns]
    tc_lower = [c.lower() for c in tc.columns]

    # Identify key columns (case-insensitive)
    col_feature_lc      = pick(tc_lower, ["feature", "rpa", "rpa_id", "protein", "geneid", "gene_id"])
    col_family_lc       = pick(tc_lower, ["parent_cluster", "family", "cluster", "parent"])
    col_subfamily_lc    = pick(tc_lower, ["subcluster", "sub_family", "subfamily", "sub_cluster"])
    col_exemplar_lc     = pick(tc_lower, ["local_exemplar", "exemplar", "representative", "rep"])
    col_exemplar_fam_lc = pick(tc_lower, ["parent_exemplar", "exemplar_fam", "representative_fam", "rep_fam"])

    # Map back to original casing (where available)
    def original_case(lc_name):
        return tc.columns[tc_lower.index(lc_name)] if lc_name in tc_lower else None

    col_feature      = original_case(col_feature_lc)
    col_family       = original_case(col_family_lc)
    col_subfamily    = original_case(col_subfamily_lc)
    col_exemplar     = original_case(col_exemplar_lc)
    col_exemplar_fam = original_case(col_exemplar_fam_lc)

    # Build mapping DF (include exemplar_fam if present; else create empty)
    map_cols = {
        col_feature:   "feature",
        col_family:    "family",
        col_subfamily: "subfamily",
        col_exemplar:  "exemplar",
    }
    core_map_df = tc[[c for c in [col_feature, col_family, col_subfamily, col_exemplar] if c is not None]].rename(columns=map_cols)

    if col_exemplar_fam is not None:
        core_map_df["exemplar_fam"] = tc[col_exemplar_fam]
    else:
        core_map_df["exemplar_fam"] = np.nan  # keeps downstream logic intact

    map_df = core_map_df.copy()
    map_df["feature_upper"]      = map_df["feature"].astype(str).str.strip().str.upper()
    map_df["exemplar_upper"]     = map_df["exemplar"].astype(str).str.strip().str.upper()
    map_df["exemplar_upper_fam"] = map_df["exemplar_fam"].astype(str).str.strip().str.upper()

    # Ensure family/subfamily are integers (for ordering)
    map_df["family"]    = pd.to_numeric(map_df["family"], errors="coerce").astype("Int64")
    map_df["subfamily"] = pd.to_numeric(map_df["subfamily"], errors="coerce").astype("Int64")

    # SHAP matrix (in your requested order)
    SHAP_COLS = COND_ORDER
    missing_cols = [c for c in SHAP_COLS if c not in tc.columns]
    if missing_cols:
        raise ValueError(f"Missing expected SHAP columns in sheet '{SHEET_TC}': {missing_cols}")

    shap_df = tc[[col_feature] + SHAP_COLS].copy()
    shap_df["feature_upper"] = shap_df[col_feature].astype(str).str.strip().str.upper()
    expr = shap_df.set_index("feature_upper")[SHAP_COLS].astype(float)

    # Align features
    common = sorted(set(expr.index).intersection(set(map_df["feature_upper"])))
    if not common:
        raise AssertionError("No overlap between mapping features and SHAP features.")

    # Preserve family appearance order
    family_order_custom = list(dict.fromkeys(map_df.loc[map_df["feature_upper"].isin(common), "family"].dropna().tolist()))

    # Ordered frame (filter to common first)
    order_df_all = (
        map_df[map_df["feature_upper"].isin(common)]
        .drop_duplicates(subset=["feature_upper"])
        .assign(
            family=pd.Categorical(
                map_df.loc[map_df["feature_upper"].isin(common), "family"],
                categories=family_order_custom,
                ordered=True,
            )
        )
        .sort_values(["family", "subfamily", "feature_upper"])
        .reset_index(drop=True)
    )

    # Family exemplar heuristic (fallback safe if exemplar_fam missing)
    fam_exemplar = {}
    for fam, fgrp in order_df_all.groupby("family", sort=False, observed=True):
        sizes = fgrp.groupby("subfamily")["feature_upper"].size().sort_values(ascending=False)
        if sizes.empty:
            fam_exemplar[fam] = fgrp["feature_upper"].iloc[0]
            continue
        max_size = sizes.iloc[0]
        biggest = sizes[sizes == max_size].index.tolist()

        candidates = []
        for sf in biggest:
            subgrp = fgrp[fgrp["subfamily"] == sf]
            # prefer provided exemplar_fam if available, else first feature
            if subgrp["exemplar_upper_fam"].notna().any() and (subgrp["exemplar_upper_fam"].str.len() > 0).any():
                candidates.append(subgrp["exemplar_upper_fam"].iloc[0])
            else:
                candidates.append(subgrp["feature_upper"].iloc[0])

        fam_exemplar[fam] = sorted(set(candidates))[0]

    order_df_all["is_sub_exemplar"] = order_df_all["feature_upper"].values == order_df_all["exemplar_upper"].values
    order_df_all["is_fam_exemplar"] = [
        order_df_all.loc[i, "feature_upper"] == fam_exemplar.get(order_df_all.loc[i, "family"], "")
        for i in range(len(order_df_all))
    ]

    # Oxygen group indices (for secondary header)
    aer_idx = [i for i, c in enumerate(SHAP_COLS) if COND_META[c][1] == "Aerobic"]
    ana_idx = [i for i, c in enumerate(SHAP_COLS) if COND_META[c][1] == "Anaerobic"]
    aer_start, aer_end = min(aer_idx), max(aer_idx)
    ana_start, ana_end = min(ana_idx), max(ana_idx)
    aer_mid = (aer_start + aer_end) / 2.0
    ana_mid = (ana_start + ana_end) / 2.0 - 3.0  # small left shift as in your code

    present_families = [f for f in family_order_custom if f in order_df_all["family"].unique()]

    # =========================
    # --- Plot: 1 heatmap / family ---
    # =========================
    for fam in present_families:
        # Filter to this family & order rows
        order_df = order_df_all[order_df_all["family"] == fam].copy().reset_index(drop=True)
        X = expr.reindex(order_df["feature_upper"]).dropna(how="all")

        if X.shape[0] == 0:
            print(f"Skipping family {fam}: no SHAP rows after alignment.")
            continue

        # Re-sync order_df to X.index
        order_df = order_df.set_index("feature_upper").loc[X.index].reset_index()

        H = X.values.astype(float)
        n_rows, n_cols = H.shape

        # Figure
        fig_h = max(6, n_rows * 0.25)
        fig_w = max(13, n_cols * 0.85)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # Heatmap (centered scaling is optional; keeping your default)
        im = ax.imshow(H, aspect="auto")

        # Y ticks (features)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(order_df["feature_upper"].tolist(), fontsize=6)
        ax.tick_params(axis="y", length=0)

        # X ticks (conditions, short labels)
        xticks = np.arange(len(SHAP_COLS))
        ax.set_xticks(xticks)
        ax.set_xticklabels([COND_META[c][0] for c in SHAP_COLS], rotation=45, ha="right")

        ax.set_title(fr"SHAP heatmap for Family F{fam} (ordered by Sub-family)\n*/** exemplars")

        # Secondary group labels (Aerobic/Anaerobic), centered over spans
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks([aer_mid, ana_mid])
        ax2.set_xticklabels(["Aerobic", "Anaerobic"], fontsize=10, weight="bold")
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")
        ax2.spines["bottom"].set_position(("outward", 40))
        ax2.tick_params(length=0)
        ax2.spines["bottom"].set_visible(False)

        # Separator between groups
        sep_pos = aer_end + 0.5
        ax.axvline(sep_pos, color="k", linewidth=1.2)

        # Expand x-limits for right annotations
        right_pad = 3.8
        ax.set_xlim(-0.5, n_cols - 0.5 + right_pad)

        # Subfamily separators (within-family)
        prev_sf = None
        for i, sf in enumerate(order_df["subfamily"]):
            if i > 0 and sf != prev_sf:
                ax.axhline(i - 0.5, linewidth=0.5)
            prev_sf = sf

        # Stars for exemplars
        ypos = np.arange(n_rows)
        x1 = n_cols - 0.55
        x2 = n_cols - 0.85
        sub_mask  = order_df["is_sub_exemplar"].values
        fam_mask  = order_df["is_fam_exemplar"].values
        both_mask = sub_mask & fam_mask

        ax.scatter(np.full(sub_mask.sum(), x1), ypos[sub_mask], marker="*", s=40,
                   c="white", edgecolors="k", linewidths=0.3)
        ax.scatter(np.full(fam_mask.sum(), x1), ypos[fam_mask], marker="*", s=60,
                   c="yellow", edgecolors="k", linewidths=0.3)
        if both_mask.any():
            ax.scatter(np.full(both_mask.sum(), x2), ypos[both_mask], marker="*", s=60,
                       c="yellow", edgecolors="k", linewidths=0.3)

        # Colors per family (single family here, but keep structure for reuse)
        family_order = order_df["family"].drop_duplicates().tolist()
        palette = plt.cm.tab20.colors
        family_to_color = {ff: palette[i % len(palette)] for i, ff in enumerate(family_order)}

        # Left-lacing subfamily brackets
        def draw_left_lacing_sf(ax_, base_x, lane_dx, arm_len):
            for ff, fgrp in order_df.groupby("family", sort=False, observed=True):
                color = family_to_color.get(ff, "k")
                sub_ids = sorted(fgrp["subfamily"].unique().tolist())
                for sf_ in sub_ids:
                    grp = fgrp[fgrp["subfamily"] == sf_]
                    top = grp.index.min() - 0.5
                    bot = grp.index.max() + 0.5
                    mid = 0.5 * (top + bot)
                    lane = sub_ids.index(sf_)
                    x = base_x + lane * lane_dx
                    ax_.plot([x, x], [top, bot], linewidth=0.9, color=color)
                    left = max(n_cols - 0.5 + 0.02, x - arm_len)
                    ax_.plot([left, x], [top, top], linewidth=0.9, color=color)
                    ax_.plot([left, x], [bot, bot], linewidth=0.9, color=color)
                    ax_.text(x - (arm_len + 0.05), mid, rf"$\mathrm{{SF}}{int(sf_)}$",
                             va="center", ha="right", fontsize=6, color=color)

        x_sf_base = n_cols + 0.25
        lane_dx_sf = 0.22
        arm_len_sf = 0.28
        draw_left_lacing_sf(ax, base_x=x_sf_base, lane_dx=lane_dx_sf, arm_len=arm_len_sf)

        # Right-side Family bracket (single bracket for this plot)
        def draw_right_f(ax_, base_x, arm_len, lw=1.1):
            for ff, fgrp in order_df.groupby("family", sort=False, observed=True):
                color = family_to_color.get(ff, "k")
                top = fgrp.index.min() - 0.5
                bot = fgrp.index.max() + 0.5
                mid = 0.5 * (top + bot)
                x = base_x
                ax_.plot([x, x], [top, bot], linewidth=lw, color=color)
                left_cap = max(n_cols - 0.5 + 0.02, x - arm_len)
                ax_.plot([left_cap, x], [top, top], linewidth=lw, color=color)
                ax_.plot([left_cap, x], [bot, bot], linewidth=lw, color=color)
                ax_.text(x + arm_len + 0.05, mid, rf"$\mathrm{{F}}{int(ff)}$",
                         va="center", fontsize=10, color=color)

        x_f_base  = n_cols + 3.10
        arm_len_f = 0.35
        draw_right_f(ax, base_x=x_f_base, arm_len=arm_len_f, lw=1.1)

        # Highlight per-family impact condition(s) with white boxes
        for ff, fgrp in order_df.groupby("family", sort=False, observed=True):
            top_row = int(fgrp.index.min())
            bot_row = int(fgrp.index.max())
            height  = (bot_row - top_row + 1)

            fam_feats = fgrp["feature_upper"].tolist()
            F = expr.loc[fam_feats, SHAP_COLS].values  # (#rows_in_fam, n_cols)

            col_scores = np.nanmean(np.abs(F), axis=0)  # mean |SHAP|
            if not np.isfinite(col_scores).any():
                continue
            mx = np.nanmax(col_scores)
            tol = 1e-12
            best_cols = np.flatnonzero(col_scores >= mx - tol)

            for x in best_cols:
                rect = mpatches.Rectangle(
                    (x - 0.5, top_row - 0.5),
                    1.0,
                    float(height),
                    fill=False,
                    edgecolor="white",
                    linewidth=1.5,
                    zorder=6,
                )
                ax.add_patch(rect)

        # Colorbar + strong outer frame
        cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.12)
        cbar.set_label("SHAP value", rotation=90)

        # Hide default spines, draw strong outer frame
        for s in ax.spines.values():
            s.set_visible(False)
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               fill=False, edgecolor="black", linewidth=2.0,
                               zorder=9999, clip_on=False))

        for side in ["top", "left", "right"]:
            ax2.spines[side].set_linewidth(2.0)
            ax2.spines[side].set_edgecolor("black")

        cbar.outline.set_linewidth(2.0)
        cbar.outline.set_edgecolor("black")

        # Save
        out_path = os.path.join(TMP_OUT_DIR, f"heatmap_shap_family_{int(fam)}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()


    # ============================================
    # SUMMARY (per-family max-impact condition_key)
    # + Write table
    # + Merge into selection table (if 'outs' exists)
    # + Bubble map (clusters  conditions)
    # ============================================
    # ----------------------------
    # Prereqs expected from earlier:
    # - family_order_custom : list of family IDs (order to consider)
    # - order_df_all        : dataframe with columns ['feature_upper','family','subfamily',...]
    # - expr                : SHAP matrix indexed by 'feature_upper' with SHAP_COLS columns
    # - SHAP_COLS           : ordered list of condition keys
    # - COND_META           : {cond_key: (short_name, "Aerobic"/"Anaerobic")}
    # - OUT_DIR or tmp_OUT_DIR (we'll resolve below)
    # ----------------------------

    # ---------- Resolve output dir ----------
    OUT_DIR = globals().get("OUT_DIR", ".")
    TMP_OUT_DIR = globals().get("tmp_OUT_DIR", os.path.join(OUT_DIR, "summary_pr"))
    os.makedirs(TMP_OUT_DIR, exist_ok=True)

    # ---------- SUMMARY ONLY ----------
    rows = []
    # guard against nullable Int64 dtypes
    present_families = [
        int(f) for f in family_order_custom
        if pd.notna(f) and (f in order_df_all["family"].unique())
    ]

    for fam in present_families:
        fgrp = (
            order_df_all.loc[order_df_all["family"] == fam, ["feature_upper", "family"]]
            .drop_duplicates(subset=["feature_upper"])
        )
        fam_feats = [f for f in fgrp["feature_upper"].tolist() if f in expr.index]
        if not fam_feats:
            continue

        F = expr.loc[fam_feats, SHAP_COLS].values  # (#features_in_fam Ã #conditions)
        # mean |SHAP| across family members, per condition
        col_scores = np.nanmean(np.abs(F), axis=0)
        if not np.isfinite(col_scores).any():
            continue

        mx = np.nanmax(col_scores)
        best_cols = np.flatnonzero(col_scores >= mx - 1e-12)  # allow ties

        for idx in best_cols:
            cond_key = SHAP_COLS[idx]
            cond_name, oxy = COND_META[cond_key]
            rows.append({
                "family": int(fam),
                "condition_key": cond_key,        # requested key
                "condition": cond_name,           # human-readable (optional)
                "oxygen": oxy,                    # Aerobic / Anaerobic (optional)
                "max_mean_abs_shap": float(col_scores[idx])
            })

    impact_df = (
        pd.DataFrame(rows)
          .sort_values(["family", "max_mean_abs_shap"], ascending=[True, False])
          .reset_index(drop=True)
    )

    # Optional: tied keys collapsed to 1 row per family
    impact_wide = (
        impact_df.groupby("family", as_index=False)
                 .agg(best_condition_keys=("condition_key", lambda s: ",".join(s)),
                      oxygen=("oxygen", lambda s: ",".join(pd.unique(s))),
                      max_mean_abs_shap=("max_mean_abs_shap", "max"))
    )

    # Save detailed rows
    out_csv = os.path.join(TMP_OUT_DIR, "family_max_condition_keys.csv")
    impact_df.to_csv(out_csv, index=False)
    print(f"Wrote: {Path(out_csv).resolve()}")

    # ---------- Merge into selection table (if outs present) ----------
    # Expect outs["cluster_selection"] with columns including 'cluster', 'pass_both', 'delta_RMSE_mean'
    if "outs" in globals() and isinstance(outs, dict) and "cluster_selection" in outs:
        sel = outs["cluster_selection"].copy()
        # normalize types
        if "cluster" in sel.columns:
            sel["cluster"] = pd.to_numeric(sel["cluster"], errors="coerce").astype("Int64")
        tmp_df = sel.merge(
            impact_df.rename(columns={"family": "cluster"}),
            on="cluster",
            how="left",
            sort=False,
            validate="1:1"
        )
        # Confidence_level: 0 if pass_both==True; 1 if not pass_both and delta_RMSE_mean>0; else 2
        def _conf_level(row):
            pb = bool(row.get("pass_both")) if not pd.isna(row.get("pass_both")) else False
            drm = row.get("delta_RMSE_mean", 0)
            if pb:
                return 0
            if (not pb) and (pd.to_numeric(pd.Series([drm]), errors="coerce").iloc[0] > 0):
                return 1
            return 2

        tmp_df["Confidence_level"] = tmp_df.apply(_conf_level, axis=1)
        tmp_df = tmp_df.sort_values(
            by=["delta_RMSE_mean", "pass_both"],
            ascending=[False, False],
            ignore_index=True
        )

        # Write to workbook
        with pd.ExcelWriter(cluster_membersFiles, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
            tmp_df.to_excel(writer, sheet_name="cluster_selection_table_pr", index=False)
        print("Updated sheet: cluster_selection_table_pr")
    else:
        print("Skipped merge: 'outs[\"cluster_selection\"]' not found.")
        tmp_df = None

    # ---------- Bubble map ----------
    # Uses Excel sheet if it exists (preferred), else falls back to tmp_df if available.
    xlsx_path = Path(cluster_membersFiles)
    sheet = "cluster_selection_table_pr"

    if xlsx_path.exists():
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
    elif tmp_df is not None:
        df = tmp_df.copy()
    else:
        raise FileNotFoundError(
            "No 'cluster_selection_table_pr' sheet found and no tmp_df to plot."
        )

    # Required columns
    needed = {"cluster", "condition_key", "max_mean_abs_shap", "Confidence_level"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for bubble map: {sorted(missing)}")

    # Derive oxygen if missing/NA
    if "oxygen" not in df.columns or df["oxygen"].isna().all():
        df["oxygen"] = df["condition_key"].astype(str).apply(
            lambda s: "Aerobic" if s.endswith("_ae") else ("Anaerobic" if s.endswith("_an") else "")
        )

    # Clean and type
    df = df.dropna(subset=["cluster","condition_key","max_mean_abs_shap","Confidence_level"]).copy()
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype(int)
    df["Confidence_level"] = pd.to_numeric(df["Confidence_level"], errors="coerce").astype(int)
    df["max_mean_abs_shap"] = pd.to_numeric(df["max_mean_abs_shap"], errors="coerce")

    # Axis ordering (conditions)
    # aero = ["pC_ae","Ac+pCA_ae","Ac+CA_ae","Ac+SF_ae","Ac+SA_ae","Ac+KL_ae","SF_ae","ANCHOR:Ac_ae"]
    # ana  = ["pC_an","Ac+pCA_an","Ac+CA_an","Ac+SF_an","Ac+SA_an","Ac+KL_an","CA_an","ANCHOR:Ac_an"]
    y_order = aero + ana
    cond_to_y = {c:i for i, c in enumerate(y_order)}
    df = df[df["condition_key"].isin(y_order)].copy()
    df["ypos"] = df["condition_key"].map(cond_to_y)

    # X ordering: by Confidence_level (0Â1Â2), then by descending max_mean_abs_shap
    x_order = (
        df.sort_values(["Confidence_level","max_mean_abs_shap"], ascending=[True, False])
          ["cluster"].drop_duplicates().tolist()
    )
    clust_to_x = {c:i for i, c in enumerate(x_order)}
    df["xpos"] = df["cluster"].map(clust_to_x)

    # Bubble-size scaling (robust)
    def bubble_sizes(values, smin=90, smax=1900, q=(5, 95)):
        v = np.asarray(values, float)
        v = np.clip(v, 0.0, None)
        if not np.isfinite(v).any():
            return np.full_like(v, 0.5*(smin+smax), dtype=float)
        lo, hi = np.nanpercentile(v, q)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(v), np.nanmax(v)
        if hi <= lo:
            return np.full_like(v, 0.5*(smin+smax), dtype=float)
        v_clip = np.clip(v, lo, hi)
        v_s = np.sqrt(v_clip)  # perceptual area
        v_s = (v_s - v_s.min()) / (v_s.max() - v_s.min() + 1e-12)
        return smin + v_s * (smax - smin)

    sizes = bubble_sizes(df["max_mean_abs_shap"].values, smin=90, smax=1900, q=(5,95))
    oxy_color = df["oxygen"].map({"Aerobic":"tab:blue", "Anaerobic":"tab:orange"}).fillna("gray")

    # Masks
    m0 = df["Confidence_level"] == 0   # selected
    m1 = df["Confidence_level"] == 1   # RMSE > 0 (not selected)
    m2 = df["Confidence_level"] == 2   # RMSE < 0

    # Pick top call per condition: min Confidence_level, tie-break by max max_mean_abs_shap
    top_idx = set()
    for cond, sub in df.groupby("condition_key", sort=False):
        sub2 = sub.sort_values(["Confidence_level","max_mean_abs_shap"], ascending=[True, False])
        if not sub2.empty:
            top_idx.add(sub2.index[0])

    # Hi-quality, math-friendly styling
    mpl.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    # Plot
    fig_w = max(12, 0.38*len(x_order))
    fig_h = max(7.0, 0.60*len(y_order))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.set_facecolor("white")

    # Level 0: filled circles
    ax.scatter(df.loc[m0, "xpos"], df.loc[m0, "ypos"],
               s=sizes[m0], c=oxy_color[m0], edgecolors="k",
               linewidths=0.9, alpha=0.95, zorder=3, label="level 0 (selected)")

    # Level 1: hollow, oxygen edge
    ax.scatter(df.loc[m1, "xpos"], df.loc[m1, "ypos"],
               s=sizes[m1], facecolors="none", edgecolors=oxy_color[m1],
               linewidths=0.9, alpha=0.9, zorder=2, label=r"level 1 ($\Delta\mathrm{RMSE}>0$)")

    # Level 2: hollow, crimson edge
    ax.scatter(df.loc[m2, "xpos"], df.loc[m2, "ypos"],
               s=sizes[m2], facecolors="none", edgecolors="crimson",
               linewidths=1.0, alpha=0.95, zorder=3, label=r"level 2 ($\Delta\mathrm{RMSE}<0$)")

    # Stars on top calls
    ax.scatter(df.loc[list(top_idx), "xpos"].values,
               df.loc[list(top_idx), "ypos"].values,
               marker="*", s=220, c="crimson", edgecolors="k", linewidths=0.6,
               zorder=5, label="per-condition top call")

    # Divider between Aerobic and Anaerobic blocks
    ax.axhline(len(aero) - 0.5, color="k", lw=1.2, zorder=1)

    # Optional vertical dividers between levels on x-axis
    n0 = len(df.loc[m0, "cluster"].drop_duplicates())
    n1 = len(df.loc[m1, "cluster"].drop_duplicates())
    if n0 > 0: ax.axvline(n0 - 0.5, color="0.25", lw=1.0, ls="--", zorder=1)
    if n1 > 0: ax.axvline(n0 + n1 - 0.5, color="0.25", lw=1.0, ls="--", zorder=1)

    # Ticks/labels
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels([f"F{c}" for c in x_order], rotation=90, ha="center")
    ax.set_yticks(range(len(y_order)))
    ax.set_yticklabels([c.replace("_ae"," (Ae)").replace("_an"," (An)") for c in y_order])
    ax.set_xlabel("Cluster (module)")
    ax.set_ylabel("Condition")

    # Grid & limits
    ax.set_axisbelow(True)
    ax.grid(axis="x", which="major", linestyle="--", color="0.85", linewidth=0.8)
    ax.grid(axis="y", which="major", linestyle=":",  color="0.85", linewidth=0.8)
    ax.set_xlim(-0.6, len(x_order)-0.4)
    ax.set_ylim(len(y_order)-0.6, -0.6)

    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("0.3")
        ax.spines[spine].set_linewidth(1.0)

    # Consolidated legend (size + levels/markers + oxygen)
    # Bubble-size scale (quartiles)
    def _size_handle(val):
        s = bubble_sizes([val], smin=90, smax=1900, q=(5, 95))[0]
        return plt.scatter([], [], s=s, facecolor="lightgray", edgecolor="k")

    q_list = (25, 50, 90)
    size_vals = [np.nanpercentile(df["max_mean_abs_shap"].values, q) for q in q_list]
    size_handles = [_size_handle(v) for v in size_vals]
    size_labels  = [fr"{q}th: {v:.2e}" for q, v in zip(q_list, size_vals)]

    hdr_size = Line2D([0], [0], color="none", label=r"size $\propto$ max mean $|\mathrm{SHAP}|$")
    hdr_lvl  = Line2D([0], [0], color="none", label="levels / markers")
    hdr_oxy  = Line2D([0], [0], color="none", label="oxygen")

    handles_lvl = [
        Line2D([0],[0], marker="o", linestyle="none",
               markerfacecolor="0.75", markeredgecolor="k", markersize=9,
               label="level 0 (selected)"),
        Line2D([0],[0], marker="o", linestyle="none",
               markerfacecolor="none", markeredgecolor="0.35", markersize=9,
               label=r"level 1 ($\Delta\mathrm{RMSE}>0$)"),
        Line2D([0],[0], marker="o", linestyle="none",
               markerfacecolor="none", markeredgecolor="crimson", markersize=9,
               label=r"level 2 ($\Delta\mathrm{RMSE}<0$)"),
        Line2D([0],[0], marker="*", linestyle="none",
               markerfacecolor="crimson", markeredgecolor="k", markersize=11,
               label="per-condition top call"),
    ]

    handles_oxy = [
        Line2D([0],[0], marker="o", linestyle="none",
               markerfacecolor="tab:blue", markeredgecolor="k", markersize=8,
               label="Aerobic"),
        Line2D([0],[0], marker="o", linestyle="none",
               markerfacecolor="tab:orange", markeredgecolor="k", markersize=8,
               label="Anaerobic"),
    ]

    handles = [hdr_size] + size_handles + [hdr_lvl] + handles_lvl + [hdr_oxy] + handles_oxy
    labels  = [h.get_label() for h in handles]
    filtered = [(h, l) for h, l in zip(handles, labels) if not getattr(h, "get_label", lambda: "")().startswith("_") and not str(l).startswith("_")]
    if filtered:
        handles_f, labels_f = zip(*filtered)
        leg = ax.legend(handles_f, labels_f, 
                        loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    ncol=2, frameon=True, fontsize=9,
                    columnspacing=1.2, handletextpad=0.7, borderaxespad=0.2)
    # leg = ax.legend(handles, labels,
                    # loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    # ncol=2, frameon=True, fontsize=9,
                    # columnspacing=1.2, handletextpad=0.7, borderaxespad=0.2)
    # Bold section headers
    for t in leg.get_texts():
        if t.get_text() in {hdr_size.get_label(), hdr_lvl.get_label(), hdr_oxy.get_label()}:
            t.set_fontweight("bold")
    leg.get_frame().set_alpha(0.95)

    # Save
    png_path = Path(OUT_DIR) / "condition_assignment_bubblemap_clustersX_conditionsY_withStars.png"
    plt.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close()
    print(f"Saved figure: {png_path.resolve()}")

    # Selected clusters list
    if "pass_both" in df.columns:
        # WANTED = list(
        #     df.loc[list(top_idx)]
        #       .query("pass_both == True")
        #       .sort_values("xpos")["cluster"].values
        # )
        df_top = (
            df.sort_values(["condition_key","Confidence_level","max_mean_abs_shap"],
                        ascending=[True,True,False])
            .groupby("condition_key", sort=False)
            .head(1)
        )

        WANTED = (
            df_top.query("delta_RMSE_mean > 0")
                .sort_values("xpos")["cluster"]
                .astype(int).tolist()
        )

    else:
        # fallback: just the top per condition in x-order
        WANTED = list(
            df.loc[list(top_idx)]
              .sort_values("xpos")["cluster"].values
        )
    print(f"The selected clusters are: {WANTED}")


    # ============================================
    # Per-feature quadrant/rank significance
    # ============================================

    PATH_OUTSIDE = Feature_level_outside_conditionalFile
    PATH_ALL     = Feature_level_Others_conditionalFile
    PATH_TC      = cluster_membersFiles
    SHEET_SHAP   = "shap_by_group"

    # SHAP columns (as provided; keep order)
    # SHAP_COLS = [
        # # Aerobic
        # "pC_ae", "Ac+pCA_ae", "Ac+CA_ae", "Ac+SF_ae", "Ac+SA_ae", "Ac+KL_ae", "SF_ae", "ANCHOR:Ac_ae",
        # # Anaerobic
        # "pC_an", "Ac+pCA_an", "Ac+CA_an", "Ac+SF_an", "Ac+SA_an", "Ac+KL_an", "CA_an", "ANCHOR:Ac_an",
    # ]

    # -- Load per-feature DELTA-RMSE (two books) --
    df_out = read_per_feature_book(PATH_OUTSIDE, tag="out")
    df_all = read_per_feature_book(PATH_ALL,     tag="all")

    paired = (df_out.merge(df_all, on=["feature","cluster","n_iters"], how="inner")
                    .drop_duplicates(subset=["feature","cluster"]))

    # -- SHAP corroboration --
    tc = pd.read_excel(PATH_TC, sheet_name=SHEET_SHAP)
    tc.columns = [c.strip() for c in tc.columns]
    for c in SHAP_COLS:
        if c not in tc.columns:
            raise ValueError(f"{PATH_TC}::{SHEET_SHAP} missing column {c}")

    tc["feature_upper"] = norm_feat(tc["feature"])
    tc["cluster"] = pd.to_numeric(tc["parent_cluster"], errors="coerce").astype("Int64")
    tc["mean_abs_shap"] = tc[SHAP_COLS].abs().mean(axis=1)

    shap_mean = tc[["feature_upper","cluster","mean_abs_shap"]].rename(columns={"feature_upper":"feature"})
    paired = paired.merge(shap_mean, on=["feature","cluster"], how="left")
    paired["mean_abs_shap"] = pd.to_numeric(paired["mean_abs_shap"], errors="coerce").fillna(0.0)

    # -- Z-scores, quadrants, indices, significance, ranking --
    # Per-cluster z scores: use transform (aligned, no deprecation)
    paired["z_out_rank"] = (paired.groupby("cluster", group_keys=False)["d_out"].transform(z_in_group))
    paired["z_all_rank"] = (paired.groupby("cluster", group_keys=False)["d_all"].transform(z_in_group))

    # Quadrant + indices
    paired["quad"] = paired.apply(quad, axis=1)
    paired["redundancy_index"] = paired["z_out_rank"] - paired["z_all_rank"]
    paired["unique_index"]     = np.minimum(paired["z_out_rank"], paired["z_all_rank"])

    # Vectorized per-feature significance that still does BH per cluster
    paired = add_per_feature_significance_df(paired)
    # Ranking, but pass include_groups=False to future-proof
    #ranked = (paired.groupby("cluster", group_keys=True).apply(rank_cluster, include_groups=True).reset_index(drop=True))
    ranked = compute_ranks_vectorized(paired)

    # -- Output per-cluster sheets + concatenated views --
    ranked = ranked[[
        "cluster","feature","n_iters",
        "d_out","d_all",
        "z_out_rank","z_all_rank","quad",
        "mean_abs_shap","redundancy_index","unique_index",
        "rep_rank","proxy_rank","balanced_rank",
        "mu_out","sd_out","se_out","zstat_out","p_out","q_out",
        "pass_sign_out","pass_effect_out","pass_fdr_out","pass_both_out",
        "mu_all","sd_all","se_all","zstat_all","p_all","q_all",
        "pass_sign_all","pass_effect_all","pass_fdr_all","pass_both_all",
        "consensus_pass","consensus_score"
    ]]

    out_xlsx = feature_QuadrantsAndRanks_file    
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xl:
        for k, g in ranked.groupby("cluster"):
            g.to_excel(xl, index=False, sheet_name=f"cluster_{int(k)}")
        ranked.to_excel(xl, index=False, sheet_name="all_clusters_concat")

        pass_any = ranked.loc[
            ranked["consensus_pass"] | ranked["pass_both_out"] | ranked["pass_both_all"]
        ].copy()
        pass_any.sort_values(["cluster","consensus_pass","balanced_rank"], ascending=[True, False, True]) \
                .to_excel(xl, index=False, sheet_name="feature_sig_passers_only")

    print(f"Wrote: {Path(out_xlsx).resolve()}")

    # ============================================
    #  (Optional) Condition-specific extraction
    # ============================================

    IN_XLSX = feature_QuadrantsAndRanks_file    
    SHEET   = "all_clusters_concat"
    NEW_SHEET = "condition_top"

    # WANTED = []  # (keep your existing variable and values)
    in_path = Path(IN_XLSX)
    if not in_path.exists():
        raise FileNotFoundError(f"Workbook not found: {in_path.resolve()}")

    df_top = pd.read_excel(in_path, sheet_name=SHEET)
    df_top["cluster"] = pd.to_numeric(df_top["cluster"], errors="coerce").astype("Int64")
    out_sel = df_top[df_top["cluster"].isin(WANTED)].copy()

    with pd.ExcelWriter(in_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as xl:
        out_sel.to_excel(xl, index=False, sheet_name=NEW_SHEET)

    print(f"Appended sheet '{NEW_SHEET}' with {len(out_sel)} rows to: {in_path.resolve()}")

    # ============================================
    # HH consensus rank assembly (with condition mapping)
    # ============================================

    IN_XLSX_GENES = feature_QuadrantsAndRanks_file    
    SHEET_GENES   = "all_clusters_concat"
    IN_XLSX_COND  = cluster_membersFiles
    SHEET_COND    = "cluster_selection_table_pr"

    R = pd.read_excel(IN_XLSX_GENES, sheet_name=SHEET_GENES)
    R.columns = [c.strip() for c in R.columns]

    need = {"cluster","feature","quad","consensus_pass","z_out_rank","z_all_rank","mean_abs_shap"}
    missing = need - set(R.columns)
    if missing:
        raise ValueError(f"Missing columns in {IN_XLSX_GENES}::{SHEET_GENES}: {sorted(missing)}")

    if "unique_index" not in R.columns:
        R["unique_index"] = np.minimum(pd.to_numeric(R["z_out_rank"], errors="coerce"),
                                       pd.to_numeric(R["z_all_rank"], errors="coerce"))

    R["cluster"] = pd.to_numeric(R["cluster"], errors="coerce").astype(int)
    R["consensus_pass"] = R["consensus_pass"].astype(bool)
    for c in ("z_out_rank","z_all_rank","unique_index","mean_abs_shap"):
        R[c] = pd.to_numeric(R[c], errors="coerce")

    C = pd.read_excel(IN_XLSX_COND, sheet_name=SHEET_COND)
    C.columns = [c.strip() for c in C.columns]
    if "cluster" not in C.columns or "condition_key" not in C.columns:
        raise ValueError(f"{IN_XLSX_COND}::{SHEET_COND} must have 'cluster' and 'condition_key'")

    C["cluster"] = pd.to_numeric(C["cluster"], errors="coerce").astype(int)

    if "max_mean_abs_shap" in C.columns:
        best_idx = C.groupby("cluster")["max_mean_abs_shap"].idxmax()
        C_best = C.loc[best_idx].copy()
    else:
        C_best = C.drop_duplicates(subset=["cluster"]).copy()

    keep_cols = ["cluster","condition_key"] + [c for c in ("condition","oxygen") if c in C_best.columns]
    C_best = C_best[keep_cols]

    R = R.merge(C_best, on="cluster", how="left")

    if WANTED is not None:
        R = R[R["cluster"].isin(WANTED)].copy()

    HH = (R[(R["quad"] == "HH") & (R["consensus_pass"])].copy()
            .reset_index(drop=True))

    HH = HH.sort_values(["unique_index","z_all_rank","z_out_rank","mean_abs_shap"],
                        ascending=[False, False, False, False])
    HH["HH_rank_overall"] = np.arange(1, len(HH) + 1)

    # HH["HH_rank_in_cluster"] = (HH.groupby("cluster", group_keys=False)
                                  # .apply(lambda g: g.assign(HH_rank_in_cluster=np.arange(1, len(g) + 1)))
                                  # ["HH_rank_in_cluster"]
                                  # .values)
    HH = HH.sort_values(["cluster","balanced_rank","rep_rank","proxy_rank"],
                    ascending=[True, True, True, True]).copy()
    HH["HH_rank_in_cluster"] = HH.groupby("cluster").cumcount() + 1

    out_cols = ["cluster","feature","quad","consensus_pass",
                "z_all_rank","z_out_rank","unique_index","mean_abs_shap",
                "condition_key"] + [c for c in ("condition","oxygen") if c in HH.columns] + \
               ["HH_rank_overall","HH_rank_in_cluster"]
    HH_out = HH[out_cols].reset_index(drop=True)

    mode = "a" if Path(IN_XLSX_GENES).exists() else "w"
    with pd.ExcelWriter(IN_XLSX_GENES, engine="openpyxl", mode=mode,
                        if_sheet_exists=("replace" if mode == "a" else None)) as xl:
        HH_out.to_excel(xl, index=False, sheet_name="HH_consensus_rank")

    print(f"Ranked {len(HH_out)} HH + consensus-pass genes; wrote sheet 'HH_consensus_rank' in {Path(IN_XLSX_GENES).resolve()}")

    # ============================================
    # Pan-condition normalization & interpretation sheets
    # ============================================
    # --- Load proteomics ---
    df = pd.read_excel(ProteomicsDataFile, sheet_name="Sheet1")
    feat_df = df.iloc[:, 5:1862].copy()
    feat_df.columns = feat_df.columns.str.upper()
    y_meta = df[["Oxygen"]]

    # --- GLOBAL z-score ---
    X_df = feat_df.sub(feat_df.mean(axis=0), axis=1).div(feat_df.std(axis=0, ddof=0), axis=1)

    # --- Ae vs An Welch tests (vectorized loop, compact) ---
    ae = y_meta["Oxygen"] == "Aerobic"
    an = y_meta["Oxygen"] == "Anaerobic"

    results = []
    for g in X_df.columns:
        ae_vals = X_df.loc[ae, g]
        an_vals = X_df.loc[an, g]
        _, p = ttest_ind(ae_vals, an_vals, equal_var=False, nan_policy="omit")
        results.append([g, ae_vals.mean(), an_vals.mean(), an_vals.mean() - ae_vals.mean(), p])

    res = pd.DataFrame(results, columns=["feature", "mean_Ae", "mean_An", "diff_An_minus_Ae", "p_raw"])

    # --- FDR + direction ---
    res["q_value_O2_DEG"] = multipletests(res["p_raw"], method="fdr_bh")[1]
    res["AeAn_direction"] = np.where(
        res["q_value_O2_DEG"] >= 0.05, "ns",
        np.where(res["diff_An_minus_Ae"] > 0, "An_up", "Ae_up")
    )
    # ==============================================================================
    # ---------------------------------------------------------------------
    # Inputs
    # ---------------------------------------------------------------------
    infile = Path(cluster_membersFiles)
    sheet_shap = "shap_by_group"
    sheet_feat = "all_clusters_concat"
    cond_cols = [
        "Ac+CA_ae","Ac+CA_an","Ac+KL_ae","ANCHOR:Ac_ae","ANCHOR:Ac_an","Ac+KL_an",
        "Ac+SA_ae","Ac+SA_an","Ac+SF_ae","Ac+SF_an","Ac+pCA_ae","Ac+pCA_an",
        "CA_an","SF_ae","pC_ae","pC_an"
    ]

    # ---------------------------------------------------------------------
    # Load required tables
    # ---------------------------------------------------------------------
    shap = pd.read_excel(infile, sheet_name=sheet_shap)
    feat = pd.read_excel(infile, sheet_name=sheet_feat)

    shap["feature"] = shap["feature"].astype(str).str.upper()
    feat["feature"] = feat["feature"].astype(str).str.upper()

    eps = 1e-12
    missing = [c for c in cond_cols if c not in shap.columns]
    if missing:
        raise ValueError(f"Missing expected condition columns in {sheet_shap}: {missing}")

    # ---------------------------------------------------------------------
    # 2. Build normalized importance matrix S (simple, defensible)
    #    - S_raw = |SHAP|
    #    - per-condition median scaling
    # ---------------------------------------------------------------------
    S_raw = shap[cond_cols].abs()
    raw = S_raw.to_numpy()
    gmin = raw.min().min()
    gmax = raw.max().max()
    S_raw = (S_raw - gmin) / (gmax - gmin + eps)
    scale = S_raw.median(axis=0).replace(0, eps)
    S = S_raw.divide(scale, axis=1).fillna(0.0)  # shape: (n_proteins, n_conditions)

    # convenience
    n_proteins, n_conds = S.shape

    ae_cols = [c for c in cond_cols if c.endswith("_ae")]
    an_cols = [c for c in cond_cols if c.endswith("_an")]

    S_ae = S[ae_cols]
    S_an = S[an_cols]

    nA = len(ae_cols)
    nN = len(an_cols)

    if (nA == 0) or (nN == 0):
        raise ValueError("Could not find both _ae and _an columns in cond_cols.")

    # ---------------------------------------------------------------------
    # 3. Global mean importance, Ae/An means, regime variance decomposition
    # ---------------------------------------------------------------------
    # global mean importance across all conditions
    global_mean = S.mean(axis=1)

    # regime-specific means
    mean_Ae = S_ae.mean(axis=1)
    mean_An = S_an.mean(axis=1)

    delta_An_minus_Ae = mean_An - mean_Ae   # positive -> more important in An

    # total variance across all conditions for each protein
    total_var = S.var(axis=1, ddof=1).replace(0, eps)

    # between-regime variance (ANOVA-style, 2 groups)
    mA = mean_Ae.to_numpy()
    mN = mean_An.to_numpy()
    grand = (nA * mA + nN * mN) / (nA + nN)

    between_num = nA * (mA - grand) ** 2 + nN * (mN - grand) ** 2
    between_var = between_num / (nA + nN)   # scaling constant; only ratios matter

    # fraction of variance explained by regime vs within-condition context
    regime_var_frac = between_var / (total_var.to_numpy() + eps)   # in [0,1] (approx)
    regime_var_frac = np.clip(regime_var_frac, 0.0, 1.0)

    # within-regime variance (for convenience; not strictly needed)
    within_var = total_var.to_numpy() - between_var
    within_var = np.maximum(within_var, 0.0)

    # ---------------------------------------------------------------------
    # 4. Formal test for Ae vs An importance difference (t-test + BH-FDR)
    # ---------------------------------------------------------------------
    tvals, pvals = ttest_ind(
        S_ae.to_numpy(),
        S_an.to_numpy(),
        axis=1,
        equal_var=False,
        nan_policy="omit",
    )

    pvals = np.asarray(pvals)

    # 2. Identify finite p-values
    finite_mask = np.isfinite(pvals)

    qvals = np.full_like(pvals, np.nan, dtype=float)
    rej   = np.zeros_like(pvals, dtype=bool)

    if finite_mask.sum() > 0:
        rej_finite, qvals_finite, _, _ = multipletests(
            pvals[finite_mask],
            alpha=0.05,
            method="fdr_bh",
        )
        qvals[finite_mask] = qvals_finite
        rej[finite_mask]   = rej_finite

    # 3. Optionally treat NaN q as "no effect"
    # (if you don't want NaNs downstream)
    qvals = np.where(np.isnan(qvals), 1.0, qvals)
    rej   = np.where(np.isnan(qvals), False, rej)

    # # BH-FDR across proteins
    # rej, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    # simple deterministic label for regime effect
    def regime_label(delta, q):
        if q < 0.05:
            if delta > 0:
                return "An-enriched importance"
            elif delta < 0:
                return "Ae-enriched importance"
        return "Regime-invariant importance"

    regime_labels = [regime_label(d, q) for d, q in zip(delta_An_minus_Ae, qvals)]

    # ---------------------------------------------------------------------
    # 5. Assemble primary pan-condition metrics in 'out'
    # ---------------------------------------------------------------------
    keep_cols = [c for c in ["feature", "parent_cluster", "UniProt ID"] if c in shap.columns]
    out = shap[keep_cols].copy()

    out["PAN_global_mean_normSHAP"] = global_mean.values
    out["Ae_mean_normSHAP"]        = mean_Ae.values
    out["An_mean_normSHAP"]        = mean_An.values
    out["AeAn_delta_An_minus_Ae"]  = delta_An_minus_Ae.values

    out["total_var_normSHAP"]      = total_var.values
    out["regime_var_frac"]         = regime_var_frac
    out["within_var_normSHAP"]     = within_var

    out["p_regime_Ae_vs_An"]       = pvals
    out["q_regime_Ae_vs_An"]       = qvals
    out["regime_importance_label"] = regime_labels

    # ---------------------------------------------------------------------
    # 6. Merge with feature-level stats and optional tables
    # ---------------------------------------------------------------------
    req_cols = [
        "feature","d_out","d_all","z_out_rank","z_all_rank",
        "quad","mean_abs_shap","consensus_pass","consensus_score"
    ]
    avail_cols = [c for c in req_cols if c in feat.columns]

    if "feature" in feat.columns:
        feat_sub = feat[avail_cols].drop_duplicates(subset=["feature"])
    else:
        feat_sub = pd.DataFrame(columns=["feature"])

    merged = out.merge(feat_sub, on="feature", how="left")

    # Description (optional)
    try:
        all_clusters = pd.read_excel(infile, sheet_name="all_clusters")
        all_clusters["feature"] = all_clusters["feature"].astype(str).str.upper()
        if {"feature","Description"}.issubset(all_clusters.columns):
            desc = all_clusters[["feature","Description"]].drop_duplicates("feature")
            merged = merged.merge(desc, on="feature", how="left")
    except Exception:
        pass

    # cluster_selection_table_pr (optional; carries family/condition context)
    try:
        clsel = pd.read_excel(infile, sheet_name="cluster_selection_table_pr")
        if {"family","condition_key"}.issubset(clsel.columns):
            ck = (
                clsel.groupby("family", dropna=False)["condition_key"]
                .apply(lambda s: "|".join(sorted(pd.unique(s.dropna()))))
                .reset_index()
                .rename(columns={"family": "parent_cluster",
                                "condition_key": "cluster_condition_keys"})
            )
            merged = merged.merge(ck, on="parent_cluster", how="left")
    except Exception:
        pass

    # FC_wide (optional; bring in log2 fold-changes)
    try:
        fc = pd.read_excel(infile, sheet_name="FC_wide")
        fc["feature"] = fc["feature"].astype(str).str.upper()
        fc_cols = [
            "feature","Ac+CA_ae","Ac+CA_an","Ac+KL_ae","Ac+KL_an",
            "Ac+SA_ae","Ac+SA_an","Ac+SF_ae","Ac+SF_an",
            "Ac+pCA_ae","Ac+pCA_an","CA_an","SF_ae","pC_ae","pC_an"
        ]
        avail_fc = [c for c in fc_cols if c in fc.columns]
        if "feature" in avail_fc:
            tmp = fc[avail_fc].copy()
            rename_map = {c: f"{c}_log2fc" for c in avail_fc if c != "feature"}
            tmp = tmp.rename(columns=rename_map)
            merged = merged.merge(tmp, on="feature", how="left")
    except Exception:
        pass

    # ---------------------------------------------------------------------
    # (Optional) BREADTH metrics on normalized S (deterministic, simple)
    # ---------------------------------------------------------------------
    S_with_feat = S.copy()
    S_with_feat["feature"] = shap["feature"].astype(str).str.upper()
    merged["feature"] = merged["feature"].astype(str).str.upper()

    merged = merged.merge(
        S_with_feat[["feature"] + cond_cols],
        on="feature",
        how="left",
        suffixes=("", "")
    )

    Svals = merged[cond_cols].fillna(0.0).to_numpy()
    merged["BREADTH_frac_ge1x"] = (Svals >= 1.0).mean(axis=1)
    merged["BREADTH_frac_ge2x"] = (Svals >= 2.0).mean(axis=1)
    merged["BREADTH_frac_ge5x"] = (Svals >= 5.0).mean(axis=1)
    # ---------------------------------------------------------------------
    # Define biologically meaningful classes using percentile thresholds
    # ---------------------------------------------------------------------
   # Helper: safe percentile
    def pct_threshold(series, q):
        return pd.to_numeric(series, errors="coerce").fillna(0.0).quantile(q)
    # Global importance thresholds
    global_med = pct_threshold(merged["PAN_global_mean_normSHAP"], 0.50)
    global_hi  = pct_threshold(merged["PAN_global_mean_normSHAP"], 0.75)
    # Regime variance threshold (how much of variance is explained by Ae vs An)
    regime_hi  = pct_threshold(merged["regime_var_frac"], 0.75)
    regime_mid = pct_threshold(merged["regime_var_frac"], 0.50)
    # Optional: effect-size magnitude threshold (for very strong Ae/An bias)
    delta_abs = merged["AeAn_delta_An_minus_Ae"].abs()
    delta_hi  = pct_threshold(delta_abs, 0.75)
    alpha_regime = 0.05  # FDR threshold for Ae vs An difference
    def classify_pan_bio(row):
        gmean   = row["PAN_global_mean_normSHAP"]
        fracR   = row["regime_var_frac"]
        delta   = row["AeAn_delta_An_minus_Ae"]
        q_reg   = row["q_regime_Ae_vs_An"]
        # 1) Background or low-importance proteins
        if gmean < global_med:
            return "Background / low-importance"
        # 2) Strong Ae vs An effect, substantial regime variance
        if (gmean >= global_hi) and (q_reg < alpha_regime) and (fracR >= regime_hi):
            # strong regime bias (effect size)
            if delta < 0:
                return "Aerobic core determinant"
            elif delta > 0:
                return "Anaerobic core determinant"
        # 3) High global importance, regime-invariant (bridging core)
        if (gmean >= global_hi) and (q_reg >= alpha_regime) and (fracR <= regime_mid):
            return "Oxygen-bridging pan-condition core"
        # 4) Remaining high-importance proteins = context-modulated contributors
        if gmean >= global_hi:
            return "Context-dependent modulator"
        # 5) Everything else = intermediate importance
        return "Intermediate-importance protein"
    merged["pan_bio_class"] = merged.apply(classify_pan_bio, axis=1)
    def short_label(cls):
        mapping = {
            "Background / low-importance": "Background",
            "Intermediate-importance protein": "Intermediate",
            "Aerobic core determinant": "Ae core",
            "Anaerobic core determinant": "An core",
            "Oxygen-bridging pan-condition core": "O2-bridging core",
            "Context-dependent modulator": "Context-modulator",
        }
        return mapping.get(cls, cls)

    merged["pan_bio_class_short"] = merged["pan_bio_class"].map(short_label)
    # ------------------------------------------------------------
    # Merge to bring in KEGGID
    # ------------------------------------------------------------
    kegg_df = pd.read_excel(Kegg2UniprotFile, sheet_name="MasterDict")
    kegg_df["UniProt"] = kegg_df["UniProt"].astype(str).str.upper()
    merged["UniProt ID"] = merged["UniProt ID"].astype(str).str.upper()
    merged = merged.merge(kegg_df[["UniProt", "KEGGID", "KO_ID", "Pathway_IDs", "Pathway_Names"]],how="left",left_on="UniProt ID",right_on="UniProt")
    merged.drop(columns=["UniProt"], inplace=True)
    desired_order = [
        'feature', 'UniProt ID', 'Description', "KEGGID",  "KO_ID", "Pathway_IDs", "Pathway_Names", 'pan_bio_class', 'pan_bio_class_short',
        'parent_cluster', 'consensus_pass', 'cluster_condition_keys',
        'regime_importance_label', 'PAN_global_mean_normSHAP', 
        'Ae_mean_normSHAP', 'An_mean_normSHAP', 'AeAn_delta_An_minus_Ae',
        'total_var_normSHAP', 'regime_var_frac', 'within_var_normSHAP',
        'p_regime_Ae_vs_An', 'q_regime_Ae_vs_An', 'd_out', 'd_all', 'z_out_rank',
        'z_all_rank', 'quad', 'mean_abs_shap', 'consensus_score',
        'Ac+CA_ae', 'Ac+CA_an', 'Ac+KL_ae', 'ANCHOR:Ac_ae', 'ANCHOR:Ac_an',
        'Ac+KL_an', 'Ac+SA_ae', 'Ac+SA_an', 'Ac+SF_ae', 'Ac+SF_an',
        'Ac+pCA_ae', 'Ac+pCA_an', 'CA_an', 'SF_ae', 'pC_ae', 'pC_an',
        'BREADTH_frac_ge1x', 'BREADTH_frac_ge2x', 'BREADTH_frac_ge5x'
    ]
    merged = merged[desired_order]
    # --- Merge into existing merged table ---
    merged["feature"] = merged["feature"].str.upper()
    merged = merged.merge(res[["feature", "q_value_O2_DEG", "AeAn_direction"]], on="feature", how="left")
    # ---------------------------------------------------------------------
    # Save XLSX with the new, non-heuristic pan-condition stats
    # ---------------------------------------------------------------------
    xlsx_path = pan_condition_outputFile
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        merged.to_excel(writer, sheet_name="pan_condition", index=False)

        legend = pd.DataFrame({
            "Metric": [
                "PAN_global_mean_normSHAP",
                "Ae_mean_normSHAP / An_mean_normSHAP",
                "AeAn_delta_An_minus_Ae",
                "regime_var_frac",
                "p_regime_Ae_vs_An / q_regime_Ae_vs_An",
                "regime_importance_label",
            ],
            "Meaning": [
                "Average normalized SHAP importance across all conditions",
                "Average normalized importance in aerobic vs anaerobic regimes",
                "Difference in mean importance (An - Ae); sign reflects regime bias",
                "Fraction of total importance variance explained by oxygen regime",
                "Formal test for Ae vs An importance difference (t-test + BH-FDR)",
                "Deterministic label: Ae-enriched, An-enriched, or regime-invariant",
            ],
        })
        legend.to_excel(writer, sheet_name="legend", index=False)

    # ================== CORE / ADAPTIVE / BACKGROUND SETS (GLOBAL) ==================
    # -------------------------
    #  Define TOP (core) sets
    # -------------------------
    # Pan-condition top = oxygen-bridging pan core
    pan_Top = merged[
        merged["pan_bio_class_short"] == "O2-bridging core"
    ].copy()
    print(f"pan_Top (O2-bridging pan-condition core): {len(pan_Top)}")

    # Aerobic and anaerobic cores
    merged_Ae_Top = merged[
        merged["pan_bio_class_short"] == "Ae core"
    ].copy()
    print(f"Ae_Top (aerobic core determinants): {len(merged_Ae_Top)}")

    merged_An_Top = merged[
        merged["pan_bio_class_short"] == "An core"
    ].copy()
    print(f"An_Top (anaerobic core determinants): {len(merged_An_Top)}")
    print("================================")

    # -------------------------
    #  Adaptive / Intermediate / Background groups (global)
    # -------------------------
    merged_contextModulators = merged[
        merged["pan_bio_class_short"] == "Context-modulator"
    ].copy()
    print(f"Context-modulator (Adaptive): {len(merged_contextModulators)} genes")

    merged_mixed = merged[
        merged["pan_bio_class_short"] == "Intermediate"
    ].copy()
    print(f"Intermediate (Adaptive): {len(merged_mixed)} genes")

    merged_houseKeep = merged[
        merged["pan_bio_class_short"] == "Background"
    ].copy()
    print(f"Background: {len(merged_houseKeep)} genes")

    # -------------------------
    #  HH-significant mask
    # -------------------------
    TopGenes_HH = (
        pd.read_excel(
            "per_feature_quadrants_ranks_AND_significance_1a.xlsx",
            sheet_name="HH_consensus_rank"
        )["feature"]
        .astype(str).str.strip().str.upper()
    )

    mask_HH = (
        merged["feature"].astype(str).str.upper().isin(TopGenes_HH)
        & (merged["quad"] == "HH")
        & merged["consensus_pass"]
    )

    merged_HH_consensus = merged[mask_HH].copy()
    print(f"Total HH-significant genes in merged table: {len(merged_HH_consensus)}")

    # -------------------------
    # CORE sets (HH-filtered) - feed into Tier 1
    # -------------------------
    core_Top_HH = merged[
        mask_HH &
        (merged["pan_bio_class_short"] == "O2-bridging core")
    ].copy()
    print(f"core_Top_HH (O2-bridging core AND HH_significant): {len(core_Top_HH)} genes")

    core_Ae_Top_HH = merged[
        mask_HH &
        (merged["pan_bio_class_short"] == "Ae core")
    ].copy()
    print(f"core_Ae_Top_HH (Ae core AND HH_significant): {len(core_Ae_Top_HH)} genes")

    core_An_Top_HH = merged[
        mask_HH &
        (merged["pan_bio_class_short"] == "An core")
    ].copy()
    print(f"core_An_Top_HH (An core AND HH_significant): {len(core_An_Top_HH)} genes")

    # Boolean masks for reuse
    core_pan_mask = (merged["pan_bio_class_short"] == "O2-bridging core")
    core_Ae_mask  = (merged["pan_bio_class_short"] == "Ae core")
    core_An_mask  = (merged["pan_bio_class_short"] == "An core")

    # -------------------------
    # CONTEXT-dependent HH hubs (orthogonal axis: regime label)
    # -------------------------
    context_pan_HH = merged[
        mask_HH &
        ~core_pan_mask &
        (merged["regime_importance_label"] == "Regime-invariant importance")
    ].copy()
    print(f"context_pan_HH (HH-significant AND regime-invariant AND not pan core): {len(context_pan_HH)} genes")

    context_Ae_HH = merged[
        mask_HH &
        ~core_Ae_mask &
        (merged["regime_importance_label"] == "Ae-enriched importance")
    ].copy()
    print(f"context_Ae_HH (HH-significant AND Ae-enriched AND not Ae core): {len(context_Ae_HH)} genes")

    context_An_HH = merged[
        mask_HH &
        ~core_An_mask &
        (merged["regime_importance_label"] == "An-enriched importance")
    ].copy()
    print(f"context_An_HH (HH-significant AND An-enriched AND not An core): {len(context_An_HH)} genes")

    # =====================================================
    # HH-derived TIERS based on global pan_bio_class_short
    # =====================================================

    # -------------------------
    # Tier 1 - Universal Core Hubs
    # -------------------------
    tier1_universal_core_HH = core_Top_HH.copy()
    print(f"Tier 1 - Universal Core Hubs (O2-bridging core ) HH): {len(tier1_universal_core_HH)} genes")

    # -------------------------
    # Tier 2 - Adaptive Generalist Hubs (Context-modulator ) HH)
    # -------------------------
    generalist_pan_HH = merged[
        mask_HH &
        (merged["pan_bio_class_short"] == "Context-modulator")
    ].copy()
    print(f"Tier 2 - Adaptive Generalist Hubs (HH ) Context-modulator): {len(generalist_pan_HH)} genes")

    # optional Ae/An splits
    generalist_Ae_HH = merged[
        mask_HH &
        (merged["pan_bio_class_short"] == "Context-modulator") &
        (merged["regime_importance_label"] == "Ae-enriched importance")
    ].copy()
    print(f"Adaptive Generalist Ae-enriched: {len(generalist_Ae_HH)} genes")

    generalist_An_HH = merged[
        mask_HH &
        (merged["pan_bio_class_short"] == "Context-modulator") &
        (merged["regime_importance_label"] == "An-enriched importance")
    ].copy()
    print(f"Adaptive Generalist An-enriched: {len(generalist_An_HH)} genes")

    # -------------------------
    # Tier 3 - Adaptive Specialist Hubs (Intermediate ) HH)
    # -------------------------
    specialist_pan_HH = merged[
        mask_HH &
        (merged["pan_bio_class_short"] == "Intermediate")
    ].copy()
    print(f"Tier 3 - Adaptive Specialist Hubs (HH ) Intermediate): {len(specialist_pan_HH)} genes")

    specialist_Ae_HH = merged[
        mask_HH &
        (merged["pan_bio_class_short"] == "Intermediate") &
        (merged["regime_importance_label"] == "Ae-enriched importance")
    ].copy()
    print(f"Adaptive Specialist Ae-enriched: {len(specialist_Ae_HH)} genes")

    specialist_An_HH = merged[
        mask_HH &
        (merged["pan_bio_class_short"] == "Intermediate") &
        (merged["regime_importance_label"] == "An-enriched importance")
    ].copy()
    print(f"Adaptive Specialist An-enriched: {len(specialist_An_HH)} genes")

    # -------------------------
    # Tier 4 - Conditional Background Hubs (Background ) HH)
    # -------------------------
    background_pan_HH = merged[
        mask_HH &
        (merged["pan_bio_class_short"] == "Background")
    ].copy()
    print(f"Tier 4 - Conditional Background Hubs (HH ) Background): {len(background_pan_HH)} genes")

    background_Ae_HH = merged[
        mask_HH &
        (merged["pan_bio_class_short"] == "Background") &
        (merged["regime_importance_label"] == "Ae-enriched importance")
    ].copy()
    print(f"Conditional Background Ae-enriched: {len(background_Ae_HH)} genes")

    background_An_HH = merged[
        mask_HH &
        (merged["pan_bio_class_short"] == "Background") &
        (merged["regime_importance_label"] == "An-enriched importance")
    ].copy()
    print(f"Conditional Background An-enriched: {len(background_An_HH)} genes")


    # ================================
    # TIERS (global importance axis)
    # ================================

    Tier1_UniversalCore_HH = tier1_universal_core_HH.copy()
    Tier2_AdaptiveGeneralist_HH = generalist_pan_HH.copy()
    Tier3_AdaptiveSpecialist_HH = specialist_pan_HH.copy()
    Tier4_ConditionalBackground_HH = background_pan_HH.copy()

    # -------------------------
    # 7. Append all these as sheets to the same XLSX
    # -------------------------
    def write_if_not_empty(df, writer, sheet_name):
        if df is not None and not df.empty:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a") as writer:
        # TOP (global) sets
        write_if_not_empty(pan_Top,        writer, "PAN_O2BridgingCore")
        write_if_not_empty(merged_Ae_Top,  writer, "PAN_AeCore")
        write_if_not_empty(merged_An_Top,  writer, "PAN_AnCore")

        # All HH-significant
        write_if_not_empty(merged_HH_consensus, writer, "HH_consensus")

        # CORE (HH-filtered)
        write_if_not_empty(core_Top_HH,    writer, "core_O2Bridging_HH")
        write_if_not_empty(core_Ae_Top_HH, writer, "core_Ae_HH")
        write_if_not_empty(core_An_Top_HH, writer, "core_An_HH")

        # CONTEXT-dependent (regime label axis)
        write_if_not_empty(context_pan_HH, writer, "HH_context_pan")
        write_if_not_empty(context_Ae_HH,  writer, "HH_context_Ae")
        write_if_not_empty(context_An_HH,  writer, "HH_context_An")

        # TIERS (global importance axis)
        write_if_not_empty(Tier1_UniversalCore_HH,        writer, "Tier1_UniversalCore_HH")
        write_if_not_empty(Tier2_AdaptiveGeneralist_HH,   writer, "Tier2_AdaptiveGeneralist_HH")
        write_if_not_empty(Tier3_AdaptiveSpecialist_HH,   writer, "Tier3_AdaptiveSpecialist_HH")
        write_if_not_empty(Tier4_ConditionalBackground_HH, writer, "Tier4_ConditionalBackground_HH")


        # Optional Ae/An splits for adaptive/background
        write_if_not_empty(generalist_Ae_HH,  writer, "Tier2_Generalist_Ae_HH")
        write_if_not_empty(generalist_An_HH,  writer, "Tier2_Generalist_An_HH")
        write_if_not_empty(specialist_Ae_HH,  writer, "Tier3_Specialist_Ae_HH")
        write_if_not_empty(specialist_An_HH,  writer, "Tier3_Specialist_An_HH")
        write_if_not_empty(background_Ae_HH,  writer, "Tier4_Background_Ae_HH")
        write_if_not_empty(background_An_HH,  writer, "Tier4_Background_An_HH")

        # PAN-level adaptive/background (global, not HH-filtered)
        write_if_not_empty(merged_contextModulators, writer, "PAN_ContextModulators")
        write_if_not_empty(merged_mixed,             writer, "PAN_Intermediate")
        write_if_not_empty(merged_houseKeep,         writer, "PAN_Background")

    # keep your formatting helper
    format_excel_with_filters_and_freeze(xlsx_path)
