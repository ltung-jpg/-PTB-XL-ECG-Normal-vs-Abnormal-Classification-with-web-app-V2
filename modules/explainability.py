"""Explainability for the HistGradientBoosting pipeline.

GLOBAL   permutation importance computed by the notebook on the held-out test set and stored in the bundle (never invented;
         if the bundle has none, the function says so).
LOCAL    SHAP TreeExplainer values for ONE recording, computed on the *scaled, imputed* features that the booster actually
         sees (the pipeline's imputer + scaler are applied, never refitted).  Values are in log-odds of "Abnormal".
         Without SHAP a transparent fallback is used: replace one feature by the training median and measure the change
         in P(Abnormal) (a one-at-a-time sensitivity, NOT an additive attribution).

What these are NOT: they describe how the MODEL uses its 145 summary features, not physiology, and they are not causal.
A feature name like II_t_amp means "the model's output is sensitive to the T-wave amplitude summary in lead II", not that this
segment of the waveform caused the prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DISCLAIMER = ("Explanations describe how the model uses its input features. They are associations learned from the training "
              "data, not causal statements about the heart, and they do not identify the waveform segment that 'caused' a result.")


def feature_group(name: str) -> str:
    if name in ("age", "sex", "height", "weight", "bmi"):
        return "Demographics"
    if "_" not in name or name.split("_")[0] not in {"I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"}:
        return "Rhythm (RR / HRV)"
    return "Lead " + name.split("_")[0]


def global_importance(bundle: dict, top: int = 20) -> pd.DataFrame | None:
    gi = bundle.get("global_importance")
    if not gi:
        return None
    df = pd.DataFrame(gi["importances"]).T.rename(columns={"mean": "importance", "std": "std"})
    df["group"] = [feature_group(n) for n in df.index]
    df.attrs["method"] = f"{gi['method']}; scoring={gi['scoring']}; dataset={gi['dataset']}; repeats={gi['n_repeats']}"
    return df.sort_values("importance", ascending=False).head(top)


def group_importance(bundle: dict) -> pd.DataFrame | None:
    gi = bundle.get("global_importance")
    if not gi:
        return None
    df = pd.DataFrame(gi["importances"]).T
    df["group"] = [feature_group(n) for n in df.index]
    return df.groupby("group")["mean"].sum().sort_values(ascending=False).to_frame("summed permutation importance")


def _transform(pipeline, frame: pd.DataFrame) -> np.ndarray:
    Z = pipeline.named_steps["impute"].transform(frame)
    return pipeline.named_steps["scale"].transform(Z)


def local_explanation(classifier, frame: pd.DataFrame, top: int = 12) -> dict:
    """Return {'method','table','base_value','note'}; table has feature, value, contribution (log-odds or delta-probability)."""
    names = classifier.feature_names
    pipe = classifier.pipeline
    frame = frame[names]
    Z = _transform(pipe, frame)
    imputed = pipe.named_steps["impute"].transform(frame)
    try:
        import shap
        expl = shap.TreeExplainer(pipe.named_steps["model"])
        sv = expl.shap_values(Z)
        sv = np.asarray(sv[1] if isinstance(sv, list) else sv)[0]
        base = float(np.ravel(expl.expected_value)[-1])
        df = pd.DataFrame({"feature": names, "value": imputed[0], "was_missing": frame.iloc[0].isna().values, "contribution": sv})
        method = "SHAP TreeExplainer (log-odds of Abnormal)"
    except Exception as e:                                               # SHAP missing / incompatible -> transparent fallback
        summ = classifier.bundle.get("training_feature_summary")
        if not summ:
            return {"method": "unavailable", "table": None, "base_value": None,
                    "note": f"SHAP unavailable ({type(e).__name__}) and the bundle has no training summary for the fallback."}
        p0 = float(pipe.predict_proba(frame)[0, 1])
        deltas = []
        for n in names:
            alt = frame.copy(); alt[n] = summ[n]["p50"]
            deltas.append(p0 - float(pipe.predict_proba(alt)[0, 1]))
        df = pd.DataFrame({"feature": names, "value": imputed[0], "was_missing": frame.iloc[0].isna().values, "contribution": deltas})
        base, method = None, "one-at-a-time sensitivity: change in P(Abnormal) when the feature is set to the training median"
    df["group"] = df["feature"].map(feature_group)
    df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)
    return {"method": method, "table": df, "base_value": base, "note": DISCLAIMER}


def unusual_features(classifier, frame: pd.DataFrame, z: float = 3.0) -> pd.DataFrame:
    """Features outside the training 1st-99th percentile (input-quality hint; not an explanation)."""
    summ = classifier.bundle.get("training_feature_summary")
    if not summ:
        return pd.DataFrame(columns=["feature", "value", "p01", "p99"])
    rows = []
    for n in classifier.feature_names:
        v = frame[n].iloc[0]
        if pd.notna(v) and (v < summ[n]["p01"] or v > summ[n]["p99"]):
            rows.append({"feature": n, "value": float(v), "p01": summ[n]["p01"], "p99": summ[n]["p99"]})
    return pd.DataFrame(rows)
