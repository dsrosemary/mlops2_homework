from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier

from preprocessing import MODEL_FEATURES, transform_preprocessor

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


def load_model(path: Path) -> CatBoostClassifier:
    logger.info("Loading CatBoost model from %s", path)
    model = CatBoostClassifier(task_type="CPU")
    model.load_model(str(path))
    return model


def _normalise_fi_df(fi: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Feature": "feature",
        "Feature Name": "feature",
        "Feature Id": "feature_id",  
        "Importance": "importance",
        "Importances": "importance",
    }
    fi = fi.rename(columns=mapping)
    if "feature" not in fi.columns or "importance" not in fi.columns:
        logger.warning("Unexpected FI columns: %s", fi.columns)
        first, second = fi.columns[:2]
        fi = fi.rename(columns={first: "feature", second: "importance"})
    return fi[["feature", "importance"]]


def save_feature_importances(model: CatBoostClassifier, dst: Path, top_k: int = 5) -> None:
    logger.info("Saving top-%d feature importances to %s", top_k, dst)
    fi = model.get_feature_importance(prettified=True)
    if isinstance(fi, (list, tuple)):
        logger.warning("Model returned raw list; skipping FI save …")
        return
    fi = _normalise_fi_df(fi).head(top_k)
    fi_dict = dict(zip(fi["feature"], fi["importance"].astype(float)))
    with dst.open("w", encoding="utf-8") as f:
        json.dump(fi_dict, f, ensure_ascii=False, indent=2)


def save_density_plot(probas: List[float], dst: Path) -> None:
    logger.info("Saving density plot to %s", dst)
    pd.Series(probas).plot(kind="density")
    plt.xlabel("fraud score")
    plt.title("Density of predicted scores")
    plt.tight_layout()
    plt.savefig(dst)
    plt.close()


def main(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_csv = Path(args.output)
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading raw test data: %s", in_path)
    raw = pd.read_csv(in_path, parse_dates=["transaction_time"])

    logger.info("Restoring preprocessor helpers from %s", args.preproc)
    helpers = joblib.load(args.preproc)

    X = transform_preprocessor(raw, helpers)
    assert list(X.columns) == MODEL_FEATURES, "Unexpected feature set after preprocessing"

    model = load_model(Path(args.model))

    logger.info("Predicting probabilities …")
    probas = model.predict_proba(X)[:, 1]
    preds = (probas > args.threshold).astype(int)

    sub = pd.DataFrame({"index": raw.index, "prediction": preds})
    sub.to_csv(out_csv, index=False)
    logger.info("Submission saved to %s", out_csv)

    if args.save_imp:
        save_feature_importances(model, out_dir / f"{out_csv.stem}_imp.json")
    if args.save_density:
        save_density_plot(probas, out_dir / f"{out_csv.stem}_density.png")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run CatBoost inference on fraud data.")
    p.add_argument("--input", required=True, help="Path to raw test.csv")
    p.add_argument("--output", required=True, help="Where to save sample_submission.csv")
    p.add_argument("--model", default="./models/my_catboost.cbm", help="Path to .cbm model file")
    p.add_argument("--preproc", default="./models/preprocessor.pkl", help="Pickled helpers from fit_preprocessor")
    p.add_argument("--threshold", type=float, default=0.98, help="Decision threshold for binary prediction")
    p.add_argument("--save-imp", action="store_true", help="Save top-5 feature importances")
    p.add_argument("--save-density", action="store_true", help="Save density plot of probs")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())