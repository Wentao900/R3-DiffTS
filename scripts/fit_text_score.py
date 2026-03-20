import argparse
import csv
import json
from pathlib import Path

import numpy as np


FEATURE_COLUMNS = [
    "text_mark",
    "text_window_len",
    "scale_code",
    "scale_pref",
    "signed_slope",
    "abs_slope",
    "history_std",
    "history_mean_abs",
    "history_total_shift",
    "history_accel",
    "history_smoothness",
    "history_trend_score",
    "history_volatility_score",
    "history_last_value",
    "raw_text_len",
    "full_text_len",
    "retrieved_text_len",
    "cot_text_len",
    "extra_text_len",
    "retrieval_to_raw_len_ratio",
    "cot_to_full_len_ratio",
    "raw_retrieved_overlap",
    "full_retrieved_overlap",
    "raw_retrieved_jaccard",
    "full_retrieved_jaccard",
    "retrieved_unique_ratio",
    "full_unique_ratio",
    "cot_unique_ratio",
    "trend_direction",
    "trend_strength",
    "trend_volatility",
    "guide_w",
]
TARGET_MAP = {
    "full_text": "delta_mse_full_text",
    "raw_only": "delta_mse_raw_only",
}


def _load_rows(path):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["source_csv"] = str(path)
    return rows




def _to_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _build_matrix(rows, feature_columns):
    data = []
    for row in rows:
        values = [_to_float(row, key, 0.0) for key in feature_columns]
        data.append(values)
    X = np.asarray(data, dtype=np.float64)
    if X.size == 0:
        return X, np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.float64), np.ones((0,), dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xn = (X - mean) / std
    return X, Xn, mean, std


def _fit_ridge(Xn, y, reg=1.0):
    n = Xn.shape[0]
    design = np.concatenate([np.ones((n, 1), dtype=np.float64), Xn], axis=1)
    eye = np.eye(design.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + reg * eye, design.T @ y)
    pred = design @ weights
    return weights, pred


def _pearson(x, y):
    if len(x) <= 1:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rank_pct(values):
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    if len(values) == 1:
        ranks[order[0]] = 1.0
        return ranks
    ranks[order] = np.linspace(0.0, 1.0, num=len(values), endpoint=True)
    return ranks


def _summarize_subset(rows, pred, target_column):
    y = np.asarray([_to_float(row, target_column, 0.0) for row in rows], dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    y_sign = y > 0.0
    pred_sign = pred > 0.0
    return {
        "count": int(len(rows)),
        "pearson": _pearson(pred, y),
        "sign_accuracy": float(np.mean(pred_sign == y_sign)) if len(y) > 0 else 0.0,
        "positive_gain_rate": float(np.mean(y_sign)) if len(y) > 0 else 0.0,
        "predicted_positive_rate": float(np.mean(pred_sign)) if len(y) > 0 else 0.0,
        "mean_target": float(np.mean(y)) if len(y) > 0 else 0.0,
        "mean_prediction": float(np.mean(pred)) if len(y) > 0 else 0.0,
    }


def _evaluate_target(rows, feature_columns, target_column, reg):
    _, Xn, feature_mean, feature_std = _build_matrix(rows, feature_columns)
    y = np.asarray([_to_float(row, target_column, 0.0) for row in rows], dtype=np.float64)
    weights, pred = _fit_ridge(Xn, y, reg=reg)
    pred_rank = _rank_pct(pred)
    y_sign = y > 0.0
    pred_sign = pred > 0.0
    metrics = {
        "target": target_column,
        "reg": reg,
        "pearson": _pearson(pred, y),
        "sign_accuracy": float(np.mean(pred_sign == y_sign)) if len(y) > 0 else 0.0,
        "positive_gain_rate": float(np.mean(y_sign)) if len(y) > 0 else 0.0,
        "predicted_positive_rate": float(np.mean(pred_sign)) if len(y) > 0 else 0.0,
        "mean_target": float(np.mean(y)) if len(y) > 0 else 0.0,
        "std_target": float(np.std(y)) if len(y) > 0 else 0.0,
        "mean_prediction": float(np.mean(pred)) if len(y) > 0 else 0.0,
        "feature_mean": {feature_columns[idx]: float(feature_mean[idx]) for idx in range(len(feature_columns))},
        "feature_std": {feature_columns[idx]: float(feature_std[idx]) for idx in range(len(feature_columns))},
        "weights": {
            "intercept": float(weights[0]),
            **{feature_columns[idx]: float(weights[idx + 1]) for idx in range(len(feature_columns))},
        },
    }
    return pred, pred_rank, metrics


def main():
    parser = argparse.ArgumentParser(description="Fit an offline text utility score from counterfactual CSV outputs.")
    parser.add_argument("--input_csv", default=None, help="Path to one eval_counterfactual_samples_*.csv")
    parser.add_argument("--input_csvs", nargs='*', default=None, help="Multiple eval_counterfactual_samples_*.csv files to fit jointly")
    parser.add_argument("--target", choices=sorted(TARGET_MAP.keys()), default="full_text", help="Which counterfactual target to fit")
    parser.add_argument("--reg", type=float, default=1.0, help="Ridge regularization strength")
    parser.add_argument("--output_json", default=None, help="Optional JSON path for summary metrics")
    parser.add_argument("--output_csv", default=None, help="Optional CSV path for rows with fitted score")
    args = parser.parse_args()

    input_paths = []
    if args.input_csv:
        input_paths.append(args.input_csv)
    if args.input_csvs:
        input_paths.extend(args.input_csvs)
    dedup_paths = []
    seen = set()
    for item in input_paths:
        if item and item not in seen:
            dedup_paths.append(item)
            seen.add(item)
    if not dedup_paths:
        raise RuntimeError("Please provide --input_csv or --input_csvs.")

    rows = []
    for item in dedup_paths:
        rows.extend(_load_rows(item))
    if not rows:
        raise RuntimeError(f"No rows found in input files: {dedup_paths}")

    target_column = TARGET_MAP[args.target]
    pred, pred_rank, metrics = _evaluate_target(rows, FEATURE_COLUMNS, target_column, args.reg)

    enriched_rows = []
    for idx, row in enumerate(rows):
        enriched = dict(row)
        enriched[f"pred_{target_column}"] = float(pred[idx])
        enriched[f"text_score_{args.target}"] = float(pred_rank[idx])
        enriched_rows.append(enriched)

    output_json = args.output_json
    if output_json is None:
        if len(dedup_paths) == 1:
            base_path = Path(dedup_paths[0])
            output_json = str(base_path.with_name(base_path.stem + f"_{args.target}_score.json"))
        else:
            output_json = str(Path("combined_text_score_" + args.target + ".json"))
    output_csv = args.output_csv
    if output_csv is None:
        if len(dedup_paths) == 1:
            base_path = Path(dedup_paths[0])
            output_csv = str(base_path.with_name(base_path.stem + f"_{args.target}_score.csv"))
        else:
            output_csv = str(Path("combined_text_score_" + args.target + ".csv"))

    by_source = {}
    source_to_indices = {}
    guide_to_indices = {}
    for idx, row in enumerate(rows):
        source_to_indices.setdefault(row["source_csv"], []).append(idx)
        guide_key = str(row.get("guide_w", "NA"))
        guide_to_indices.setdefault(guide_key, []).append(idx)

    for source_csv, indices in source_to_indices.items():
        by_source[source_csv] = _summarize_subset([rows[i] for i in indices], pred[indices], target_column)

    by_guide_w = {}
    for guide_key, indices in guide_to_indices.items():
        by_guide_w[guide_key] = _summarize_subset([rows[i] for i in indices], pred[indices], target_column)

    summary = {
        "input_csvs": dedup_paths,
        "target": args.target,
        "target_column": target_column,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "by_source_csv": by_source,
        "by_guide_w": by_guide_w,
        "output_csv": output_csv,
    }

    with Path(output_json).open("w") as f:
        json.dump(summary, f, indent=4)

    fieldnames = list(enriched_rows[0].keys())
    with Path(output_csv).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_rows)

    print(json.dumps(summary, ensure_ascii=True, indent=4))


if __name__ == "__main__":
    main()
