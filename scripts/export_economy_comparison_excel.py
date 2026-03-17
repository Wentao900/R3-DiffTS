import json
from pathlib import Path

import pandas as pd


RUN_DIR = Path("save/forecasting_Economy_20260312_194619")
OUTPUT_PATH = Path("economy_experiment_comparison.xlsx")


def load_json(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text)
    return obj


def build_summary_df(selected, valid_metrics, test_metrics, config):
    rows = [
        ("scope", "Economy only"),
        ("run_dir", str(RUN_DIR)),
        ("selection_split", selected["selection_split"]),
        ("selected_guide_w", selected["selected_guide_w"]),
        ("best_valid_mse", selected["best_valid_mse"]),
        ("valid_mse_at_selected", valid_metrics["MSE"]),
        ("valid_mae_at_selected", valid_metrics["MAE"]),
        ("test_mse_at_selected", test_metrics["MSE"]),
        ("test_mae_at_selected", test_metrics["MAE"]),
        ("seq_len", config["seq_len"]),
        ("pred_len", config["pred_len"]),
        ("text_len", config["text_len"]),
        ("model.domain", config["model"]["domain"]),
        ("model.use_rag_cot", config["model"]["use_rag_cot"]),
        ("model.use_two_stage_rag", config["model"]["use_two_stage_rag"]),
        ("model.scale_aware_rag", config["model"]["scale_aware_rag"]),
        ("train.multi_res_loss_weight", config["train"]["multi_res_loss_weight"]),
        ("train.use_scale_router", config["train"]["use_scale_router"]),
        ("diffusion.use_router_guide", config["diffusion"]["use_router_guide"]),
        ("diffusion.sample_steps", config["diffusion"]["sample_steps"]),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def build_guide_sweep_df(selected):
    best = selected["best_valid_mse"]
    rows = []
    for guide_w in selected["candidates"]:
        key = str(guide_w)
        mse = selected["valid_scores"][key]
        rows.append(
            {
                "guide_w": guide_w,
                "valid_mse": mse,
                "delta_vs_best": mse - best,
            }
        )
    df = pd.DataFrame(rows).sort_values("valid_mse", ascending=True).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def build_horizon_df(valid_metrics, test_metrics):
    rows = []
    for idx in range(1, 13):
        key = f"h{idx}"
        v = valid_metrics["horizon_metrics"][key]
        t = test_metrics["horizon_metrics"][key]
        rows.append(
            {
                "horizon": key,
                "valid_mse": v["MSE"],
                "valid_mae": v["MAE"],
                "test_mse": t["MSE"],
                "test_mae": t["MAE"],
                "test_minus_valid_mse": t["MSE"] - v["MSE"],
            }
        )
    return pd.DataFrame(rows)


def build_band_df(valid_metrics, test_metrics):
    order = ["1", "2-3", "4-6", "7-12"]
    rows = []
    for key in order:
        v = valid_metrics["band_metrics"][key]
        t = test_metrics["band_metrics"][key]
        rows.append(
            {
                "band": key,
                "valid_mse": v["MSE"],
                "valid_mae": v["MAE"],
                "test_mse": t["MSE"],
                "test_mae": t["MAE"],
                "test_minus_valid_mse": t["MSE"] - v["MSE"],
            }
        )
    return pd.DataFrame(rows)


def build_router_df(valid_metrics, test_metrics):
    rows = []

    for key in ["1", "2-3", "4-6", "7-12"]:
        rows.append(
            {
                "metric_group": "mean_weights",
                "metric_name": key,
                "valid_value": valid_metrics["router_metrics"]["mean_weights"][key],
                "test_value": test_metrics["router_metrics"]["mean_weights"][key],
                "test_minus_valid": test_metrics["router_metrics"]["mean_weights"][key]
                - valid_metrics["router_metrics"]["mean_weights"][key],
            }
        )

    for key in ["1", "2-3", "4-6", "7-12"]:
        rows.append(
            {
                "metric_group": "argmax_freq",
                "metric_name": key,
                "valid_value": valid_metrics["router_metrics"]["argmax_freq"][key],
                "test_value": test_metrics["router_metrics"]["argmax_freq"][key],
                "test_minus_valid": test_metrics["router_metrics"]["argmax_freq"][key]
                - valid_metrics["router_metrics"]["argmax_freq"][key],
            }
        )

    scalar_keys = [
        "mean_entropy",
        "teacher_alignment",
        "mean_text_window_len",
        "mean_sample_guide_w",
        "std_sample_guide_w",
        "mean_guide_ratio",
        "mean_scale_score",
    ]
    for key in scalar_keys:
        rows.append(
            {
                "metric_group": "scalar",
                "metric_name": key,
                "valid_value": valid_metrics["router_metrics"][key],
                "test_value": test_metrics["router_metrics"][key],
                "test_minus_valid": test_metrics["router_metrics"][key]
                - valid_metrics["router_metrics"][key],
            }
        )

    return pd.DataFrame(rows)


def autosize_sheet(writer, sheet_name, df):
    worksheet = writer.sheets[sheet_name]
    for idx, column in enumerate(df.columns):
        values = [str(column)] + [str(v) for v in df[column].tolist()]
        width = min(max(len(v) for v in values) + 2, 40)
        worksheet.set_column(idx, idx, width)


def main():
    selected = load_json(RUN_DIR / "selected_guide_w.json")
    guide_token = str(selected["selected_guide_w"]).replace(".", "p")
    valid_metrics = load_json(RUN_DIR / f"valid_metrics_guide_{guide_token}.json")
    test_metrics = load_json(RUN_DIR / f"eval_metrics_guide_{guide_token}.json")
    config = load_json(RUN_DIR / "config_results.json")

    sheets = {
        "summary": build_summary_df(selected, valid_metrics, test_metrics, config),
        "guide_sweep": build_guide_sweep_df(selected),
        "horizon_compare": build_horizon_df(valid_metrics, test_metrics),
        "band_compare": build_band_df(valid_metrics, test_metrics),
        "router_compare": build_router_df(valid_metrics, test_metrics),
    }

    with pd.ExcelWriter(OUTPUT_PATH, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            autosize_sheet(writer, sheet_name, df)


if __name__ == "__main__":
    main()
