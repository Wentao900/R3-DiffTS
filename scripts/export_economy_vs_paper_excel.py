import json
from pathlib import Path

import pandas as pd


RUN_DIR = Path("save/forecasting_Economy_20260312_194619")
PDF_NAME = "2504.19669v1.pdf"
OUTPUT_PATH = Path("economy_vs_2504_19669v1.xlsx")


def load_json(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text)
    return obj


def current_run_data():
    selected = load_json(RUN_DIR / "selected_guide_w.json")
    guide_token = str(selected["selected_guide_w"]).replace(".", "p")
    valid_metrics = load_json(RUN_DIR / f"valid_metrics_guide_{guide_token}.json")
    test_metrics = load_json(RUN_DIR / f"eval_metrics_guide_{guide_token}.json")
    config = load_json(RUN_DIR / "config_results.json")
    return selected, valid_metrics, test_metrics, config


def paper_reference_data():
    return {
        "paper_title": "Multimodal Conditioned Diffusive Time Series Forecasting",
        "paper_id": "arXiv:2504.19669v1",
        "economy_h12_test": {
            "MSE": 0.245,
            "MAE": 0.380,
            "source": "Table 6/7, Economy, pred_len=12",
        },
        "economy_avg_6_12_18_test": {
            "MSE": 0.249,
            "MAE": 0.374,
            "source": "Table 2, Economy",
        },
        "economy_h12_baselines": [
            ("Reformer", 0.792, 0.734),
            ("Autoformer", 0.300, 0.438),
            ("FEDformer", 0.303, 0.438),
            ("PatchTST", 0.282, 0.419),
            ("HCAN", 0.367, 0.462),
            ("FiLM", 0.386, 0.509),
            ("DLinear", 1.252, 0.922),
            ("TimeMixer++", 0.216, 0.366),
            ("Timemachine", 0.273, 0.412),
            ("CSDI", 1.590, 1.015),
            ("D3V AE", 0.628, 0.620),
            ("FPT", 0.312, 0.448),
            ("TimeLLM", 0.258, 0.389),
            ("MM-TSF", 0.965, 0.805),
            ("GLAFF", 0.490, 0.579),
            ("TimeLinear", 0.303, 0.440),
            ("MCD-TSF (paper)", 0.245, 0.380),
        ],
    }


def build_fair_compare_df(selected, valid_metrics, test_metrics, config, paper):
    rows = [
        {
            "item": "dataset",
            "paper_2504_19669v1": "Economy",
            "current_repo": config["model"]["domain"],
            "delta_repo_minus_paper": "",
        },
        {
            "item": "pred_len",
            "paper_2504_19669v1": 12,
            "current_repo": config["pred_len"],
            "delta_repo_minus_paper": config["pred_len"] - 12,
        },
        {
            "item": "seq_len",
            "paper_2504_19669v1": 36,
            "current_repo": config["seq_len"],
            "delta_repo_minus_paper": config["seq_len"] - 36,
        },
        {
            "item": "text_len",
            "paper_2504_19669v1": 36,
            "current_repo": config["text_len"],
            "delta_repo_minus_paper": config["text_len"] - 36,
        },
        {
            "item": "test_mse_h12",
            "paper_2504_19669v1": paper["economy_h12_test"]["MSE"],
            "current_repo": test_metrics["MSE"],
            "delta_repo_minus_paper": test_metrics["MSE"] - paper["economy_h12_test"]["MSE"],
        },
        {
            "item": "test_mae_h12",
            "paper_2504_19669v1": paper["economy_h12_test"]["MAE"],
            "current_repo": test_metrics["MAE"],
            "delta_repo_minus_paper": test_metrics["MAE"] - paper["economy_h12_test"]["MAE"],
        },
        {
            "item": "selected_guide_w",
            "paper_2504_19669v1": 0.8,
            "current_repo": selected["selected_guide_w"],
            "delta_repo_minus_paper": selected["selected_guide_w"] - 0.8,
        },
        {
            "item": "valid_mse_selected",
            "paper_2504_19669v1": "",
            "current_repo": valid_metrics["MSE"],
            "delta_repo_minus_paper": "",
        },
    ]
    return pd.DataFrame(rows)


def build_method_diff_df(config):
    rows = [
        ("paper core model", "Original MCD-TSF with TAA + TTF", "Economy V2 on top of MCD-TSF"),
        ("text encoder", "BERT-base", config["model"]["llm"]),
        ("text window", "Fixed 36 intervals", f"Dynamic {config['model']['dynamic_text_lens']}"),
        ("retrieval augmentation", "No RAG", f"use_rag_cot={config['model']['use_rag_cot']}"),
        ("two-stage retrieval", "No", config["model"]["use_two_stage_rag"]),
        ("scale-aware RAG", "No", config["model"]["scale_aware_rag"]),
        ("multi-resolution loss", "No", config["train"]["multi_res_loss_weight"]),
        ("scale router", "No", config["train"]["use_scale_router"]),
        ("router-aware guide", "No", config["diffusion"]["use_router_guide"]),
        ("guide strategy", "Fixed default w=0.8", f"valid-selected guide_w={config['model']['guide_w_candidates']} -> {1.4}"),
        ("sampling steps", "20 (paper setting)", config["diffusion"]["sample_steps"]),
        ("timestamp integration", "TAA", "TAA + extra router-aware logic in current repo"),
        ("text integration", "TTF + CFG", "TTF + RAG/CoT path + CFG"),
    ]
    return pd.DataFrame(rows, columns=["aspect", "paper_2504_19669v1", "current_repo"])


def build_paper_baseline_df(test_metrics, paper):
    rows = []
    for method, mse, mae in paper["economy_h12_baselines"]:
        rows.append(
            {
                "method": method,
                "paper_test_mse_economy_h12": mse,
                "paper_test_mae_economy_h12": mae,
            }
        )
    rows.append(
        {
            "method": "Current repo Economy V2",
            "paper_test_mse_economy_h12": test_metrics["MSE"],
            "paper_test_mae_economy_h12": test_metrics["MAE"],
        }
    )
    df = pd.DataFrame(rows)
    df["mse_rank"] = df["paper_test_mse_economy_h12"].rank(method="min")
    df["mae_rank"] = df["paper_test_mae_economy_h12"].rank(method="min")
    return df.sort_values(["mse_rank", "mae_rank", "method"]).reset_index(drop=True)


def build_notes_df(paper):
    rows = [
        ("source_pdf", PDF_NAME),
        ("paper_title", paper["paper_title"]),
        ("paper_id", paper["paper_id"]),
        ("fairness_note_1", "Main comparison uses Economy, pred_len=12, test metrics from Table 6/7."),
        ("fairness_note_2", "Paper Table 2 Economy values are averages over 6/12/18 and are not directly comparable to a single-horizon run."),
        ("fairness_note_3", "Current repo result comes from save/forecasting_Economy_20260312_194619 with selected guide_w=1.4."),
        ("fairness_note_4", "Current repo includes post-paper modifications such as dynamic text window, RAG/CoT, scale router, and multi-resolution loss."),
        ("paper_table2_economy_mse", paper["economy_avg_6_12_18_test"]["MSE"]),
        ("paper_table2_economy_mae", paper["economy_avg_6_12_18_test"]["MAE"]),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def autosize_sheet(writer, sheet_name, df):
    worksheet = writer.sheets[sheet_name]
    for idx, column in enumerate(df.columns):
        values = [str(column)] + [str(v) for v in df[column].tolist()]
        width = min(max(len(v) for v in values) + 2, 50)
        worksheet.set_column(idx, idx, width)


def main():
    selected, valid_metrics, test_metrics, config = current_run_data()
    paper = paper_reference_data()

    sheets = {
        "fair_compare_h12": build_fair_compare_df(selected, valid_metrics, test_metrics, config, paper),
        "method_diff": build_method_diff_df(config),
        "paper_baselines_h12": build_paper_baseline_df(test_metrics, paper),
        "notes": build_notes_df(paper),
    }

    with pd.ExcelWriter(OUTPUT_PATH, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            autosize_sheet(writer, sheet_name, df)


if __name__ == "__main__":
    main()
