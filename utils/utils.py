import numpy as np
import torch
from torch.optim import Adam, AdamW
from tqdm import tqdm
import os
import json


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=10,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=float(config["lr"]), weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=1.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def evaluate(
    model,
    test_loader,
    nsample=100,
    scaler=1,
    mean_scaler=0,
    foldername="",
    window_lens=[1, 1],
    guide_w=0,
    save_attn=False,
    save_token=False,
    save_trend_prior=False,
    model_folder=None,
    split="test",
    append_to_config_results=True,
):
    load_folder = model_folder or foldername
    if load_folder:
        model_path = os.path.join(load_folder, "model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=model.device))
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        nmse_total = 0
        nmae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        all_tt_attns = []
        all_tf_attns = []
        all_tokens = []
        all_trend_priors = []
        all_text_marks = []
        pred_len = int(window_lens[1]) if len(window_lens) > 1 else 0
        horizon_mse_total = np.zeros(pred_len, dtype=np.float64)
        horizon_mae_total = np.zeros(pred_len, dtype=np.float64)
        horizon_evalpoints_total = np.zeros(pred_len, dtype=np.float64)
        band_infos = model.get_multi_res_band_info() if hasattr(model, "get_multi_res_band_info") else []
        band_mse_total = {label: 0.0 for label, _ in band_infos}
        band_mae_total = {label: 0.0 for label, _ in band_infos}
        band_evalpoints_total = {label: 0.0 for label, _ in band_infos}
        router_weight_sum = None
        router_argmax_total = None
        router_entropy_sum = 0.0
        router_sample_total = 0
        router_target_hits = 0
        router_target_total = 0
        router_text_window_sum = 0.0
        router_text_window_count = 0
        guide_weight_sum = 0.0
        guide_weight_sq_sum = 0.0
        guide_weight_count = 0
        guide_ratio_sum = 0.0
        guide_ratio_count = 0
        scale_score_sum = 0.0
        scale_score_count = 0
        with tqdm(test_loader, mininterval=1.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample, guide_w)
                if hasattr(model, "get_scale_router_diagnostics"):
                    router_diag = model.get_scale_router_diagnostics(test_batch, guide_w=guide_w)
                else:
                    router_diag = None

                if save_trend_prior and isinstance(test_batch, dict) and "trend_prior" in test_batch:
                    all_trend_priors.append(test_batch["trend_prior"].detach().cpu().numpy())
                    if "text_mark" in test_batch:
                        all_text_marks.append(test_batch["text_mark"].detach().cpu().numpy())

                if save_attn:
                    if save_token:
                        samples, c_target, eval_points, observed_points, observed_time, attns, tokens = output
                    else:
                        samples, c_target, eval_points, observed_points, observed_time, attns = output
                else:
                    samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)
                if save_attn:
                    f = lambda x: x.detach().mean(dim=1).unsqueeze(1)
                    attns = [(f(attn1), f(attn2)) for attn1, attn2 in attns] 
                    tt_attns, tf_attns = zip(*attns)
                    tt_attns = torch.cat(tt_attns, 1)
                    tf_attns = torch.cat(tf_attns, 1)
                    tt_attns = tt_attns.chunk(2, dim=0)[0]
                    tf_attns = tf_attns.chunk(2, dim=0)[0]
                    all_tt_attns.append(tt_attns) 
                    all_tf_attns.append(tf_attns) 
                if save_token:
                    all_tokens.extend(tokens)
                if router_diag is not None:
                    weights = router_diag["weights"].double()
                    argmax = router_diag["argmax"].long()
                    if router_weight_sum is None:
                        router_weight_sum = np.zeros(weights.shape[1], dtype=np.float64)
                        router_argmax_total = np.zeros(weights.shape[1], dtype=np.float64)
                    router_weight_sum += weights.sum(dim=0).cpu().numpy()
                    router_argmax_total += np.bincount(argmax.cpu().numpy(), minlength=weights.shape[1]).astype(np.float64)
                    router_entropy_sum += router_diag["entropy"].double().sum().item()
                    router_sample_total += weights.shape[0]
                    if "target_index" in router_diag:
                        target_index = router_diag["target_index"].long()
                        router_target_hits += (argmax == target_index).sum().item()
                        router_target_total += target_index.numel()
                    if "text_window_len" in router_diag:
                        router_text_window_sum += router_diag["text_window_len"].double().sum().item()
                        router_text_window_count += router_diag["text_window_len"].numel()
                    if "sample_guide_w" in router_diag:
                        sample_guide = router_diag["sample_guide_w"].double()
                        guide_weight_sum += sample_guide.sum().item()
                        guide_weight_sq_sum += (sample_guide ** 2).sum().item()
                        guide_weight_count += sample_guide.numel()
                    if "guide_ratio" in router_diag:
                        guide_ratio = router_diag["guide_ratio"].double()
                        guide_ratio_sum += guide_ratio.sum().item()
                        guide_ratio_count += guide_ratio.numel()
                    if "scale_score" in router_diag:
                        scale_score = router_diag["scale_score"].double()
                        scale_score_sum += scale_score.sum().item()
                        scale_score_count += scale_score.numel()

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler
                nmse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                )
                nmae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                )

                if pred_len > 0:
                    pred_slice = slice(window_lens[0], window_lens[0] + pred_len)
                    pred_sq = nmse_current[:, pred_slice, :].sum(dim=2)
                    pred_abs = nmae_current[:, pred_slice, :].sum(dim=2)
                    pred_eval = eval_points[:, pred_slice, :].sum(dim=2)
                    horizon_mse_total += pred_sq.sum(dim=0).detach().cpu().numpy()
                    horizon_mae_total += pred_abs.sum(dim=0).detach().cpu().numpy()
                    horizon_evalpoints_total += pred_eval.sum(dim=0).detach().cpu().numpy()
                    for label, (start, end) in band_infos:
                        band_mse_total[label] += pred_sq[:, start:end].sum().item()
                        band_mae_total[label] += pred_abs[:, start:end].sum().item()
                        band_evalpoints_total[label] += pred_eval[:, start:end].sum().item()

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                nmse_total += nmse_current.sum().item()
                nmae_total += nmae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "nmse_total": nmse_total / evalpoints_total,
                        "nmae_total": nmae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            if not all_target:
                raise RuntimeError(
                    f"{split} loader produced no batches. "
                    "Check dataset size, batch_size, num_workers, and drop_last settings."
                )

            all_target = torch.cat(all_target, dim=0)
            all_evalpoint = torch.cat(all_evalpoint, dim=0)
            all_observed_point = torch.cat(all_observed_point, dim=0)
            all_observed_time = torch.cat(all_observed_time, dim=0)
            all_generated_samples = torch.cat(all_generated_samples, dim=0)
            # if save_attn:
            #     all_tt_attns = torch.cat(all_tt_attns, dim=0)
            #     all_tf_attns = torch.cat(all_tf_attns, dim=0)


            # np.save(foldername + "/generated_nsample" + str(nsample) + "_guide" + str(guide_w) + ".npy", all_generated_samples.cpu().numpy())
            # np.save(foldername + "/target_" + str(nsample) + "_guide" + str(guide_w) + ".npy", all_target.cpu().numpy())
            # if save_attn:
            #     np.save(foldername + "/all_tt_attns" + ".npy", all_tt_attns.cpu().numpy())
            #     np.save(foldername + "/all_tf_attns" + ".npy", all_tf_attns.cpu().numpy())
            # if save_token:
            #     np.save(foldername + "/tokens" + ".npy", np.asarray(all_tokens))
            if save_trend_prior and all_trend_priors and foldername:
                trend_prior_arr = np.concatenate(all_trend_priors, axis=0)
                np.save(os.path.join(foldername, "trend_priors.npy"), trend_prior_arr)
                if all_text_marks:
                    text_mark_arr = np.concatenate(all_text_marks, axis=0)
                    np.save(os.path.join(foldername, "trend_text_marks.npy"), text_mark_arr)

            horizon_metrics = {}
            for idx in range(pred_len):
                denom = horizon_evalpoints_total[idx]
                if denom <= 0:
                    continue
                horizon_metrics[f"h{idx + 1}"] = {
                    "MSE": float(horizon_mse_total[idx] / denom),
                    "MAE": float(horizon_mae_total[idx] / denom),
                }

            band_metrics = {}
            for label, _ in band_infos:
                denom = band_evalpoints_total.get(label, 0.0)
                if denom <= 0:
                    continue
                band_metrics[label] = {
                    "MSE": float(band_mse_total[label] / denom),
                    "MAE": float(band_mae_total[label] / denom),
                }

            router_metrics = {}
            if router_weight_sum is not None and router_sample_total > 0:
                router_labels = [label for label, _ in band_infos]
                if not router_labels:
                    router_labels = [f"band_{idx}" for idx in range(len(router_weight_sum))]
                router_metrics = {
                    "mean_weights": {
                        label: float(router_weight_sum[idx] / router_sample_total)
                        for idx, label in enumerate(router_labels)
                    },
                    "argmax_freq": {
                        label: float(router_argmax_total[idx] / router_sample_total)
                        for idx, label in enumerate(router_labels)
                    },
                    "mean_entropy": float(router_entropy_sum / router_sample_total),
                }
                if router_target_total > 0:
                    router_metrics["teacher_alignment"] = float(router_target_hits / router_target_total)
                if router_text_window_count > 0:
                    router_metrics["mean_text_window_len"] = float(router_text_window_sum / router_text_window_count)
                if guide_weight_count > 0:
                    mean_sample_guide = guide_weight_sum / guide_weight_count
                    guide_var = max(guide_weight_sq_sum / guide_weight_count - mean_sample_guide ** 2, 0.0)
                    router_metrics["mean_sample_guide_w"] = float(mean_sample_guide)
                    router_metrics["std_sample_guide_w"] = float(np.sqrt(guide_var))
                if guide_ratio_count > 0:
                    router_metrics["mean_guide_ratio"] = float(guide_ratio_sum / guide_ratio_count)
                if scale_score_count > 0:
                    router_metrics["mean_scale_score"] = float(scale_score_sum / scale_score_count)

            results = {
                "split": split,
                "guide_w": guide_w,
                "MSE": nmse_total / evalpoints_total,
                "MAE": nmae_total / evalpoints_total,
                "horizon_metrics": horizon_metrics,
                "band_metrics": band_metrics,
            }
            if router_metrics:
                results["router_metrics"] = router_metrics
            if append_to_config_results and foldername:
                with open(os.path.join(foldername, "config_results.json"), "a") as f:
                    json.dump(results, f, indent=4)
            guide_tag = str(guide_w).replace("-", "m").replace(".", "p")
            if foldername:
                metrics_prefix = "eval" if split == "test" else split
                with open(os.path.join(foldername, f"{metrics_prefix}_metrics_guide_{guide_tag}.json"), "w") as f:
                    json.dump(results, f, indent=4)
            print("MSE:", nmse_total / evalpoints_total)
            print("MAE:", nmae_total / evalpoints_total)
            if horizon_metrics:
                print("Horizon metrics:", json.dumps(horizon_metrics, ensure_ascii=True))
            if band_metrics:
                print("Band metrics:", json.dumps(band_metrics, ensure_ascii=True))
            if router_metrics:
                print("Router metrics:", json.dumps(router_metrics, ensure_ascii=True))
    return nmse_total / evalpoints_total
