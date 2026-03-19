# Self-Conditioning Experiment Analysis on Economy (36 -> 12)

## 1. Background and Motivation

The recent diffusion-side exploration focused on finding a **minimal but effective** modification to the current MCD-TSF forecasting pipeline.

Earlier attempts showed that several common diffusion tricks did **not** transfer well to the current setup:

- **Min-SNR weighting** led to clear degradation.
- **Cosine noise schedule** caused mild to moderate degradation.
- **Cosine + Min-SNR** degraded even more.
- Adjusting the DDIM sampling stride from `quad` to `linear` only provided limited recovery and did not solve the main issue.

These results suggested that the current model is sensitive to modifications that alter:

- the training target weighting,
- the diffusion noise schedule,
- or the global denoising dynamics.

Based on that observation, the next candidate was chosen to be a **low-intrusion structural enhancement**:

- keep the original `quad` diffusion schedule,
- keep the original loss design,
- keep CFG / router-guide / RAG / trend guidance unchanged,
- and only add **x0 self-conditioning**.

The intuition was that self-conditioning might improve denoising consistency without changing the optimization target itself.

## 2. What Was Changed

A new self-conditioning variant was implemented on top of the original baseline diffusion path.

### 2.1 Core idea

At training time:

- the model first performs a no-grad preview prediction of `x0`,
- then uses that detached prediction as an extra input channel in the actual denoising forward pass.

At sampling time:

- each reverse diffusion step feeds the previous `x0` prediction back into the next step.

### 2.2 Configuration

The self-conditioning run used:

- `self_condition: true`
- `self_condition_prob: 0.5`
- `self_condition_target_only: true`

Run file:

- `save/forecasting_Economy_20260319_122736/config_results.json`

Relevant config lines:

- `save/forecasting_Economy_20260319_122736/config_results.json:61`
- `save/forecasting_Economy_20260319_122736/config_results.json:62`
- `save/forecasting_Economy_20260319_122736/config_results.json:63`

The comparison baseline is the original Economy `36 -> 12` run:

- `save/forecasting_Economy_20260312_194619/config_results.json`

## 3. Compared Runs

### 3.1 Baseline

- Run: `save/forecasting_Economy_20260312_194619`
- Diffusion schedule: `quad`
- Sampling method: `quad`
- No self-conditioning

### 3.2 Self-conditioning

- Run: `save/forecasting_Economy_20260319_122736`
- Diffusion schedule: `quad`
- Sampling method: `quad`
- `self_condition=true`

This keeps the comparison clean: the only meaningful diffusion-side change is the addition of self-conditioning.

## 4. Validation Results

### 4.1 Best validation point

Baseline:

- selected guide weight: `1.4`
- best valid MSE: `0.14063129197983515`

Source:

- `save/forecasting_Economy_20260312_194619/selected_guide_w.json:3`
- `save/forecasting_Economy_20260312_194619/selected_guide_w.json:4`

Self-conditioning:

- selected guide weight: `0.5`
- best valid MSE: `0.12176733925229027`

Source:

- `save/forecasting_Economy_20260319_122736/selected_guide_w.json:3`
- `save/forecasting_Economy_20260319_122736/selected_guide_w.json:4`

### 4.2 Quantitative improvement

Validation MSE improved from:

- `0.140631` -> `0.121767`

Absolute improvement:

- `-0.018864`

Relative improvement:

- about **13.4%** better than the baseline

### 4.3 Guide-weight sweep behavior

A particularly important observation is that self-conditioning improves the entire validation sweep range.

Baseline valid scores:

- `0.4 -> 0.193345`
- `0.5 -> 0.182690`
- `0.6 -> 0.173504`
- `0.7 -> 0.165634`
- `0.8 -> 0.159071`
- `0.9 -> 0.153699`
- `1.0 -> 0.149342`
- `1.2 -> 0.143331`
- `1.4 -> 0.140631`

Self-conditioning valid scores:

- `0.4 -> 0.122042`
- `0.5 -> 0.121767`
- `0.6 -> 0.122505`
- `0.7 -> 0.124245`
- `0.8 -> 0.126940`
- `0.9 -> 0.130513`
- `1.0 -> 0.134903`
- `1.2 -> 0.146010`
- `1.4 -> 0.160056`

Sources:

- `save/forecasting_Economy_20260312_194619/selected_guide_w.json:16`
- `save/forecasting_Economy_20260319_122736/selected_guide_w.json:16`

This indicates two things:

1. Self-conditioning clearly improves validation-time denoising quality.
2. The optimal global guidance strength shifts significantly downward, from `1.4` to `0.5`.

That means the new model becomes much more sensitive to conditioning, and strong guidance is no longer necessary.

## 5. Test Results

### 5.1 Overall metrics

Baseline test metrics:

- MSE: `0.2496992787744245`
- MAE: `0.3801771555191431`

Source:

- `save/forecasting_Economy_20260312_194619/eval_metrics_guide_1p4.json:4`
- `save/forecasting_Economy_20260312_194619/eval_metrics_guide_1p4.json:5`

Self-conditioning test metrics:

- MSE: `0.26061055599114835`
- MAE: `0.3976185138408954`

Source:

- `save/forecasting_Economy_20260319_122736/eval_metrics_guide_0p5.json:4`
- `save/forecasting_Economy_20260319_122736/eval_metrics_guide_0p5.json:5`

### 5.2 Quantitative difference

Test MSE changed from:

- `0.249699` -> `0.260611`

Absolute change:

- `+0.010911`

Relative change:

- about **4.37% worse** than baseline

Test MAE changed from:

- `0.380177` -> `0.397619`

Absolute change:

- `+0.017441`

Relative change:

- about **4.59% worse** than baseline

Therefore, the self-conditioning variant is **not** better on the final aggregate test metric, even though it is clearly better on validation.

## 6. Horizon-Wise Analysis

The most important insight comes from the horizon breakdown.

### 6.1 Horizon-level changes

Compared with baseline, self-conditioning gives:

- `h1`: improved (`-0.002416`)
- `h2`: slightly worse (`+0.001170`)
- `h3`: improved (`-0.009968`)
- `h4`: improved (`-0.005683`)
- `h5`: improved (`-0.008094`)
- `h6`: nearly unchanged (`+0.000797`)
- `h7`: worse (`+0.015558`)
- `h8`: worse (`+0.018770`)
- `h9`: slightly worse (`+0.003110`)
- `h10`: worse (`+0.016360`)
- `h11`: worse (`+0.035203`)
- `h12`: clearly worse (`+0.066128`)

### 6.2 Band-level view

Baseline band MSE:

- `1`: `0.119713`
- `2-3`: `0.146453`
- `4-6`: `0.206207`
- `7-12`: `0.327525`

Self-conditioning band MSE:

- `1`: `0.117297`
- `2-3`: `0.142054`
- `4-6`: `0.201880`
- `7-12`: `0.353380`

Sources:

- `save/forecasting_Economy_20260312_194619/eval_metrics_guide_1p4.json:56`
- `save/forecasting_Economy_20260319_122736/eval_metrics_guide_0p5.json:56`

### 6.3 Interpretation

The pattern is very consistent:

- **short-term horizon improves**,
- **mid-term horizon slightly improves**,
- **longer horizon (7-12) degrades**.

So the effect of self-conditioning is not a uniform gain or loss.
Instead, it creates a **horizon trade-off**:

- better local denoising and near-term consistency,
- but weaker long-range extrapolation.

This explains why validation improves strongly while test aggregate does not.

## 7. Router / Guidance Observations

Router-related diagnostics are also informative.

Baseline:

- mean sample guide weight: `1.147880`
- mean guide ratio: `0.819914`
- mean scale score: `0.199857`

Self-conditioning:

- mean sample guide weight: `0.413517`
- mean guide ratio: `0.827034`
- mean scale score: `0.211723`

Sources:

- `save/forecasting_Economy_20260312_194619/eval_metrics_guide_1p4.json:74`
- `save/forecasting_Economy_20260319_122736/eval_metrics_guide_0p5.json:74`

This suggests:

- the router behavior itself is broadly similar,
- but the globally optimal `guide_w` becomes much smaller under self-conditioning,
- meaning the denoiser is now more responsive to conditioning and requires less external push.

In other words, self-conditioning does not fundamentally break the guidance mechanism; instead, it shifts the operating regime of the model.

## 8. Main Takeaway

The self-conditioning experiment should be regarded as a **partially successful** diffusion-side modification.

### 8.1 What improved

- Validation MSE improved significantly.
- Short- and mid-horizon forecasting became better.
- The model needed much smaller global guidance weight.

### 8.2 What did not improve

- Final test-set aggregate MSE did not surpass the baseline.
- The degradation is concentrated in the `7-12` horizon range.

### 8.3 Practical interpretation

This means self-conditioning is useful as a **short-to-mid horizon stabilizer**, but not yet a full replacement for the baseline if overall test performance is the final objective.

A concise summary is:

> Self-conditioning substantially improves validation performance and short/mid-horizon accuracy, but its gains do not fully transfer to the final test metric because the far horizon (`7-12`) becomes worse.

## 9. Recommended Next Step

Given the above findings, the most promising next direction is **not** to abandon self-conditioning immediately, but to preserve its short-horizon gains while compensating for its long-horizon weakness.

The most reasonable next-step hypotheses are:

1. keep self-conditioning, but add a **horizon-aware or step-aware guidance schedule**;
2. keep self-conditioning only in training, but weaken it during sampling;
3. keep self-conditioning and specifically add a mechanism that boosts the `7-12` horizon region.

At the current stage, the evidence suggests that self-conditioning is a meaningful and non-trivial modification, but it should be treated as a **trade-off method** rather than a universally superior replacement.

## 10. One-Paragraph Report Version

We further evaluated an `x0 self-conditioning` variant on the Economy `36 -> 12` setting while keeping the original `quad` diffusion schedule unchanged. Compared with the original baseline (`save/forecasting_Economy_20260312_194619`), the self-conditioning model (`save/forecasting_Economy_20260319_122736`) reduced the best validation MSE from `0.140631` to `0.121767`, an improvement of about `13.4%`, and shifted the optimal guidance weight from `1.4` to `0.5`. However, the final test MSE slightly increased from `0.249699` to `0.260611`. A horizon-wise analysis shows that self-conditioning improves short- and mid-range forecasting (`1-6`) but degrades the longer horizon (`7-12`), indicating that it enhances local denoising consistency while weakening long-range extrapolation. Therefore, self-conditioning is a useful diffusion-side enhancement for validation stability and near-term prediction, but it is not yet a direct replacement for the baseline when overall test performance is the primary objective.
