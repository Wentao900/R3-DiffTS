# Text Score Gate Workflow

本文档整理当前仓库中保留的 `text_score` 主线流程，只保留到在线 `text_score gate` 为止。

当前流程分为 3 步：

1. 导出 `counterfactual` 样本，获得文本效用标签
2. 用导出的样本拟合离线 `text_score`
3. 在推理/评测时启用在线 `text_score gate`

## 1. 当前保留的核心代码

- `data_provider/data_loader.py`
  - 为每个样本导出文本、趋势先验、连续历史统计特征
- `utils/utils.py`
  - `counterfactual_eval` 导出 sample-level 收益标签和文本交互特征
- `scripts/fit_text_score.py`
  - 用 `counterfactual` CSV 拟合离线 `text_score`
- `main_model.py`
  - 加载离线 `text_score` 模型，并在前向/采样中做在线 gate
- `exe_forecasting.py`
  - 提供 `text_score gate` 的 CLI 开关

## 2. Step 1: 导出 counterfactual 样本

目标：为每个样本计算文本是否带来收益，形成 `delta_mse_*` 标签。

### 2.1 推荐命令

```bash
python -u exe_forecasting.py \
  --root_path ../Time-MMD-main \
  --data_path Economy/Economy.csv \
  --config economy_36_12_scale_router_guide.yaml \
  --seq_len 36 \
  --pred_len 12 \
  --text_len 36 \
  --freq m \
  --guide_w 1.4 \
  --counterfactual_eval
```

如需提升 `text_score` 训练样本的区分度，建议加入：

```bash
--max_text_tokens 512 --dynamic_text_len --dynamic_text_lens 6 18 36
```

完整示例：

```bash
python -u exe_forecasting.py \
  --root_path ../Time-MMD-main \
  --data_path Economy/Economy.csv \
  --config economy_36_12_scale_router_guide.yaml \
  --seq_len 36 \
  --pred_len 12 \
  --text_len 36 \
  --freq m \
  --guide_w 1.4 \
  --counterfactual_eval \
  --max_text_tokens 512 \
  --dynamic_text_len \
  --dynamic_text_lens 6 18 36
```

### 2.2 输出文件

运行后会在对应 `save/forecasting_*` 目录下生成：

- `eval_counterfactual_guide_*.json`
- `eval_counterfactual_samples_guide_*.csv`

其中 `csv` 是训练 `text_score` 的核心输入。

### 2.3 当前导出的关键字段

#### 标签字段

- `delta_mse_raw_only`
- `delta_mse_full_text`
- `delta_mae_raw_only`
- `delta_mae_full_text`

#### 基础文本字段

- `text_mark`
- `text_window_len`
- `scale_code`
- `raw_text_len`
- `full_text_len`
- `retrieved_text_len`
- `cot_text_len`

#### 趋势/历史字段

- `scale_pref`
- `signed_slope`
- `abs_slope`
- `history_std`
- `history_mean_abs`
- `history_total_shift`
- `history_accel`
- `history_smoothness`
- `history_trend_score`
- `history_volatility_score`
- `history_last_value`
- `trend_direction`
- `trend_strength`
- `trend_volatility`
- `guide_w`

#### 文本交互字段

- `extra_text_len`
- `retrieval_to_raw_len_ratio`
- `cot_to_full_len_ratio`
- `raw_retrieved_overlap`
- `full_retrieved_overlap`
- `raw_retrieved_jaccard`
- `full_retrieved_jaccard`
- `retrieved_unique_ratio`
- `full_unique_ratio`
- `cot_unique_ratio`

## 3. Step 2: 拟合离线 text_score

目标：根据 `counterfactual` 样本中的特征，拟合一个样本级文本价值分数。

### 3.1 单文件拟合

```bash
python scripts/fit_text_score.py \
  --input_csv save/<run_dir>/eval_counterfactual_samples_guide_1p4.csv \
  --target full_text
```

### 3.2 多文件联合拟合

推荐将多个 `guide_w` 的 CSV 一起拟合：

```bash
python scripts/fit_text_score.py \
  --input_csvs \
  save/<run_dir_1>/eval_counterfactual_samples_guide_1p0.csv \
  save/<run_dir_2>/eval_counterfactual_samples_guide_1p4.csv \
  --target full_text \
  --output_json save/combined_full_text_score.json \
  --output_csv save/combined_full_text_score.csv
```

也可以拟合 `raw_only` 目标：

```bash
python scripts/fit_text_score.py \
  --input_csvs <csv1> <csv2> \
  --target raw_only
```

### 3.3 输出文件

- `*_score.json`
- `*_score.csv`

其中 `json` 包含：

- 拟合指标：`pearson`、`sign_accuracy`
- 线性权重：`weights`
- 在线 gate 需要的归一化统计：
  - `feature_mean`
  - `feature_std`
  - `mean_target`
  - `std_target`

在线 gate 实际加载的是这个 `json`。

## 4. Step 3: 启用在线 text_score gate

目标：在模型推理时，根据离线拟合的 `text_score` 对文本条件进行样本级门控。

### 4.1 推荐命令

```bash
python -u exe_forecasting.py \
  --root_path ../Time-MMD-main \
  --data_path Economy/Economy.csv \
  --config economy_36_12_scale_router_guide.yaml \
  --seq_len 36 \
  --pred_len 12 \
  --text_len 36 \
  --freq m \
  --guide_w 1.4 \
  --use_text_score_gate \
  --text_score_model_path save/combined_full_text_score.json \
  --text_score_gate_strength 0.5 \
  --text_score_gate_floor 0.1
```

### 4.2 参数说明

- `--use_text_score_gate`
  - 开启在线 gate
- `--text_score_model_path`
  - 指向 `scripts/fit_text_score.py` 生成的 `json`
- `--text_score_gate_strength`
  - gate 强度，范围建议 `0.0 ~ 1.0`
  - 越大越相信 `text_score`
- `--text_score_gate_floor`
  - 低分样本保留的最小文本通量
  - 避免 gate 过于激进地把文本完全压掉

### 4.3 当前建议的起始值

第一轮建议：

- `text_score_gate_strength = 0.5`
- `text_score_gate_floor = 0.1`

如果离线评分器质量还不稳定，不建议一开始就使用：

- `strength = 1.0`
- `floor = 0.0`

因为这样门控会太激进。

## 5. 当前方法的工作原理

### 5.1 离线标签来源

`counterfactual_eval` 会比较：

- `text_off`
- `raw_only`
- `full_text`

并定义：

- `delta_mse_full_text = mse(text_off) - mse(full_text)`
- `delta_mse_raw_only = mse(text_off) - mse(raw_only)`

因此：

- `delta > 0` 表示文本有帮助
- `delta < 0` 表示文本在误导预测

### 5.2 在线 gate 作用位置

当前在线 gate 会影响：

- 文本编码后的 `context`
- `trend guidance` 中与文本相关的权重
- 下游依赖 `text_mask` 的 router / multi-res 分支

也就是说，现在的 `text_score` 不是只做分析，而是会实际影响模型使用文本的强度。

## 6. 推荐实验顺序

### 路线 A：先验证 score 质量

1. 跑多个 `guide_w` 的 `counterfactual_eval`
2. 联合拟合 `text_score`
3. 观察 `pearson / sign_accuracy`
4. 若指标有提升，再尝试在线 gate

### 路线 B：直接验证在线 gate

1. 准备一个离线 `text_score json`
2. 在固定 `guide_w` 下分别跑：
   - baseline
   - `text_score gate`
3. 比较 `eval_metrics_guide_*.json`

## 7. 当前已知限制

当前 `text_score` 主线已经可运行，但仍有几个已知限制：

- 离线 score 仍然主要依赖导出的统计特征，表达能力有限
- 若 `text_window_len`、`scale_code` 在数据上不分化，score 的辨识度会受限
- 在线 gate 的效果高度依赖离线 score 质量
- 若离线 `pearson / sign_accuracy` 很低，在线 gate 很可能无收益甚至退化

## 8. 最简命令清单

### 导出 counterfactual 样本

```bash
python -u exe_forecasting.py \
  --root_path ../Time-MMD-main \
  --data_path Economy/Economy.csv \
  --config economy_36_12_scale_router_guide.yaml \
  --seq_len 36 --pred_len 12 --text_len 36 --freq m \
  --guide_w 1.4 \
  --counterfactual_eval
```

### 拟合 text_score

```bash
python scripts/fit_text_score.py \
  --input_csv save/<run_dir>/eval_counterfactual_samples_guide_1p4.csv \
  --target full_text
```

### 启用在线 gate

```bash
python -u exe_forecasting.py \
  --root_path ../Time-MMD-main \
  --data_path Economy/Economy.csv \
  --config economy_36_12_scale_router_guide.yaml \
  --seq_len 36 --pred_len 12 --text_len 36 --freq m \
  --guide_w 1.4 \
  --use_text_score_gate \
  --text_score_model_path save/combined_full_text_score.json \
  --text_score_gate_strength 0.5 \
  --text_score_gate_floor 0.1
```

## 9. 建议保留的最终主线

如果只保留“到在线 `text_score gate` 为止”的实现，建议将仓库中的方法主线理解为：

- `counterfactual` 负责制造监督信号
- `fit_text_score.py` 负责学习离线评分器
- `main_model.py` 负责把评分器变成在线 gate

这就是当前仓库中 `text_score` 方案的最终保留版本。
