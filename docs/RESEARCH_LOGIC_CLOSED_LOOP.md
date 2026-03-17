# 科研逻辑闭环图

本文档导出当前仓库主线 `Economy V2` 的科研逻辑闭环，重点对应：

- 数据侧时间尺度判断
- 两阶段 RAG / CoT 趋势构造
- 多尺度训练监督
- Router-aware 推理引导
- 指标回收与假设验证

> 注：当前仓库里已经沉淀了多数据集配置与通用能力，但这里展示和引用的“实验数据对比 / 指标分析”当前仅对应 `Economy` 任务，不代表所有数据集都已完成同等粒度的对比验证。

## Mermaid 版本

```mermaid
flowchart TD
    A[研究问题<br/>不同样本的时间尺度不同<br/>固定文本窗 / 固定 horizon 权重 / 固定 CFG 不合理]
    B[输入样本<br/>历史数值序列 x(1:L)<br/>历史文本报告与检索语料]
    C[数据侧尺度判断<br/>提取 slope / std / accel 等统计特征<br/>得到 short / mid / long]
    D[动态文本窗选择<br/>6 / 18 / 36]
    E[Scale-aware RAG Query<br/>把时间尺度提示注入检索 query]
    F[两阶段 RAG + CoT<br/>Stage-1 检索 E0<br/>生成 trend hypothesis<br/>Stage-2 检索 E1]
    G[构造文本条件 composed_text]
    H[提取趋势先验 trend_prior<br/>direction / strength / volatility]
    I[扩散预测模型 CSDI_Forecasting<br/>条件 = 数值 + 时间特征 + 文本编码 + trend prior]
    J[主预测损失<br/>整体预测区间误差]
    K[多尺度 horizon 切分<br/>1 / 2-3 / 4-6 / 7-12]
    L[Scale Router<br/>输入历史统计 + trend prior + text mask]
    M[样本级 band 权重<br/>决定训练更重哪段 horizon]
    N[训练目标<br/>主损失 + lambda * 多尺度辅助损失]
    O[推理阶段<br/>router -> scale_score -> sample-wise guide_w]
    P[测试输出<br/>MSE / MAE<br/>horizon metrics<br/>band metrics<br/>router metrics]
    Q[科研闭环验证<br/>检验时间尺度假设是否同时改善<br/>文本组织 / 训练关注 / 推理引导]

    A --> B
    B --> C
    C --> D
    C --> E
    C --> L
    D --> F
    E --> F
    F --> G
    F --> H
    G --> I
    H --> I
    H --> L
    I --> J
    I --> K
    K --> M
    L --> M
    J --> N
    M --> N
    L --> O
    O --> P
    N --> P
    P --> Q
    Q -. 反向修正设计 .-> A
```

## 一行版闭环

```text
历史数值 -> 判断时间尺度 -> 选择文本窗并做 RAG/CoT -> 生成 trend_prior
-> router 分配各 horizon 的训练权重 -> 推理时 router 调整 sample-wise guide_w
-> 用 horizon / band / router 指标验证“时间尺度自适应”假设
```

## 汇报版解释

1. 仓库当前主线不是单独优化某个模块，而是在验证“样本时间尺度”这个统一中间变量。
2. 这个中间变量先影响文本构造，再影响训练损失，最后影响采样引导。
3. 因此它形成的是一个真正的科研闭环，而不是几个彼此独立的小技巧。
4. 实验输出也围绕这个闭环回收证据，不只看总 MSE，还看 horizon、band 和 router 的行为统计。
5. 当前这套“实验数据对比”在仓库中主要落地于 `Economy`，其余数据集更多是配置与能力预留，不应直接视为已完成同口径对比。

## 对应代码位置

- 入口与验证集选 guide: `exe_forecasting.py`
- 数据侧尺度判断与文本构造: `data_provider/data_loader.py`
- 两阶段 RAG / CoT: `utils/rag_cot.py`
- 趋势先验解析: `utils/trend_prior.py`
- 多尺度损失、scale router、router-aware guide: `main_model.py`
- 当前主配置: `config/economy_36_12_scale_router_guide.yaml`
