# PPT大纲

## 第1页：标题页

### 标题
时间尺度感知的多模态条件扩散时间序列预测

### 副标题
从 MCD-TSF 到尺度感知 RAG-CoT-扩散预测框架

### 页面内容
- 汇报主题：如何把时间尺度感知机制引入多模态时间序列预测
- 核心思路：将文本构造、推理摘要、多分辨率训练和自适应 guidance 统一到条件扩散模型中
- 研究目标：提升模型对不同样本时间尺度差异的适应能力

### 开场一句话
$$
\text{Scale-aware multimodal conditioning} \rightarrow \text{better forecasting}
$$

### 讲述重点
本次汇报分为两部分：先简要回顾上一篇工作 MCD-TSF，再介绍当前方法如何进一步把时间尺度引入文本构造、训练监督和推理控制全过程。

---

## 第2页：研究背景

### 标题
研究背景：为什么时间序列预测需要多模态信息

### 页面内容
- 时间序列预测不仅依赖历史数值，还依赖时间结构和文本事件信息
- 传统方法主要建模数值序列，难以利用外部语义信息
- 扩散模型适合处理时间序列中的随机性和不确定性
- 多模态预测的关键不只是“加入文本”，而是“让文本真正服务于预测”

### 核心表达
给定历史序列、时间戳和文本背景：
$$
(X, T, E) \rightarrow Y
$$

相比单模态预测：
$$
X \rightarrow Y
$$

多模态预测希望利用更多条件信息：
$$
(X, T, E) \rightarrow \hat{Y}
$$

### 讲述重点
这一页要强调：单纯依靠数值序列不够，时间结构和文本背景都可能包含重要预测线索，但如何有效利用这些信息是核心问题。

---

## 第3页：上一篇工作回顾：MCD-TSF 的出发点

### 标题
前序工作回顾：MCD-TSF

### 页面内容
- 上一篇文章题目：
  $$
  \text{Multimodal Conditioned Diffusive Time Series Forecasting}
  $$
- 核心目标：把时间序列预测从单模态扩展到多模态条件扩散预测
- 输入模态包括：
  - 历史序列
  - 时间戳
  - 文本描述
- 基本思想：让时间戳和文本在扩散去噪过程中作为条件信息参与预测

### 核心框架
$$
(X, T, E) \rightarrow \text{Diffusion Model} \rightarrow \hat{Y}
$$

### 讲述重点
这一页主要告诉听众：上一篇工作已经证明，多模态条件扩散在时间序列预测中是有效的，它为当前工作提供了基础。

---

## 第4页：上一篇工作回顾：MCD-TSF 怎么做

### 标题
MCD-TSF 的核心机制

### 页面内容
- 历史序列是扩散模型的主输入
- 时间戳增强不同时间点之间的结构关系
- 文本提供历史背景和外部语义补充
- 多模态条件在每个扩散步中共同引导去噪
- 通过 classifier-free guidance 动态控制文本条件影响

### 扩散过程
给定未来目标序列 $Y$，扩散模型从高斯噪声开始逐步恢复：
$$
Y^K \rightarrow Y^{K-1} \rightarrow \cdots \rightarrow Y^0
$$

在每一步中：
$$
Y^{k-1} = f_{\theta}(Y^k, X, T, E)
$$

### 讲述重点
这一页重点说明 MCD-TSF 的贡献在于“多模态条件扩散”，但文本和时间戳仍然主要是统一作为条件输入模型。

---

## 第5页：上一篇工作的局限与本文动机

### 标题
问题在哪里：为什么还需要继续改进

### 页面内容
- 上一篇工作已经加入文本和时间戳，但文本利用方式偏统一注入
- 缺少针对不同样本的时间尺度自适应机制
- 文本检索、证据筛选和推理摘要之间的链条还不够显式
- 不同样本在短期、中期、长期上的预测需求不同，但原方法没有系统建模

### 研究动机
我们希望进一步回答三个问题：
1. 文本证据应该如何按样本动态构造？
2. 检索到的文本如何转成可计算条件？
3. 不同样本的时间尺度差异如何同时作用于训练和推理？

### 逻辑转折
从上一篇工作的：
$$
\text{Multimodal Conditioning}
$$

走向当前工作的：
$$
\text{Scale-aware Multimodal Conditioning}
$$

### 讲述重点
这一页是承上启下。核心是说明：上一篇工作证明了多模态条件有效，而本文进一步把“时间尺度”变成统一组织文本构造、训练和推理的关键机制。

---

## 第6页：本文整体框架

### 标题
本文整体框架：主干 + 两条控制支路

### 页面内容
- 输入包括历史序列 $X_i$、时间信息 $U_i$、原始文本
- 整体分为两条支路：
  - 文本构造支路
  - 控制支路
- 两条支路最终都作用于条件扩散预测

### 总体逻辑
$$
\text{历史序列}
\rightarrow
\text{时间尺度表征}
\rightarrow
\begin{cases}
\text{文本构造支路: } \text{scale-aware RAG} \rightarrow \text{CoT} \rightarrow \text{条件表示} \\
\text{控制支路: } \text{scale router} \rightarrow \text{multi-res training} + \text{adaptive guidance}
\end{cases}
\rightarrow
\text{条件扩散预测}
$$

### 讲述重点
这一页是全文最核心的结构图。要明确说明：本文不是在原模型上简单再加一个模块，而是把时间尺度变成连接文本构造、训练监督和推理 guidance 的统一线索。

---

## 第7页：历史序列提取时间尺度表征

### 标题
Step 1：从历史序列中提取时间尺度表征

### 页面内容
- 首先从历史序列中提取样本的尺度相关特征
- 这些特征包括：
  - 趋势强度
  - 局部波动
  - 差分变化
  - 加速度
  - 相对波动率
- 最终形成统一尺度表征 $z_i$

### 公式
$$
z_i = \Phi(X_i)
$$

其中历史序列为：
$$
X_i = [x_{i,1}, x_{i,2}, \dots, x_{i,H}]
$$

### 含义
这一步的目标不是直接预测，而是判断当前样本更偏向：
- 短期扰动
- 中期过渡
- 长期趋势

### 讲述重点
强调 $z_i$ 是后续所有动作的起点，但不是一个单独变量直接控制所有模块；后续会分化为文本构造信号和 router 控制信号两种用途。

---

## 第8页：文本构造支路，尺度感知窗口与 RAG

### 标题
Step 2：尺度感知的文本窗口与 RAG 检索

### 页面内容
- 先由尺度表征生成文本构造信号
- 再由该信号决定文本窗口长度和尺度提示
- 之后结合历史序列、时间信息和原始文本构造 query
- 最终检索得到与当前样本更匹配的证据集合

### 公式1：文本构造信号
$$
s_i^{\mathrm{rag}} = G_{\mathrm{rag}}(z_i),
\qquad
\ell_i = \Gamma(s_i^{\mathrm{rag}})
$$

### 公式2：尺度感知检索
$$
\mathcal{D}_i^{\mathrm{rag}}
=
\operatorname{Retrieve}\!\left(
\Psi_q(X_i, U_i, T_i^{\mathrm{raw}}, s_i^{\mathrm{rag}}),
\mathcal{C}_i(\ell_i)
\right)
$$

### 解释
- $s_i^{\mathrm{rag}}$：文本构造支路中的尺度信号
- $\ell_i$：文本窗口长度
- $T_i^{\mathrm{raw}}$：原始文本片段
- $\mathcal{D}_i^{\mathrm{rag}}$：检索到的证据集合

### 讲述重点
说明时间尺度不是直接替代检索器，而是参与“文本窗口构造 + query 组织”，从而使检索结果更贴合当前样本的预测需求。

---

## 第9页：CoT 将证据变成结构化条件

### 标题
Step 3：CoT 把检索证据变成可计算条件

### 页面内容
- RAG 返回的是分散的证据集合
- 这些证据仍然是冗余的、碎片化的
- 因此需要引入 CoT，把证据整理成结构化推理摘要
- CoT 结果不仅是文字解释，还会转成趋势先验供后续模块使用

### 公式1：CoT 推理摘要
$$
c_i = \Psi_{\mathrm{cot}}(X_i, U_i, \mathcal{D}_i^{\mathrm{rag}})
$$

### 公式2：趋势先验
$$
\pi_i = \Omega(c_i)
$$

可表示为：
$$
\pi_i = [\text{direction}_i,\ \text{strength}_i,\ \text{volatility}_i]
$$

### 含义
CoT 提供两个结果：
- 文本侧的推理摘要
- 控制支路可用的趋势先验

### 讲述重点
这一页要讲清楚：CoT 不是附加解释层，而是从“证据”到“可计算中间变量”的关键桥梁。

---

## 第10页：统一文本条件表示

### 标题
Step 4：形成统一文本条件表示

### 页面内容
- 最终进入模型的文本条件不是单一来源
- 它需要统一整合：
  - 原始文本片段
  - 数值摘要
  - 尺度提示
  - 检索证据
  - CoT 摘要

### 公式
$$
E_i^{\mathrm{upd}}
=
\operatorname{Compose}\!\left(
T_i^{\mathrm{raw}},
S_i^{\mathrm{num}},
s_i^{\mathrm{rag}},
\mathcal{D}_i^{\mathrm{rag}},
c_i
\right)
$$

### 含义
- $T_i^{\mathrm{raw}}$：原始文本片段
- $S_i^{\mathrm{num}}$：由数值历史得到的摘要
- $s_i^{\mathrm{rag}}$：尺度提示
- $\mathcal{D}_i^{\mathrm{rag}}$：RAG 证据
- $c_i$：CoT 推理摘要

### 讲述重点
要强调进入模型的文本条件并不是“只有原始文本”或“只有检索文本”，而是多个来源的统一融合。这一步保证文本侧信息完整且结构化。

---

## 第11页：控制支路，scale router 与多分辨率训练

### 标题
Step 5：scale router 驱动多分辨率训练

### 页面内容
- 训练时不再把整个预测窗口看成一个整体，而是拆成多个时间 band
- 对每个 band 分别计算误差，从而显式区分短期、中期、长期预测难度
- 控制支路融合历史序列特征、趋势先验和文本可用性
- router 输出样本级时间尺度 band 权重分布
- 最终由“样本级 router 权重 + 全局 band 难度统计”共同决定辅助损失的加权方式

### 这一页要回答的问题
$$
\text{对于当前样本，训练时应该更关注哪些 horizon band？}
$$

### 为什么需要这一部分
- 如果所有 horizon 使用统一损失，模型只能学到平均意义上的预测策略
- 但不同样本的难点并不相同：
  - 有些样本短期波动更强，应该更关注前几个预测点
  - 有些样本长期趋势更明显，应该更关注后段 horizon 的拟合
- 因此需要一种样本级的时间尺度加权机制，让训练目标随样本变化而变化

### 公式1：scale router
$$
r_i^{\mathrm{router}}
=
\operatorname{softmax}\!\big(
G_{\mathrm{router}}(\tilde z_i)
\big)
=
(r_{i,1}, r_{i,2}, \dots, r_{i,B})
$$

其中：
$$
\tilde z_i = \operatorname{Fuse}(X_i, \pi_i, m_i)
$$

### 解释
- $X_i$：历史序列提供的数值统计特征
- $\pi_i$：由 CoT 摘要提取出的趋势先验
- $m_i$：文本是否可用的标记
- $r_i^{\mathrm{router}}$：样本在不同时间 band 上的偏好分布

### 讲法
router 不是直接预测未来值，而是输出一个分布，表示当前样本更应该让模型关注短期 band、中期 band 还是长期 band。

### 公式2：band 损失
对于每个 band $b$：
$$
L_{i,b}
=
\frac{1}{N_{i,b}}
\sum \rho\!\left(
(Y_i - \hat{Y}_i)\odot M_{i,b}
\right)
$$

### 解释
- $M_{i,b}$：第 $b$ 个时间 band 对应的 mask
- $N_{i,b}$：该 band 中有效预测点的数量
- $\rho(\cdot)$：点级损失函数，可以是平方损失或 Huber 损失

### 含义
这一步先把整体预测误差拆成多个时间尺度上的误差：
$$
\text{overall error} \rightarrow \{\text{short-term error},\ \text{mid-term error},\ \text{long-term error}\}
$$

### 公式3：router 驱动的 band 加权
$$
\mathcal{L}_{\mathrm{aux},i}
=
\sum_{b=1}^{B} w_{i,b} L_{i,b}
$$

其中：
$$
w_{i,b} = \operatorname{Blend}\!\left(r_{i,b}^{\mathrm{router}},\ \bar w_b^{\mathrm{global}}\right)
$$

### 解释
- $r_{i,b}^{\mathrm{router}}$：样本级 band 权重
- $\bar w_b^{\mathrm{global}}$：全局 band 难度统计
- $w_{i,b}$：最终用于损失加权的 band 权重

### 讲法
- 如果只用 router 权重，训练可能过于激进
- 如果只用全局统计，又缺少样本个性
- 因此这里采用混合策略：
  - 用 router 提供样本级时间尺度偏好
  - 用全局统计提供整体稳定性
  - 最终得到更稳健的 band 权重

### 公式4：总训练目标
$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{main}}
+
\lambda_{\mathrm{mr}}\mathcal{L}_{\mathrm{aux}}
$$

### 含义
- $r_i^{\mathrm{router}}$：样本在不同时间 band 上的权重分布
- $\mathcal{L}_{\mathrm{aux}}$：多分辨率辅助监督损失
- 训练目标不仅关注整体预测误差，也关注不同时间尺度上的误差结构

### 这一页可以这样总结
1. 先把未来预测窗口切成多个时间 band。
2. 再让 router 判断当前样本更偏向哪种时间尺度。
3. 用 router 权重动态调节不同 band 的损失占比。
4. 从而让模型学到样本级的时间尺度自适应预测策略。

### 讲述重点
强调尺度信息不只影响文本构造，还影响训练时模型应该更关注哪些 horizon。也就是把时间尺度差异直接写进训练目标本身：
$$
\text{sample-aware horizon reweighting}
$$

一句话总结这页：
$$
\text{scale router 的本质，是让损失函数按样本的时间尺度结构动态重分配到不同 horizon band 上}
$$

---

## 第12页：推理阶段，自适应 guidance

### 标题
Step 6：推理阶段的自适应 guidance

### 页面内容
- router 输出不会只用于训练，还会继续参与推理控制
- 首先将 band 权重分布压缩成一个样本级尺度分数
- 再根据该尺度分数对基础 guide weight 做样本级调整
- 从而实现不同样本对文本条件依赖程度的动态调节

### 公式1：尺度分数
$$
\sigma_i
=
\sum_{b=1}^{B} r_{i,b}^{\mathrm{router}} m_b
$$

### 公式2：样本级 guidance
$$
g_i
=
g \cdot
\operatorname{clip}
\left(
1 + \alpha(\sigma_i - 0.5),
\rho_{\min},
\rho_{\max}
\right)
$$

### 含义
- $\sigma_i$：样本级尺度分数
- $g$：基础 guide weight
- $g_i$：样本级自适应 guidance 强度

### 讲述重点
说明 guidance 不应该是固定常数。长期结构明显的样本可以更依赖文本条件，短期扰动明显的样本则需要更保守的文本修正。

---

## 第13页：最终预测与总结

### 标题
最终预测与方法总结

### 页面内容
- 推理时同时计算：
  - 无条件预测
  - 带文本条件预测
- 两者之差表示文本条件带来的修正量
- 样本级 guidance 强度 $g_i$ 决定修正量的放大程度
- 最终文本条件支路和控制支路共同决定预测结果

### 最终预测公式
$$
\hat{Y}_{i,k}^{(g_i)}
=
f_{\theta}(Y_{i,k}, X_i, U_i, \varnothing)
+
g_i
\left(
f_{\theta}(Y_{i,k}, X_i, U_i, E_i^{\mathrm{upd}})
-
f_{\theta}(Y_{i,k}, X_i, U_i, \varnothing)
\right)
$$

### 总结逻辑
$$
X_i
\rightarrow
z_i
\rightarrow
\begin{cases}
s_i^{\mathrm{rag}} \rightarrow \mathcal{D}_i^{\mathrm{rag}} \rightarrow c_i \rightarrow E_i^{\mathrm{upd}} \\
r_i^{\mathrm{router}} \rightarrow \mathcal{L}_{\mathrm{aux}},\ g_i
\end{cases}
\rightarrow
\hat{Y}_i
$$

### 总结语
- 前序工作证明：多模态条件扩散有效
- 当前工作进一步引入：时间尺度感知机制
- 最终形成完整链条：
  $$
  \text{Scale-aware RAG} + \text{CoT} + \text{multi-res training} + \text{adaptive guidance}
  $$

### 讲述重点
这一页作为收尾页，要把全文压缩成一句话：文本条件解决“参考什么信息”，尺度控制解决“关注什么时间尺度以及多大程度相信文本”，扩散模型负责把这两者统一变成最终预测。
