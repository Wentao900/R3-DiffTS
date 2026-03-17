# 更新公式主线讲解（完整逻辑链版本）

本文档讲的是当前仓库里这一条完整方法链，不删模块，也不把不同职责的模块硬压成一个变量流。

核心思想不是“一个尺度分布从头控制到尾”，而是：

1. 历史序列先暴露样本的时间尺度特征；
2. 这些尺度特征一方面参与文本条件的构造；
3. 另一方面参与训练阶段的多分辨率监督与推理阶段的 guidance 调节；
4. 最终文本条件和尺度控制共同作用于条件扩散预测。

因此，整条逻辑应讲成：

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

---

## 1. 历史序列先形成时间尺度表征

### 公式 1：尺度特征提取
$$
z_i = \Phi(X_i)
$$

### 方法说明
对于样本 $i$，先从历史序列 $X_i$ 中提取尺度相关特征。这里的目标不是直接预测未来，而是判断该样本当前更偏向短期扰动、中期过渡还是长期趋势。

### 具体做法
1. 输入历史序列
   $$
   X_i = [x_{i,1}, x_{i,2}, \dots, x_{i,H}]
   $$
2. 提取刻画尺度的统计或隐表示，例如：
   - 整体斜率与趋势强度；
   - 局部波动与一阶差分强度；
   - 加速度或二阶变化；
   - 相对波动率与平均幅值。
3. 将这些信息压缩为尺度表征 $z_i$。

### 这一公式的作用
它不是最终控制量，而是后续两条支路的共同起点：
- 文本构造支路会用它来决定文本窗口和尺度提示；
- 控制支路会用它来决定多尺度训练权重与推理 guidance 强度。

---

## 2. 第一条支路：时间尺度参与文本构造

这一支路回答的问题是：

> 当前样本应该用什么时间范围的原始文本、以什么尺度语义去检索外部证据、再把这些证据整理成什么样的文本条件输入模型？

---

### 公式 2：面向文本构造的尺度信号
$$
s_i^{\mathrm{rag}} = G_{\mathrm{rag}}(z_i),
\qquad
\ell_i = \Gamma(s_i^{\mathrm{rag}})
$$

### 方法说明
这里不把尺度表征直接用于预测，而是先把它转换成文本构造信号 $s_i^{\mathrm{rag}}$。然后再由该信号决定文本窗口长度 $\ell_i$，以及是否给后续 query 注入“short / mid / long”这样的尺度提示。

### 具体做法
1. 从 $z_i$ 中得到一个样本级尺度偏好 $s_i^{\mathrm{rag}}$；
2. 根据该尺度偏好映射出文本窗口长度 $\ell_i$；
3. 若样本更偏短期，则原始文本窗口更短；
4. 若样本更偏长期，则原始文本窗口更长；
5. 同时把尺度语义作为检索提示词，帮助 query 更聚焦于当前样本关心的时间层次。

### 这一公式的作用
它把“时间尺度”第一次真正落到文本侧动作上，决定“先给模型看到哪段文本”和“检索时强调什么尺度语义”。

---

### 公式 3：尺度感知的 RAG 检索
$$
\mathcal{D}_i^{\mathrm{rag}}
=
\operatorname{Retrieve}\!\left(
\Psi_q(X_i, U_i, T_i^{\mathrm{raw}}, s_i^{\mathrm{rag}}),
\mathcal{C}_i(\ell_i)
\right)
$$

### 符号说明
- $T_i^{\mathrm{raw}}$：由窗口长度 $\ell_i$ 选出的原始文本片段；
- $\mathcal{C}_i(\ell_i)$：建立在该文本上下文上的候选文本语料；
- $\Psi_q(\cdot)$：构造检索 query 的函数；
- $\mathcal{D}_i^{\mathrm{rag}}$：最终检索出的文本证据集合。

### 方法说明
这一步的重点不是“无条件地全库检索”，而是“带着样本的数值模式、当前文本上下文和尺度提示去检索”。也就是说，时间尺度并不直接替代语义相似度，而是参与 query 和文本上下文的构造，使检索更贴近当前样本的预测需求。

### 具体做法
1. 先由 $\ell_i$ 取出原始文本片段 $T_i^{\mathrm{raw}}$；
2. 再基于历史序列 $X_i$、时间信息 $U_i$、原始文本 $T_i^{\mathrm{raw}}$ 和尺度提示 $s_i^{\mathrm{rag}}$ 构造 query；
3. 用检索器从候选文本中取出最相关证据；
4. 在两阶段 RAG 场景下，还会在第一阶段证据基础上生成 trend hypothesis，再做第二轮更聚焦的检索；
5. 将最终证据记为 $\mathcal{D}_i^{\mathrm{rag}}$。

### 这一公式的作用
它解决“文本证据从哪里来”的问题，而且强调：证据不是静态喂给模型的，而是被当前样本的时间尺度偏好主动组织过的。

---

## 3. RAG 之后用 CoT 把证据变成结构化中间变量

### 公式 4：CoT 推理摘要
$$
c_i = \Psi_{\mathrm{cot}}(X_i, U_i, \mathcal{D}_i^{\mathrm{rag}})
$$

### 方法说明
RAG 给出的是证据集合，但预测模型真正需要的是“可对齐到未来 horizon 的结构化判断”。因此需要 CoT 把分散证据整理成中间推理摘要 $c_i$。

### 具体做法
1. 输入历史序列 $X_i$，避免推理脱离真实数值变化；
2. 输入时间信息 $U_i$，保留事件与预测目标的时序关系；
3. 输入检索到的证据集合 $\mathcal{D}_i^{\mathrm{rag}}$；
4. 用 CoT 模块组织出更紧凑的中间结论；
5. 输出摘要 $c_i$，其中可包含：
   - 方向判断；
   - 趋势强弱；
   - 波动水平；
   - 关键影响因素。

### 为什么这一步重要
因为模型后面不仅需要“文本存在”，还需要“文本中的逻辑关系被显式整理出来”。否则检索证据只是材料堆积，不能稳定转化为有效条件。

### 这一公式的作用
它完成了从“证据”到“可计算条件”的过渡。

---

### 公式 5：由 CoT 摘要提取趋势先验
$$
\pi_i = \Omega(c_i)
$$

### 方法说明
在当前方法里，CoT 不只是文本摘要器，还会被进一步解析为趋势先验 $\pi_i$。这个趋势先验后面会进入控制支路，用于辅助 router 判断样本更适合关注哪些时间尺度。

### 具体做法
1. 从 $c_i$ 中读取结构化字段；
2. 将方向、强度、波动等离散语义映射为数值向量；
3. 得到趋势先验
   $$
   \pi_i = [\text{direction}_i,\ \text{strength}_i,\ \text{volatility}_i]
   $$

### 这一公式的作用
它是文本支路与控制支路之间的桥。也就是说，CoT 的输出不仅送入文本条件，还反过来参与后续的尺度控制。

---

### 公式 6：统一文本条件表示
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

### 符号说明
- $T_i^{\mathrm{raw}}$：原始文本片段；
- $S_i^{\mathrm{num}}$：由数值历史得到的简洁统计摘要；
- $s_i^{\mathrm{rag}}$：尺度提示；
- $\mathcal{D}_i^{\mathrm{rag}}$：RAG 检索证据；
- $c_i$：CoT 推理摘要。

### 方法说明
这一步把文本侧所有有用信息统一打包成最终条件表示 $E_i^{\mathrm{upd}}$。因此，进入文本编码器的条件并不是“只拼接 RAG 和 CoT”，而是“原始文本 + 数值摘要 + 尺度提示 + 检索证据 + 推理摘要”的组合。

### 为什么要这样设计
- 只保留原始文本：缺少针对性；
- 只保留检索文本：缺少结构化推理；
- 只保留 CoT：又会丢失原始证据来源；
- 统一组合：同时保留事实、语义聚焦和推理结果。

### 这一公式的作用
它完成文本支路的收束，把前面得到的所有文本相关信息封装成扩散模型可直接利用的条件输入。

---

## 4. 第二条支路：时间尺度参与训练阶段的多分辨率控制

这一支路回答的问题是：

> 模型在训练时，应该更关注哪些 horizon band？不同样本在短期、中期、长期误差上的权重应该如何动态分配？

---

### 公式 7：样本级 scale router 分布
$$
r_i^{\mathrm{router}}
=
\operatorname{softmax}\!\big(
G_{\mathrm{router}}(\tilde z_i)
\big)
=
\left(r_{i,1}, r_{i,2}, \dots, r_{i,B}\right),
\qquad
\sum_{b=1}^{B} r_{i,b} = 1
$$

其中
$$
\tilde z_i = \operatorname{Fuse}(X_i, \pi_i, m_i)
$$

### 符号说明
- $\pi_i$：由 CoT 解析得到的趋势先验；
- $m_i$：文本可用性标记；
- $B$：multi-resolution band 的数量；
- $r_{i,b}$：样本 $i$ 对第 $b$ 个 band 的偏好权重。

### 方法说明
这里的 router 分布不是前面文本构造支路里的尺度提示，而是专门为训练与推理控制服务的样本级 band 权重分布。它会综合历史序列特征、趋势先验和文本可用性，判断当前样本更应该让模型在哪些预测区间上投入更多注意力。

### 具体做法
1. 从历史序列中提取 router 特征；
2. 将趋势先验 $\pi_i$ 融合进来，补充文本侧对趋势与波动的判断；
3. 将文本可用标记 $m_i$ 融合进来，区分“有文本辅助”和“无文本辅助”样本；
4. 由 router 输出 band 上的 softmax 分布 $r_i^{\mathrm{router}}$。

### 这一公式的作用
它把“样本属于哪种时间尺度结构”转成了训练和推理都可直接调用的 band 级控制量。

---

### 公式 8：多分辨率 band 损失
对于每个 band $b$，定义
$$
L_{i,b}
=
\frac{1}{N_{i,b}}
\sum \rho\!\left(
\big(Y_i - \hat Y_i\big)\odot M_{i,b}
\right)
$$

其中：
- $M_{i,b}$ 表示第 $b$ 个预测区间对应的 mask；
- $N_{i,b}=\sum M_{i,b}$；
- $\rho(\cdot)$ 可以是平方损失，也可以是 Huber 损失。

### 方法说明
这一步不是只看整段预测误差，而是把未来预测窗口拆成多个时间 band，分别计算误差。这样模型不会只学一个平均意义上的“整体正确”，而是能显式感知短期、中期、长期各自的拟合难度。

### 这一公式的作用
它把预测误差从“单一整体损失”拆成“按时间尺度分解的误差结构”，为后面的自适应加权提供基础。

---

### 公式 9：router 加权的辅助损失
$$
\mathcal{L}_{\mathrm{aux},i}
=
\sum_{b=1}^{B} w_{i,b} L_{i,b},
\qquad
w_{i,b} = \operatorname{Blend}\!\left(r_{i,b}^{\mathrm{router}},\ \bar w_b^{\mathrm{global}}\right)
$$

### 方法说明
当前方法并不是完全依赖样本级 router，也不是完全依赖全局统计，而是把两者混合。

### 具体做法
1. 先由全局 EMA 统计得到各 band 的整体难度权重 $\bar w_b^{\mathrm{global}}$；
2. 再由 router 给出样本级 band 权重 $r_{i,b}^{\mathrm{router}}$；
3. 将全局权重与样本权重做混合；
4. 再与均匀分布做保守回拉，避免权重过于极端；
5. 得到最终 band 权重 $w_{i,b}$。

### 这一公式的作用
它说明模型训练时关注哪些 horizon，不是静态预设，而是由“样本自身尺度结构 + 全局难度统计”共同决定的。

---

## 5. 文本条件和多分辨率监督一起进入条件扩散训练

### 公式 10：基础条件扩散主损失
$$
\mathcal{L}_{\mathrm{main}}
=
\mathbb{E}_{k}
\left[
\left\|
f_{\theta}\!\left(
Y_{i,k},
X_i,
U_i,
E_i^{\mathrm{upd}}
\right)
- Y_i
\right\|_2^2
\right]
$$

### 方法说明
扩散主干会在每个训练步 $k$ 接收带噪目标状态 $Y_{i,k}$，并同时接收历史序列、时间信息和更新后的文本条件 $E_i^{\mathrm{upd}}$，学习如何在数值历史和文本条件共同作用下恢复目标序列。

### 这一公式的作用
它说明前面的 RAG 和 CoT 不是解释性附件，而是直接进入预测主干的条件输入。

---

### 公式 11：总训练目标
$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{main}}
+
\lambda_{\mathrm{mr}}\mathcal{L}_{\mathrm{aux}}
$$

其中
$$
\mathcal{L}_{\mathrm{aux}}
=
\frac{1}{N}\sum_{i=1}^{N}\mathcal{L}_{\mathrm{aux},i}
$$

### 方法说明
训练阶段并不是只最小化一个基础重建误差，而是同时最小化：
- 主扩散损失 $\mathcal{L}_{\mathrm{main}}$；
- 多分辨率辅助损失 $\mathcal{L}_{\mathrm{aux}}$。

此外，为了在推理时使用 classifier-free guidance，训练阶段还会随机遮蔽条件分支，使模型同时学习“有条件预测”和“无条件预测”。

### 这一公式的作用
它把整条训练逻辑闭合起来：文本条件负责提供外部语义，multi-res loss 负责强调不同 horizon 的尺度结构，而 CFG 训练保证推理时可以进行条件引导。

---

## 6. 推理阶段：router 再次调节 guidance 强度

训练结束后，时间尺度不再只体现在损失加权上，还会继续进入采样阶段，调节文本条件对预测结果的影响强度。

---

### 公式 12：router 尺度分数
$$
\sigma_i
=
\sum_{b=1}^{B} r_{i,b}^{\mathrm{router}} m_b
$$

### 符号说明
- $m_b$：第 $b$ 个 band 对应的中心位置或尺度中心；
- $\sigma_i$：样本级尺度分数。

### 方法说明
这一步把 router 在多个 band 上的离散分布压缩成一个连续分数，用来表示当前样本更偏短期还是更偏长期。

### 这一公式的作用
它把多维 band 分布进一步转成一个可直接控制 guidance 的标量。

---

### 公式 13：样本级自适应 guidance
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

### 符号说明
- $g$：基础 guide weight；
- $\alpha$：router guide 的缩放系数；
- $\rho_{\min}, \rho_{\max}$：样本级 guide ratio 的上下界。

### 方法说明
如果样本更偏向长期结构，则 router 会给出更大的尺度分数，guidance 可以相对增强；如果样本更偏向短期扰动，则文本条件对预测的修正不一定需要太强，guidance 可以相对保守。

### 这一公式的作用
它说明 guidance 不是固定常数，而是由样本级尺度结构动态调节的。

---

### 公式 14：条件引导的最终预测
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

### 方法说明
推理时同时计算：
- 无文本条件预测；
- 带文本条件预测。

两者之差表示文本条件带来的修正量，再由样本级 guidance 强度 $g_i$ 决定这部分修正应被放大多少。

### 这一公式的作用
它把文本支路和控制支路最终汇合到预测输出上：
- 文本支路提供 $E_i^{\mathrm{upd}}$；
- 控制支路提供 $g_i$；
- 扩散主干把两者融合成最终预测。

---

## 7. 用一句话讲完整条线

整条方法链可以概括为：

> 先从历史序列中提取时间尺度表征；这一表征一方面用于文本构造，决定文本窗口、尺度感知检索和 CoT 推理，并形成统一文本条件；另一方面用于训练与推理控制，通过 scale router 生成多分辨率 band 权重和样本级 guidance 强度；最后，文本条件与尺度控制共同作用于条件扩散模型，得到最终预测。

对应的完整公式链可以写成：

$$
X_i
\rightarrow
z_i
\rightarrow
\begin{cases}
s_i^{\mathrm{rag}}
\rightarrow
T_i^{\mathrm{raw}}
\rightarrow
\mathcal{D}_i^{\mathrm{rag}}
\rightarrow
c_i
\rightarrow
\pi_i
\rightarrow
E_i^{\mathrm{upd}}
\\[4pt]
\tilde z_i
\rightarrow
r_i^{\mathrm{router}}
\rightarrow
\mathcal{L}_{\mathrm{aux}},\ g_i
\end{cases}
\rightarrow
\mathcal{L}
\rightarrow
\hat{Y}_i
$$

如果用更直白的话来讲，就是：

> 文本条件解决“模型应该参考哪些外部信息”，尺度控制解决“模型在不同时间尺度上应该关注什么、以及在推理时应该多大程度相信文本条件”，而条件扩散模型负责把这两部分统一变成最终预测。
