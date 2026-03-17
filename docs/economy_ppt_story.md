# Economy 版本科研闭环故事与 PPT 大纲

## 0. 文档目的

这份文档用于支持 `Economy` 版本方法汇报。内容分为四部分：

1. `PPT` 大纲
2. 完整科研故事
3. 每一步的公式、变量定义和实际意义
4. 最终闭环总结

本文严格对应当前仓库中 `Economy` 主实验口径：

- 配置文件：`config/economy_36_12_scale_router_guide.yaml`
- 任务设置：`seq_len = 36`，`pred_len = 12`，`text_len = 36`
- 当前口径中，`Economy` 的主线不是 `CoT` 直接控制 `trend-aware CFG`
- 当前主线是：

$$
\text{历史序列}
\rightarrow
\text{时间尺度识别}
\rightarrow
\text{动态文本窗口}
\rightarrow
\text{两阶段 RAG / 趋势假设}
\rightarrow
\text{条件扩散训练}
\rightarrow
\text{scale router 多尺度加权}
\rightarrow
\text{router 自适应 guidance 推理}
\rightarrow
\text{最终预测}
$$

---

## 1. PPT 大纲

## Slide 1. 研究问题

### 标题

`为什么 Economy 预测需要时间尺度自适应？`

### 核心信息

- 经济类时间序列不是同一种动态过程
- 有些样本更依赖短期扰动
- 有些样本更依赖长期趋势
- 固定文本窗、固定训练关注点、固定 guidance 强度会带来失配

### 一句话讲法

> 预测困难并不只是因为未来不确定，而是因为不同样本依赖的时间尺度不同。

---

## Slide 2. 现有方法的根本问题

### 标题

`固定时间尺度处理异质样本，会造成三类失配`

### 核心信息

- 文本失配：短期样本被塞入过长文本，长期样本又只看到局部文本
- 优化失配：所有样本都按同一个 horizon 权重训练
- 推理失配：所有样本都使用相同的 guidance 强度

### 一句话讲法

> 文本、训练目标和推理控制都在用固定规则，但样本本身是尺度异质的。

---

## Slide 3. 核心假设

### 标题

`时间尺度是统一驱动文本构造、训练优化和推理控制的关键变量`

### 核心信息

- 如果先识别样本尺度
- 再据此组织文本
- 再据此调整训练关注重点
- 再据此调整推理 guidance
- 那么模型会更稳定，也更符合样本本身的动态结构

### 一句话讲法

> 时间尺度不是一个局部技巧，而是整个系统的组织变量。

---

## Slide 4. 方法总览

### 标题

`Economy 版本整体流程`

### 推荐展示

- 使用 `docs/economy_actual_pipeline.svg`

### 一句话讲法

> 这套方法不是简单堆模块，而是围绕“时间尺度自适应”形成输入、训练、推理三位一体的闭环。

---

## Slide 5. 第一步：时间尺度识别

### 标题

`先判断当前样本偏短期还是长期`

### 核心信息

- 从历史序列抽取斜率、波动、加速度等统计量
- 形成一个连续尺度偏好分数
- 再映射成 `short / mid / long`

### 一句话讲法

> 先回答“这个样本属于什么时间尺度”，再决定后面所有模块怎么工作。

---

## Slide 6. 第二步：动态文本窗口与两阶段 RAG

### 标题

`让文本和当前样本的时间尺度对齐`

### 核心信息

- 按尺度动态选择文本窗口长度
- 第一阶段检索拿到初始证据
- 用初始证据生成趋势假设
- 第二阶段围绕趋势假设精炼证据

### 一句话讲法

> 文本不是越多越好，而是要和当前样本真正关心的预测尺度一致。

---

## Slide 7. 第三步：趋势假设如何进入模型

### 标题

`趋势假设同时变成文本条件和趋势先验`

### 核心信息

- 生成的 `trend hypothesis` 不只是给人看
- 它一方面进入 `composed_text`
- 另一方面被解析成 `trend_prior`
- 为后续 router 提供稳定的结构化语义

### 一句话讲法

> 我们把松散的文本检索结果，压缩成可被模型稳定使用的结构化中间语义。

---

## Slide 8. 第四步：为什么还要有 scale router

### 标题

`样本尺度不同，训练时关注的 horizon 也应该不同`

### 核心信息

- 把未来 12 步拆成多个 band
- router 判断当前样本更该重视哪一段
- 多尺度辅助损失不再是固定加权，而是样本级加权

### 一句话讲法

> 同样预测 12 步，不同样本的训练重点应该不同。

---

## Slide 9. 第五步：推理阶段的自适应 guidance

### 标题

`Economy 不是 CoT 直接控 CFG，而是 router 调 guidance`

### 核心信息

- 先在验证集选一个全局最优 `guide_w`
- router 再把它改成样本级 `sample_guide`
- 让不同样本以不同强度使用文本条件

### 一句话讲法

> 推理阶段的适应性不是来自“统一的 guidance”，而是来自“样本级 guidance”。

---

## Slide 10. 结果与闭环

### 标题

`同一个时间尺度变量，贯穿输入、训练和推理`

### 核心信息

- 时间尺度决定文本窗口
- 时间尺度影响检索与趋势假设
- 时间尺度控制训练时的 horizon 权重
- 时间尺度控制推理时的 guidance 强度

### 一句话讲法

> 这就是一个完整的科研闭环：同一个核心变量，在多个阶段被反复使用并被实验验证。

---

## 2. 完整科研故事

## 2.1 故事的起点：问题不是“有没有文本”，而是“文本有没有按对的尺度进入模型”

我们面对的是一个多模态经济预测任务。输入里既有历史数值序列，也有对应时间范围内的文本报告。直觉上，文本应该帮助预测，但真实情况更复杂。

原因在于，`经济样本之间的动态模式并不统一`。有些样本的未来变化更多由短期扰动决定，有些样本则明显由中长期趋势主导。如果仍然对所有样本都使用：

- 相同长度的文本窗口
- 相同的检索组织方式
- 相同的训练 horizon 权重
- 相同的 guidance 强度

那么模型就会默认所有样本都遵循同一种时间结构。这种假设与任务本身是不匹配的。

因此，真正的问题不是：

$$
\text{文本是否有用}
$$

而是：

$$
\text{文本是否以正确的时间尺度进入模型}
$$

---

## 2.2 第一层想法：先识别样本尺度，再组织文本

既然样本之间的尺度不同，那么第一步就不该直接做预测，而应该先从历史数值中估计当前样本更偏：

$$
\text{short}, \quad \text{mid}, \quad \text{long}
$$

一旦这个判断成立，文本构造方式就应该跟着变。短期样本看短文本，长期样本看长文本。这样，文本输入才不是“统一模板”，而是“按样本定制”。

---

## 2.3 第二层想法：文本还要进一步从“堆积信息”变成“形成趋势假设”

仅仅改变文本窗口还不够。因为原始文本本身可能依然噪声很大，信息也未必足够聚焦。所以需要通过检索把文本进一步组织起来。

但一次检索还不够，因为第一次检索更像是在“找相关材料”，还没有形成清晰的预测语义。因此我们引入两阶段 `RAG`：

1. 第一阶段检索，拿到初始证据
2. 基于这些证据形成一个趋势假设
3. 再围绕这个趋势假设做第二阶段检索

这意味着系统不再只是“把资料拿过来”，而是：

$$
\text{先检索} \rightarrow \text{先形成解释性中间语义} \rightarrow \text{再精炼检索}
$$

这样得到的文本条件更稳定，也更贴近当前样本真正的预测方向。

---

## 2.4 第三层想法：同一个尺度信息，不能只服务于文本，还应该进入训练目标

到这里，文本侧已经被尺度自适应地组织起来了。但我们继续问一个问题：

如果一个样本本身更偏短期，那么训练时为什么还要和长期样本一样，对整个预测区间平均处理？

因此我们把未来区间拆成多个 horizon band，让模型不只是学习整体预测，还学习：

$$
\text{不同时间段的误差应该被如何区分对待}
$$

再进一步，我们不想使用固定的 band 权重，而是让模型自己判断某个样本更应关注哪一段，这就是 `scale router` 的作用。

所以训练阶段形成了第二个闭环：

$$
\text{时间尺度识别} \rightarrow \text{router} \rightarrow \text{样本级多尺度损失加权}
$$

---

## 2.5 第四层想法：训练中学到的尺度偏好，应该延续到推理阶段

如果 router 只在训练中使用，那它学到的只是“怎么加权损失”，但这些知识没有被显式传到推理阶段。

于是我们进一步把 router 输出用于推理控制。先在验证集选出一个全局最优的 `guide_w`，然后让 router 根据当前样本尺度，把这个全局值缩放成样本级 guidance。

因此，`Economy` 版本的核心不是：

$$
\text{CoT} \rightarrow \text{直接控制 CFG}
$$

而是：

$$
\text{CoT / 趋势假设}
\rightarrow
\text{trend prior}
\rightarrow
\text{router}
\rightarrow
\text{样本级 guidance}
$$

---

## 2.6 最终形成的科研闭环

到最后，我们得到的是一个围绕“时间尺度自适应”的统一系统。

同一个核心变量，即样本级时间尺度，被反复用于：

1. 组织文本窗口
2. 组织检索方式
3. 构造趋势先验
4. 控制训练时不同 horizon 的关注重点
5. 控制推理时 guidance 的强弱

因此，整个方法不是一组松散模块，而是一个完整闭环：

$$
\text{问题发现}
\rightarrow
\text{机制假设}
\rightarrow
\text{输入构造}
\rightarrow
\text{训练优化}
\rightarrow
\text{推理控制}
\rightarrow
\text{实验验证}
$$

---

## 3. 每一步如何走：公式、变量定义与意义

## 3.1 Step 1：时间尺度识别

### 输入

给定历史序列：

$$
\mathbf{x} = (x_1, x_2, \dots, x_L), \qquad L = 36
$$

### 统计量定义

代码中首先计算：

$$
\text{slope} = \frac{|x_L - x_1|}{L - 1}
$$

$$
\sigma = \text{std}(\mathbf{x})
$$

$$
\mu_{\text{abs}} = \frac{1}{L}\sum_{t=1}^{L} |x_t|
$$

$$
\text{total\_shift} = |x_L - x_1|
$$

若 $L > 2$，定义二阶差分平均绝对值：

$$
\text{accel} = \frac{1}{L-2}\sum_{t=1}^{L-2} |x_{t+2} - 2x_{t+1} + x_t|
$$

### 平滑度

$$
\text{smoothness} = \frac{1}{1 + \frac{\text{accel}}{\text{slope} + \varepsilon}}
$$

### 趋势分数与波动分数

$$
\text{trend\_score}
=
\frac{\text{total\_shift}}{\sigma + \varepsilon}
\cdot
\text{smoothness}
$$

$$
\text{volatility\_score}
=
\frac{\sigma}{\mu_{\text{abs}} + \varepsilon}
+
\frac{\text{accel}}{\sigma + \varepsilon}
$$

### 尺度偏好分数

$$
\text{scale\_pref}
=
\frac{\text{trend\_score}}
{\text{trend\_score} + \text{volatility\_score} + \varepsilon}
$$

其中：

$$
\text{scale\_pref} \in (0,1)
$$

### 标签映射

设候选文本长度集合为：

$$
\mathcal{T} = \{6, 18, 36\}
$$

则将 `scale_pref` 离散映射为：

$$
\text{scale\_label} \in \{\text{short}, \text{mid}, \text{long}\}
$$

若用索引形式表示，可写为：

$$
j = \min\left(\left\lfloor \text{scale\_pref} \cdot |\mathcal{T}| \right\rfloor, |\mathcal{T}| - 1\right)
$$

并同步得到：

$$
\text{text\_window\_len} \in \{6, 18, 36\}
$$

### 这一步的意义

这一步回答的是：

$$
\text{当前样本的预测主要更像短期问题，还是长期问题？}
$$

它是整套方法的起点，因为后续的文本构造、router 权重、推理 guidance 都依赖这个尺度判断。

---

## 3.2 Step 2：动态文本窗口

### 做法

若当前样本在时间步 $t=L$ 结束，文本窗口长度为 $w$，则文本截取范围为：

$$
[L-w+1, \; L]
$$

也就是说，只取与当前尺度更匹配的历史文本片段。

### 这一步的意义

动态窗口的目标不是增加文本量，而是降低文本失配：

$$
\text{让输入文本与当前样本真正需要的时间尺度对齐}
$$

---

## 3.3 Step 3：数值摘要与第一阶段检索

### 数值摘要

检索时先对历史序列形成一个数值摘要，包括：

$$
\text{last}, \quad \text{mean}, \quad \text{std}, \quad \text{slope}
$$

可以抽象记为：

$$
\mathbf{s}_{\text{num}} = \phi(\mathbf{x})
$$

### 第一阶段查询

若加入尺度提示，则第一阶段查询可以抽象写为：

$$
Q_1
=
\text{concat}
\big(
\text{domain},
\text{scale\_hint},
\mathbf{s}_{\text{num}},
\text{base\_text}
\big)
$$

然后执行检索：

$$
E_0 = \text{Retrieve}(Q_1, k_1)
$$

其中：

$$
k_1 = 9
$$

### 这一步的意义

第一阶段不是直接给最终文本，而是先拿到与当前样本及当前尺度相关的一批初始证据：

$$
E_0 = \{e_1, e_2, \dots, e_{k_1}\}
$$

---

## 3.4 Step 4：趋势假设生成

### 趋势假设

基于数值摘要与第一阶段证据，系统生成一个结构化趋势假设：

$$
z = \text{TrendHypothesis}(\mathbf{s}_{\text{num}}, E_0)
$$

其结构可以写为：

$$
z = \{
\text{direction},
\text{strength},
\text{volatility},
\text{key\_factors}
\}
$$

其中：

- `direction` 取值为 `up / down / flat`
- `strength` 取值为 `weak / moderate / strong`
- `volatility` 取值为 `low / medium / high`

### 这一步的意义

这一层不是最终预测，而是：

$$
\text{先把“检索到的材料”整理成一个中间趋势解释}
$$

它使系统从“找资料”转向“形成面向预测的结构化语义”。

---

## 3.5 Step 5：第二阶段检索

### 第二阶段查询

将第一阶段查询与趋势假设组合为第二阶段查询：

$$
Q_2 = \text{concat}(Q_1, z)
$$

然后执行：

$$
E_1 = \text{Retrieve}(Q_2, k_2)
$$

其中：

$$
k_2 = 3
$$

最终证据可记为：

$$
E_{\text{final}} = \text{Merge}(E_1, E_0)
$$

### 这一步的意义

第二阶段的作用是：

$$
\text{围绕趋势假设精炼文本证据，而不是继续泛化检索}
$$

也就是说，第一阶段回答“有哪些相关材料”，第二阶段回答“哪些材料最支持当前趋势解释”。

---

## 3.6 Step 6：形成文本条件与趋势先验

### 文本条件

最终组合文本记为：

$$
\text{composed\_text}
=
\psi(\text{base\_text}, E_{\text{final}}, z)
$$

这部分送入文本编码器，得到：

$$
\mathbf{c} = \text{TextEncoder}(\text{composed\_text})
$$

其中 $\mathbf{c}$ 即扩散模型中的文本上下文 `context`。

### 趋势先验向量

将结构化趋势假设解析成三维向量：

$$
\mathbf{p}_{\text{trend}} =
\begin{bmatrix}
d \\
s \\
v
\end{bmatrix}
$$

其中：

$$
d \in \{-1, 0, 1\}
$$

分别对应：

$$
\text{down}, \quad \text{flat}, \quad \text{up}
$$

并且：

$$
s \in \{0.5, 1.0, 1.5\}
$$

对应：

$$
\text{weak}, \quad \text{moderate}, \quad \text{strong}
$$

还有：

$$
v \in \{0.0, 0.5, 1.0\}
$$

对应：

$$
\text{low}, \quad \text{medium}, \quad \text{high}
$$

### 这一步的意义

同一个趋势假设被分成两条路径：

1. 语义路径：

$$
z \rightarrow \text{composed\_text} \rightarrow \mathbf{c}
$$

2. 结构化路径：

$$
z \rightarrow \mathbf{p}_{\text{trend}}
$$

前者服务于文本条件建模，后者服务于 router 和推理控制。

---

## 3.7 Step 7：条件扩散的主任务

### 带噪输入

在训练时，给定扩散步 $t$，对原始观测序列加噪：

$$
\tilde{\mathbf{x}}_t
=
\sqrt{\alpha_t}\,\mathbf{x}
+
\sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}
$$

其中：

$$
\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
$$

### 模型预测

扩散模型接收：

- 数值历史
- 掩码信息
- 时间特征
- 文本上下文 $\mathbf{c}$

输出预测：

$$
\hat{\mathbf{x}}_t = f_\theta(\tilde{\mathbf{x}}_t, \mathbf{c}, \text{side\_info})
$$

### 主损失

由于当前 `Economy` 配置中：

$$
\text{noise\_esti} = \text{false}
$$

因此模型直接预测数据本身而不是噪声，主损失写为：

$$
\mathcal{L}_{\text{main}}
=
\frac{
\left\|
(\mathbf{x} - \hat{\mathbf{x}}_t)\odot \mathbf{M}_{\text{target}}
\right\|_2^2
}{
\sum \mathbf{M}_{\text{target}}
}
$$

其中：

$$
\mathbf{M}_{\text{target}} = \mathbf{M}_{\text{observed}} - \mathbf{M}_{\text{cond}}
$$

### 这一步的意义

这是模型的主预测任务，本质仍然是条件扩散式时间序列预测。  
前面做的所有尺度自适应设计，最终都要服务于这一步能不能预测得更准。

---

## 3.8 Step 8：多尺度 horizon 切分

### band 定义

未来长度为：

$$
H = 12
$$

配置中的切分边界为：

$$
[1, 3, 6, 12]
$$

因此得到多个 horizon band：

$$
\mathcal{B}_1 = [1,1], \quad
\mathcal{B}_2 = [2,3], \quad
\mathcal{B}_3 = [4,6], \quad
\mathcal{B}_4 = [7,12]
$$

### band 残差

对第 $b$ 个 band，定义该区间残差：

$$
\mathbf{r}^{(b)} = \left(\mathbf{x} - \hat{\mathbf{x}}_t\right)^{(b)}
$$

### Huber 点损失

配置中使用 `Huber`，因此点损失为：

$$
\rho_\delta(r)
=
\begin{cases}
\frac{1}{2}r^2, & |r|\le \delta \\[4pt]
\delta |r| - \frac{1}{2}\delta^2, & |r| > \delta
\end{cases}
$$

其中：

$$
\delta = 1.0
$$

### 单样本 band 损失

对样本 $i$ 和 band $b$，定义：

$$
\ell_{i,b}
=
\frac{
\sum_{k,\tau \in \mathcal{B}_b}
\rho_\delta(r_{i,k,\tau}) \, M_{i,k,\tau}
}{
\sum_{k,\tau \in \mathcal{B}_b} M_{i,k,\tau} + \varepsilon
}
$$

### 这一步的意义

这一步把“预测未来 12 步”拆成多个时间尺度子任务，使得模型能够知道：

$$
\text{短期预测误差和长期预测误差不是一回事}
$$

---

## 3.9 Step 9：scale router 的输入特征

### 历史统计特征

对历史部分 $\mathbf{x}_{1:L}$，构造以下特征：

$$
f_1 = \frac{\text{signed\_slope}}{\text{volatility} + \varepsilon}
$$

$$
f_2 = \frac{|\text{signed\_slope}|}{\text{volatility} + \varepsilon}
$$

$$
f_3 = \frac{\text{volatility}}{\text{mean\_abs} + \varepsilon}
$$

$$
f_4 = \frac{\text{diff\_std}}{\text{volatility} + \varepsilon}
$$

$$
f_5 = \frac{\text{accel}}{\text{diff\_std} + \varepsilon}
$$

$$
f_6 = \log(1 + \text{mean\_abs})
$$

### 最终 router 输入

把趋势先验和文本可用性一起拼接：

$$
\mathbf{h}_i =
\left[
f_1, f_2, f_3, f_4, f_5, f_6,
\mathbf{p}_{\text{trend}},
m_{\text{text}}
\right]
$$

其中：

$$
m_{\text{text}} \in \{0,1\}
$$

表示该样本是否真的有有效文本条件。

### 这一步的意义

router 并不是只看原始文本，它看的是：

$$
\text{数值尺度特征} + \text{结构化趋势先验} + \text{文本可用性}
$$

这样做比直接基于自由文本判断尺度更稳定。

---

## 3.10 Step 10：router 如何输出 band 权重

### logits 与 softmax

设 router 网络为 $g_\theta$，温度为 $T$，则：

$$
\mathbf{z}_i = \frac{g_\theta(\mathbf{h}_i)}{T}
$$

band 权重为：

$$
w_{i,b}
=
\frac{\exp(z_{i,b})}{\sum_{b'} \exp(z_{i,b'})}
$$

因此：

$$
\sum_b w_{i,b} = 1
$$

### teacher regularization

代码中利用数据侧的粗标签 `scale_code` 构造 teacher 分布：

$$
\mathbf{q}_i = \text{Teacher}(\text{scale\_code}_i)
$$

然后加入：

$$
\mathcal{L}_{\text{teacher}}
=
\lambda_{\text{teacher}} \cdot \gamma_{\text{warmup}}
\cdot
\text{KL}
\left(
\log \mathbf{w}_i,
\mathbf{q}_i
\right)
$$

其中：

$$
\gamma_{\text{warmup}}
=
\max\left(
0,\,
1 - \frac{\text{step}}{\text{warmup\_steps}}
\right)
$$

### 熵正则

router 还带有熵项：

$$
\mathcal{L}_{\text{entropy}} = -\lambda_{\text{ent}} \, H(\mathbf{w}_i)
$$

其中：

$$
H(\mathbf{w}_i) = -\sum_b w_{i,b}\log w_{i,b}
$$

### 这一步的意义

teacher 项的作用是：

$$
\text{给 router 一个稳定的弱监督起点}
$$

熵项的作用是：

$$
\text{避免 router 过早塌缩到极端单一 band}
$$

---

## 3.11 Step 11：多尺度辅助损失如何加权

### 全局 band 权重

首先根据 band 的全局损失统计得到一组全局权重：

$$
\mathbf{w}^{\text{global}}
$$

### 样本权重与全局权重混合

router 给出样本级权重：

$$
\mathbf{w}^{\text{sample}}_i
$$

先混合为：

$$
\mathbf{w}^{\text{dyn}}_i
=
\alpha \mathbf{w}^{\text{sample}}_i
+
(1-\alpha)\mathbf{w}^{\text{global}}
$$

再与均匀分布混合：

$$
\mathbf{u} =
\left[
\frac{1}{B},\dots,\frac{1}{B}
\right]
$$

$$
\mathbf{w}^{\text{mix}}_i
=
(1-\eta)\mathbf{u}
+
\eta \mathbf{w}^{\text{dyn}}_i
$$

然后加入保底项：

$$
\mathbf{w}^{\text{final}}_i
=
(1-\beta)\mathbf{w}^{\text{mix}}_i
+
\beta \mathbf{u}
$$

最后重新归一化。

### 多尺度辅助损失

$$
\mathcal{L}_{\text{aux},i}
=
\sum_b
w^{\text{final}}_{i,b}
\ell_{i,b}
$$

整体辅助损失为：

$$
\mathcal{L}_{\text{aux}}
=
\frac{1}{N}
\sum_i
\mathcal{L}_{\text{aux},i}
+
\mathcal{L}_{\text{teacher}}
+
\mathcal{L}_{\text{entropy}}
$$

### 总训练目标

最终训练目标为：

$$
\mathcal{L}
=
\mathcal{L}_{\text{main}}
+
\lambda_{\text{mr}} \mathcal{L}_{\text{aux}}
$$

### 这一步的意义

这一部分实现的是：

$$
\text{同一个样本，不同 horizon 的重要性可以不同}
$$

因此模型不再用统一方式学习所有样本，而是按样本尺度偏好进行训练。

---

## 3.12 Step 12：推理前先选全局 guide weight

### 验证集扫描

设候选集合为：

$$
\mathcal{G}
=
\{0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4\}
$$

则在验证集上选择：

$$
g^\star
=
\arg\min_{g \in \mathcal{G}}
\text{MSE}_{\text{valid}}(g)
$$

当前 `Economy` 口径下选出的结果为：

$$
g^\star = 1.4
$$

### 这一步的意义

我们先确定一个“全局合理的 guidance 基准强度”，避免推理期完全失控。

---

## 3.13 Step 13：router 自适应 guidance

### router scale score

设 band 中心为：

$$
c_b \in [0,1]
$$

则 router 的尺度分数定义为：

$$
\text{scale\_score}_i
=
\sum_b w_{i,b} c_b
$$

### guide ratio

设 router guidance 系数为 $\lambda_r$，则：

$$
\text{guide\_ratio}_i
=
1 + \lambda_r(\text{scale\_score}_i - 0.5)
$$

再做上下界裁剪：

$$
\text{guide\_ratio}_i
=
\text{clip}
\left(
\text{guide\_ratio}_i,
r_{\min},
r_{\max}
\right)
$$

### 样本级 guidance

最终样本级 guidance 为：

$$
g_i = \text{guide\_ratio}_i \cdot g^\star
$$

### CFG 采样

设条件预测为 $\hat{\mathbf{x}}^{\text{cond}}$，无条件预测为 $\hat{\mathbf{x}}^{\text{uncond}}$，则：

$$
\hat{\mathbf{x}}_i
=
\hat{\mathbf{x}}^{\text{uncond}}_i
+
g_i
\left(
\hat{\mathbf{x}}^{\text{cond}}_i
-
\hat{\mathbf{x}}^{\text{uncond}}_i
\right)
$$

### 这一步的意义

这一步是 `Economy` 版本最重要的结论之一：

$$
\text{推理期的自适应 guidance 来自 router，而不是 CoT 直接控制}
$$

也就是说，`CoT` 的作用是帮助形成 `trend_prior` 与文本条件；  
真正决定每个样本 guidance 强度的是 router 对时间尺度的判断。

---

## 3.14 Step 14：最终输出与实验验证

### 输出

模型最终输出：

$$
\hat{\mathbf{y}}_{1:H}
$$

并统计：

- 整体 `MSE`
- 整体 `MAE`
- 逐 horizon 指标
- 逐 band 指标
- router 权重分布
- `guide_ratio`
- `sample_guide`

### 这一步的意义

这一层不是简单看最终误差，而是要验证：

$$
\text{时间尺度自适应是否真的在输入、训练、推理三个层面同时起作用}
$$

---

## 4. 如何把它讲成一个完整故事

下面这段可以直接作为汇报时的口语版主线。

### 第一段：问题提出

> 我们研究 `Economy` 任务时发现，预测误差并不只是来自模型容量不够，而是来自样本之间的时间尺度差异。  
> 一部分样本主要是短期扰动主导，一部分样本则明显更受长期趋势影响。  
> 但传统做法对所有样本都使用固定文本窗口、固定训练目标、固定 guidance 强度，这会带来系统性失配。

### 第二段：核心假设

> 因此我们提出一个核心假设：样本级时间尺度是统一驱动文本组织、训练优化和推理控制的关键变量。  
> 如果先识别样本更偏短期还是长期，再让后续模块围绕这个尺度工作，预测会更稳定、更合理。

### 第三段：输入构造

> 基于这个假设，我们先从历史数值中估计当前样本的时间尺度。  
> 这个尺度判断首先用来动态选择文本窗口，然后进入两阶段 RAG。  
> 第一阶段检索找到相关材料，第二阶段先形成趋势假设，再围绕这个趋势假设精炼证据。  
> 最终我们得到两类结果：一类是 `composed_text`，作为文本条件进入扩散模型；另一类是 `trend_prior`，作为结构化趋势先验服务于后续 router。

### 第四段：训练优化

> 但我们认为，时间尺度信息不能只用于文本构造。  
> 如果样本本身更偏短期，那么训练时关注的 horizon 也应该更偏短期。  
> 所以我们把未来 12 步拆成多个 horizon band，并引入 `scale router`，让它根据历史统计特征、trend prior 和文本可用性，为不同样本动态分配 band 权重。  
> 这样，多尺度辅助损失就不再是固定加权，而是样本级加权。

### 第五段：推理控制

> 更进一步，我们不希望 router 学到的尺度偏好只停留在训练里。  
> 因此在推理阶段，我们先在验证集选择一个全局最优的 `guide_w`，再让 router 根据当前样本的尺度偏好，把这个全局值缩放成样本级 `sample_guide`。  
> 所以 `Economy` 版本不是 CoT 直接控制 guidance，而是 CoT 先形成 trend prior，再由 router 完成最终的样本级 guidance 调节。

### 第六段：闭环结论

> 最终，时间尺度这个核心变量被统一地用于文本组织、训练优化和推理控制。  
> 因此这不是几个独立模块的简单拼接，而是一个围绕时间尺度自适应构建起来的完整科研闭环。

---

## 5. 最终闭环总结

如果把整套方法压缩成一条最核心的逻辑链，那么就是：

$$
\text{不同样本的时间尺度不同}
\Rightarrow
\text{文本应按尺度组织}
\Rightarrow
\text{训练应按尺度分配 horizon 关注重点}
\Rightarrow
\text{推理应按尺度调整 guidance 强度}
$$

因此 `Economy` 版本的核心贡献不应表述为：

$$
\text{RAG} + \text{CoT} + \text{Diffusion}
$$

而应表述为：

$$
\text{以时间尺度自适应为主线，统一输入构造、训练优化与推理控制}
$$

更完整地写为：

$$
\text{历史序列}
\rightarrow
\text{时间尺度识别}
\rightarrow
\text{动态文本窗口}
\rightarrow
\text{两阶段 RAG / 趋势假设}
\rightarrow
\text{composed text + trend prior}
\rightarrow
\text{条件扩散主任务}
\rightarrow
\text{scale router 多尺度加权}
\rightarrow
\text{router 自适应 guidance}
\rightarrow
\text{最终预测}
$$

这就是 `Economy` 版本最完整、最统一、也最适合答辩和汇报的科研故事。
