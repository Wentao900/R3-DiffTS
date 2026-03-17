# 完整逻辑：如何清晰讲清楚整个结构

## 1. 先讲问题

这套方法要解决的不是“怎么把文本塞进模型”，而是两个更具体的问题：

1. 当前样本应该参考什么文本信息？
2. 当前样本在训练和推理时，应该更关注哪个时间尺度？

所以方法必须同时解决：
- 文本条件构造
- 时间尺度控制

这就是双分枝结构存在的原因。

---

## 2. 共享起点

所有东西都从同一个样本出发：

- 历史数值序列：$X_i$
- 时间信息：$U_i$
- 原始文本：$T_i^{raw}$

先只看历史数值，提取时间尺度表征：

$$
z_i = \Phi(X_i)
$$

这个 $z_i$ 的作用是判断当前样本更偏：
- 短期扰动
- 中期过渡
- 长期趋势

**这一步是共享前端。后面两条分枝都建立在这个尺度认知上。**

---

## 3. 第一条分枝：文本构造分枝

这条分枝只负责一件事：

> 给模型构造高质量文本条件

它的逻辑是：

### 第一步：由尺度决定文本窗口和检索提示

$$
s_i^{rag} = G_{rag}(z_i), \qquad \ell_i = \Gamma(s_i^{rag})
$$

含义：
- 如果样本更偏短期，就看更短的文本窗口
- 如果样本更偏长期，就看更长的文本窗口
- 同时把 short / mid / long 的尺度提示注入 query

### 第二步：做尺度感知 RAG

$$
\mathcal D_i^{rag} = \operatorname{Retrieve}(\Psi_q(X_i,U_i,T_i^{raw},s_i^{rag}))
$$

含义：
- 从与当前样本更匹配的文本范围里检索证据
- 不是统一检索，不是所有样本都用同一套文本条件

### 第三步：做 CoT 推理摘要

$$
c_i = \Psi_{cot}(X_i,U_i,\mathcal D_i^{rag})
$$

含义：
- 把检索到的零散证据整理成结构化推理结果

### 第四步：形成统一文本条件

$$
E_i^{upd} = \operatorname{Compose}(T_i^{raw}, S_i^{num}, s_i^{rag}, \mathcal D_i^{rag}, c_i)
$$

这条分枝的最终输出是：

$$
E_i^{upd}
$$

也就是：**扩散模型真正使用的文本条件**。

---

## 4. 第二条分枝：控制分枝

这条分枝只负责一件事：

> 根据样本的时间尺度结构，控制训练和推理

它的逻辑是：

### 第一步：从文本摘要里提取趋势先验

CoT 摘要不会停留在文本层，还会被解析成：

$$
\pi_i = \Omega(c_i)
$$

通常是：
- direction
- strength
- volatility

这一步非常关键，因为它把文本分枝的结果传给控制分枝。

### 第二步：router 读取控制特征

router 的输入不是原始文本，而是：

- 历史序列统计特征
- 趋势先验 $\pi_i$
- 文本可用性标记 $m_i$

记为：

$$
\tilde z_i = \operatorname{Fuse}(X_i,\pi_i,m_i)
$$

### 第三步：输出样本级尺度分布

$$
r_i^{router} = \operatorname{softmax}(G_{router}(\tilde z_i))
$$

含义：
- 当前样本更应该关注短期 band
- 还是长期 band
- 或者按混合比例分配

### 第四步：训练时控制多分辨率损失

先把预测窗口拆成多个 horizon band，分别算误差：

$$
L_{i,b}
$$

然后 router 给每个 band 分配权重：

$$
\mathcal L_{aux,i} = \sum_b w_{i,b} L_{i,b}
$$

总训练目标是：

$$
\mathcal L = \mathcal L_{main} + \lambda_{mr}\mathcal L_{aux}
$$

含义：
- 不同样本，训练时关注的 horizon 不一样
- 这是样本级时间尺度自适应训练

### 第五步：推理时控制 guidance 强度

router 还会进一步给出样本级 guidance：

$$
g_i = g \cdot \operatorname{clip}(1+\alpha(\sigma_i-0.5), \rho_{min}, \rho_{max})
$$

所以这条分枝的最终输出是两类控制量：

- 训练时的 band 权重
- 推理时的 guidance 强度 $g_i$

---

## 5. 两条分枝怎么汇合

两条分枝最终都汇入条件扩散主干。

### 文本分枝提供

$$
E_i^{upd}
$$

### 控制分枝提供

- 训练时：$\mathcal L_{aux}$
- 推理时：$g_i$

### 扩散主干负责最终预测

训练时，扩散主干学习：

$$
f_\theta(Y_{i,k}, X_i, U_i, E_i^{upd})
$$

推理时，最终预测是：

$$
\hat Y_{i,k}^{(g_i)}
=
f_\theta(Y_{i,k},X_i,U_i,\varnothing)
+
g_i \Big(
f_\theta(Y_{i,k},X_i,U_i,E_i^{upd})
-
f_\theta(Y_{i,k},X_i,U_i,\varnothing)
\Big)
$$

所以最终关系非常清楚：

- 文本分枝决定：**给模型什么文本条件**
- 控制分枝决定：**模型如何训练、推理时多大程度相信文本**
- 扩散主干决定：**如何把这两者融合成最终预测**

---

## 6. 为什么一定要讲成双分枝

因为这两个任务本来就不同：

### 文本构造分枝解决的是

$$
\text{What to use}
$$

也就是：
- 用什么文本
- 检索什么证据
- 生成什么条件

### 控制分枝解决的是

$$
\text{How to use}
$$

也就是：
- 训练时重视哪个 horizon
- 推理时给文本多大权重

如果你把它讲成一条直线，就会把：
- 文本构造
- 训练控制
- 推理控制

这三种职责混在一起，逻辑会乱。

---

## 7. 最终一句话版本

你直接这么讲：

> 整个方法先从历史序列中提取共享的时间尺度表征；基于这个尺度表征，一条分枝负责文本构造，即动态选择文本窗口、执行 RAG 检索、做 CoT 推理并形成统一文本条件；另一条分枝负责控制，即结合趋势先验和文本可用性，通过 scale router 生成样本级 horizon 权重和推理 guidance 强度；最后，文本条件和控制信号共同作用于条件扩散模型，得到最终预测。

---

## 8. 最后给一张最简结构式

$$
(X_i,U_i,T_i^{raw})
\rightarrow
z_i
\rightarrow
\begin{cases}
\text{文本构造分枝: } s_i^{rag} \rightarrow \mathcal D_i^{rag} \rightarrow c_i \rightarrow E_i^{upd} \\
\text{控制分枝: } \pi_i,m_i \rightarrow r_i^{router} \rightarrow \mathcal L_{aux}, g_i
\end{cases}
\rightarrow
f_\theta
\rightarrow
\hat Y_i
$$
