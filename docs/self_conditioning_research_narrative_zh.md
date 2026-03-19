# Self-Conditioning 改动的完整科研逻辑叙事（中文）

## 一、研究背景：为什么还要继续改扩散模型

当前项目的核心目标不是单纯做一个新的时间序列预测模型，而是在 **MCD-TSF 现有多模态条件扩散框架** 上，寻找一种**低风险、可解释、可复现**的小改动，使模型在不破坏既有系统结构的前提下获得更好的预测性能。

具体而言，当前系统已经具备以下几个较强的组成部分：

- 扩散式时间序列生成主干；
- 文本条件输入与多模态融合；
- RAG / CoT / trend prior 等外部条件增强；
- classifier-free guidance（CFG）；
- 基于多尺度 band 的 router guide；
- 多分辨率辅助损失。

这意味着一个现实问题：

> 模型已经不是“纯扩散模型”，而是一个高度耦合的条件扩散预测系统。

因此，任何对 diffusion core 的改动，都不再只是“局部优化”，而很可能会改变整个系统的训练动力学、条件利用方式以及采样行为。也正因如此，后续扩散侧改动的研究策略不能再走“大改结构”的路线，而必须强调：

1. **改动尽量小**；
2. **因果归因尽量清晰**；
3. **实验结论必须可解释**；
4. **优先保留现有系统优势，而不是重写扩散范式**。

这就是本轮研究的出发点。

---

## 二、前期探索：为什么最终会走到 Self-Conditioning

在进入 self-conditioning 之前，已经做过若干类常见扩散增强尝试。它们的共同特征是：

- 在图像扩散领域或通用 diffusion literature 中被证明有效；
- 但在当前 MCD-TSF 上，并不一定成立。

### 2.1 Min-SNR weighting

最先尝试的是 **Min-SNR timestep weighting**。这类方法的逻辑是：

- 不同 diffusion timestep 的训练目标存在冲突；
- 通过对不同 timestep 的 loss 进行重加权，可以提高训练效率与收敛质量。

但实验结果表明，该方法在当前系统中显著退化。结合实验表现，可以推断：

- 当前模型不是一个单纯的 epsilon-pred diffusion model；
- 其条件建模、router guide、CFG 与扩散过程已经耦合；
- 对 timestep loss 再加权，会直接改变已有条件系统的平衡。

因此，Min-SNR 这条线虽然在通用 diffusion 训练中成立，但在本项目中不适合作为低风险增益方案。

### 2.2 Cosine noise schedule

随后测试了 **cosine schedule**。从结果上看：

- 相比原始 `quad` schedule，有一定退化；
- 退化幅度没有 Min-SNR 那么严重；
- 但仍未优于 baseline。

这说明：

- 当前模型对噪声日程本身也比较敏感；
- 更“标准”的 diffusion schedule，不一定更适合当前数据和条件机制。

### 2.3 综合判断

这一阶段最重要的认识是：

> 当前模型不适合继续优先尝试“改变训练目标”或“改变整体噪声动力学”的扩散改法。

于是，研究策略转向寻找一种更温和的增强方式：

- 不改 loss；
- 不改 schedule；
- 不改采样公式；
- 不改条件系统；
- 只在 denoiser 输入侧增加一个更稳定的辅助信息。

在这样的约束下，**x0 self-conditioning** 成为了最自然的候选。

---

## 三、研究假设：为什么 Self-Conditioning 值得尝试

Self-conditioning 的基本思想是：

- 模型在当前 diffusion step 去预测 `x0`；
- 再把上一步或预估得到的 `x0` 作为额外条件输入，帮助模型当前步更稳定地完成去噪。

对于当前项目，这个想法有三个关键吸引力。

### 3.1 它不改变训练目标

与 Min-SNR 不同，self-conditioning 不会改变 loss 的定义，也不会对 timestep 重新加权。因此它不直接动训练优化目标本身。

### 3.2 它更贴合当前 `x0-pred` 路线

当前系统的实现本身就是偏向 **data prediction / x0 prediction** 的路径，而不是完全标准的 epsilon-pred 路径。因此，self-conditioning 在逻辑上与现有预测目标更一致。

### 3.3 它是“输入增强”而不是“范式替换”

它做的事情不是推翻现有 diffusion core，而是给 denoiser 增加一个额外的上下文：

- 当前条件观测；
- 当前 noisy target；
- 上一步自己的 `x0` 草稿。

在研究策略上，这属于一种更稳健的探索方式：

> 先增强局部去噪一致性，再观察是否能在不破坏全局系统的前提下带来收益。

---

## 四、实验设计：如何保证比较是干净的

为了让实验结论具有可解释性，本轮 self-conditioning 实验刻意保持了对比的“干净性”。

### 4.1 对照组（baseline）

使用原始 Economy `36 -> 12` 主线实验作为对照：

- 路径：`save/forecasting_Economy_20260312_194619`
- diffusion schedule：`quad`
- sampling method：`quad`
- 无 self-conditioning

### 4.2 实验组（self-conditioning）

在 baseline 基础上，只加入 self-conditioning：

- 路径：`save/forecasting_Economy_20260319_122736`
- diffusion schedule：仍然是 `quad`
- sampling method：仍然是 `quad`
- 开启：
  - `self_condition=true`
  - `self_condition_prob=0.5`
  - `self_condition_target_only=true`

对应配置记录见：

- `save/forecasting_Economy_20260319_122736/config_results.json:61`
- `save/forecasting_Economy_20260319_122736/config_results.json:62`
- `save/forecasting_Economy_20260319_122736/config_results.json:63`

因此，本轮实验可以视为一个较为标准的单变量对照：

> 对比的核心变量就是“是否加入 self-conditioning”。

### 4.3 为什么只用 Economy 36 -> 12

当前项目的主报告、指标分析与方法比较，本身就是围绕 Economy `36 -> 12` 展开的。因此，在这一阶段继续使用同一任务设置，有两个好处：

1. 能直接与已有基线结果对齐；
2. 结论不会被数据集切换、horizon 切换等额外因素干扰。

---

## 五、实现逻辑：Self-Conditioning 在代码里是怎么工作的

Self-conditioning 的实现遵循“最小侵入”原则。

### 5.1 训练阶段

训练时，模型先做一次 no-grad 的预估：

- 用当前 noisy input 做一次 preview forward；
- 得到一个 preview `x0`；
- 将这个 preview `x0.detach()` 作为 self-condition；
- 再进入正式 forward 计算 loss。

而且并不是每个 batch 都强制使用 self-conditioning，而是按：

- `self_condition_prob = 0.5`

来随机触发。这样做的目的是：

- 防止模型过度依赖 self-condition；
- 保留原始两通道输入路径的鲁棒性。

### 5.2 采样阶段

采样时，每个 reverse diffusion step 都会把上一步得到的 `predicted x0` 反馈给下一步，形成递归自条件输入。

### 5.3 为什么只对 target 区域做 self-condition

配置中设置了：

- `self_condition_target_only=true`

这样做的考虑是：

- 条件观测区已经由 `cond_obs` 显式提供；
- self-condition 的主要价值在于辅助 target 区域生成；
- 若把 self-condition 同时覆盖到 observed 区域，容易引入冗余信息，甚至干扰已有条件通道。

因此，当前实现中 self-condition 更像是：

> 对未来预测区间的一种“自回顾草稿”。

---

## 六、实验结果：验证集和测试集给出了什么信号

## 6.1 验证集结果：显著改善

baseline 的最优验证结果为：

- `guide_w = 1.4`
- `best_valid_mse = 0.14063129197983515`

见：

- `save/forecasting_Economy_20260312_194619/selected_guide_w.json:3`
- `save/forecasting_Economy_20260312_194619/selected_guide_w.json:4`

self-conditioning 的最优验证结果为：

- `guide_w = 0.5`
- `best_valid_mse = 0.12176733925229027`

见：

- `save/forecasting_Economy_20260319_122736/selected_guide_w.json:3`
- `save/forecasting_Economy_20260319_122736/selected_guide_w.json:4`

即：

- 验证集 MSE 从 `0.140631` 下降到 `0.121767`
- 绝对改善 `0.018864`
- 相对改善约 `13.4%`

这是一个明显且不可忽略的提升。

### 6.1.1 更深一层的现象：整条 valid sweep 都变强了

不仅最优点更好，而且几乎整个 `guide_w` sweep 区间都表现更优：

baseline：

- `0.4 -> 0.193345`
- `0.5 -> 0.182690`
- `0.6 -> 0.173504`
- `0.7 -> 0.165634`
- `0.8 -> 0.159071`
- `0.9 -> 0.153699`
- `1.0 -> 0.149342`
- `1.2 -> 0.143331`
- `1.4 -> 0.140631`

self-conditioning：

- `0.4 -> 0.122042`
- `0.5 -> 0.121767`
- `0.6 -> 0.122505`
- `0.7 -> 0.124245`
- `0.8 -> 0.126940`
- `0.9 -> 0.130513`
- `1.0 -> 0.134903`
- `1.2 -> 0.146010`
- `1.4 -> 0.160056`

这说明 self-conditioning 并不是偶然在一个 guide 点上碰巧更好，而是真的改变了模型的验证期行为。

同时，它还带来了一个很有价值的变化：

> 最优 `guide_w` 从 `1.4` 降到了 `0.5`。

这意味着：

- 新模型对条件已经更敏感；
- 不再需要很强的外部 guidance；
- 从系统角度看，self-conditioning 提高了 denoiser 对条件信息的内部利用效率。

---

## 七、测试集结果：总体略差，但不是全面退化

baseline 测试集结果为：

- `MSE = 0.2496992787744245`
- `MAE = 0.3801771555191431`

见：

- `save/forecasting_Economy_20260312_194619/eval_metrics_guide_1p4.json:4`
- `save/forecasting_Economy_20260312_194619/eval_metrics_guide_1p4.json:5`

self-conditioning 测试集结果为：

- `MSE = 0.26061055599114835`
- `MAE = 0.3976185138408954`

见：

- `save/forecasting_Economy_20260319_122736/eval_metrics_guide_0p5.json:4`
- `save/forecasting_Economy_20260319_122736/eval_metrics_guide_0p5.json:5`

因此：

- 测试 MSE 增加了 `0.010911`
- 相对 baseline 约差 `4.37%`
- 测试 MAE 增加了 `0.017441`
- 相对 baseline 约差 `4.59%`

这说明：

> 验证集收益并没有完整迁移到测试集整体指标上。

但这里必须强调一个关键点：

- 这不是那种“大幅掉点”的失败；
- 它更像是一个**有明确结构性收益，但伴随远期代价**的 trade-off。

---

## 八、为什么验证集显著变好，但测试总体没有同步提升

这个问题是本轮实验最核心的科学问题。

如果只看 aggregate 指标，很容易得到一个过于粗糙的结论：

- valid 好了；
- test 差了一点；
- 那就是过拟合了。

但从更细粒度结果看，真实情况比“简单过拟合”更复杂。

### 8.1 Horizon 级别分析

相对 baseline，self-conditioning 的 horizon MSE 变化如下：

- `h1`: 改善
- `h2`: 略差
- `h3`: 改善
- `h4`: 改善
- `h5`: 改善
- `h6`: 基本持平
- `h7`: 变差
- `h8`: 变差
- `h9`: 略差
- `h10`: 变差
- `h11`: 变差
- `h12`: 明显变差

也就是说：

- `1-6` 这段总体更好；
- `7-12` 这段总体更差。

### 8.2 Band 级别分析

band MSE 对比更清晰：

baseline：

- `1`: `0.119713`
- `2-3`: `0.146453`
- `4-6`: `0.206207`
- `7-12`: `0.327525`

self-conditioning：

- `1`: `0.117297`
- `2-3`: `0.142054`
- `4-6`: `0.201880`
- `7-12`: `0.353380`

可以看出：

- 短中期 band 全部变好；
- 只有长尾 `7-12` 明显变差。

### 8.3 这意味着什么

这意味着 self-conditioning 的真实作用并不是“整体增强”或“整体削弱”，而是：

> 它改善了局部去噪一致性和近程预测稳定性，但同时削弱了长程外推能力。

从时间序列预测角度理解，这种现象是合理的：

- self-conditioning 会让模型更依赖“上一步自己的草稿”；
- 这种机制对短距离预测有帮助，因为局部结构更连续；
- 但当预测 horizon 拉长时，模型反复参考自身先前的估计，容易逐步变得保守，或者把误差累积到远期。

因此，本轮结果更像是一个**预测范围上的再分配**：

- 把短中期做得更稳了；
- 代价是长期尾部略受损。

---

## 九、Guide 与 Router 现象：为什么最优引导权重会大幅下降

一个非常值得记录的实验事实是：

- baseline 最优 `guide_w = 1.4`
- self-conditioning 最优 `guide_w = 0.5`

而 router 相关统计并没有发生剧烈异常：

baseline：

- `mean_guide_ratio = 0.819914`
- `mean_scale_score = 0.199857`

self-conditioning：

- `mean_guide_ratio = 0.827034`
- `mean_scale_score = 0.211723`

差异并不大。

真正变化明显的是：

- `mean_sample_guide_w`
  - baseline：`1.147880`
  - self-conditioning：`0.413517`

这说明什么？

说明问题不在 router 本身“坏了”，而在于：

> self-conditioning 改变了模型对条件信号的吸收方式，使得模型本身已经更容易被条件驱动，因此无需再施加很强的外部 guidance。

从科研叙事角度，这是一个很重要的结果，因为它意味着：

- self-conditioning 不是简单重复条件信息；
- 它的确改变了 denoiser 对条件上下文的利用方式；
- 这种变化是真实存在且可以通过最优 `guide_w` 的迁移观察到的。

---

## 十、如何正确评价这次实验

如果只看一句话结论，这次实验最准确的表述不是：

- “成功了”，也不是
- “失败了”。

更准确的说法应该是：

> Self-conditioning 是一个**部分成功**的扩散侧改动。

### 10.1 为什么说它成功

因为它确实带来了以下真实收益：

- 验证集显著提升；
- 短中期 horizon 提升；
- 模型对条件更敏感，guide 需求下降；
- 它不是随机抖动，而是有稳定结构性信号。

### 10.2 为什么又不能直接说它优于 baseline

因为目前最终 test aggregate 指标仍然没有超过 baseline：

- `0.249699` vs `0.260611`

只要整体测试 MSE 还没赢，就不能宣称它是最终替代方案。

### 10.3 它的真正价值是什么

这次实验真正有价值的地方在于：

1. 它证明当前系统并不是“所有 diffusion 改动都不工作”；
2. 它提供了一个清晰的方向：
   - 近端预测可以被进一步稳定化；
   - 但要同时照顾远端 horizon；
3. 它帮助我们把后续研究问题重新聚焦为：
   - **如何保留 `1-6` 的收益，同时修复 `7-12` 的退化？**

这比简单得到一个“全输”或“全赢”的结果，更具有研究价值。

---

## 十一、下一步研究问题应该怎么提

如果延续这条研究线，下一步不应该再问：

> self-conditioning 有没有用？

因为这个问题已经被回答了：

- 有用，但不是全面用。

下一步更合理的问题是：

> 如何在保留 self-conditioning 对短中期去噪稳定性的帮助下，补偿其对长程预测的不利影响？

这个问题可以进一步分解为三种后续方向：

### 11.1 方向一：只在训练阶段使用 self-conditioning

即：

- 训练时保留 self-conditioning，帮助模型学会更稳定的局部 denoising；
- 采样时关闭或弱化 self-conditioning，避免远期误差累积。

这是最直接验证“训练收益是否能保留、采样副作用是否能减轻”的办法。

### 11.2 方向二：对采样阶段做 step-aware / horizon-aware 调节

例如：

- 在前半程 reverse step 中使用 self-conditioning；
- 后半程减弱它；
- 或针对更远 horizon 区域补一个更强的 guidance schedule。

其核心目标是：

- 不让 self-conditioning 的递归反馈在长程外推时持续累积误差。

### 11.3 方向三：把 self-conditioning 作为“近端增强器”而不是全局替代器

也就是说，研究目标从“全面替代 baseline”变成：

- 针对短中期场景取得更优结果；
- 在需要更长 horizon 时再用其他机制做补偿。

这种思路在实际应用上也成立，因为不同 horizon 的业务价值本来就可能不同。

---

## 十二、最终科研叙事总结

本轮研究从一个明确的问题出发：

- 在当前多模态条件扩散预测系统中，是否存在一种**足够小、足够稳、但又真实有效**的 diffusion-side 改动？

前期实验表明：

- 改 loss weighting（Min-SNR）不行；
- 改 noise schedule（cosine）也不理想；
- 说明当前系统不适合优先从训练目标和全局噪声动力学入手。

因此，研究策略转向更低侵入的输入增强方案，即 `x0 self-conditioning`。

实验结果表明：

- self-conditioning 在验证集上显著优于 baseline；
- 它系统性地改善了 guide sweep，并将最优 `guide_w` 从 `1.4` 拉低到 `0.5`；
- 它确实增强了模型对条件信息的内部利用能力；
- 但这种收益主要集中在短中期 horizon；
- 在 `7-12` 区间，长程外推能力有所下降；
- 因而最终 test aggregate MSE 尚未超过 baseline。

因此，本轮实验最重要的科研结论不是“self-conditioning 成功替代 baseline”，而是：

> self-conditioning 揭示了当前系统中一个真实存在的结构性机会：
> 我们可以通过改善局部去噪一致性来显著提升验证表现和近端预测，但若想把收益迁移到最终测试集，还需要额外机制去控制长程预测误差的累积。

这使后续研究从“盲目尝试通用 diffusion trick”，推进到了一个更聚焦、更有方向感的问题：

> 如何把 self-conditioning 的短中期收益保留下来，并通过采样调度或远期补偿机制修复其长尾退化？

从科研逻辑上讲，这正是一次有价值的实验推进：

- 它没有直接给出最终最优解；
- 但它显著缩小了问题空间；
- 同时给出了明确、可执行、可继续深化的下一步研究方向。

---

## 十三、可直接用于报告的精炼结论

在保持原始 `quad` 扩散日程不变的前提下，我们进一步引入了 `x0 self-conditioning` 机制，希望以最小侵入方式增强扩散模型的局部去噪一致性。实验结果表明，该方法在验证集上带来了显著收益：最优验证 MSE 从 `0.140631` 降至 `0.121767`，相对提升约 `13.4%`，同时最优引导权重从 `1.4` 下移至 `0.5`，说明模型对条件信息的内部利用能力增强。然而，这一收益没有完全迁移到测试集整体指标上：测试 MSE 从 `0.249699` 小幅上升至 `0.260611`。进一步的 horizon 分析发现，self-conditioning 对 `1-6` 区间具有稳定增益，但在 `7-12` 区间出现退化，说明其主要提升了近端预测稳定性，却削弱了远端外推能力。因此，self-conditioning 不是当前 baseline 的直接替代方案，但它揭示了一条明确的后续研究方向：通过保留其短中期收益，并进一步设计长程补偿机制，有望在未来实现更完整的性能提升。
