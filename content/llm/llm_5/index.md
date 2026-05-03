+++
title = "Decoding 总览：从「分布」到「生成」"
description = "模型只生产分布，把分布变成 token 的，是 decoding policy。"
date = 2026-05-03
tags = ["LLM", "Decoding", "采样", "Autoregressive"]
categories = ["LLM"]
showTableOfContents = true
+++

{{< katex >}}

![分布作为模型输出](feature-distribution-output.png)

> 前置知识提示：读这篇前，建议了解：next-token、softmax、概率分布（见第 1、4 篇）。

> 模型只生产分布，把分布变成 token 的，是 decoding policy。

---

打开 ChatGPT 问一个问题，文字是一段一段流出来的。看上去模型在「写答案」——一个 token 接一个 token 蹦出来，组成完整的句子。

但拉到底层看，模型在每一步**根本没有挑过 token**。它在第 t 步交出的，是一个铺满整个词表的概率分布：每一个 token——「的」「是」「明天」「🍌」乃至专门表示「我说完了」的特殊 token——都被分配了一个概率。

这一篇要讲的，就是从「分布」到「生成」之间发生的事。它有一个统一的名字：decoding。

### 模型只产分布，不产 token

先把第一个直觉夯实：模型给出的不是一个 token，是一个分布。

> 模型像一个概率气象台。它不告诉你「明天下雨」，它告诉你「下雨 70%，多云 20%，晴 10%」。要不要带伞，是你这边的决策。

把这个直觉放到上一篇我们已经熟悉的链路里：

```
hidden state ─lm_head─▶ logits ─softmax─▶ probs（分布）
```

这个分布写作 \(P(x_t \mid x_{\lt t})\) —— 给定上下文之后，下一个 token 是某个候选 \(x_t\) 的概率。它的维度等于词表大小 \(|V|\) —— 对今天主流模型来说，大约在 5 万到 25 万之间。所以模型每一步交出的，是一个 5 万到 25 万维的概率向量，**不是一个 token**。

举个具体例子。把「今天天气」喂给模型，它给出的分布大概长这样：

| token | 概率 |
| --- | --- |
| 真 | 0.42 |
| 很 | 0.18 |
| 不 | 0.10 |
| 挺 | 0.08 |
| 特别 | 0.05 |
| ……剩下 5 万多个 | 合计 0.17 |

到这里，模型的工作做完了。它把它能给的东西全摆在桌上。

要把这桌候选变成一个具体的 token——是「真」？是「很」？还是干脆按掷骰子的方式抽一个？——是另一个完全独立的环节。这个环节就叫 **decoding**。

> 模型负责把分布算出来；把分布变成 token，归 decoding policy 管。两件事是分开的。

![分布作为模型输出](feature-distribution-output.png)

*图：给定上下文「今天天气」，模型给出的不是某个 token，而是整个词表上的概率分布。decoding 在这之后才发生。*

### 自回归循环：选一个、追加上去、再预测一次

第一节讲的是「单步」。但生成一句话需要很多步——所以第二个问题来了：

第 t 步选完了一个 token，怎么进入第 t+1 步？

答案出乎意料地朴实：**把刚选的那个 token 接到上下文末尾，让模型重新做一次完整的预测。**

> 像一个一边写一边回头看的作者：写完一个字，他会把前面所有写过的字（包括刚写的这个）从头扫一遍，再决定下一个字。模型在生成时干的就是这件事。

形式化一下。整个生成过程是一个循环，每一轮包含四步（先写最朴素的版本，logits 上的处理细节下面会展开）：

```python
while not stop:
    logits  = model(context).logits[:, -1, :]   # 1. 取最后一个位置的 logits
    logits  = apply_processors(logits)          # 2. temperature / top-k / top-p / 重复惩罚
    token   = decoding_policy(logits)           # 3. argmax，或 softmax 后采样
    context = context + [token]                 # 4. 追加回上下文
```

这四步合起来就是 **autoregressive loop**——「自回归循环」。「自回归」的意思是：模型对下一个 token 的预测，依赖于已有上下文——其中既包括输入的 prompt（system / user message、检索回来的资料、工具返回值），也包括模型自己之前生成的 token。

第 2 步的 `apply_processors` 在朴素版本里可以省略——直接对原始 logits 做 argmax 就是 greedy。但工业实现几乎都在这一步插一层 **logits processors**：温度（在 logits 上除一个数）、重复惩罚（把已经出现过的 token 的 logit 减一点）、top-k / top-p（把不在候选集里的 logit 设为 \(-\infty\)）。这些都是改 **logits / 分布形状**，不是改「如何从分布里挑」——下一篇会逐个拆。

这个循环看起来简单，但有几个被低估的细节。

#### 每生成一个 token 都需要一次 decode step

推理在工程上分成两段：**prefill** 一次性把整个 prompt 喂进去，把每个位置的 K、V 算出来缓存好；**decode** 阶段每生成一个新 token，只对这一个新 token 做增量前向——历史 K、V 直接从缓存里读，不重新计算。

所以严格说，「每一步都是一次完整的前向」是不准确的：语义上模型确实依赖完整 context，但计算上 decode 走的是增量路径，省掉了对历史 token 的重算。这层优化省下了海量算力，也是后面「KV cache 是推理系统核心资产」的源头。

但有一件事不变：**每生成一个 token，就需要一次 decode step**。所以输出越长，步数越多，**生成延迟和输出长度直接成正比**。这是 LLM 推理和普通模型推理（一次 forward 拿到全部输出）最不一样的地方。

#### 循环必须有出口

一个不会停下来的循环就是死循环。生成的终止条件通常是三选一里任何一个先触发：

- **EOS（end-of-sequence）**：词表里有一个特殊 token，意思是「我说完了」。模型自己学会在合适的位置预测它，policy 选中后整个循环退出。这是最自然的停止方式。
- **stop sequence**：调用方手动指定的字符串。比如做对话续写时设 `stop=["\nUser:"]`，一旦生成出现这个序列就立刻停——防止模型自顾自把对方的话也续出来。注意 stop sequence 是**字符层面**的规则，触发后 API 返回的结果通常**不包含**这段 stop 文本本身。
- **max tokens**：硬性长度上限，兜底用。EOS 没出来、stop 没命中、token 数也到了，强行截断，避免无限生成。

三个条件并存的设计很关键：模型可能「忘了」输出 EOS（尤其是新模型 + 新模板时）；调用方的 stop 也未必能兜住所有边界情况；max tokens 是最终保险。

#### 循环里真正「做选择」的，只有第 3 步

第 1 步是模型固定行为，第 2 步是对分布形状的可插拔变换（processors），第 4 步是字符串拼接。**整段循环里真正在「从分布里挑 token」的，只有第 3 步**——`decoding_policy(logits)`。这个函数怎么写，加上第 2 步装了哪些 processors，一起构成了所有 decoding 方法的差异。

> 生成是一个串行的决策循环；token 是一个一个挑出来的，不是一次写出来的。循环每一轮的可调点只有 policy，其余都是机械执行。

![自回归循环示意图](autoregressive-loop.png)

*图：每一轮「前向 → softmax → policy → 追加」四步构成 autoregressive loop。policy 是循环里唯一可调的环节，外侧由 EOS / stop sequence / max tokens 决定何时退出。*

### 决定怎么选——decoding policy 的三个权衡

回到 policy 函数。同一份分布，不同的 policy 可以选出非常不同的 token。这一节我们不一一介绍各种具体策略——那是下一篇的事——而是先把 policy 的设计空间看清楚：你到底在权衡什么？

最干净的切分是两类：

- **确定性（deterministic）**：固定一份分布，输出永远一样。最典型的是 **argmax**——总挑概率最高的那个 token，也叫 greedy decoding。
- **随机性（stochastic）**：按概率从分布里**抽样**。即便分布不变，每次抽出来的也可能不同。

这两类风格不同，背后是三个交织在一起的权衡轴。

#### 第一个轴：可复现性

确定性策略一旦定下来，输入相同输出就相同——调试好做，回归测试好写。随机策略每次跑结果都漂移，**固定随机种子（seed）通常可以拿到近似可复现的结果**，但不保证跨版本、跨硬件、跨服务后端严格一致——OpenAI 官方文档自己也只承诺 「mostly deterministic」，还要配合 `system_fingerprint` 判断后端是否换过。这就是为什么很多 API 提供 `seed` 参数：业务上需要「看起来在采样、实际可大致复现」的折中，但别把它当成完全确定。

#### 第二个轴：连贯 vs 多样（coherence vs diversity）

argmax 永远走概率最高的那条路，听起来最稳，但实际跑下来很容易陷在高频套话里：模型会开始重复（「非常非常非常……」）、堆模板化短语、回答没有惊喜。原因不复杂——每一步局部最优，整段未必有趣。

随机采样反过来：偶尔会从分布的「长尾」里抽出一些不那么常见的 token，给生成结果带来变化和创造力。代价是有一定概率「翻车」——抽到一个不太合理的 token，后面的循环就跟着歪。

这就是 **coherence-diversity trade-off**：你不可能同时拿到「绝不出错」和「非常有创意」。整个 decoding 文献后续的所有花活——top-k、top-p、temperature、repetition penalty——都是在这条轴上找更精细的平衡点。

#### 第三个轴：可控性

argmax 简单粗暴，但你想干预很难——除非改模型本身。采样类策略提供了一组可以拨的旋钮，它们都作用在 **logits / 分布形状**这一层（而不是「怎么从分布里挑」那一层）：温度调分布尖锐度（上一篇讲过，本质是在 logits 上除一个数）、top-k / top-p 切候选范围（下一篇讲）、重复惩罚把已出现 token 的 logit 压低。这些都是在不动模型权重的前提下，**用 decoding 这一独立层去塑造输出风格**的。

> 一个看起来反直觉但很重要的事实：同一个模型，配不同 decoding policy，可以表现得像两个产品。需要稳定可解析输出的场景——结构化抽取、分类、JSON 生成、测试回归——greedy / 低温确定性解码仍然是默认选择；开放式长文本生成（对话、写作、故事、创意）才更需要随机采样和 top-p。模型权重一行没动。

所以下次「模型表现不行」的时候，先别急着换模型。问一句：**decoding 调对了吗？**

> decoding policy 是和模型解耦的独立设计变量，它在确定性、可复现性、coherence-diversity、可控性这几个维度上做权衡。

![argmax vs sampling](argmax-vs-sampling.png)

*图：同一份分布，argmax 永远指向最高的那根柱子；sampling 按概率扔骰子，偶尔会落到长尾。两条路径共用一个分布，但产物完全不同。*

### 收尾

这篇拆完，整个生成过程在你脑子里应该是一条很清晰的链路：

```
模型 ─▶ 分布 ─policy─▶ token ─append─▶ context ↻ ……直到 stop
```

这条链路给了你一个**分层排查框架**：

- **模型 + 上下文**决定基础分布——它知道什么、信什么，被 prompt / system / 检索资料引向哪个分布；
- **logits processors + decoding policy**决定怎么把分布变成 token——温度 / 惩罚 / top-k / top-p 在改分布形状，argmax / sampling 在做最终选择；
- **终止条件**决定循环什么时候退出——交给模型自己（EOS）、外部规则（stop sequence），还是兜底硬切（max tokens）。

这三层在推理时的可调性不一样：模型本身一般动不了（除非换模型、微调或换 system prompt / 模板），但 logits processors、decoding policy 与终止条件都是线上可改的旋钮。所以模型答得不好的时候，可以分层往下看：是上游分布就没指对（prompt / 上下文 / 模型本身的问题），还是分布是对的但被 decoding 把好答案挑掉了？这两类问题，解法完全不同。

那么具体的 policy 长什么样？greedy 怎么演化到 beam search？为什么工业上几乎没人用纯 argmax，反而 top-k 和 top-p 成了默认？下一篇我们把这些放到一张表里对照着拆。

### 参考资料

- Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020. [https://arxiv.org/abs/1904.09751](https://arxiv.org/abs/1904.09751) ——提出 nucleus（top-p）采样，对 coherence-diversity trade-off 给出了系统性分析
- Hugging Face Transformers Docs, *Generation Strategies*. [https://huggingface.co/docs/transformers/main/en/generation_strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies) ——主流 decoding 策略的实现接口与参数语义
- OpenAI Platform Docs, *Text generation*. [https://platform.openai.com/docs/guides/text-generation](https://platform.openai.com/docs/guides/text-generation) ——`stop`、`max_tokens`、`seed` 等参数的官方说明