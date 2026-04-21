+++
title = "3 困惑度 PPL：它到底衡量了什么，为什么有时会骗人"
description = "困惑度把交叉熵翻译成「模型平均在几个词里犹豫」——直觉友好，但 tokenizer 和语料不同时，这个数字不能直接比。"
date = 2026-04-21
tags = ["LLM", "困惑度", "PPL", "Bits-per-byte", "评估指标"]
categories = ["LLM"]
showTableOfContents = true
+++

{{< katex >}}

> 前置知识提示：读这篇前，建议了解：token、交叉熵、NLL（见第 1、2 篇）。

---

### 论文里的那个数字，到底是什么意思

你一定见过这样的句子：「Our model achieves a perplexity of 15.3 on Penn Treebank.」这个 15.3 是高是低？模型在「困惑」什么？如果另一篇论文报告 PPL = 21.8，是不是一定比 15.3 差？

上一篇我们推导出了交叉熵——每个 token 位置上模型的平均 loss。交叉熵越低越好，但它的单位是 nats（或 bits），不太直觉。你很难感受「1.73 nats」到底好不好。

困惑度（Perplexity, PPL）做的事情很简单：**把交叉熵翻译成一个更有直觉的数字——模型在每个位置平均在多少个 token 里犹豫。**

![困惑度直觉概念图](ppl-intuition.png)

*图：PPL = 5.6 意味着模型在每个位置平均觉得有约 5.6 个 token 「差不多可能」——就像每道题都是 5.6 选 1*

---

### 一步指数：从交叉熵到困惑度

回忆上一篇的结论：对句子「今天天气真好。」，交叉熵 = 1.73（nats）。

> 想象一场考试。交叉熵是你的「平均罚分」——每道题平均被扣了 1.73 分。但如果有人问你「这场考试难不难」，你回答「平均扣了 1.73 分」，对方大概率一脸茫然。更直觉的回答是：「感觉每道题都像在 5–6 个选项里猜。」困惑度做的就是这个翻译。

数学上，困惑度就是交叉熵的指数：

$$
\text{PPL} = \exp(\text{CE}) = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{\lt t})\right)
$$

用大白话说：如果交叉熵用自然对数（单位 nats），困惑度就是 \(e^{CE}\)；如果用以 2 为底的对数（单位 bits），就是 \(2^{CE}\)——两种算法结果相同。交叉熵越低，困惑度越低，模型越好。

用上一篇的例子走一遍：

$$
\text{PPL} = \exp(1.73) \approx 5.64
$$

意思是：模型在预测「今天天气真好。」这句话时，平均每个位置觉得有大约 5.6 个 token「差不多可能」。不是说恰好有 5.6 个候选词——而是模型的不确定程度，等效于在 5.6 个均匀分布的选项里做选择。

几个锚点帮你校准直觉：

- **PPL = 1**：模型完全确定每个位置该出什么 token。现实中不可能——语言本身有随机性
- **PPL = V**（词表大小）：在整个词表里均匀猜。词表有 V 个 token，均匀猜测的 PPL 就是 V——比如词表 50000，PPL = 50000
- **PPL = 15–30**：一些经典 benchmark 上常见的范围，但和 tokenizer 类型（word-level / BPE）、测试集强相关，不建议当作统一的好坏区间
- **PPL = 5–10**：非常强的模型，或者测试文本很规律（如代码、格式化文本）

```python
import math

ce = 1.73                        # 上一篇算出的交叉熵
ppl = math.exp(ce)               # PPL = 5.64

# 如果用 log2（bits），需要换底
ce_bits = ce / math.log(2)       # ≈ 2.50 bits-per-token
ppl_bits = 2 ** ce_bits          # 还是 5.64——PPL 和底数无关
```

注意最后一行：**不管交叉熵用自然对数（nats）还是以 2 为底的对数（bits），算出来的 PPL 是一样的。** 因为指数和对数互逆，底数在一步指数里被抵消了。PPL 是一个不依赖对数底数的量——这是它作为报告指标的一个好处。

---

### PPL 的陷阱：为什么有时会骗人

到这里你可能觉得，PPL 是一个干净的指标——越低越好，可以拿来比模型。但实际没这么简单。PPL 有两个经常被忽略的前提条件，一旦不满足，数字就会骗人。

#### 陷阱一：tokenizer 不同，PPL 不可比

交叉熵的公式里有一个关键的 \(T\)——token 的数量。同一句话，不同的 tokenizer 切出来的 token 数量可以差很多。

> 举个例子。「unbelievable」这个词，一种 tokenizer 可能把它切成 1 个 token（整词），另一种切成 3 个（un / believ / able）。如果模型 A 用 tokenizer A（切成 1 个），模型 B 用 tokenizer B（切成 3 个），那模型 B 的交叉熵是在 3 个位置上取平均，模型 A 是在 1 个位置上取平均——分母不同，交叉熵不可比，PPL 也不可比。

更极端的情况：如果一个 tokenizer 是字符级（character-level），把「hello」切成 5 个 token，每个 token 的预测都很容易（26 个字母里选），PPL 可能只有 3–4。但这不意味着模型更好——它只是在一个更简单的粒度上做预测。

这不是理论上的问题。GPT-2 和 LLaMA 用的 tokenizer 完全不同（BPE vs SentencePiece，词表大小也不同），直接比它们的 PPL 没有意义。

#### 陷阱二：测试语料不同，PPL 不可比

语言本身的「不确定程度」因文体而异。

- **法律合同**：用词规范、结构固定，PPL 天然偏低
- **日常对话**：随意、跳跃、俚语多，PPL 天然偏高
- **代码**：语法严格、关键字有限，PPL 可以非常低
- **诗歌**：刻意打破常规表达，PPL 偏高

一个模型在维基百科上 PPL = 20，在推特上 PPL = 45，不能说它在推特上表现差——推特本身就更难预测。

还有一个容易忽略的变量：**评估时的上下文窗口长度**。Transformer 评估长文本时，上下文越长，后续 token 的预测越容易，PPL 越低。同一个模型用 512 token 窗口和 2048 token 窗口评估，PPL 可以差不少——所以比较时窗口长度也必须对齐。

那问题来了：有没有一个更公平的指标？

---

### Bits-per-byte：一个更公平的尺子

解决 tokenizer 不可比的一个办法：**把度量单位从「每 token」换成「每 byte」。**

不管你用什么 tokenizer，最终文本都是由 byte 组成的。UTF-8 编码下，一个英文字母是 1 byte，一个中文字是 3 bytes，这是固定的、不依赖 tokenizer 的。

**Bits-per-byte（BPB）** 的计算方式是：把整段文本的总 loss（以 bits 为单位）除以这段文本的总 byte 数。

$$
\text{BPB} = \frac{\text{NLL}_{\text{total}} / \ln 2}{\text{byte count}}
$$

用大白话说：把模型对整段文本的总损失转成 bits，再除以文本本身的字节数。不管你怎么切 token，分母都是一样的字节数。

```python
import math

total_nll = 8.63        # 整句话的 NLL（nats）
total_bytes = 21        # "今天天气真好。" 的 UTF-8 字节数
bpb = (total_nll / math.log(2)) / total_bytes  # ≈ 0.593 bits/byte
```

BPB 的好处是：**不管你用什么 tokenizer（BPE、SentencePiece、字符级），分母都是同样的 byte 数。** 所以不同模型的 BPB 可以直接比——前提是用同一个测试语料、同一个上下文窗口长度和同一套评估协议。BPB 消除了 tokenizer 差异，但语料和评估设置的差异仍然会影响结果。

GPT-4 Technical Report 中报告过内部代码库上的 next-word prediction loss（bits per word），但它不是 BPB 的标准用例。在需要跨 tokenizer 比较的语言建模评估中，BPB/BPC 更常被用作补充或替代指标——The Pile 就明确将 BPB 作为 preferred metric。

---

### 什么时候可以放心比 PPL

说了这么多「不可比」，那 PPL 什么时候能比？条件很明确：

- **同一个 tokenizer** + **同一个测试集** → PPL 可以直接比，越低越好
- tokenizer 不同 → 用 bits-per-byte
- 测试集不同 → 不可比，除非你的目的就是比较模型在不同领域的适应能力

还有一个容易踩的坑：PPL 是**在测试集上算的**，不是在训练集上。如果一个模型在训练集上 PPL 很低、在测试集上 PPL 很高，那是过拟合，不是好模型。

![tokenizer 对 PPL 影响示意图](tokenizer-ppl-comparison.png)

*图：同一句话被不同 tokenizer 切成不同数量的 token，导致交叉熵的分母 T 不同——PPL 数字无法直接比较*

---

### 读完这篇，你拿到了什么

我们从上一篇的交叉熵出发，走了三步：

1. **PPL = exp(CE)**：一步指数，把「平均罚分」翻译成「在多少个选项里犹豫」
2. **PPL 的陷阱**：tokenizer 不同、语料不同时，PPL 数字不能直接比
3. **Bits-per-byte**：一个消除 tokenizer 差异的更公平指标

从「模型训练时最小化什么」（交叉熵）到「对外报告模型有多好」（PPL / BPB），这条链路现在完整了。但 PPL 告诉你的只是模型在「预测下一个 token」这件事上有多好——它不直接告诉你模型能不能写出好的摘要、能不能准确回答问题。从 PPL 到真实任务的表现之间还有很长的路。

下一篇我们进入一个新的话题：当词表有 50000 个 token 时，模型怎么从一个向量算出这 50000 个概率？答案是 Softmax——一个把任意实数变成概率分布的函数，简单到只有一行公式，但藏着不少值得拆的细节。

---

### 参考资料 / 推荐阅读

- Jurafsky, D. & Martin, J. H. *Speech and Language Processing*, Chapter 3: N-gram Language Models（§3.3 PPL 的定义与直觉）
- Gao, L. et al. (2020). *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*. [arXiv:2101.00027](https://arxiv.org/abs/2101.00027)（使用 BPB 作为评估指标的代表性工作）
- OpenAI (2023). *GPT-4 Technical Report*. [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)（报告了内部代码库上的 next-word prediction loss）
