+++
title = "Harness Engineering 全景梳理"
description = "从起源、发展到现状的系统性回顾——什么是 Harness Engineering，为什么它比模型本身更决定 AI Agent 的实际表现"
date = 2026-03-31
tags = ["Harness Engineering", "AI Agent", "Context Engineering", "Claude Code", "SWE-bench"]
categories = ["LLM"]
showTableOfContents = true
+++

## 什么是 Harness Engineering

简单说，Harness Engineering 是一门关于**怎么让 AI 编码 Agent 在真实项目中可靠工作**的学科。

这个词的隐喻来自马术装备（harness = 缰绳/马具）。AI 模型是一匹很能跑的马，harness 是缰绳和马鞍——没有它们，马想去哪就去哪。工程师是骑手，提供方向。

之所以需要这个学科，核心原因很直接：**2025 年证明 Agent 能写代码了，但 2026 年大家发现让它们可靠地、大规模地写代码完全是另一回事。** 模型本身不是瓶颈了，围绕模型的那套系统才是。

和前面两个阶段对比一下就很清楚：

| 阶段 | 时间 | 一句话 |
| --- | --- | --- |
| Prompt Engineering | 2022–2024 | 怎么**问** |
| Context Engineering | 2025 | **给**模型什么 |
| Harness Engineering | 2026 | 整个**系统**怎么转 |

三者是嵌套关系：**Harness ⊇ Context ⊇ Prompt**。Prompt 是最内层的一次性指令，Context 是动态组装给模型的信息，Harness 是包裹一切的运行时环境——工具、权限、状态、测试、回退机制、日志，所有那些不是模型本身但决定模型能否干好活的东西。

---

## 起源与发展时间线

"Harness" 在软件工程里不是新词。EleutherAI 的 **lm-evaluation-harness** 从 2020 年起就是 LLM 评估的事实标准，Hugging Face 的 Open LLM Leaderboard 跑的就是它。所以 "harness" 在 AI 社区里一直有 "把模型套进框架" 的意味。

下面是这个概念从酝酿到爆发的关键节点。

| 时间 | 事件 | 意义 |
| --- | --- | --- |
| 2025.11 | Anthropic 发布《Effective Harnesses for Long-Running Agents》 | 种子文章。首次系统性地用 "harness" 描述 Agent 基础设施，提出双 Agent 架构解决跨 Context Window 工作问题 |
| 2026.01 | Aakash Gupta 发文《2025 Was Agents. 2026 Is Agent Harnesses》；Phil Schmid 发文 | 行业预判。核心论点：harness 而非模型才是竞争壁垒（Manus 同一模型重写 harness 五次，每次都提升可靠性） |
| 2026.02.05 | Mitchell Hashimoto 发布《My AI Adoption Journey》 | **术语结晶时刻**。给出 "Engineer the Harness" 的命名——每次 Agent 犯错就工程化修复，使错误不再发生 |
| 2026.02.11 | OpenAI 发布《Harness Engineering: Leveraging Codex in an Agent-First World》 | **引爆点**。5 个月、100 万行代码、零行手写。全面展示 harness 三大组件（Context Engineering、架构约束、垃圾回收） |
| 2026.02 中旬 | Birgitta Böckeler 在 Martin Fowler 网站发布分析 | 重要批评：OpenAI 文章缺少功能和行为验证的讨论；遗留代码库 harness 改造可能不经济 |
| 2026.02.17 | LangChain 发布《Improving Deep Agents with Harness Engineering》 | **数据证明**。仅改 harness（模型不变），Terminal Bench 2.0 得分 52.8% → 66.5% |
| 2026.02–03 | HumanLayer、LangChain、Anthropic 等密集发文；多篇学术论文出现 | 话题爆发。HumanLayer 讲实操配置手法，Anthropic 发第二篇引入多 Agent 架构，学术界开始正式化（HARNESSCARD 提案等） |

---

## 关键文献详细总结

### 工程实践报告

#### Anthropic —《Effective Harnesses for Long-Running Agents》(2025.11)

[链接](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)

这篇是整个话题的种子文章。它要解决的问题很具体：怎么让 AI Agent 跨多个 Context Window 持续工作。核心困难在于，每个新的 Agent 会话都从零开始——就像轮班制的工程团队，新来的工程师完全不知道上一班发生了什么。

Anthropic 给出的方案是**双 Agent 架构**（其实是同一个模型，只是初始 prompt 不同）：

**Initializer Agent** 只在第一次运行时启动，负责搭环境——写 `init.sh` 脚本、创建 `claude-progress.txt` 进度日志、做首次 git commit。最关键的是，它会把用户的高层需求拆解成一个详细的 **Feature List**（JSON 格式，200+ 条具体功能描述，每条标记为 `"passes": false`）。

**Coding Agent** 后续每次会话都走这个。上来先 `pwd`、读 git log、读进度文件，搞清楚当前状态，然后只挑一个 feature 来做，做完 commit + 更新进度文件，保证环境干净。

他们发现了三个很有代表性的失败模式：（1）Agent 试图 one-shot 搞完所有事，context 用完留下半成品；（2）做了几个 feature 后 Agent 看一圈觉得差不多了直接宣布完工；（3）不做端到端测试就标记 feature 完成。

对应的解决思路：用 JSON 而非 Markdown 存 feature list（模型不太敢随便改 JSON 的结构化数据）、强制每次只做一个 feature、强制用 Puppeteer MCP 做浏览器自动化测试。每个 session 的启动流程也标准化了：`pwd → 读进度 → 读 feature list → 启动 dev server → 基础功能测试 → 开始新 feature`。

这些听起来都不算多高级的技术，但组合起来效果显著。这就是 harness 的精髓：**不靠模型更聪明，靠系统设计更周全。**

#### Mitchell Hashimoto —《My AI Adoption Journey》(2026.02.05)

[链接](https://mitchellh.com/writing/my-ai-adoption-journey)

不是技术规范，更像一个顶级工程师的学习笔记，价值在方法论。

Hashimoto 记录了从 AI 怀疑论者到深度用户的完整过程。他做了一件很硬核的事：为了搞清楚 AI 编码到底行不行，把每个任务做两遍——先手写一遍，再用 Agent 写一遍，比较质量。双份工作量，直到彻底搞清 Agent 在哪犯错、为什么犯错、怎么通过环境改造防止再犯。

他的进阶路径：Chatbot → IDE Copilot → Agentic Coding → **Engineer the Harness**。在最后一个阶段，核心思想凝结成一句话：

> *"（Harness Engineering）is the idea that anytime you find an agent makes a mistake, you take the time to engineer a solution such that the agent never makes that mistake again."*
> 

实际例子：他的终端模拟器 Ghostty 项目里，AGENTS.md 文件的每一行规则都对应一次过去的 Agent 失败。这篇文章被广泛认为是术语的**结晶时刻**。

#### OpenAI —《Harness Engineering: Leveraging Codex in an Agent-First World》(2026.02.11)

[链接](https://openai.com/index/harness-engineering/)

这篇是整个话题的引爆点。核心数据：5 个月从空仓库开始，约 100 万行代码（应用逻辑、基础设施、工具链、文档、内部工具），**零行手写代码**。3 名工程师起步后扩展到 7 人，约 1500 个 PR 合并，平均每人每天 3.5 个。估算速度是手写的 10 倍。

"零手写代码" 是刻意的约束——强迫团队必须构建出让 Agent 真正能干活的基础设施，而不是在 Agent 卡住时偷偷手写两行绕过去。

**Context Engineering 部分**：AGENTS.md 只有约 100 行，当 "目录" 而非 "百科全书"。真正的知识库在 `docs/` 目录下，结构化存放设计文档（带验证状态追踪）、架构文档（domain 和 package 分层 map）、质量文档（给每个产品 domain 和架构层打分并追踪变化）。他们发现上下文塞太多反而有害——信息过载时模型开始局部匹配而非全局导航。

**架构约束部分**：强制分层架构，每个 domain 有严格的依赖方向。通过自定义 Linter（静态代码分析工具，在代码运行前检查是否符合规则）和结构测试在 CI（持续集成，代码每次提交时自动跑检查的流水线）层面强制执行。Linter 的错误信息里直接嵌入修复指导——Agent 违反边界时，错误消息会解释这个边界是什么、为什么存在、怎么修。这样 Agent 不需要 "理解" 架构，违反了就被机械地拦住。

**垃圾回收部分**：Agent 生成的代码会积累 entropy——命名不一致、文档过期、死代码堆积。Codex 会复制仓库中已有的模式——包括不好的模式，导致 drift 不可避免。他们最初每周五人工清理 "AI slop"，后来发现不可持续，改用定期运行的清理 Agent 自动扫描不一致和违规。

**Agent 自主开发流程**：Agent 已经可以自己拉 review feedback、inline 回应、push 更新、squash and merge 自己的 PR。不过他们强调这严重依赖这个仓库的特定结构和工具链，不能假设可以直接推广。

一个反直觉的核心发现：**约束 Agent 能做的事情反而提高了产出效率。** 自由度太高时 Agent 浪费 token 探索死胡同；清晰边界让它更快收敛到正确方案。

#### Birgitta Böckeler —《Harness Engineering》(2026.02，Martin Fowler 网站)

[链接](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html)

Thoughtworks Distinguished Engineer 的跟进分析。她把 OpenAI 的 harness 归纳为三类：Context Engineering（知识库 + 动态上下文）、Architectural Constraints（linter + 结构测试）、Garbage Collection（定期清理 Agent）。指出这混合了确定性方法和 LLM-based 方法。

重要批评：**文章缺少对功能和行为验证的讨论。** Harness 管住了代码怎么写、怎么组织，但没有验证代码是否做了用户需要的事。另外她指出，对现有非标准化代码库做 harness 改造可能在经济上不可行——类比在从未跑过静态分析的代码库上第一次跑 linter 然后被 alert 淹没。

她还提出了一个有趣的前瞻：未来团队可能会从一组通用 harness 模板中挑选来启动新项目，就像今天的 service template 一样。

#### LangChain —《Improving Deep Agents with Harness Engineering》(2026.02.17)

[链接](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)

这篇的独特价值在于它是**数据驱动**的——不是在讲理念，而是展示实验数据。核心结果：deepagents-cli 在 Terminal Bench 2.0 上从 52.8% 提到 66.5%（+13.7pp），模型固定为 gpt-5.2-codex，只改 harness。

他们只调了三个维度的旋钮：System Prompt、Tools、Middleware。改进方法来自一个自制的 **Trace Analyzer Skill**——从 LangSmith 拉实验 traces，spawn 并行的错误分析 Agent，主 Agent 综合发现并建议改进，人类审核后实施。类似 ML 中的 boosting，聚焦上一轮的错误。

四个具体改进：

**Self-Verification Loop**（贡献最大）：最常见的失败是 Agent 写完代码、重读一遍觉得行、就停了。解决方案是系统提示注入四步流程 Planning → Build → Verify → Fix，外加一个 `PreCompletionChecklistMiddleware` 在 Agent 要退出时拦截它强制验证。类似 "Ralph Wiggum Loop"——hook 强制 Agent 在退出前继续执行一轮验证。

**环境上下文注入**：`LocalContextMiddleware` 在启动时自动 map 目录结构、查找可用工具。听起来很基础，但减少了大量 "Agent 找不到文件在哪" 的错误。

**Reasoning Sandwich（推理三明治）**：gpt-5.2-codex 有 4 档推理模式。纯 xhigh 反而差（53.9%）因为超时多。他们用 xhigh-high-xhigh 的三明治策略——规划和验证阶段花更多算力推理，中间执行阶段降档。

**Doom Loop 检测**：`LoopDetectionMiddleware` 追踪同一文件编辑次数，超过 N 次后注入 "换个思路" 的提示。补丁性质的 workaround，但当前模型水平下有用。

重要发现：harness 在模型间不通用。同一 harness 版本，Codex 跑 66.5%，Claude Opus 4.6 跑 59.6%——因为没针对 Claude 做优化轮次。前沿模型在各自 harness 上做 post-training，所以模型和 harness 有耦合性。他们公开了 [Traces 数据集](https://smith.langchain.com/public/29393299-8f31-48bb-a949-5a1f5968a744/d?tab=2)。

#### Anthropic —《Harness Design for Long-Running Application Development》(2026)

[链接](https://www.anthropic.com/engineering/harness-design-long-running-apps)

这是 Anthropic 第二篇 harness 文章，作者 Prithvi Rajasekaran。在第一篇的双 Agent 基础上更进一步。

核心改进：引入三角色——**Planner**（把用户 spec 拆成可交付物级别的 sprint 计划）、**Generator**（一次一个 feature 的实现 Agent）、**QA**（独立的质量检查 Agent）。

重要发现：

- **单 Agent 跑太久会失去 coherence**——context window 填满后推理质量下降（和他们的 context engineering 文章呼应）
- **模型倾向于标记任务完成但实际没有 self-evaluate**——需要独立的 QA 步骤
- Planner 故意不指定细粒度技术细节，只约束交付物——如果 Planner 在技术细节上犯错，错误会 cascade 到下游实现
- 核心哲学：**"harness 中每个组件都编码了模型不能做什么的假设——这些假设值得持续压力测试，因为它们可能不正确，也会随模型进步而过时。有趣的 harness 组合空间不会缩小，而是移动。"**

#### HumanLayer —《Skill Issue: Harness Engineering for Coding Agents》(2026.03)

[链接](https://www.humanlayer.dev/blog/skill-issue-harness-engineering-for-coding-agents)

这篇最实操，逐个讲了 harness 的配置手段：

- **CLAUDE.md / AGENTS.md**：项目根目录的指令文件，被确定性注入 system prompt。先从小文件开始，Agent 每次犯同样的错就加一条规则（Hashimoto 模式）
- **Skills**：渐进式知识披露。一开始把所有指令塞进 system prompt 会让 Agent 变差——instruction budget 还没开始干活就用完了。Skills 让 Agent 只在需要时才加载特定知识
- **Sub-agents**：核心价值是 **context firewall**。父 Agent 只看到发给子 Agent 的 prompt 和子 Agent 的最终结果，中间的工具调用、中间结果等噪声全被隔离。在需要跨多个 context window 解决的复杂问题中，这是保持 coherence 的关键手段
- **安全警告**：Skill 注册表（ClawHub、[skills.sh](http://skills.sh)）已被发现分发恶意 Skill。对待它们应该像对待 `npm install random-package` 一样谨慎

### 学术论文 / 预印本

#### 《Harness Engineering for Language Agents》([Preprints.org](http://Preprints.org), 2026.03.23)

[链接](https://www.preprints.org/manuscript/202603.1756)

第一篇试图把 harness engineering 正式化为学术研究对象的论文。它做了三件事：

第一，提出 **CAR 分解框架**，把 harness 层拆成三个维度：Control（持久指令和规则，比如 [AGENTS.md](http://AGENTS.md)）、Agency（动作能力和工具接口，比如 Shell 执行、文件系统）、Runtime（状态管理和执行生命周期，比如 compaction 策略、session 交接）。这三个维度给了一个统一语言来讨论 harness 设计。

第二，提出 **HARNESSCARD**，类似大家熟悉的 Model Card，但针对 harness 层。要求研究者发表 Agent 系统成果时至少披露以下信息：用的什么 base model、[AGENTS.md](http://AGENTS.md) 等 control artifacts 怎么写的、runtime policy 是什么（比如 compaction 策略）、能调哪些工具（action substrate）、反馈和验证怎么分层（feedback stack）、权限和安全怎么管（governance）、以及评估流程（evaluation protocol）。目标不是加官僚，而是让结果可审计、可复现。

第三，做了一个**可见性审计**：梳理了截至 2026 年 3 月的论文、benchmark、工程笔记，发现学术论文倾向关注 interfaces 和 observability（Agent 能做什么、怎么评估），工程笔记更多关注 runtime、control 和 governance（怎么让 Agent 在生产中不出事）。换句话说，对生产部署最关键的那部分反而在学术文献中暴露最少。这本身就解释了为什么需要 HARNESSCARD。

核心论点：**很多报告中的 Agent 性能提升实际上部分是 harness-sensitive 的，不是纯 model-driven 的。** 两个系统用类似前沿模型但表现差很多时，原因往往至少部分在 harness 层。但现在大多数论文只报告模型的名字，不报告 harness 细节——这让结果难以复现和公平比较。

#### 《Natural-Language Agent Harnesses》(arXiv, 2026.03)

[链接](https://arxiv.org/html/2603.25723v1)

这篇探索了一个有意思的方向：能否用自然语言本身作为 harness 的表示格式？

AGENTS.md 和 Skill bundles 已经展示了这种可行性——用纯文本来打包项目约定和可复用流程。但论文指出，这些只是在可复用知识层面展示了可行性，并没有把 harness 级别的契约、角色边界、状态语义、失败处理等统一到一个可执行的表示里。

论文还回应了一个担忧：更强的模型是否会降低 harness 的价值？他们的实验支持一个不同的解读——当自然语言用来指定 harness 级别的控制（角色、契约、验证门、持久状态语义、委托边界）而不仅仅是一次性 prompt 时，它仍然重要。

#### 《Building AI Coding Agents for the Terminal: OpenDev》(arXiv, 2026.03)

[链接](https://arxiv.org/html/2603.05344v1)

少有的**开源编码 Agent 的正式技术报告**。大部分生产级系统要么闭源要么没发过技术报告，外部研究者很难了解实际设计决策。

OpenDev 把架构分成两个阶段——**Scaffolding**（部署前组装：system prompt、tool schemas、subagent registry）和 **Harness**（运行时编排：tool dispatch、context management、safety enforcement）。这个分法比较清晰地划分了 "部署前" 和 "运行时" 的边界。

四层架构：Entry & UI → Agent → Tool & Context → Persistence。具体实现了 workload-specialized model routing（不同复杂度的任务路由到不同模型）、dual-agent 分离 planning 和 execution、lazy tool discovery（不提前加载所有工具）、adaptive context compaction（渐进式压缩旧观察结果）。

#### 《Meta-Harness: End-to-End Optimization of Model Harnesses》(Stanford, 2026.03)

[链接](https://yoonholee.com/meta-harness/) | [GitHub](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact)

来自 Stanford IRIS Lab（Chelsea Finn 实验室，合作者包括 MIT 的 Omar Khattab）。这篇问了一个很自然的下一步问题：既然 harness 这么重要，能不能**让 AI 自己优化 harness**？

核心做法是一个迭代搜索循环：（1）一个编码 Agent（实际用的 Claude Code）读取一个文件系统，里面包含所有历史候选 harness 的源代码、执行轨迹和分数；（2）Agent 分析失败原因，提出新版本 harness；（3）新版本被评估，结果回存文件系统，循环重复。

和之前的文本优化方法（OpenEvolve、TTT-Discover）的关键区别在于信息量：之前的方法每步只看约 26K token 的压缩摘要，Meta-Harness 给 proposer 开放完整的文件系统，每步最多 **10M token** 的诊断上下文（400 倍）。Proposer 可以通过 grep、cat 等标准工具自己决定读什么，把失败追溯到具体的 harness 决策，而不是从一个分数里猜。

实验结果：

- 文本分类任务：超过最佳手工 harness (ACE) 7.7pp，同时只用 4 倍少的 token。仅 4 次评估就追平了下一个最佳方法跑 60 轮的成绩
- Terminal Bench 2.0：用 Claude Haiku 4.5 作为基底模型，优化后 harness 达到 37.6% pass rate，超过所有已报告的 Haiku harness（包括 Claude Code 27.5%、Goose 35.5%）
- 数学任务：一个优化出的策略在五个它从未见过的模型上都提升了准确率

重要性：这篇论文说明 harness engineering 有可能从人工调优变成自动化搜索。前沿模型间性能差距在缩小，但同一模型上不同 harness 实现的差距并没有缩小——而现在这个差距可以被算法化开发。

#### 《VeRO: An Evaluation Harness for Agents to Optimize Agents》(arXiv, 2026.02)

[链接](https://arxiv.org/abs/2602.22480)

和 Meta-Harness 方向类似但角度不同：VeRO 不是一个优化方法，而是一个用来**评估「编码 Agent 优化另一个 Agent」这件事做得好不好**的 benchmark 框架。

它提供了五个抽象（通过 tools 和 hooks 暴露），让一个 "optimizer Agent" 可以通过代码修改来迭代改进 "target Agent"。关键发现很有意思：

- 当前的 optimizer 缺乏多样性——默认只改 prompt，很少做结构性改动（改工具逻辑、控制流、scaffolding）
- 在 tool-use 任务上的改进无法迁移到 reasoning 任务
- 完整的 VeRO harness 带来约 8% 的提升，而只给 resource access 只带来 2%——说明更丰富的 scaffolding 改动空间很重要
- Claude 模型（Sonnet、Opus）作为 optimizer 显著优于 GPT-5.2-Codex

这篇把 "Agent 优化 Agent" 定位为一个 capability frontier——可行但远未解决。

#### 《Agentic Harness for Real-World Compilers (llvm-autofix)》(arXiv, 2026.03)

[链接](https://arxiv.org/abs/2603.20075)

把 harness engineering 应用到了一个之前没人试过的领域：**编译器 bug 修复**。编译器不是普通软件，修一个 LLVM bug 需要的跨领域专业知识（词法分析、类型系统、IR 设计、优化、代码生成）通常需要人类工程师多年经验。SWE-bench 和 SWE-agent 里的标准 bash 工具在编译器场景下几乎没用。

他们做了三件事：（1）开发了一套编译器专用的 Agent 工具，提供对 LLVM 的构建、复现、探索、调试、测试接口，屏蔽不必要的技术细节；（2）构建了 **llvm-bench** 数据集——334 个可复现的 LLVM 中端 bug，按难度分三档（easy 76.3%、medium 13.2%、hard 10.5%），还维护了 llvm-bench live 子集避免训练数据污染；（3）做了一个编译器场景专用的最小 Agent（llvm-autofix-mini）。

关键发现：同样模型从 SWE-bench Verified 切到 llvm-bench live，平均解决率下降 62%。最好的 DeepSeek V3.2 在 SWE-bench 60%，在 llvm-bench 只有 38%。但用了 llvm-autofix 工具后每个模型平均提升约 22%，GPT-5 达到 52%。

専家审查还发现 LLM 在编译器场景下有三类典型错误，而现有回归测试不足以系统性地捕捉。这篇用编译器这个极端场景证明：**领域专用 harness 是必需的，通用 harness 在高度专业化领域不够用。**

#### Terminal-Bench (Stanford & Laude Institute, 2026)

[链接](https://arxiv.org/html/2601.11868v1) | [官网](https://www.tbench.ai/)

89 个任务，横跨 ML 训练、调试、生物信息学等领域，在真正的 sandboxed 终端环境中执行。前沿模型 + Agent 得分低于 65%。

不像 SWE-bench 专注于代码补丁，Terminal-Bench 测试更广泛的终端操作——从编译 Linux 内核到训练模型到搭建服务器。它由任务数据集 + 执行 harness（Harbor 框架）两部分组成。对 harness engineering 最重要的发现：**harness 的选择显著影响得分**——LangChain 仅改 harness 就从第 30 名外跳到前 5，Meta-Harness 用自动化搜索在 Haiku 模型上超过了所有人工 harness。

#### SWE-bench 系列 (Princeton NLP, 2023–2026)

[链接](https://github.com/SWE-bench/SWE-bench) | [官网](https://www.swebench.com/)

500 个真实 GitHub issue，Docker 容器化评估，ICLR 2024 oral。每个 issue 来自真实的开源仓库，模型需要生成能通过单元测试的补丁。

SWE-bench 对 harness 问题很自觉——它专门用了一个最小化的 bash-only Agent harness（mini-SWE-agent）做评估，确保比较的是模型能力而非 harness 工程。但即便如此，Anthropic 声称定制 harness 比标准 harness 多带来约 10pp 的准确率提升。后来衡生出 SWE-bench Verified、SWE-bench Pro（顶尖模型仅 23%）、SWE-bench Multilingual 等变体。

---

## Harness 对性能影响的关键数据

把散落在各处的硬数据集中一下：

- [**Can.ac](http://Can.ac) 实验**：同一模型，不同 harness，性能从 6.7% → 68.3%（模型权重不变，10x 提升）
- **LangChain deepagents**：仅改 harness，Terminal Bench 2.0 从 52.8% → 66.5%（+13.7pp）
- **Claude Opus 4.6 在 Terminal Bench 2.0**：在 Claude Code（模型 post-training 时的 harness）排名 #33，换到其他 harness 排名约 #5——说明模型会过度拟合训练时的 harness
- **Anthropic 官方说法**：基础设施配置对编码 benchmark 的影响可达数个百分点，有时超过排行榜上顶尖模型之间的差距
- **SWE-bench**：Anthropic 定制 harness vs 标准 mini-SWE-agent，差距约 10pp

---

## 工具生态

### 主要 Harness 实现

| 工具 | 角色 |
| --- | --- |
| **Claude Code** | Anthropic 的编码 Agent harness。管理文件系统、sub-agents、工具编排、会话间记忆。Skills 由它首创后成为开放标准 |
| **OpenAI Codex CLI** | 百万行实验的核心。[AGENTS.md](http://AGENTS.md)  • CI 集成验证。模型和 harness 高度耦合（apply_patch 工具） |
| **LangChain deepagents** | 开源（MIT），Python/JS。提供文件系统、多模型路由、context compaction、large tool call offloading 等原语 |
| **OpenCode** | 开源 Claude Code 替代品。为了兼容 Codex 模型专门加了 apply_patch 工具来模拟 Codex harness |
| **Stripe Minions** | 内部系统，每周 1000+ 合并 PR。400+ 内部工具通过 MCP 接入，运行在隔离预热 devbox 中 |

### Harness 的六大配置面

根据 HumanLayer 和 LangChain 的总结：

1. **指令文件**（[CLAUDE.md](http://CLAUDE.md) / [AGENTS.md](http://AGENTS.md)）— 确定性注入 system prompt 的项目规则
2. **Skills** — 渐进式披露知识，避免 instruction budget 爆炸
3. **MCP Server** — 通过 Model Context Protocol 接入外部工具
4. **Sub-agents** — Context Firewall，隔离中间噪声
5. **Hooks / Middleware** — 工具调用前后的确定性检查（lint、验证、loop 检测等）
6. **Compaction** — Context window 接近满时的智能摘要和卸载

验证层的速度梯度：`PostToolUse Hook（毫秒）→ Pre-commit Hook（秒）→ CI（分钟）→ 人类审查（小时/天）`。检查越早，Agent 自我纠正越有效。

---

## 开放问题

### 功能验证还是缺

Böckeler 的批评至今仍然成立：harness 管住了代码怎么写、怎么组织，但验证代码是否做了用户需要的事情仍然薄弱。Anthropic 自己也承认 Agent 会在没有端到端测试的情况下标记完成。

### 遗留代码库怎么办

对全新项目做 harness 设计是一回事，对现有的、历史包袱沉重的代码库做 harness 改造是完全不同的挑战。Böckeler 类比了在从未跑过静态分析的代码库上第一次跑 linter 然后被 alert 淹没的场景。

### 模型和 Harness 的耦合问题

前沿编码模型在各自 harness 上做 post-training（Claude ↔ Claude Code, GPT-5 Codex ↔ Codex）。好处是模型在"自己的" harness 里表现最佳，坏处是换环境性能下降（Claude Opus 4.6 的排名从 #33 到 #5 的案例）。这对开源社区和跨平台使用场景是个真实问题。

### Harness 设计是场移动的战争

用 Anthropic Prithvi Rajasekaran 的话说：有趣的 harness 组合空间不会随模型进步而缩小，它只是**移动**了。今天需要复杂 pipeline 解决的问题，明天可能一个 prompt 就搞定了。但新能力又带来新的失败模式。Harness 工程师的工作是持续找到下一个有价值的组合。

---

## 参考资源

### 核心文献

- OpenAI, *Harness Engineering: Leveraging Codex in an Agent-First World* (2026.02) — [链接](https://openai.com/index/harness-engineering/)
- Anthropic, *Effective Harnesses for Long-Running Agents* (2025.11) — [链接](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- Anthropic, *Harness Design for Long-Running Application Development* (2026) — [链接](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- Mitchell Hashimoto, *My AI Adoption Journey* (2026.02) — [链接](https://mitchellh.com/writing/my-ai-adoption-journey)
- Birgitta Böckeler, *Harness Engineering* (2026.02) — [链接](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html)

### 学术论文

- *Meta-Harness: End-to-End Optimization of Model Harnesses* (Stanford, 2026.03) — [链接](https://yoonholee.com/meta-harness/)
- *Agentic Harness for Real-World Compilers* (arXiv, 2026.03) — [链接](https://arxiv.org/abs/2603.20075)
- *VeRO: An Evaluation Harness for Agents to Optimize Agents* (arXiv, 2026.02) — [链接](https://arxiv.org/abs/2602.22480)
- *Harness Engineering for Language Agents* ([Preprints.org](http://Preprints.org), 2026.03) — [链接](https://www.preprints.org/manuscript/202603.1756)
- *Natural-Language Agent Harnesses* (arXiv, 2026.03) — [链接](https://arxiv.org/html/2603.25723v1)
- *OpenDev* (arXiv, 2026.03) — [链接](https://arxiv.org/html/2603.05344v1)
- *Terminal-Bench* (arXiv, 2026) — [链接](https://arxiv.org/html/2601.11868v1)
- *SWE-bench* (Princeton NLP, ICLR 2024) — [链接](https://github.com/SWE-bench/SWE-bench)

### 工程博客

- LangChain, *Improving Deep Agents with Harness Engineering* (2026.02) — [链接](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)
- LangChain, *The Anatomy of an Agent Harness* (2026.03) — [链接](https://blog.langchain.com/the-anatomy-of-an-agent-harness/)
- HumanLayer, *Skill Issue* (2026.03) — [链接](https://www.humanlayer.dev/blog/skill-issue-harness-engineering-for-coding-agents)
- Phil Schmid, *The Importance of Agent Harness in 2026* (2026.01) — [链接](https://www.philschmid.de/agent-harness-2026)
- *The Emerging "Harness Engineering" Playbook* (2026.02) — [链接](https://www.ignorance.ai/p/the-emerging-harness-engineering)

### Benchmark

- SWE-bench — [链接](https://www.swebench.com/)
- Terminal-Bench — [链接](https://www.tbench.ai/)
- SWE-bench Verified (Epoch AI) — [链接](https://epoch.ai/benchmarks/swe-bench-verified)
- LangChain Traces 数据集 — [链接](https://smith.langchain.com/public/29393299-8f31-48bb-a949-5a1f5968a744/d?tab=2)

---

*编制于 2026.3.31，基于公开文献整理。*