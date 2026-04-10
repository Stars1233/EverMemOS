<div align="center" id="readme-top">

<!-- Logo -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="methods/evermemos/figs/evermind-logo-dark.svg">
  <img src="methods/evermemos/figs/evermind-logo-light.svg" alt="EverMind" width="260">
</picture>

<br><br>

# 面向 Agentic AI 的记忆操作系统

**为 AI 提供长期记忆基础设施，让它能记住、适应并持续进化。**

<br>

[![License: Apache 2.0][license-badge]][license]
[![arXiv: EverMemOS][arxiv-evermemos-badge]][arxiv-evermemos-link]
[![arXiv: HyperMem][arxiv-hypermem-badge]][arxiv-hypermem-link]
[![arXiv: EverMemBench][arxiv-evermembench-badge]][arxiv-evermembench-link]
[![HuggingFace: EverMemBench-Dynamic][hf-badge]][hf-link]

[官网][website] · [博客][blog] · [文档][docs] · [讨论区][discussions]

</div>

<br>

<!-- Benchmark Stats -->
<table>
  <tr>
    <td align="center"><strong>93.05%</strong><br><sub>LoCoMo</sub></td>
    <td align="center"><strong>83.00%</strong><br><sub>LongMemEval-S</sub></td>
    <td align="center"><strong>+40%</strong><br><sub>EvoAgent GDPVAL</sub></td>
  </tr>
</table>

---

## 为什么需要 EverOS

> *"没有记忆的实体无法展现一致性，也无法进化——因为它缺失了表层之下的根基。"*

大语言模型正从单轮对话机器人进化为长期交互式智能体。但当一个 Agent 需要在数周的对话中保持连贯时，它会撞上一个现实的天花板：有限的上下文窗口和碎片化的记忆。

扩大上下文窗口不是答案——超长上下文既昂贵，效果也会随距离衰减。长期 Agent 的未来取决于**结构化的记忆组织**。

EverOS 是一个记忆操作系统，通过三阶段记忆生命周期将无限的交互流转化为结构化的"数字大脑"：**情景痕迹形成**、**语义巩固**、**重构性回忆**——让 Agent 能够持续地将原始交互转化为结构化的、不断进化的知识。

---

## 项目结构

EverOS 围绕两大支柱构建——**方法**为 Agent 提供记忆与进化能力，**评测**客观衡量这些能力是否真正有效。

```
EverOS/
├── methods/
│   ├── evermemos/          # 长期记忆操作系统
│   └── hypermem/           # 超图记忆架构
│
└── benchmarks/
    ├── evermembench/        # 记忆质量评测
    └── evoagentbench/       # Agent 自进化评测
```

### 方法

<table>
<tr>
<td width="50%">

#### 🧠 EverMemOS

受生物印迹原理启发的自组织记忆操作系统。从对话中提取、构建和检索长期知识——让 Agent 能够记住、理解并持续进化。

[论文][arxiv-evermemos-link] · [文档](methods/evermemos/) · [快速开始](#快速开始)

</td>
<td width="50%">

#### 🔮 HyperMem

基于超图的层次化记忆架构，通过超边捕获高阶关联。将记忆组织为主题、事件和事实三层结构，支持粗到细的长期对话检索。LoCoMo 92.73%。

[论文][arxiv-hypermem-link] · [文档](methods/hypermem/)

</td>
</tr>
</table>

### 评测

<table>
<tr>
<td width="50%">

#### 📊 EverMemBench

记忆质量三层评估：事实召回、应用推理、个性化泛化。在统一标准下评测记忆系统和大语言模型。

[论文][arxiv-evermembench-link] · [数据集][hf-link] · [文档](benchmarks/evermembench/)

</td>
<td width="50%">

#### 📈 EvoAgentBench

Agent 自进化能力评测——不是静态快照，而是纵向成长曲线。通过有无进化的受控对照实验衡量迁移效率、错误规避和技能命中质量。

[文档](benchmarks/evoagentbench/)

</td>
</tr>
</table>

> 所有评测都设计为**开放的公共标准**。任何记忆架构、任何 Agent 框架都可以在同一把标尺下接受检验。

---

## 核心结果

### 记忆性能

| 系统 | LoCoMo | LongMemEval-S |
| :--- | :----: | :----: |
| **EverMemOS** | **93.05%** | **83.00%** |
| **HyperMem** | **92.73%** | — |
| Mem0 | 78.4% | — |
| MemOS | 74.2% | — |
| Zep | 71.6% | — |

### 自进化增益

| 任务类型 | Agent + LLM | 基线 | + EverOS 技能 | Δ |
| :--- | :--- | :----: | :----: | :----: |
| 代码 (Django) | OpenClaw + Qwen3.5-397B | 37% | 58% | **+21%** |
| 代码 (Django) | Nanobot + Qwen3.5-397B | 21% | 47% | **+26%** |
| 通用 (GDPVAL) | OpenClaw + Qwen3.5-397B | 29% | 69% | **+40%** |
| 通用 (GDPVAL) | OpenClaw + Qwen3.5-27B | 41% | 61% | **+20%** |

---

## 快速开始

```bash
git clone https://github.com/EverMind-AI/EverOS.git
cd EverOS
```

然后根据需求进入对应的组件：

| | 场景 | 入口 |
| :-- | :--- | :--- |
| 🧠 | 构建拥有长期记忆的 Agent | [methods/evermemos/](methods/evermemos/) |
| 🔮 | 使用超图记忆架构 | [methods/hypermem/](methods/hypermem/) |
| 📊 | 评测记忆系统质量 | [benchmarks/evermembench/](benchmarks/evermembench/) |
| 📈 | 衡量 Agent 自进化效果 | [benchmarks/evoagentbench/](benchmarks/evoagentbench/) |

> 每个组件都有独立的安装指南、依赖配置和使用示例。

---

## 社区

我们热爱开源力量！无论是修 Bug、开发新功能、完善文档，还是抛出奇思妙想——每一个 PR 都在推动 EverOS 前进。

浏览 [Issues][issues-link] 找到你的切入点，加入 [Discussions][discussions] 分享想法，或联系维护者——[@elliotchen200][elliot-x-link]（𝕏）和 [@cyfyifanchen][cyfyifanchen-link]（GitHub）。

| 社区 | 链接 |
| :--- | :--- |
| Discord | [![Discord Members][discord-members-badge]][discord] |
| WeChat | [![WeChat][wechat-badge]][wechat] |

---

## 引用

如果 EverOS 对你的研究有帮助，请引用：

```bibtex
@article{evermemos2025,
  title   = {EverMemOS: A Self-Organizing Memory Operating System for AI Agents},
  author  = {EverMind Team},
  journal = {arXiv preprint arXiv:2601.02163},
  year    = {2025}
}

@article{hypermem2026,
  title   = {HyperMem: Hypergraph Memory for Long-Term Conversations},
  author  = {Yue, Hu, Sheng, Zhou, Zhang, Liu, Guo, Deng},
  journal = {arXiv preprint arXiv:2604.08256},
  year    = {2026}
}

@article{evermembench2025,
  title   = {EverMemBench: A Comprehensive Benchmark for Long-Term Memory
             in Conversational AI},
  author  = {EverMind Team},
  journal = {arXiv preprint arXiv:2602.01313},
  year    = {2025}
}

@article{evoagentbench2025,
  title   = {EvoAgentBench: The First Objective Benchmark for Agent Self-Evolution},
  author  = {EverMind Team},
  year    = {2025}
}
```

---

## 许可证

[Apache 2.0][license]

---

<div align="center">

<a href="https://evermind.ai"><strong>EverMind</strong></a> · Keep in Mind. Evolve over Time.

</div>

<!-- Badge Definitions -->
[license-badge]: https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square
[arxiv-evermemos-badge]: https://img.shields.io/badge/arXiv-EverMemOS-b31b1b?style=flat-square&logo=arxiv&logoColor=white
[arxiv-hypermem-badge]: https://img.shields.io/badge/arXiv-HyperMem-b31b1b?style=flat-square&logo=arxiv&logoColor=white
[arxiv-evermembench-badge]: https://img.shields.io/badge/arXiv-EverMemBench-b31b1b?style=flat-square&logo=arxiv&logoColor=white
[hf-badge]: https://img.shields.io/badge/🤗_EverMemBench--Dynamic-F5C842?style=flat-square
[discord-members-badge]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Fv10%2Finvites%2FgYep5nQRZJ%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&suffix=%20members&label=Discord&color=404EED&style=for-the-badge&logo=discord&logoColor=white
[wechat-badge]: https://img.shields.io/badge/WeChat-EverMind%20社区-07C160?style=for-the-badge&logo=wechat&logoColor=white

<!-- Link Definitions -->
[license]: https://github.com/EverMind-AI/EverOS/blob/main/LICENSE
[arxiv-evermemos-link]: https://arxiv.org/abs/2601.02163
[arxiv-hypermem-link]: https://arxiv.org/abs/2604.08256
[arxiv-evermembench-link]: https://arxiv.org/abs/2602.01313
[hf-link]: https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic
[website]: https://evermind.ai
[blog]: https://evermind.ai/blogs
[docs]: methods/evermemos/docs/
[discussions]: https://github.com/EverMind-AI/EverOS/discussions
[issues-link]: https://github.com/EverMind-AI/EverOS/issues
[elliot-x-link]: https://x.com/elliotchen200
[cyfyifanchen-link]: https://github.com/cyfyifanchen
[discord]: https://discord.gg/gYep5nQRZJ
[wechat]: https://github.com/EverMind-AI/EverOS/discussions/67
