# 📈 FinQuant_Qwen2.5-7B_QLoRA: 基于卖方金工研报的 A 股量化大模型微调实践

<div align="center">
  <img src="https://img.shields.io/badge/Base_Model-Qwen2.5_7B-blue.svg" alt="Base Model">
  <img src="https://img.shields.io/badge/Method-QLoRA-orange.svg" alt="Method">
  <img src="https://img.shields.io/badge/Framework-LLaMA_Factory-green.svg" alt="Framework">
  <img src="https://img.shields.io/badge/Hardware-RTX_4090_(24GB)-red.svg" alt="Hardware">
</div>

## 📖 项目愿景 (Project Vision)

通用大语言模型（LLMs）在处理高阶金融数理逻辑时，常暴露出严重的“客服腔调”与“代码表现欲”，难以直接辅助专业量化投研。

本项目旨在探索如何通过**参数高效微调 (PEFT)**，将卖方深度研报中的非线性逻辑、因子状态依赖（State-Dependency）等硬核知识注入大模型。致力于打造一个语言极其克制、深谙 A 股微观结构、且具备强指令边界控制能力的**专属量化金融 AI 助手**。

---

## 🛠️ 技术架构与工程亮点 (Engineering Highlights)

### 1. 极致的单卡算力压榨 (RTX 4090 24G)
在消费级显卡极度受限的显存环境下，成功跑通 7B 模型的指令微调全链路闭环，彻底攻克长序列训练的 OOM（显存溢出）瓶颈：
* **量化策略**：采用 `4-bit` 量化加载基座模型，将显存基础占用压缩至 5GB。
* **显存攻防**：强制开启 `Gradient Checkpointing`（梯度检查点），牺牲约 20% 训练速度换取巨量显存释放；采用 `paged_adamw_32bit` 优化器防止显存峰值溢出。
* **参数调优**：为平衡收敛速度与显存，采用 `Batch Size = 1` 配合 `Gradient Accumulation = 16`，等效大批次训练；设定 LoRA `Rank = 16`, `Alpha = 32`, `LR = 1e-4`。

### 2. 高质量知识工程 (Data-Centric AI)
* **语料来源**：深度清洗并结构化了包含多因子挖掘、隐马尔可夫模型(HMM)、高频数据降频等主题的权威金工研报。
* **格式对齐**：使用高智商大模型 API 批量生成 10,000+ 条高质量 `Instruction-Output` 对，严格过滤“AI 幻觉”及冗余废话，对齐工业界研究员的叙事品味。

---

## 📊 模型评估：A/B 盲测矩阵 (Evaluation)

为验证模型对量化知识的吸收程度，本项目设计了涵盖宏观状态切换、另类因子挖掘、代码工程等维度的“量化灵魂拷问”。

| 测试维度 | 🔴 原始 Qwen2.5-7B (Control) | 🟢 Fin-Qwen-7B (Ours) | 💡 核心跃升点 |
| :--- | :--- | :--- | :--- |
| **状态依赖与择时** | 仅提供 HMM 的泛泛统计学定义，缺乏特定市场结合。 | **精准指出将波动率、换手率作为观测变量，通过后验概率动态更新因子权重的实证逻辑。** | 成功内化 A 股非线性状态切换逻辑。 |
| **拥挤度预警机制** | 给出宽泛的交易量占比和持仓集中度概念。 | **精准还原研报逻辑，提出 VOL、TurnOver、Beta 核心指标，并给出带惩罚系数的目标函数。** | 指标定义贴近工业界实战。 |
| **单因子预处理代码**| 包含低效的 `for` 循环操作，伴随大量冗余说明。 | **全篇采用 `numpy/pandas` 向量化操作，代码极度简洁高效。** | 代码风格对齐专业量化开发规范。 |
| **指令边界控制** | 未经要求便自动生成长串无用的演示代码。 | **严格聚焦理论推演，行文极度克制，未输出冗余代码。** | 洗脱通用 AI 的“代码幻觉”与过度表现欲。 |

> 📌 **完整深度对局评测报告请参阅：[EVALUATION.md](./EVALUATION.md)**

---

## 🚀 快速上手 (Quick Start)

### 1. 环境依赖
```bash
conda create -n fin_qwen python=3.10 -y
conda activate fin_qwen
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
pip install transformers accelerate bitsandbytes
