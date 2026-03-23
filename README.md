# 📈 FinQuant-Qwen2.5-7B: 基于卖方金工研报的 A 股量化大模型微调实践

<div align="center">
  <img src="https://img.shields.io/badge/Base_Model-Qwen2.5_7B-blue.svg" alt="Base Model">
  <img src="https://img.shields.io/badge/Method-QLoRA-orange.svg" alt="Method">
  <img src="https://img.shields.io/badge/Framework-LLaMA_Factory-green.svg" alt="Framework">
  <img src="https://img.shields.io/badge/Hardware-RTX_4090_(24GB)-red.svg" alt="Hardware">
</div>

## 📖 项目愿景 (Project Vision)

通用大语言模型（LLMs）在处理高阶金融数理逻辑时，常暴露出严重的“客服腔调”与“代码表现欲”，且缺乏对 A 股特定市场微观结构（如状态切换、资金博弈）的深度认知，难以直接辅助专业量化投研。

本项目旨在探索如何通过**参数高效微调 (PEFT)**，将顶级卖方金工深度研报中的非线性逻辑、因子状态依赖（State-Dependency）等硬核知识注入大模型。致力于打造一个语言极其克制、深谙量化实证逻辑、且具备强指令边界控制能力的**专属量化投研大脑 (FinQuant-Qwen)**。

---

## 🛠️ 技术架构与工程亮点 (Engineering Highlights)

### 1. 极致的单卡算力压榨 (OOM 攻防战)
在单卡 RTX 4090 (24GB 显存) 极度受限的环境下，成功跑通 7B 模型的指令微调全链路闭环：
* **量化策略**：采用 `4-bit` 量化加载基座模型，将显存基础占用压缩至 5GB 极小值。
* **显存释放**：强制开启 `Gradient Checkpointing`（梯度检查点），牺牲约 20% 训练速度换取巨量显存释放；采用 `paged_adamw_32bit` 分页优化器防止显存峰值溢出。
* **参数调优**：为平衡收敛速度与显存约束，采用 `Batch Size = 1` 配合 `Gradient Accumulation = 16`，等效大批次稳健训练；设定核心超参 LoRA `Rank = 16`, `Alpha = 32`。

### 2. 高质量数据工程 (Data-Centric AI)
* **知识提取**：独立完成大规模华泰金工深度研报的 PDF 解析、OCR 切块与清洗，剥离冗余噪音。
* **格式对齐**：通过大模型 API 批量构造高质量 `Instruction-Output` 对，严格剔除带有“AI 机器味”的泛泛之谈，行文风格深度对齐资深量化研究员的叙事品味。

---

## 📊 核心能力跃升评估 (A/B Test Evaluation)

通过对 10 个核心量化投研维度的 Side-by-Side 盲测对比，FinQuant-Qwen2.5-7B 相比基底模型实现了四个核心层面的“质变”：

1. **行业认知的深度对齐 (Domain Alignment)**
   成功内化了实证逻辑，能够精准调用工业界标准指标（如 VOL、TurnOver、Beta），并给出诸如利用 HMM 模型进行因子波动率状态切换的特定策略，摆脱了维基百科式的通用定义。
2. **零代码幻觉与边界控制 (Boundary Control)**
   展现出极强的工业级克制力。在纯理论推演（如适应度函数设计）中，绝对不触发“代码幻觉”去编写冗余 Demo，而是直击要害。
3. **工程代码的降维打击 (Code Engineering)**
   代码风格高度贴合资深算法工程师，严格采用 `Pandas` / `Numpy` 的底层向量化（Vectorization）操作（如 `np.where`），兼顾执行效率与极简审美。
4. **数学表达的严谨性 (Mathematical Rigor)**
   能够熟练且优雅地输出带有惩罚项的凸优化目标函数、多目标适应度函数等高阶 LaTeX 数学公式，极大地提升了输出的学术质感。

> 📌 **完整深度对局评测报告与 LaTeX 公式推导请参阅：[EVALUATION.md](./EVALUATION.md)**

---

## 🚀 快速上手 (Quick Start)

### 1. 环境依赖
```bash
conda create -n finquant python=3.10 -y
conda activate finquant
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
pip install transformers accelerate bitsandbytes

### 2. 模型推理
```bash
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "Qwen/Qwen2.5-7B-Instruct"
lora_path = "Mrsultimarati-911/FinQuant-Qwen2.5-7B" 

# 4-bit 量化加载基座
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    load_in_4bit=True, 
    device_map="auto"
)

# 动态挂载量化大脑
model = PeftModel.from_pretrained(base_model, lora_path)

prompt = "请说明在多因子选股中，如何利用树模型处理特征交叉，并给出防过拟合的超参建议。"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

---

## ⚠️ 免责声明 (Disclaimer)
本项目仅供学术研究与工程技术交流，不产生商业价值，所涉及的因子逻辑与模型输出绝对不构成任何投资建议。金融市场具有高度不确定性，投资者应自主决策并严格承担相关风险。
