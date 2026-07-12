# Counting Field on Lattice

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/hepzanechen/counting-field-on-lattice.svg?style=social&label=Star)](https://github.com/hepzanechen/counting-field-on-lattice)

基于 PyTorch 的全计数统计（full counting statistics）、非平衡格林函数（NEGF）、计数场导数和晶格量子输运计算库。支持 BdG 哈密顿量、Kitaev 链、SSH 模型、拓扑超导性和 Keldysh 形式主义。

基于 Keldysh 形式主义和计数场方法的紧束缚晶格量子输运计算框架。

## 项目简介

本项目实现了一套高效的量子输运计算工具，用于研究晶格系统中的电子输运性质。核心方法基于 Keldysh 路径积分框架下的计数场（Counting Field）技术，并通过 PyTorch 实现自动微分功能，能够计算高阶输运相关量。

## Science Agent 设计理解

本项目正在从「数值计算库」扩展为一个 **OpenCode-native science-agent architecture with deterministic physics tools**。核心思想不是让 LLM 直接判断物理真假，而是让 LLM 组织科研推理，让确定性 Python 工具产生和裁决证据。

### 当前已实现：OpenCode-native 证据流水线

项目根目录的 `opencode.json` 定义了五个原生 OpenCode agents：

- `science-intake`：把自然语言物理猜想转成结构化假设候选，**必须包含证伪策略**。
- `experiment-designer`：根据模型目录设计 positive/control 实验和预注册证伪判据。
- `evidence-reporter`：把结构化证据账本写成带 `[E<n>]` 引用的报告。
- `hypothesis-reviser`：在 `FALSIFIED` 或 `INCONCLUSIVE` 后提出更严格的修正猜想。
- `skeptical-phd`：**怀疑性审查**，只从已知 confounder catalog 中找替代解释（有限尺寸、eta 展宽、平凡 ABS、无序零模等）。

这些 agents **只能提议、叙述和修正**。真正的哈密顿量构造、CF/GF 双路径计算、物理不变量检查和最终 verdict 都由确定性代码完成：

```text
science_agent/
  core/        # Hypothesis, model catalog, verdict schema, discovery ledger
  runtime/     # OpenCode client wrapper
  stages/      # LLM-facing science stages (intake, reporting, revision, skeptical)
  physics/     # deterministic runner + physics judge
  prompts/     # versioned agent prompts (falsification-first)

utils/physics/invariants.py  # hermiticity, p-h symmetry, current conservation, etc.
examples/                    # runnable demos
data/science_ledger/         # structured discovery records
```

### 证伪优先原则 (Falsification-First)

LLM 天生具有谄媚倾向。我们的设计强制所有 agents **以证伪为目标**：
- `science-intake` 必须输出 `falsification_strategy`
- `experiment-designer` 必须为每个实验设计 `falsification_criterion`
- `hypothesis-reviser` 不能弱化判据，只能收紧假设
- `skeptical-phd` 只从已知 confounder catalog 中找替代解释

### 结构化科学发现账本 (Discovery Ledger)

每个猜想从提出到验证的全过程都被结构化记录：
- `INITIAL` → `TESTING` → `SUPPORTED`/`FALSIFIED`/`INCONCLUSIVE`/`REFINED`
- 包含证伪策略、实验证据、修正历史、怀疑性审查结果
- 支持按状态查询、生成 markdown 摘要、追加式不可篡改

```python
from science_agent.core.ledger import Ledger
ledger = Ledger(path="data/science_ledger/discoveries.json")
ledger.add(discovery_record)
ledger.update_status(record_id, "SUPPORTED", evidence={...})
```

设计原则：

> **LLM proposes; deterministic physics disposes.**

也就是说，LLM 可以提出模型、参数、控制实验和修正假设，但不能覆盖物理裁判。每个数字都必须来自可复现计算，每个结论都必须引用 evidence ledger。

### 进阶：Virtual Lab / 课题组式 Science Agent

我们进一步把 agent 架构从「线性流水线」升级为**认识论工作单元**：不只是让 agents 扮演不同角色，而是把 personality 变成**可执行的认知契约**——scope、time horizon、novelty、evidence threshold、interaction policy——而不是剧本台词。

| 角色 | scope | time | novelty | evidence | interaction |
|---|---|---|---|---|---|
| `deep-specialist` | narrow | persistent | low | high | isolated |
| `creative-explorer` | broad | single | very_high | low | sandbox |
| `numerical-auditor` | narrow | single | very_low | very_high | readonly |
| `skeptical-falsifier` | medium | single | medium | high | isolated |
| `literature-cartographer` | broad | periodic | medium | citation | readonly |
| `integrator` | global | periodic | low | synthesis | hub |

**关键约束**（每条都是代码强制执行，不是 prompt 里的恳求）：

1. **Context isolation + persistent scoped memory**: `deep-specialist` 只看自己的 track 和对应 ledger entry，看不到 `data/proposals/` 或 `data/audits/`，也不会被其他 agents 的观点污染
2. **Two-buffer architecture**: `creative-explorer` 提案进入 `data/proposals/` (状态 `PROPOSED`)，只有经过 gate 后才能由 `integrator` 提升为 ledger 中的 `INITIAL`
3. **File-based blackboard**: agents 互不直接对话，各自把结构化产物写入独立目录，只有 `integrator` 汇总
4. **Gated synthesis**: `integrator` 提出综合后，确定性 gate 检查 (a) audit 是否 PASS、(b) skepticism 是否非 WEAK、(c) 是否存在 unresolved disagreement——否则 `SUPPORTED` 被降级为 `INCONCLUSIVE` 或 `NEEDS_MORE_DATA`
5. **结构化分歧**: disagreements 通过 `Disagreement{dimension, position_a, position_b, resolution}` 结构记录，而非 LLM 散文
6. **Hypothesis immutability**: `Hypothesis` class 是 frozen，reviser 不能修改既有判据，只能提出新假设

```text
                         Integrator / PI
              分配问题、设置 checkpoint、综合证据
                               │
      ┌────────────┬───────────┼───────────┬────────────┐
      ▼            ▼           ▼           ▼            ▼
Deep Specialist  Creative   Numerical   Skeptical   Literature
                  Explorer    Auditor     Falsifier   Cartographer
      │            │           │           │            │
 track.json   proposal.json audit.json skeptic.json literature.json
      └────────────┴───────────┴───────────┴────────────┘
                               │
                   deterministic synthesis gate
                               │
                       discovery ledger

  data/virtual_lab/
    tracks/        deep-specialist persistent state
    proposals/     creative-explorer sandbox
    audits/        numerical-auditor reports
    skepticism/    skeptical-falsifier reports
    literature/    literature-cartographer maps
    synthesis/     integrator synthesis reports
    ledger.json    master discovery ledger
```

这些 PhD agents 不按 spectrum/transport/scaling 这种固定观测量命名，而按**认知角色**命名：专注、探索、审计、怀疑、文献、综合。这更接近真实科研组织：科学发现不是单个 LLM 推理，而是多个独立证据流经过确定性 gate 后的综合。

**运行 Virtual Lab demo**:

```bash
.venv/bin/python -m examples.demo_virtual_lab
```

## 主要功能

### 1. 输运计算方法

- **计数场导数方法** (`genfunc_cf_deriv_method/`)
  - 使用自动微分计算电流、噪声等输运量的导数
  - 支持高达四阶的导数计算
  - 结合 vmap 方法提升计算效率

- **格林函数逆方法** (`greens_functions_inv_method/`)
  - 直接计算格林函数和输运系数
  - 支持递归格林函数方法
  - 可计算电流密度分布

### 2. 哈密顿模型 (`hamiltonians/`)

- **中心系统** (`Central.py`)
  - `Central`: 标准二维晶格
  - `DisorderedCentral`: 无序晶格系统
  - `CentralBdG`: Bogoliubov-de Gennes 配对形式
  - `TopologicalSurface2D`: 二维拓扑表面态
  - `MZMVortexHamiltonian`: Majorana 零模涡旋结构
  - `ChernTexturedInsulator`: 陈数纹理绝缘体

- **一维模型**
  - `SSHChain`: Su-Schrieffer-Heeger 链
  - `KitaevChain`: Kitaev 链（支持 Majorana 零模）

- **引线** (`Lead.py`)
  - `SpinlessLead`: 无自旋引线
  - `SpinfulLead`: 自旋极化引线
  - `MultiOrbitalLead`: 多轨道引线

### 3. 可视化工具 (`dataplot/`)

- 能带结构与色散关系绘图
- 局部态密度（LDOS）分析
- 电流密度分布可视化
- 输运量随能量变化关系
- 导纳矩阵热图

### 4. 实用工具 (`utils/`)

- 批量张量运算（批量克罗内克积、批量求迹）
- 费米分布函数计算
- 引线消约（Lead Decimation）算法
- 配置参数加载

## 安装依赖

```bash
pip install torch numpy matplotlib
```

项目主要依赖：
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- SciPy

## 快速开始

### 基本输运计算

```python
import torch
from hamiltonians.Central import Central
from hamiltonians.Lead import SpinfulLead
from greens_functions_inv_method.transport_calculation import calculate_transport_properties

# 定义系统参数
Nx, Ny = 10, 10
t_x = torch.tensor(1.0)
t_y = torch.tensor(1.0)

# 构建中心区域哈密顿量
central = Central(Ny, Nx, t_y, t_x)
H_total = central.H_full

# 定义引线
leads_info = [
    SpinfulLead(mu=torch.tensor(0.0), t_lead=t_x, 
                connection_coordinates=[(0, i) for i in range(Ny)],
                central_Nx=Nx, central_Ny=Ny)
]

# 计算输运性质
E_values = torch.linspace(-2, 2, 100)
temperature = torch.tensor(0.01)
eta = torch.tensor(0.01)

results = calculate_transport_properties(
    E_batch=E_values,
    H_total=H_total,
    leads_info=leads_info,
    temperature=temperature,
    eta=eta
)
```

### 使用计数场方法计算高阶导数

```python
from genfunc_cf_deriv_method.calculations.calculation_cf_autograd import calculation_cf_autograd

# 计算输运量的导数（电流噪声等）
results = calculation_cf_autograd(
    H_BdG=H_BdG,
    E_batch=E_values,
    eta=0.01,
    leads_info=leads_info,
    max_derivative_order=4
)
```

## 项目结构

```
.
├── genfunc_cf_deriv_method/    # 计数场导数方法
│   ├── calculations/           # 自动微分计算
│   └── workflow/              # 格林函数逆构建流程
├── greens_functions_inv_method/ # 格林函数逆方法
│   ├── direct_calculation.py  # 直接计算
│   ├── transport_calculation.py # 输运计算
│   └── total_self_energy.py   # 自能计算
├── hamiltonians/              # 哈密顿模型
│   ├── Central.py             # 中心系统
│   └── Lead.py                # 引线模型
├── dataplot/                  # 数据可视化
│   ├── dispersion_plot.py     # 能带绘图
│   ├── ldos_plot.py           # LDOS绘图
│   ├── current_density_plot.py # 电流密度绘图
│   └── transport_plot.py      # 输运性质绘图
├── utils/                     # 工具函数
│   ├── batch/                 # 批量运算
│   └── physics/               # 物理工具
└── doc/                       # 文档笔记
```

## 应用领域

- 量子霍尔效应研究
- 拓扑绝缘体输运性质
- Majorana 零模探测
- SSH 链噪声分析
- Kitaev 链中的量子输运
- 无序系统中的局域化现象

## 文档

更多理论背景和详细使用方法请参考：

- `doc/note/lattice_generating_slides.md` - 计数场方法理论介绍


## 许可证

本项目仅供研究使用。
