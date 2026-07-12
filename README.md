# Quantum Transport Science Agent

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![OpenCode](https://img.shields.io/badge/OpenCode-native-purple.svg)](https://opencode.ai/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个 **OpenCode-native、证伪优先的 Science Agent Virtual Lab**。LLM agents 负责提出猜想、设计实验、审计、怀疑和综合；确定性的量子输运引擎负责构建哈密顿量、计算证据、检查物理不变量并约束最终结论。

> **LLM proposes; deterministic physics disposes.**

[English](README.en.md) · [完整设计文档](doc/agent/science-agent-engineering.md)

## 30 秒了解项目

```text
自然语言物理猜想
        ↓
OpenCode agents 预注册证伪策略与 positive/control 实验
        ↓
QuantumTransportDomain 确定性运行 CF + NEGF 双路径计算
        ↓
Physics Judge 检查厄米性、粒子-空穴对称、流守恒、透射界、噪声与数值一致性
        ↓
Virtual Lab 独立 workcells：深度追踪、数值审计、怀疑性证伪、文献映射
        ↓
Integrator synthesis → deterministic gate → Discovery Ledger
```

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# OpenCode-native pipeline：猜想输入，带引用报告输出
python -m examples.demo_native_agentic \
  "I conjecture that a Kitaev chain hosts Majorana zero modes when |mu| < 2|t|."

# 完整 Virtual Lab：独立认知角色 + group meeting memo
python -m examples.demo_virtual_lab \
  "I conjecture that a Kitaev chain hosts Majorana zero modes when |mu| < 2|t|."
```

LLM stages 需要本机可用的 [OpenCode CLI](https://opencode.ai/)。纯物理 demos 不需要 LLM：

```bash
python -m examples.demo_kitaev_mzm
python -m examples.demo_mzm_phase_boundary
```

## Virtual Lab 架构

```text
                         Integrator / PI
                               │
       ┌────────────┬──────────┼──────────┬────────────┐
       ▼            ▼          ▼          ▼            ▼
 Deep Specialist  Creative  Numerical  Skeptical  Literature
                  Explorer   Auditor    Falsifier  Cartographer
       │            │          │          │            │
   track.json   proposal.json audit.json skeptic.json literature.json
       └────────────┴──────────┴──────────┴────────────┘
                               │
                  deterministic synthesis gate
                               │
                    Scientific Discovery Ledger
```

Agent “personality”不是说话风格，而是可执行的认识论契约：

| Role | Scope | Memory | Evidence standard | Interaction |
|---|---|---|---|---|
| `deep-specialist` | 一个 hypothesis | 持久 track | ≥3 独立证据 | 隔离 |
| `creative-explorer` | 全局 | 单次 | 仅提案 | proposal sandbox |
| `numerical-auditor` | 单条 evidence | 无状态 | 严格数值 gate | 只读原始数据 |
| `skeptical-falsifier` | confounder catalog | 无状态 | 每条排除必须引用证据 | 不读创意提案 |
| `literature-cartographer` | claim ↔ literature | 周期性 | supporting + contradicting | 只读 |
| `integrator` | 全局 | ledger history | 综合全部证据 | 唯一汇总点 |

关键机制：

- **Falsification-first**：所有假设必须先定义什么结果会推翻它。
- **Context isolation**：agents 不互相聊天；每个 workcell 写入独立结构化文件。
- **Two-buffer architecture**：创意先进入 `proposals/`，通过 gate 后才能进入 scientific ledger。
- **Gated synthesis**：audit FAIL、skeptic WEAK 或 unresolved disagreement 都会阻止 `SUPPORTED`。
- **Immutable history**：被证伪的假设不会被覆盖；修正产生新的 hypothesis lineage。

## 通用 Agent Core 与领域引擎

仓库现在明确包含两个 Python packages：

```text
src/science_agent/       # 通用认识论基础设施
src/quantum_transport/   # 可替换的科学领域引擎
```

`science_agent` 只依赖 `ScienceDomain` protocol：

```python
class ScienceDomain(Protocol):
    def catalog_for_prompt(self) -> str: ...
    def validate_parameters(self, model_name, proposed): ...
    def build_system(self, model_name, params, **options): ...
```

当前实现 `QuantumTransportDomain` 提供：

- Kitaev / SSH / BdG Hamiltonians；
- Keldysh counting-field autograd（支持高阶 cumulants）；
- NEGF / Green's-function transport；
- CF 与 NEGF 双路径交叉验证；
- 厄米性、粒子-空穴对称、流守恒、透射界、噪声非负、η→0 外推。

未来可以在不修改 agent core 的情况下接入 molecular dynamics、materials science 或其他 domain adapter。

## 科学发现账本

```text
PROPOSED → INITIAL → TESTING
                       ├── SUPPORTED
                       ├── FALSIFIED
                       ├── INCONCLUSIVE
                       ├── NEEDS_MORE_DATA
                       └── REFINED
```

Runtime artifacts 写入 `data/`（默认 gitignored）：tracks、proposals、audits、skepticism、literature maps、synthesis reports 和 master ledger。

## 项目结构

```text
src/
├── science_agent/
│   ├── core/           # contracts, domain protocol, hypothesis, ledger
│   ├── runtime/        # OpenCode client
│   ├── stages/         # intake, reporting, revision, skepticism
│   ├── prompts/        # versioned agent contracts
│   └── orchestrator.py # Virtual Lab blackboard coordination
└── quantum_transport/
    ├── agent_adapter.py
    ├── agent_runner.py
    ├── hamiltonians/
    ├── methods/
    │   ├── counting_field/
    │   └── negf/
    ├── utils/
    └── visualization/

examples/               # runnable vertical slices
tests/                  # behavior and adapter contract tests
doc/agent/              # agent engineering design
doc/physics/            # counting-field and lattice physics notes
opencode.json           # native OpenCode agent definitions
```

## 测试

```bash
pytest -q
```

测试覆盖 domain protocol、模型参数 gate、Kitaev 拓扑/平凡谱差异、CF/NEGF 双路径物理裁判和 discovery ledger 持久化。

## 文档

- [Science Agent Engineering](doc/agent/science-agent-engineering.md)
- [Counting-field / lattice physics notes](doc/physics/lattice_generating_slides.md)
- [English README](README.en.md)

## License

MIT — see [LICENSE](LICENSE).
