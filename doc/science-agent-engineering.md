# Science Agent Engineering for `countingFieldOnLattice`

> 一份把量子输运计算项目重构为「Science Agent 工程」的设计文档。
>
> 参考方法论：
> - Cloudflare《大规模编排 AI 代码审查》 — 多专项 reviewer + 协调器架构
>   （原文 <https://blog.cloudflare.com/ai-code-review/>，中译 <https://deusyu.app/posts/ai-code-review/>）
> - HumanLayer《12-Factor Agents》 — <https://github.com/humanlayer/12-factor-agents>

---

## 0. TL;DR

本项目的正确性标准是**物理定律**（可数学验证），而这些定律目前全部以**散文形式**散落在
`memo.md` / `doc/memo.md` 中，**没有一行代码强制执行**。这正是构建 Science Agent 的黄金切入点：

> 把「散文形式的物理知识」编译成「一组各管一条定律的专项审查 agent + 一个协调器」，
> 让计算结果在被采信前先通过物理裁判。

这与 Cloudflare 的核心结论一致——朴素的「把结果塞进一个大 prompt 让 LLM 找问题」会产生噪声，
而**小而专、有客观锚点的 reviewer + 协调器去重判严重性**才可靠。区别在于：软件项目的锚点是测试，
而**我们的锚点是守恒律、对称性、厄米性**——比测试更硬。

---

## 1. 为什么本项目特别适合做 Agent 工程

| 项目特征 | 一般软件项目 | 对 Agent 设计的含义 |
|---|---|---|
| 正确性 = 物理定律（可数学验证） | 正确性 = 通过测试 | Agent 能做**确定性物理裁判**，非模糊「looks good」 |
| 大量**散文领域知识**（`memo.md`） | 知识在代码/文档 | 知识可直接编译成检查器 + system prompt |
| 工作流 = **参数扫描 + 眼睛看图** | CRUD / API | Agent 价值在**闭环扫描 + 自动异常检测** |
| PyTorch autograd，可微分 | — | 可用数值/梯度一致性做交叉验证 |
| Notebook 探索式，手动迭代 | 结构化代码 | Agent 把「手动实验循环」→「可复现编排」 |

**一句话：本项目最不缺「客观对错标准」——而这恰是 LLM agent 最怕缺、我们天生就有的东西。**

---

## 2. 现状盘点（代码事实）

### 2.1 两条计算路径（entry points）

1. **计数场导数法** `genfunc_cf_deriv_method/`
   - 编排层 `workflow/`：
     - `construct_ginv_total.py::construct_ginv_total(H_BdG, E_batch, eta, leads_info)` — 顶层装配
     - `construct_ginv_central.py::construct_ginv_central(...)` — 中心区 RAK 空间块对角 G⁻¹
     - `add_ginv_leads.py::add_ginv_leads(...)` — 引线自能叠加
     - `construct_ginv_tlc.py::construct_ginv_tlc(...)` — 隧穿耦合
   - 计算层 `calculations/calculation_cf_autograd.py` — autograd 求高阶导（电流/噪声）

2. **格林函数逆方法** `greens_functions_inv_method/`
   - `transport_calculation.py::calculate_transport_properties(...)` — 主入口
   - `direct_calculation.py` — 直接法（噪声公式见 `memo.md`）
   - `total_self_energy.py` — 引线总自能

### 2.2 实验生命周期（当前是手动的）

```
定义系统(Hamiltonian) → 配置参数(Nx,Ny,mu,eta,E_batch) → 计算 → 眼睛看图(dataplot/) → 手动调参 → 重复
```

- 系统模型：`hamiltonians/`（`Central.py`, `SSHChain`, `KitaevChain`, `MZMVortexHamiltonian`, ...）
- 探索入口：根目录 notebooks（`test_kitaev.ipynb`, `vor_direct.3.ipynb`, `qhe_autograd.ipynb`, ...）
- 可视化：`dataplot/`, `plot_vary_params.py`, `plotly_onsite_split.py`
- 产物：`data/`

### 2.3 物理约束现状：**全是散文，零强制**（核心 gap）

以下不变量在 `memo.md` / `doc/memo.md` 中被文字描述，但**代码中没有任何 assert / 检查器执行它们**：

| 物理不变量 / 约定 | 文档出处 | 代码强制？ |
|---|---|---|
| 电流守恒 ∑ᵢ Iᵢ = 0 | `memo.md`（"eta 会导致守恒失败"） | ❌ 无 |
| eta = 0 才守恒；eta 需匹配 E 网格点 | `memo.md` | ❌ 无 |
| 粒子-空穴对称 E ↔ −E（μ₁=−μ₂ 应有电流） | `memo.md` | ❌ 无 |
| 累积量定义与符号：∂ʲ ln Z / ∂(iλ)ʲ | `memo.md` | ❌ 无 |
| 单位约定 e=1, ℏ=1, Φ₀=2π | `memo.md` | ❌ 无 |
| 张量 device 一致性（funcDevice 坑） | `memo.md` | ❌ 无（隐式） |
| 哈密顿量厄米性 / BdG 结构 | 隐含 | ❌ 无 |
| 透射本征值 0 ≤ Tₙ ≤ 1 | 隐含 | ❌ 无 |
| 噪声系数 S ∝ ∑ Tₙ(1−Tₙ) ≥ 0 | `memo.md` | ❌ 无 |

**这张表就是 agent 工程要填的坑。**

---

## 3. 借鉴方法论 → 映射到本项目

### 3.1 Cloudflare「大杂烩审查」→「物理审查」

Cloudflare 的关键不是造一个大 agent，而是 **7 个专项 reviewer + 1 个协调器（去重、判严重性、出结构化结论）**。
映射：

| Cloudflare reviewer | 本项目 reviewer | 检查内容 |
|---|---|---|
| Security | **守恒律 reviewer** | ∑ᵢ Iᵢ 残差；eta≠0 告警 |
| Performance | **对称性 reviewer** | E↔−E 偏差；μ₁=−μ₂ 电流非零 |
| Code Quality | **数值健康 reviewer** | 厄米性、NaN/发散、eta↔E 网格匹配 |
| Codex（内部规范） | **约定 reviewer** | 单位 e=ℏ=1, Φ₀=2π；device 一致性 |
| Docs | **一致性 reviewer** | `memo.md` 公式 ↔ 代码实现吻合 |
| Braintrust（可观测） | **审查归档** | 每次计算的审查历史可追溯 |
| 协调器 | **协调器 agent** | 汇总、去重、判「这次结果可信吗」 |

对应 Cloudflare「面对已加错误处理的函数还建议'加错误处理'」的反面：**有物理定律做锚，reviewer 不瞎报。**

### 3.2 12-Factor Agents → 让它成为「工程」而非「脚本」

| Factor | 本项目具体做法 |
|---|---|
| 1. NL → tool calls | 实验配置（`params.yaml`）→ 结构化计算调用 |
| **2. Own your prompts** | `memo.md` 物理约定**版本化**为 reviewer prompt 模板，不外包黑盒 |
| **3. Own your context window** | 格林函数是巨型复矩阵 → **只把标量特征**（守恒残差、对称性偏差）喂给 agent |
| **4. Tools are structured outputs** | reviewer 返回 `{invariant, residual, pass, severity, hint}`，非散文 |
| 5. Unify state | 计算态与业务态统一在审查报告对象里 |
| 6. Launch/Pause/Resume | 长扫描可暂停/恢复 |
| **7. Contact humans with tools** | 物理判断存疑时，结构化地请人类裁决（而非静默通过） |
| **8. Own your control flow** | 「扫描→算→审查→失败则调 eta 重算」由**我们**定义，非 LLM 自由发挥 |
| **9. Compact errors into context** | 发散/NaN → 压缩成结构化摘要回喂，协调器决定重试 |
| **10. Small, focused agents** | 一个 reviewer 只管一条定律；守恒 reviewer 不碰对称性 |
| 11. Trigger from anywhere | CLI / notebook / CI 均可触发审查 |
| **12. Stateless reducer** | 审查 = `f(计算结果, 物理约定) → 报告`，无隐藏状态，可复现 |

---

## 4. 目标架构

```
                    ┌─────────────────────────────────────────┐
                    │   实验配置 params.yaml                     │
                    │   Nx,Ny, mu, eta, E_batch, model=Kitaev   │
                    └───────────────┬───────────────────────────┘
                                    │  Factor 1: NL/配置 → 结构化调用
                    ┌───────────────▼───────────────┐
                    │      协调器 Agent (Orchestrator) │
                    │  编排扫描 / 去重 / 判严重性 / 出结论 │
                    └───┬───────────┬───────────┬─────┘
          并发 (Factor 10) │           │           │
        ┌─────────────────▼┐  ┌──────▼───────┐  ┌─▼──────────────┐
        │ 守恒律 Reviewer   │  │ 对称性Reviewer │  │ 数值健康Reviewer │
        │ ∑Iᵢ=0 残差       │  │ E↔−E 偏差     │  │ 厄米/NaN/eta网格 │
        └─────────┬────────┘  └──────┬───────┘  └─┬──────────────┘
                  │ 结构化输出(Factor 4)            │
                  └───────────┬───────────────────┘
                              ▼
                 ┌────────────────────────┐
                 │  结构化审查报告 (JSON)     │  ← 无状态 reducer (Factor 12)
                 │  pass/fail + 残差 + 建议  │
                 └────────┬───────────────┘
                          │  失败 → Factor 8 控制流：调 eta / 换网格 / 重算
                          ▼
                 ┌────────────────────────┐
                 │ 自动绘图 + 归档到 data/   │
                 └────────────────────────┘
```

### 关键设计决策（即「我们对 agent 的理解」的体现）

1. **物理定律是 tool，不是 prompt。** 守恒残差用 PyTorch 确定性算出，agent 只负责*判严重性 + 给建议*，避免 LLM 幻觉物理结论。
2. **计算结果不进 context，特征进 context。** agent 看的是 `{守恒残差: 1e-3, 本征值范围: [0, 1.02]}` 这类标量摘要（Factor 3）。
3. **协调器做去重与根因判定。** 例：eta≠0 同时触发「守恒失败」+「数值健康告警」，协调器要识别为**同一根因**（对应 Cloudflare 去重）。
4. **闭环重试是可执行的散文知识。** `memo.md` 说「eta=0 才守恒」→ agent 检测守恒失败可自动按规则调参重算。

---

## 5. 结构化契约（Schema）

### 5.1 物理不变量清单 `physics_invariants.yaml`（Factor 2）

```yaml
invariants:
  - id: current_conservation
    statement: "sum_i I_i == 0"
    tolerance: 1e-6
    source: "memo.md: eta causes lead current conservation fails"
    depends_on: [eta_zero]
    severity_if_violated: high

  - id: eta_zero
    statement: "eta == 0 for exact conservation"
    source: "memo.md: we should let eta=0"
    severity_if_violated: medium

  - id: particle_hole_symmetry
    statement: "result(E) relates to result(-E) under BdG p-h symmetry"
    source: "memo.md: electron and hole are E<->-E symmetry"
    severity_if_violated: high

  - id: transmission_bounds
    statement: "0 <= T_n <= 1 for all channels"
    tolerance: 1e-9
    severity_if_violated: high

  - id: noise_nonnegative
    statement: "S = sum_n T_n(1-T_n) >= 0"
    source: "memo.md: S_LL = (e^3|V|/pi.hbar) sum T_n(1-T_n)"
    severity_if_violated: high

  - id: hamiltonian_hermitian
    statement: "H == H.conj().transpose(-1,-2)"
    tolerance: 1e-10
    severity_if_violated: critical

  - id: device_consistency
    statement: "all tensors on funcDevice (E_batch.device)"
    source: "memo.md: device controlled by funcDevice"
    severity_if_violated: medium

conventions:
  units: {e: 1, hbar: 1, flux_quantum: "2*pi"}
```

### 5.2 审查报告（Factor 4 / 12）

```json
{
  "run_id": "kitaev_mu0_eta0_2026-07-09",
  "config": { "model": "KitaevChain", "Nx": 100, "eta": 0.0, "mu": 0.0 },
  "checks": [
    { "invariant": "current_conservation", "residual": 3.2e-8, "pass": true,  "severity": "info" },
    { "invariant": "transmission_bounds",  "max": 1.0000002, "pass": false, "severity": "high",
      "hint": "T_n slightly exceeds 1; suspect eta broadening or numerical precision" }
  ],
  "root_cause_groups": [
    { "cause": "eta_broadening", "affected": ["transmission_bounds"] }
  ],
  "verdict": "REVIEW_REQUIRED",
  "suggested_action": "rerun with eta=0 and denser E grid"
}
```

---

## 6. 落地路径（分阶段，每阶段可独立展示成果）

| 阶段 | 目标 | 展示的「agent 理解」 | 依赖 |
|---|---|---|---|
| **0. 知识工程化** | 把 `memo.md` 约束抽成 `physics_invariants.yaml` + 纯 Python 确定性检查器（**不用 LLM**） | Factor 2（own prompts/knowledge）；tool 优先 | 无 |
| **1. 单 agent** | 一个「物理审查 agent」：输入计算结果 → 调检查器 tool → 输出结构化报告 | Factor 2/4/10 | 阶段 0 |
| **2. 编排** | 多 reviewer 并发 + 协调器去重判严重性 | Cloudflare 架构；Factor 10 | 阶段 1 |
| **3. 闭环** | 参数扫描 → 审查 → 自动重试/绘图 → 归档 | Factor 8/12；notebook 循环自动化 | 阶段 2 |
| **4. 可观测** | 审查历史归档 `data/`，可追溯「哪次计算为何不可信」 | Cloudflare braintrust 可观测性 | 阶段 3 |

**建议先做阶段 0**：即使不接任何 LLM，一份可执行的 `physics_invariants.yaml` + 检查器已经把「散文知识」变成「工程资产」，
且是后续所有 agent 的确定性 tool 层——这是整套设计里 ROI 最高、风险最低的一块。

---

## 7. 反模式（明确不做什么）

- ❌ 把整个格林函数矩阵塞进 LLM context（违反 Factor 3）
- ❌ 让 LLM「自己判断」是否守恒（应由 PyTorch 确定性计算，LLM 只判严重性）
- ❌ 一个大 reviewer 检查所有定律（违反 Factor 10，会像 Cloudflare 初版一样噪声大）
- ❌ reviewer 返回散文结论（违反 Factor 4，无法去重/编排）
- ❌ 计算失败静默通过（应 Factor 7 结构化上报人类）
