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

## 4. 目标架构：假设驱动的认知循环（Hypothesis-Driven Epistemic Loop）

> 早期版本是一个 5 层线性流水线（意图→生成→执行→验证→结论）。经过压力测试后放弃：
> **科研不是瀑布，是认知循环** —— 猜想→预言→计算→偏差→修正。且线性流水线让 LLM
> 判守恒违反的严重性（会被合理化掉）、无反馈回路（发现问题不能重试）、人类裁决层自废武功。

```
┌─────────────────────────────────────────────────┐
│           HYPOTHESIS（预注册假设）                  │
│  猜想 + 模型 + 参数 + 证伪判据 + 证据日志             │ ← 判据在计算前冻结，防移动球门
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   实验设计器（确定性）      │  最小可证伪实验优先：先 Nx=8 再 Nx=12
          │   + 对照组：平凡相中信号必须消失│
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   双路径执行器（确定性）    │  CF-autograd ∥ GF-inversion
          │   → 容差内互相比对        │  两条数学路线算同一可观测量 = 复式记账
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   物理裁判（确定性 PyTorch）│  厄米性/p-h 对称/流守恒/透射界/噪声非负
          │   → pass / fail          │  纯代码，零 LLM
          │   + eta→0 外推消歧        │  守恒违反 ∝ eta ⟹ 物理极限下守恒成立
          └────────────┬────────────┘
                       │
              ┌────────▼────────┐
              │ 证据登记         │  裁判失败 → INADMISSIBLE（证据不可采信）
              │ (append-only)   │  裁判通过 → 对照预注册判据 → SUPPORTED/FALSIFIED
              └────────┬────────┘
                       │  失败 → 回到实验设计器（调 eta/网格/尺寸）
              ┌────────▼────────┐
              │  LLM 总结器      │  唯一让 LLM「判断」的位置：
              │  （可选）         │  猜想→结构化假设（入口）+ 证据→自然语言结论（出口）
              └─────────────────┘
```

### 关键设计决策（即「我们对 agent 的理解」的体现）

1. **证伪优先于证实（Popper）。** LLM 是谄媚的——会想方设法确认用户的猜想。所以证伪判据在计算前**冻结**在假设结构里，且强制包含**对照实验**（平凡相中信号必须消失，否则信号是假象）。
2. **物理定律是 tool，不是 prompt。** 守恒残差用 PyTorch 确定性算出。LLM 只在入口（意图解析）和出口（总结）出现，中间全部确定性。
3. **双路径交叉验证 = 复式记账。** CF 法（lnZ 求导）与 GF 法（Fisher-Lee）互为对账。**诚实标注局限**：两条路径共享部分输入构造，一致性排除方法级 bug，不排除共享输入 bug。
4. **三态判定，不是二态。** `SUPPORTED / FALSIFIED / INADMISSIBLE`——物理裁判失败时证据**不可采信**（既不支持也不证伪），这是科学计算特有的认识论层次。
5. **散文知识 → 可执行控制流。** `memo.md`「eta≠0 破坏守恒」→ 裁判检测到守恒违反时自动跑 eta/10 外推：违反量 ∝ eta ⟹ 物理极限下守恒成立，消歧「数值伪影」vs「真 bug」。
6. **计算结果不进 context，特征进 context。** 格林函数是巨型复矩阵，进入报告的只有标量：`{守恒残差: 4.1e-3, 外推比: 9.93, min|E|: 0.063}`（Factor 3）。

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

## 6. 已落地实现（可运行 Demo）

```
.venv/bin/python -m agent.demo_kitaev_mzm
```

**测试的真实物理猜想**：Kitaev 链在拓扑相（|μ|<2|t|）中承载 Majorana 零模，
表现为 (a) 近零 BdG 本征值 (b) 非零零偏压 Andreev 透射；平凡相中两者必须消失。

### 代码结构

| 文件 | 角色 | LLM? |
|---|---|---|
| `utils/physics/invariants.py` | 物理裁判 tool 层：8 个确定性检查器（厄米/p-h 谱对称/流守恒/透射界/噪声非负/双路径一致/eta 外推） | ❌ |
| `agent/hypothesis.py` | 预注册假设 + 证伪判据 + 三态判定（SUPPORTED/FALSIFIED/INADMISSIBLE）+ 证据登记 | ❌ |
| `agent/executor.py` | 双路径执行器（CF-autograd ∥ GF-inversion）+ 裁判编排 + eta 外推消歧 | ❌ |
| `agent/demo_kitaev_mzm.py` | 完整认知循环：设计（cheapest-first + 对照组）→ 执行 → 裁判 → 判据 → 账本 | ❌ |
| `data/agent_ledger/*.json` | 追加式证据账本（可观测性） | ❌ |

### 实测运行结果（2026-07-09）

- `topological_small`（Nx=8, μ=−5）：7 项裁判全过，min|E|=0.063、Andreev=0.019 → **SUPPORTED**
- `topological_scaled`（Nx=12, μ=−5）：GF 流守恒违反 4.1e-3 → 裁判自动跑 eta/10 外推 → 违反比 9.93 ≈ eta 比 10 → 判定为 eta 伪影，**证据可采信** → SUPPORTED
- `trivial_control`（Nx=8, μ=−25）：min|E|=15.6、Andreev=0.0 → 信号如预期消失 → SUPPORTED
- 双路径一致性：CF vs GF 电流最大相对误差 0.4%（容差内）
- **负向测试**：把「平凡相承载 MZM」这个错误猜想喂给同一循环 → **FALSIFIED** ✓（证伪路径真实有效，不是摆设）

### 后续阶段

| 阶段 | 目标 | 状态 |
|---|---|---|
| 0. 知识工程化（检查器 tool 层） | `utils/physics/invariants.py` | ✅ 已完成 |
| 1. 认知循环（假设→实验→裁判→账本） | `agent/` | ✅ 已完成（demo） |
| 2. LLM 入口 | 自然语言猜想 → 结构化 Hypothesis（模型选择、判据生成） | ⬜ |
| 3. LLM 出口 | 证据账本 → 自然语言结论报告 | ⬜ |
| 4. 泛化 | 从 KitaevChain 扩展到 hamiltonians/ 全部 8+ 模型 | ⬜ |

---

## 7. 反模式（明确不做什么）

- ❌ 把整个格林函数矩阵塞进 LLM context（违反 Factor 3）
- ❌ 让 LLM「自己判断」是否守恒（应由 PyTorch 确定性计算，LLM 只判严重性）
- ❌ 一个大 reviewer 检查所有定律（违反 Factor 10，会像 Cloudflare 初版一样噪声大）
- ❌ reviewer 返回散文结论（违反 Factor 4，无法去重/编排）
- ❌ 计算失败静默通过（应 Factor 7 结构化上报人类）
- ❌ 用梯度搜索证伪拓扑不变量——Chern 数量子化，可观测量呈台阶状，梯度几乎处处为零、
  相变点是分布不是函数。梯度只用于连续观测量对连续参数；拓扑量用自适应结构化扫描
- ❌ 对双路径一致性给出超额信心——两条路径共享部分输入构造，一致只排除方法级 bug
- ❌ 把「裁判失败」和「猜想被证伪」混为一谈——前者是证据不可采信（INADMISSIBLE），
  后者是可采信证据否定了猜想（FALSIFIED），认识论层次完全不同
