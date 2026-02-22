# Korg.jl 与 Jorg 分子处理对比（机制深挖版）

## 10 行结论摘要
1. `Korg.jl` 的分子处理是“统一链路”：`Species/Formula` 表达、CE 方程组、默认分子数据、谱合成分子贡献在主流程内闭环。  
2. `Jorg` 在 CE 主求解上已基本对齐 Korg 思路（含分子项、charged molecule、显式 Jacobian 或 JAX 版本）。  
3. `Jorg` 的关键短板不在“有没有分子方程”，而在“失败路径与工程集成一致性”。  
4. 失败路径上，`Jorg` 的 `LayerProcessor` 会回退到简化 Saha（分子耦合丢失），这与 Korg 主路径的完整 CE 思路不一致。  
5. `Jorg.synthesize` 暴露了 `molecular_cross_sections` 参数，但当前主流程未实际叠加它；Korg 在参考波长与主波长两处都显式叠加。  
6. `Jorg` 的 `statmech` 默认分子数据链（Barklem+ExoMol+polyatomic logK）是完整的，但 line 侧另有一套分子截面模块，存在实现/接入不一致风险。  
7. `Jorg` 的 PINN 文档已明确：physics loss 目前仅原子守恒与电荷守恒，分子残差未显式纳入。  
8. CE 机制层面对比中，Korg 使用 `NLsolve + ForwardDiff`，Jorg SciPy 端采用 `root(hybr/lm)` 并行进式 fallback，JAX 端采用 JAXopt + implicit diff。  
9. 术语对象上，Jorg `MAX_ATOMS_PER_MOLECULE=10` 高于 Korg 的 6，功能上更宽，但并不自动代表更稳定。  
10. 建议优先级：P0 先补“分子截面主流程接入 + fallback 监控硬化 + PINN 分子残差落地”，再做性能与一致性优化。

---

## 1. 范围与版本基线
- 对比对象：
  - `Korg.jl-1.0.1`
  - `jorg/src/jorg`
- 本文只讨论“分子处理”，包含两层：
  - CE 核心（Saha + 分子平衡 + 守恒方程 + 求解器）
  - CE 之外的分子处理（分子线、分子截面、连续谱离子分子项）
- 本文不做 API 变更，不提交代码实现，只给差距与路线图。

---

## 2. 术语与对象模型（Species/Formula）

### 2.1 Korg.jl
- `Formula` 表示不含电荷的化学式，`Species` = `Formula + charge`。  
- Korg 中 `MAX_ATOMS_PER_MOLECULE = 6`，并提供统一的 `get_atoms` / `ismolecule` / `n_atoms` / `get_mass`。  
- 证据：`Korg.jl-1.0.1/src/species.jl:3`, `Korg.jl-1.0.1/src/species.jl:8`, `Korg.jl-1.0.1/src/species.jl:168`, `Korg.jl-1.0.1/src/species.jl:181`, `Korg.jl-1.0.1/src/species.jl:295`。

### 2.2 Jorg
- `jorg.statmech.species` 采用与 Korg 类似的 `Formula/Species` 语义，但上限是 `MAX_ATOMS_PER_MOLECULE = 10`。  
- 支持常见分子与一般公式解析，语义上可覆盖更多多原子分子。  
- 证据：`jorg/src/jorg/statmech/species.py:28`, `jorg/src/jorg/statmech/species.py:54`, `jorg/src/jorg/statmech/species.py:261`, `jorg/src/jorg/statmech/species.py:333`。

### 2.3 结论
- 两边对象模型“同构”，这是 CE 与 opacity 链路可比的前提。  
- 但 Jorg 的上限更高是“表示能力”优势，不等于“求解/数据/集成稳定性”优势。

---

## 3. CE 核心机制对比（重点）

### 3.1 Saha 电离项

#### Korg.jl
- 直接给出 `wII, wIII`，包含电子平动配分项 `translational_U`。  
- 氢的二次电离单独处理（`wIII=0` for H）。  
- 证据：`Korg.jl-1.0.1/src/statmech.jl:18`, `Korg.jl-1.0.1/src/statmech.jl:27`, `Korg.jl-1.0.1/src/statmech.jl:28`, `Korg.jl-1.0.1/src/statmech.jl:48`。

#### Jorg（SciPy CE）
- 同样预计算 `wII_ne`, `wIII_ne2`，解时用 `1/ne` 与 `1/ne^2` 缩放。  
- 证据：`jorg/src/jorg/statmech/korg_chemical_equilibrium.py:263`, `jorg/src/jorg/statmech/korg_chemical_equilibrium.py:338`, `jorg/src/jorg/statmech/korg_chemical_equilibrium.py:339`。

#### Jorg（JAX CE）
- 在 log 空间计算并做 soft clamp，控制指数溢出。  
- 证据：`jorg/src/jorg/statmech/chem_eq_jax.py:526`, `jorg/src/jorg/statmech/chem_eq_jax.py:539`, `jorg/src/jorg/statmech/chem_eq_jax.py:540`。

### 3.2 分子平衡常数 `logKp -> log_nK`

#### Korg.jl
- `get_log_nK = logKp - (n_atoms-1)*log10(kT)`，统一用于分子残差与回代。  
- 证据：`Korg.jl-1.0.1/src/statmech.jl:61`, `Korg.jl-1.0.1/src/statmech.jl:62`。

#### Jorg
- SciPy CE 与 JAX CE 都采用同一转换公式。  
- 证据：`jorg/src/jorg/statmech/korg_chemical_equilibrium.py:309`, `jorg/src/jorg/statmech/chem_eq_jax.py:569`, `jorg/src/jorg/statmech/chem_eq_jax.py:581`。

### 3.3 元素守恒 + 电荷守恒

#### Korg.jl
- 原子项：`atom_number_densities - (1+wII+wIII)*neutral_number_densities`。  
- 分子项：neutral molecule 扣元素守恒；charged diatomic 同时进电荷残差。  
- 证据：`Korg.jl-1.0.1/src/statmech.jl:308`, `Korg.jl-1.0.1/src/statmech.jl:317`, `Korg.jl-1.0.1/src/statmech.jl:328`, `Korg.jl-1.0.1/src/statmech.jl:331`。

#### Jorg（SciPy）
- 完整复现上述结构，且 Jacobian 中显式加入分子项导数。  
- 证据：`jorg/src/jorg/statmech/korg_chemical_equilibrium.py:341`, `jorg/src/jorg/statmech/korg_chemical_equilibrium.py:369`, `jorg/src/jorg/statmech/korg_chemical_equilibrium.py:371`, `jorg/src/jorg/statmech/korg_chemical_equilibrium.py:457`, `jorg/src/jorg/statmech/korg_chemical_equilibrium.py:498`。

#### Jorg（JAX）
- `optimality_fun` 同时构造 `R_elem` 与 `R_charge`，含 neutral/charged molecules。  
- 证据：`jorg/src/jorg/statmech/chem_eq_jax.py:560`, `jorg/src/jorg/statmech/chem_eq_jax.py:561`, `jorg/src/jorg/statmech/chem_eq_jax.py:567`, `jorg/src/jorg/statmech/chem_eq_jax.py:579`。

### 3.4 未知量参数化与缩放

#### Korg.jl
- 未知量是 92 个中性分数 + 缩放电子数（`ne/n_total*1e5`），并用 `abs(x)` 约束负值探索。  
- 证据：`Korg.jl-1.0.1/src/statmech.jl:186`, `Korg.jl-1.0.1/src/statmech.jl:294`, `Korg.jl-1.0.1/src/statmech.jl:297`, `Korg.jl-1.0.1/src/statmech.jl:300`。

#### Jorg（SciPy）
- 同样采用 `x=[neutral_fractions, ne/n_total*1e5]` 和 `abs` 约束。  
- 证据：`jorg/src/jorg/statmech/korg_chemical_equilibrium.py:666`, `jorg/src/jorg/statmech/korg_chemical_equilibrium.py:328`, `jorg/src/jorg/statmech/korg_chemical_equilibrium.py:332`。

#### Jorg（JAX）
- 使用无约束变量 `y`，通过 `sigmoid` 映射到 `f` 与 `ne`，并天然限制到物理区间。  
- 证据：`jorg/src/jorg/statmech/chem_eq_jax.py:552`, `jorg/src/jorg/statmech/chem_eq_jax.py:553`, `jorg/src/jorg/statmech/chem_eq_jax.py:705`。

### 3.5 非线性求解器与 Jacobian

#### Korg.jl
- `NLsolve(method=:newton, autodiff=:forward)`；残差 Jacobian 由 `ForwardDiff` 自动得到。  
- 同时支持对状态方程做隐式微分（`drdx`, `drdp`）。  
- 证据：`Korg.jl-1.0.1/src/statmech.jl:193`, `Korg.jl-1.0.1/src/statmech.jl:200`, `Korg.jl-1.0.1/src/statmech.jl:255`, `Korg.jl-1.0.1/src/statmech.jl:262`。

#### Jorg（SciPy）
- 以 `root` 为主，尝试 `hybr -> hybr(small ne) -> lm`，并可挂显式 Jacobian。  
- 工程上更偏“鲁棒求解控制”，数学上不是 Korg 的 Newton+AD 逐行等价。  
- 证据：`jorg/src/jorg/statmech/korg_chemical_equilibrium.py:684`, `jorg/src/jorg/statmech/korg_chemical_equilibrium.py:687`, `jorg/src/jorg/statmech/korg_chemical_equilibrium.py:699`, `jorg/src/jorg/statmech/korg_chemical_equilibrium.py:713`。

#### Jorg（JAX）
- `JAXopt(LevenbergMarquardt/Broyden) + custom_root`，支持 implicit diff。  
- 证据：`jorg/src/jorg/statmech/chem_eq_jax.py:624`, `jorg/src/jorg/statmech/chem_eq_jax.py:633`, `jorg/src/jorg/statmech/chem_eq_jax.py:652`。

### 3.6 初值与失败重试策略

#### Korg.jl
- 初值忽略分子；失败后以更小 `ne` 再试。  
- 证据：`Korg.jl-1.0.1/src/statmech.jl:124`, `Korg.jl-1.0.1/src/statmech.jl:197`, `Korg.jl-1.0.1/src/statmech.jl:199`。

#### Jorg
- SciPy 版保留相同思路并扩展了多阶段 attempt/stats。  
- JAX 版也有 residual 过大时的 fallback（设极小 `ne` 分数重解）。  
- 证据：`jorg/src/jorg/statmech/korg_chemical_equilibrium.py:647`, `jorg/src/jorg/statmech/korg_chemical_equilibrium.py:685`, `jorg/src/jorg/statmech/chem_eq_jax.py:913`, `jorg/src/jorg/statmech/chem_eq_jax.py:918`。

---

## 4. CE 之外的分子处理

### 4.1 分子线列表与格式支持（VALD/Kurucz/ExoMol）

#### Korg.jl
- 核心入口 `read_linelist` 支持 `vald/kurucz/...`，并提供 `load_ExoMol_linelist`。  
- 说明里明确了格式边界：Kurucz 分子支持存在限制（某些路径会抛错）。  
- 证据：`Korg.jl-1.0.1/src/linelist.jl:215`, `Korg.jl-1.0.1/src/linelist.jl:321`, `Korg.jl-1.0.1/src/linelist.jl:325`, `Korg.jl-1.0.1/src/linelist.jl:475`。

#### Jorg
- `lines/linelist.py` 已有 VALD/Kurucz 自动识别与 ExoMol loader。  
- 证据：`jorg/src/jorg/lines/linelist.py:92`, `jorg/src/jorg/lines/linelist.py:179`, `jorg/src/jorg/lines/linelist.py:511`, `jorg/src/jorg/lines/linelist.py:1281`。

### 4.2 预计算分子截面（MolecularCrossSection）

#### Korg.jl
- 有稳定的 `MolecularCrossSection` 构建、保存/读取、插值叠加接口。  
- 在 `synthesize` 中明确两次叠加：参考波长 `α_ref` 与主 `α`。  
- 证据：`Korg.jl-1.0.1/src/molecular_cross_sections.jl:4`, `Korg.jl-1.0.1/src/molecular_cross_sections.jl:95`, `Korg.jl-1.0.1/src/synthesize.jl:266`, `Korg.jl-1.0.1/src/synthesize.jl:296`。

#### Jorg
- 存在 `lines/molecular_cross_sections.py`，但实现是“简化近似版”并非 Korg 主实现等价。  
- 同时 `synthesize` 中虽暴露 `molecular_cross_sections` 参数，但当前文件只出现于函数签名/文档，主计算未实际调用。  
- 证据：`jorg/src/jorg/lines/molecular_cross_sections.py:227`, `jorg/src/jorg/lines/molecular_cross_sections.py:263`, `jorg/src/jorg/synthesis.py:332`, `jorg/src/jorg/synthesis.py:392`。

### 4.3 连续谱中的离子分子（H2+）

#### Korg.jl
- `H2plus_bf_and_ff` 明确在连续谱层面单独计算，注释中说明“离子分子尚未完整纳入分子 CE 时先 on-the-fly 处理”。  
- 证据：`Korg.jl-1.0.1/src/ContinuumAbsorption/absorption_H.jl:357`, `Korg.jl-1.0.1/src/ContinuumAbsorption/absorption_H.jl:390`, `Korg.jl-1.0.1/src/ContinuumAbsorption/absorption_H.jl:393`。

#### Jorg
- 有 `continuum/stancil1994.py` 数据与插值器，数据层准备充分。  
- 证据：`jorg/src/jorg/continuum/stancil1994.py:2`, `jorg/src/jorg/continuum/stancil1994.py:28`, `jorg/src/jorg/continuum/stancil1994.py:261`。

---

## 5. Jorg 的不足（对照 Korg.jl）

### 5.1 功能缺口
1. **`molecular_cross_sections` 主流程未接入**  
- 现状：`synthesize` 有参数但无实际叠加调用。  
- 对照：Korg 在 `α_ref` 与 `α` 都执行 `interpolate_molecular_cross_sections!`。  
- 证据：`jorg/src/jorg/synthesis.py:332`, `jorg/src/jorg/synthesis.py:392`, `Korg.jl-1.0.1/src/synthesize.jl:266`, `Korg.jl-1.0.1/src/synthesize.jl:296`。

2. **分子截面模块与主 statmech 链路存在双轨实现风险**  
- 现状：`statmech` 默认链路是 Barklem+ExoMol+polyatomic logK；但 `lines/molecular_cross_sections.py` 是独立简化近似模型。  
- 风险：同名概念（MolecularCrossSection）在物理一致性上可能分叉。  
- 证据：`jorg/src/jorg/statmech/korg_equilibrium_constants.py:193`, `jorg/src/jorg/statmech/korg_equilibrium_constants.py:321`, `jorg/src/jorg/lines/molecular_cross_sections.py:227`, `jorg/src/jorg/lines/molecular_cross_sections.py:296`。

### 5.2 数值稳健性缺口
1. **CE 失败回退会降级为简化 Saha，丢失分子耦合**  
- 现状：`LayerProcessor` 捕获异常后调用 `_saha_fallback`，只算原子（且仅到 Ni 的主元素）。  
- 对照：Korg 主路径设计目标是完整 CE 方程组求解。  
- 证据：`jorg/src/jorg/opacity/layer_processor.py:338`, `jorg/src/jorg/opacity/layer_processor.py:344`, `jorg/src/jorg/opacity/layer_processor.py:358`, `Korg.jl-1.0.1/src/statmech.jl:120`, `Korg.jl-1.0.1/src/statmech.jl:317`。

2. **PINN surrogate 物理约束仍是原子-only**  
- 现状：分子影响主要靠监督标签“学到”，不是显式残差硬约束。  
- 证据：`jorg/docs/implementation/chem_eq_pinn_design.tex:18`, `jorg/docs/implementation/chem_eq_pinn_design.tex:178`, `jorg/docs/implementation/chem_eq_pinn_design.tex:216`。

### 5.3 工程集成缺口
1. **主流程宣称 Korg-compatible，但分子截面关键环节未闭环**。  
2. **分子模块层次较多（statmech/lines/continuum），目前缺统一“单一真相路径（single source of truth）”约束。**

---

## 6. 改进优先级路线图（P0 / P1 / P2）

### P0-1：把 `molecular_cross_sections` 真正接入 `synthesize` 主流程
- 改动点：`jorg/src/jorg/synthesis.py`  
- 方案：对齐 Korg 的两处注入点  
- 方案说明：
  - 在参考波长锚点（相当于 Korg `α_ref`）叠加分子截面。
  - 在主波长网格 `alpha_matrix` 叠加分子截面。
- 收益：补齐最关键“对外参数有定义但不生效”缺口。  
- 风险：单位与 species key 对齐错误会导致系统偏差。  
- 验收标准：
  - 当传入非空分子截面时，`alpha_matrix` 与 `flux` 必须变化。
  - 日志/调试输出能报告“分子截面贡献非零层数与波段”。

### P0-2：CE fallback 观测与控制硬化
- 改动点：`jorg/src/jorg/opacity/layer_processor.py`  
- 方案：将 fallback 视为“受控降级”而非静默成功。  
- 具体：
  - 记录每层 fallback 触发原因、温压点、影响物种数。
  - 提供 `strict_ce=True` 选项：若 fallback 触发则抛错终止（用于精度任务）。
- 收益：防止 silently-wrong。  
- 风险：严格模式会降低鲁棒性（更多失败）。  
- 验收标准：
  - 默认模式可统计 fallback 比率。
  - 严格模式在 fallback 触发时 deterministic fail。

### P0-3：PINN 分子残差最小可用版
- 改动点：`jorg/src/jorg/statmech/chem_eq_pinn_loss.py`（及训练脚本）  
- 方案：先纳入“分子元素守恒残差”而非全分子动力学项。  
- 收益：减少 surrogate 在分子主导区间的物理漂移。  
- 风险：训练不稳定、梯度爆炸。  
- 验收标准：
  - 分子敏感区（低温、金属丰度变化）上 `ne` 与关键分子数密度误差下降。
  - 不牺牲现有 atomic residual 指标。

### P1-1：统一分子截面实现来源
- 改动点：`jorg/src/jorg/lines/molecular_cross_sections.py` 与 `jorg/src/jorg/synthesis.py`  
- 方案：明确其定位为“实验模块”或“生产模块”；若生产化则与 statmech 统一数据约束。  
- 收益：减少同名不同物理实现的维护风险。  
- 风险：重构工作量中等。  
- 验收标准：文档中明确唯一推荐路径，测试覆盖其单位与 species 映射。

### P1-2：CE 解算一致性基准测试
- 改动点：新增测试，如 `jorg/tests/test_ce_korg_parity.py`  
- 方案：固定网格对比 `korg_chemical_equilibrium.py`、`chem_eq_jax.py`、Korg 参考输出。  
- 收益：避免版本迭代引入回归。  
- 风险：基准数据维护成本。  
- 验收标准：误差阈值与 fallback 率阈值纳入 CI。

### P2-1：按场景自适应选择 CE 后端（SciPy/JAX/PINN）
- 改动点：`jorg/src/jorg/synthesis.py` 与配置层  
- 方案：引入后端策略器（精度优先/速度优先）。  
- 收益：把“正确性与性能”冲突显式化。  
- 风险：调度复杂度增加。  
- 验收标准：同一输入下可复现后端选择逻辑与性能收益。

---

## 7. 附录：关键代码路径索引

### Korg.jl
- `Korg.jl-1.0.1/src/species.jl`
- `Korg.jl-1.0.1/src/statmech.jl`
- `Korg.jl-1.0.1/src/synthesize.jl`
- `Korg.jl-1.0.1/src/molecular_cross_sections.jl`
- `Korg.jl-1.0.1/src/linelist.jl`
- `Korg.jl-1.0.1/src/ContinuumAbsorption/absorption_H.jl`
- `Korg.jl-1.0.1/src/read_statmech_quantities.jl`

### Jorg
- `jorg/src/jorg/statmech/species.py`
- `jorg/src/jorg/statmech/korg_chemical_equilibrium.py`
- `jorg/src/jorg/statmech/chem_eq_jax.py`
- `jorg/src/jorg/statmech/korg_equilibrium_constants.py`
- `jorg/src/jorg/opacity/layer_processor.py`
- `jorg/src/jorg/synthesis.py`
- `jorg/src/jorg/lines/linelist.py`
- `jorg/src/jorg/lines/molecular_cross_sections.py`
- `jorg/src/jorg/continuum/stancil1994.py`
- `jorg/docs/implementation/chem_eq_pinn_design.tex`

---

## 文档验收对照（与任务要求逐项对齐）
- 一致性检查：本文每个“明确差距”均给出 Korg 与 Jorg 双边证据。  
- 覆盖检查：已覆盖 CE 与 CE 外分子处理。  
- 可执行性检查：所有 P0 项含改动点/收益/风险/验收标准。  
- 读者可用性检查：首屏提供 10 行摘要结论。
