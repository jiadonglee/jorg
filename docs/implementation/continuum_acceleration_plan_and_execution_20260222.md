# Jorg Continuum 提速计划与执行记录（2026-02-22）

## 1. 目标与约束（锁定）
- 硬件目标：CPU 优先。
- 精度目标：保持现有 continuum 物理精度，不以放宽误差换速度。
- 工程目标：continuum 主路径移除运行时 SciPy 插值依赖（保留 legacy 回退）。
- API 约束：不破坏外部 synthesis API。
- 分阶段改造：先插值热点替换，再 batch/缓存接入，再基准与 notebook 收敛。

## 2. 实施计划（当前版本）
1. 基线与剖析固化。
2. 通用 JAX 插值层替代 SciPy（支持 Flat/Line/Zero）。
3. 热点模块逐个替换（Stancil/Peach/hydrogenic/helium/hydrogen H2+）。
4. Continuum fast path 接口化（cache/layout/backend 开关）。
5. 集成到 benchmark 与 notebook 输出（Jorg vs Korg 分步表）。
6. 补测试（插值一致性、组件一致性、fast vs legacy 一致性）。

## 3. 执行步骤与状态

### Step 1: JAX 插值基础层
**状态：完成**
- 新增文件：`/Users/jdli/Project/jorg/jorg/src/jorg/continuum/interp_jax.py`
- 提供：
  - `interp1_linear_clamped`
  - `interp2_linear_clamped`
  - `interp2_linear_extrap_line`
- 边界行为：`flat / line / zero`。

### Step 2: SciPy 热点替换
**状态：完成**
- `stancil1994.py`：去 `RegularGridInterpolator`，改为 `interp_jax`，并匹配 Korg `Line` 外推。
- `peach1970.py`：去 `RegularGridInterpolator`，改为 `interp_jax`，并匹配 Peach `oob=0`。
- `hydrogenic_bf_ff.py`：去 `RegularGridInterpolator`，改为 `interp_jax`。
- `helium.py`：去 `RectBivariateSpline`，改为表格化 `interp_jax`。

### Step 3: H2+ 路径向量化
**状态：完成**
- `stancil1994.py` 新增批量接口：`h2plus_bf_ff_batch(...)`。
- `hydrogen.py` 的 `h2_plus_bf_ff_absorption(...)` 去掉逐波长 Python loop，改为批量调用。

### Step 4: fast path / cache / backend 开关
**状态：完成（第一版）**
- `exact_physics_continuum.py` 新增：
  - `ContinuumTableCache`
  - `ContinuumSpeciesLayout`
  - `total_continuum_absorption_fast(...)`
  - `total_continuum_absorption_batch_fast(...)`
- `layer_processor.py` 新增：
  - `continuum_backend: {"legacy", "jax_fast"}`
  - `continuum_cache`
  - coarse-grid 复用缓存

### Step 5: 基准脚本与产物
**状态：完成**
- 新增：`/Users/jdli/Project/jorg/jorg/examples/benchmark_continuum_hotspots.py`
- 输出目录：`/Users/jdli/Project/jorg/jorg/output/benchmarks/continuum_accel_baseline/<run_id>/`
- 产物：
  - `manifest.json`
  - `pinn_ce_source_breakdown.json`
  - `layer_processor_breakdown.json`
  - `jorg_vs_korg_per_step_timing_table.csv`

### Step 6: 测试补齐
**状态：完成**
- 新增测试文件：`/Users/jdli/Project/jorg/jorg/tests/test_continuum_jax_fast.py`
- 覆盖内容：
  - `test_interp_jax_matches_scipy_stancil_grid`
  - `test_interp_jax_matches_scipy_peach_grid`
  - `test_hydrogenic_ff_interp_parity`
  - `test_h2plus_bf_ff_vectorized_matches_legacy`
  - `test_total_continuum_absorption_fast_matches_legacy`

### Step 7: Notebook 更新
**状态：完成**
- 更新文件：`/Users/jdli/Project/jorg/jorg/output/jupyter-notebook/jorg-vs-korg-pinn-step-by-step.ipynb`
- 保留目标表格样式：
  - `Jorg vs Korg per-step timing (seconds)`
  - 列：`J_atm/K_atm, J_ce/K_ce, J_line/K_line, J_rt/K_rt, J_tot/K_tot`
- 新增 Stage1 拆分输出：`J_ce_only`, `J_cntm_only`。
- CSV 也同步新增上述字段。

### Step 8: continuum 热点二次剖析与批量核改造
**状态：完成（2026-02-22 当日追加）**
- 目标：按上一轮结论，优先处理 `metal_bf / positive_ion_ff / H_I_bf`。
- 剖析（56 层，200 像素，continuum-only）：
  - `H_I_bf_fast`：约 `21.5%` wall time。
  - `metal_bf_absorption`：约 `19.1%` wall time。
  - `positive_ion_ff_absorption`：约 `10.1%` wall time。
- 改造项：
  1. `metals_bf.py`  
     - 新增默认 fast path：把 metal species 截面表堆叠为单个张量，单次 JAX kernel 计算总和（替代逐 species 多次 dispatch）。  
     - 保留 `species_list` 自定义时的 fallback 路径。  
     - 对默认路径做一致性校验：fast path 与 fallback `max_abs=0, max_rel=0`。
  2. `positive_ion_ff.py`  
     - 迁移频率选择与常量到循环外，减少重复切片和常量计算。  
     - 对零/负密度提前跳过。
  3. `h_i_bf_api.py`  
     - `H_I_bf_fast` 新增 Nahar 数据一次性加载标记，避免每次调用重复进入 loader 分支。

### Step 9: H I batch 核接入与可控开关
**状态：完成（2026-02-22 当日追加）**
- 新增 `H_I_bf_fast_batch(...)`，并在 `total_continuum_absorption_batch_fast(...)` 中接入批量 H I bf 计算。
- `LayerProcessor` 的 `enable_batch_continuum` 保留可选开关（默认关闭）：
  - 开启后在 continuum-only 路径使用批量 kernel；
  - 关闭时保持逐层路径，兼容当前行为。
- 基准脚本新增参数：`--enable-batch-continuum`，用于直接做 A/B。

### Step 10: one-shot 冷启动缓解（JAX 持久编译缓存）
**状态：完成（2026-02-22 当日追加）**
- 新增：`/Users/jdli/Project/jorg/jorg/src/jorg/jax_runtime.py`
  - 在包初始化时配置 JAX persistent compilation cache（跨进程复用）。
  - 默认缓存目录：`~/.cache/jorg/jax_compilation_cache`。
  - 支持环境变量：
    - `JORG_ENABLE_JAX_PERSISTENT_CACHE`（默认启用）
    - `JORG_JAX_CACHE_DIR`
    - `JORG_JAX_CACHE_MIN_COMPILE_SECS`
    - `JORG_JAX_CACHE_MIN_ENTRY_BYTES`
- 接入点：`/Users/jdli/Project/jorg/jorg/src/jorg/__init__.py`
  - 导入即调用 `configure_jax_runtime()`。
- 独立进程验证（清空缓存后连续两次）：
  - run1：`3.48s`
  - run2：`2.11s`
  - 约 `1.65x`（run1/run2）冷启动后续加速。

## 4. 关键改动文件清单
- `/Users/jdli/Project/jorg/jorg/src/jorg/continuum/interp_jax.py`
- `/Users/jdli/Project/jorg/jorg/src/jorg/continuum/stancil1994.py`
- `/Users/jdli/Project/jorg/jorg/src/jorg/continuum/peach1970.py`
- `/Users/jdli/Project/jorg/jorg/src/jorg/continuum/hydrogenic_bf_ff.py`
- `/Users/jdli/Project/jorg/jorg/src/jorg/continuum/helium.py`
- `/Users/jdli/Project/jorg/jorg/src/jorg/continuum/hydrogen.py`
- `/Users/jdli/Project/jorg/jorg/src/jorg/continuum/exact_physics_continuum.py`
- `/Users/jdli/Project/jorg/jorg/src/jorg/continuum/metals_bf.py`
- `/Users/jdli/Project/jorg/jorg/src/jorg/continuum/positive_ion_ff.py`
- `/Users/jdli/Project/jorg/jorg/src/jorg/continuum/h_i_bf_api.py`
- `/Users/jdli/Project/jorg/jorg/src/jorg/opacity/layer_processor.py`
- `/Users/jdli/Project/jorg/jorg/examples/benchmark_continuum_hotspots.py`
- `/Users/jdli/Project/jorg/jorg/tests/test_continuum_jax_fast.py`
- `/Users/jdli/Project/jorg/jorg/output/jupyter-notebook/jorg-vs-korg-pinn-step-by-step.ipynb`

## 5. 验证与结果
- 测试：`pytest -q /Users/jdli/Project/jorg/jorg/tests/test_continuum_jax_fast.py` 通过（7/7）。
- 基准脚本冒烟：成功生成 baseline run 目录与产物。
- Step 8 剖析对比（同配置）：
  - `metal_bf_absorption`：`9.83 ms/call -> 9.14 ms/call`（约 `1.08x`）。
  - 三热点合计 wall 占比仍约一半，但 `metal_bf` 明显下降。
- Step 9 批量开关评估：
  - one-shot（新进程单次）开启 `--enable-batch-continuum` 时，受 JIT 首次编译影响，当前样例下可能变慢。
  - warm-run（同进程热身后）对 continuum-only 场景有小幅收益（约 `1%~2%`，200/1000/2001 像素中位数）。
  - 因此保持默认关闭，作为可选实验开关保留。
- Step 10 冷启动评估：
  - 持久缓存目录已生成 JIT 条目，二次独立进程运行显著快于首次冷启动。

## 6. 当前结论
- 已经完成“可切换 backend + 去 SciPy 运行时插值 + H2+ 向量化 + 可复现基准”的工程闭环。
- Step 8 的热点内核优化已落地，但端到端 one-shot 结果仍受 JIT 冷启动影响，速度收益与样本规模/运行方式相关。
- Step 10 已将 cold-start 问题转为“首次编译成本 + 后续跨进程复用”模式，重复任务可稳定受益。
- 下一轮重点应放在：
  1. 继续压降 `H_I_bf_fast`（当前仍为第一热点）。
  2. 对超大编译条目做缓存策略约束（目录轮转/清理策略）。
  3. 用固定 warm-up + 3 次重复中位数固化 benchmark。

## 7. 快速复现实验命令
```bash
python /Users/jdli/Project/jorg/jorg/examples/benchmark_continuum_hotspots.py \
  --pixel-list 200,1000,2001 \
  --spec-list 1,3 \
  --wl-min 5000 --wl-max 5020 \
  --continuum-backend jax_fast
```

```bash
pytest -q /Users/jdli/Project/jorg/jorg/tests/test_continuum_jax_fast.py
```
