# Metal Backend Master Plan

## Goal

Keep Metal capability, backend-owned MPSGraph/custom-kernel routes, native resources, storage,
preparation, and execution truthful across the cold Prepare and hot Runtime lifecycle.

## Authority and contracts

[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) is authoritative. The exact applicable headings
are [`modules/runtime`](../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
[`modules/prepare`](../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
[Concrete backend modules](../../../architecture/contracts/backend-execution.md#concrete-backend-modules),
[Performance evidence and optimization tooling](../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling),
[Metal backend](../../../architecture/contracts/backend-execution.md#metal-backend), and
[Prepare lifecycle](../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle).

The non-authoritative [Metal strategy note](../../../design/notes/metal-backend-strategy.md) records
the residency/synchronization direction. [ADR 0002](../../../design/decisions/0002-backend-owned-lowering.md)
owns backend lowering, and [ADR 0013](../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)
owns persistent prepared resources. The [module-boundary](../../../architecture/module-boundaries.md),
[dependency](../../../architecture/dependency-rules.md), and
[Runtime/Prepare/backend](../../../architecture/runtime-prepare-backend-boundary.md) documents
explain the shared boundary.

## Scope and non-goals

Metal owns truthful capability, MPSGraph/custom lowering and route choice, Metal storage and
workspace, native application binary interface (ABI), prepared resources, execution, trace
contribution, typed route candidates, and the private compatibility needed before any outer
workload-cache decision can be accepted during prepare. Tuning tooling owns cache orchestration.

It does not own public Tensor semantics, global autograd, Planning ownership policy, shared
Prepare/Runtime interpretation of private knobs, Engine composition, CPU Apple routes, or a
Training-to-Metal optimizer bridge.

## Stable invariants and dependencies

- Planning selects only `owner = Metal`; Metal analysis selects a backend-private route and
  declares every shared requirement before slot assignment. Runtime invokes already-prepared work.
- Capability is implemented semantic truth, not hardware/library availability. Target, operation,
  type, Shape, layout, numerical policy, and resource validity filter before heuristic, cache, or
  measurement comparison.
- For one Metal occurrence/domain, `STRICT_IEEE` capability is a subset of `ACCELERATOR`
  capability. Accelerator may add only Model-authorized operation-local results; it may always
  execute the strict behavior and must never remove a strict operation or descriptor domain.
- Candidate generators return complete valid typed route configurations, are version-controlled,
  tested, and colocated with their routes. Shared orchestration treats them opaquely.
- Safe heuristics remain correct without tuning. Model 0026 independently owns future IEEE
  FLOAT16 semantics; two-byte storage implies neither BFLOAT16 nor FLOAT16 capability.
- Task 0005's strict `ADD`/`SUB`/`MUL`/`DIV` delivery remains historically withdrawn after its
  subnormal audit. Complete 0015 restores only tensor `FLOAT32` binary arithmetic under
  `ACCELERATOR`, with bounded DAZ/FTZ and no strict binary capability.
- Blocked 0010 added no BOOL capability; blocked 0011 added no scalar arithmetic capability;
  blocked 0012 added no extrema reduction capability; blocked 0013 added no cumulative-scan
  capability. Complete 0014 composes exact affine layouts plus `CONTIGUOUS`. Complete 0015 gives
  accelerator graphs the four canonical tensor-binary operations. Complete 0019 adds exact
  canonical `ABS` to both profile matrices. Complete 0020 adds accelerator canonical
  `SUM`/`MEAN`/`SUM_TO_SHAPE`; Complete 0021 adds accelerator positive static rank-two MATMUL after
  its fresh oracle, worker evidence, and independent Class C approval passed. Complete 0022
  delivered profile monotonicity at `41517594` plus documentation remediation `90cd5fd9` and final
  independent Class C approval. INT32 GATHER and ONE_HOT Task 0023 is Complete at implementation
  `9a7911c9` plus evidence `80e6cedd` after final independent lean Class C `APPROVE` with zero
  findings. Exact INT32 Scatter Elements replacement Task 0024 is Complete at implementation
  `a947e574732273bee4469d42afe8935082d53109` plus documentation remediation
  `aa191ca469010d081150e97dcddd504ec626dd9e` after final independent lean Class C `APPROVE` with
  zero findings. Exact FLOAT32 UNFOLD_AXIS Task 0025 is Complete at implementation
  `44edd86092509348e1e72cf7f0f4c3b13d141fa8`, first remediation
  `d0a947fa953bddb2714c1917c7114ac345c530b7`, and final remediation/reviewed revision
  `f88066e3ad0547987bb03b2d18ed2813f97de223`. Profile-common canonical FLOAT32 SOFTMAX Task 0026
  is Blocked after its mandatory run flushed a reference subnormal. Profile-common canonical
  FLOAT32 BatchNorm inference Task 0027 is Blocked after its sole run flushed signed minimum
  subnormals. Task 0030 is also Blocked: its sole direct `MAX_POOL2D` run returned negative zero
  for `[+0,-0]` and finite `3.0f` for all four qNaN/sNaN windows. Task 0031 is Blocked before a
  probe: direct MPSGraph structurally maps full Conv2d geometry, but its shape-dependent contraction
  algorithm is undocumented and one lean execution cannot authorize that full domain. Task 0032
  is likewise Blocked before a probe: MPSGraph has no direct MSE selector, and a subtraction/square
  decomposition cannot import accelerator binary relaxations into exact profile-common loss
  semantics. Task 0033 is Blocked before a probe: direct unmasked SDPA exists, but attention is
  unrelaxed and the selector leaves independent FTZ, accumulation, stable-softmax, special-class,
  and shape-algorithm gaps. No Metal task is Ready.
- Historical 0006, 0007, and 0009 remain Blocked records. Profile-qualified 0016 is also Blocked:
  its broad gate proved only exact `ABS`, while `EXP`/`SIGMOID` failed unchanged no-FTZ and one-ULP
  requirements. Metal 0017 remains Blocked under its old accelerator-reduction contract.
- Metal 0018 is Blocked under its unchanged historical MATMUL contract. Its mandatory fresh
  real-M3 smoke passed, but the first full direct `K=1` cell produced positive zero for
  `+0.0f * -1.0f`, while that task's permitted set contained only negative zero. Controls passed,
  the probe was removed, and no production change exists.
- Model 0029 is Complete at `30826783` after its proof, 23-task validation, documentation/diff
  scope check, and independent Class C approval passed. Metal 0021's completely fresh full
  real-M3 oracle passed 384 executables and 3,072 runs before production edits; implementation
  `ef2c6a1a` plus evidence corrections `be5543f9` and `5631d51f` passed final independent Class C
  review with zero findings.
- Complete Metal 0022 landed at implementation `41517594` plus documentation remediation
  `90cd5fd9`; final independent Class C review returned `APPROVE` with zero findings.
- Complete Metal 0023 landed at implementation `9a7911c9` plus evidence `80e6cedd`; its final lean
  Class C review approved with zero findings after one concrete-risk production-ABI SNaN probe.
- Complete Metal 0024 landed at implementation `a947e574732273bee4469d42afe8935082d53109`
  plus documentation remediation `aa191ca469010d081150e97dcddd504ec626dd9e`; final independent
  lean Class C review approved with zero findings.
- Complete Metal 0025 delivered bounded common-profile exact FLOAT32 UNFOLD_AXIS and passed final
  independent lean Class C review at reviewed revision `f88066e3ad0547987bb03b2d18ed2813f97de223`.
- Metal 0026 is Blocked without production changes: its only permitted direct-selector run returned
  `0x00000000` where the four-ULP oracle required `0x0008ec28`.
- Later Metal 0028 and 0029 remain reserved for separately authorized FLOAT16 and BFLOAT16 scopes.
- Metal 0030 is Blocked without production changes. Its one raw-winner execution passed ordinary
  and signed-subnormal preservation plus shape/input/guard controls, but failed signed-zero order
  and every NaN-class cell; schema 12/wire 20/POOL2D 7 and identities 13 remain unimplemented.
- Metal 0031 is Blocked without a device probe or production change. Its accelerator-only
  unbiased no-grad canonical FLOAT32 Conv2d candidate has an exact documented NCHW/OIHW mapping,
  but full-geometry numerical authorization requires either an exact custom kernel or preceding
  Model/architecture broadening; a prohibited matrix or one-geometry narrowing is not a route.
- Metal 0032 is Blocked without a device probe or production change. Its smallest loss candidate
  is profile-common canonical no-grad same-typed FLOAT32 MSE `NONE`; independent FTZ,
  special-class, and rounding gaps cannot be closed by one lean execution, and a matrix is
  prohibited. Exact custom-kernel proof, an authoritative complete MPSGraph MSE contract, or
  preceding operation-specific Model/architecture broadening is required.
- Metal 0033 is Blocked without a device probe or production change. Its smallest attention
  candidate is profile-common canonical FLOAT32 no-grad unmasked noncausal default-scale
  one-output SDPA. The current runtime can use only the macOS-15 direct selector, whose additive
  mask contract mismatches Model and whose documentation leaves multiple independent exact-result
  gaps. Authoritative complete selector documentation, an exact custom-kernel proof, or preceding
  attention-specific Model/architecture broadening is required; adjacent row relaxations cannot
  transfer.
- Model 0026 remains an independent FLOAT16 Draft. Model 0027–0029, Config 0006, Engine 0018, CPU
  0017, and Metal 0015/0019/0020/0021/0022/0023/0024/0025 are Complete.
- Production dependencies may point to Model, Config, Planning, Runtime, Prepare,
  Backend Contract, and Trace, never Engine or Training. Task 0002's Compiler edge is test-only.

## Package map

```text
io.github.pho001.synaptik.backend.metal
  public capability, explicit native-library configuration, and closeable Engine integration;
  package-private native ABI, device/queue, storage/workspace, preparation, routes, binding,
  and resources
io.github.pho001.synaptik.backend.metal.prepare
  deferred extraction only after multiple preparation families prove the seam
io.github.pho001.synaptik.backend.metal.route.mpsgraph
  deferred broader MPSGraph extraction; never a custom-kernel or CPU home
```

The first two executable routes remain coupled package-private types beside the three public
facade types. A later task must update this map before extracting a package or widening
visibility.

## Task list

| ID | Task | Status | Depends on | Conflicts with | Parallel group | Integration order | Integration validation | Intent/result |
|---|---|---|---|---|---|---|---|---|
| 0001 | [Metal capability, storage, and native foundation](tasks/0001-metal-capability-storage-and-native-foundation.md) | Complete | Complete shared planning, runtime, prepare, backend-contract, and trace contracts; no Model 0026 dependency while fail-closed | None | None | Any | Focused Metal foundation suite | Added a fail-closed provider, native context, and run-owned storage without executable capability. |
| 0002 | [MPSGraph prepared execution route](tasks/0002-mpsgraph-prepared-execution-route.md) | Complete | 0001; Runtime 0016; Prepare 0006; Engine 0010; Compiler 0006B7 | None | None | Any | Focused Metal and conformance suites | Added maximal-partition positive-shape contiguous FLOAT32 NEG through reusable MPSGraph preparation/execution. |
| 0003 | [Single-NEG custom Metal kernel route](tasks/0003-single-neg-custom-metal-kernel-route.md) | Complete | 0001–0002 | None | None | Any | Focused Metal/native suites | Added a private custom route for one NEG/feed/target within `1..UINT32_MAX`; all other supported partitions retain MPSGraph. |
| 0004 | [Typed Metal route candidate generators and cache compatibility](tasks/0004-typed-metal-route-candidate-generators-and-cache-compatibility.md) | Complete | 0002–0003, opaque prepare/tuning boundary and artifact versioning | None | None | Any | Focused candidate, Metal, and conformance suites | Added typed NEG candidates and a session-compatible authenticated codec foundation without outer tuning integration. |
| 0005 | [MPSGraph mixed NEG/binary FLOAT32 whole-partition route](tasks/0005-mpsgraph-mixed-binary-whole-partition.md) | Complete | 0001–0004; Engine 0017; Compiler 0006B7; Prepare 0008; Runtime 0016 | Metal capability/preparation/native ABI/Engine Metal scopes | None | ABI/schema → capability/analysis → Prepare → public Engine tests → docs/review | Native export audit; focused Metal/conformance; real public Engine integration; architecture checks; `git diff --check` | Originally delivered and approved after remediation `9f3a264`; an independent post-completion subnormal audit later required withdrawing all four binary operations. ABI v4 remains exact; schema 3 rejects wires `2..5`. |
| 0006 | [MPSGraph FLOAT32 unary algebra route](tasks/0006-mpsgraph-float32-unary-algebra.md) | Blocked | 0005; current Model unary semantics; Engine 0017; Compiler 0006B7; Prepare 0008; Runtime 0016 | Metal capability/preparation/native ABI/Engine Metal scopes; scalar schema work | None | Replacement route → numerical gates → Class C review | Reproducible real probe and fail-closed evidence; no production change | MPSGraph passed ABS ULP0, EXP ULP1, SIGMOID ULP1 but failed exact RECIPROCAL/LOG/SQRT/RSQRT/RELU/TANH cases; custom kernels or relaxed gates are required. |
| 0007 | [MPSGraph FLOAT32 SUM/MEAN reduction foundation](tasks/0007-mpsgraph-float32-reductions.md) | Blocked | 0005; current Model reduction attrs and Compiler autograd; Engine 0017; Compiler 0006B7; Prepare 0008; Runtime 0016 | Metal capability/preparation/native ABI/Engine Metal scopes; 0006 is not a dependency | None | Exact replacement route → numerical gates → Class C review | Reproducible real direct-output probe and fail-closed evidence; no production change | Eight runs returned positive zero for cancellation cases requiring SUM `2.0f` and MEAN `0.5f`; selector/schema/rank-zero feasibility does not satisfy exact Model semantics. |
| 0008 | [MPSGraph FLOAT32 affine transforms](tasks/0008-mpsgraph-float32-affine-transforms.md) | Complete | 0005; current Model/Compiler affine Shape and layout contracts; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007 | Metal capability/preparation/native ABI/Engine Metal materialization; shared aliasing or transfer work | None | Real selector/bit probe → typed schema → capability/analysis → Prepare/Runtime/materialization → Engine proof → docs/review | ABI-v4 export/schema audit; exact Shape/layout/raw-bit gates; direct affine publication; transfer non-widening; focused Metal/conformance/Engine/architecture checks | Approved after implementation `7e39f705`, remediation `9713e528`/`aa42ed711`, and independent Class C APPROVE with zero findings; five bounded affine transforms, ABI v4 exact. |
| 0009 | [MPSGraph FLOAT32 rank-two MATMUL training checkpoint](tasks/0009-mpsgraph-float32-rank2-matmul-training-checkpoint.md) | Blocked | 0008; Model/Compiler rank-two MATMUL and first-order rules; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007 | 0006/0007; Metal capability/preparation/native schema/candidate/materialization/Engine scopes | None | Exact replacement route → precision gates → Class C review | Reproducible real direct-target probe and fail-closed evidence; no production change | `K=1` multiplication by one flushed positive/negative minimum and ordinary subnormals to zero; no permitted FLOAT32 association/FMA yields zero. Independent `APPROVE-BLOCKER`; signed zero is not a blocker. |
| 0010 | [MPSGraph FLOAT32/BOOL predicates and selection](tasks/0010-mpsgraph-float32-bool-predicates-and-selection.md) | Blocked | 0008; current Model predicate/BOOL/WHERE semantics; Compiler forward capture; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007/0009 or Model 0026 | 0006/0007/0009; Metal capability/preparation/native schema/candidate/storage/materialization/Engine scopes | None | Exact replacement route → numerical gates → Class C review | Reproducible real direct-target comparison probe and fail-closed evidence; no production change | Apple M3 Max MPSGraph comparisons treated positive and negative minimum subnormal as equal to `±0` at optimization levels 0/1; raw identity, ordinary/special controls, canonical BOOL targets, canaries, and permutations passed. |
| 0011 | [MPSGraph FLOAT32 scalar pointwise arithmetic](tasks/0011-mpsgraph-float32-scalar-pointwise-arithmetic.md) | Blocked | 0005/0008; current Model scalar semantics and Compiler forward capture; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007/0009/0010 or Model 0026 | Every concurrent Metal capability/preparation/native schema/candidate/materialization/Engine scope | None | Exact replacement route → numerical gates → Class C review | Reproducible real scalar-bit/direct-target probe and independent blocker; no production change | The optimization-level 0/1 matrix found 17,328 mismatches in 77,824 checks, including signed-zero, subnormal, finite-division, and infinity-division failures; controls passed. |
| 0012 | [MPSGraph FLOAT32 extrema reductions](tasks/0012-mpsgraph-float32-extrema-reductions.md) | Blocked | 0005/0008; current Model ordinary-extrema semantics and Compiler forward capture; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007/0009/0010/0011 or Model 0026 | Every concurrent Metal capability/preparation/native schema/candidate/materialization/Engine scope | None | Probe → schema/geometry → capability/topology → candidates/lifecycle/scalar publication → Engine proof → docs/review | Reproducible exhaustive direct-target probe and independent blocker; no production change | Propagating MIN/MAX selectors were order-dependent for opposite signed zeros and zero/minimum-subnormal pairs: 2,016 MIN plus 2,016 MAX mismatches and 2,976 subnormal failures in 43,824 executions. |
| 0013 | [MPSGraph FLOAT32 cumulative scans](tasks/0013-mpsgraph-float32-cumulative-scans.md) | Blocked | 0005/0008; current Model cumulative-scan semantics and Compiler forward capture; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006/0007/0009–0012 or Model 0026 | Every concurrent Metal capability/preparation/native schema/candidate/materialization/Engine scope | None | Exact replacement route -> numerical gates -> Class C review | Reproducible exact cumulative-selector/direct-target probe and independent blocker; no production change | Inclusive length-one SUM/PRODUCT flushed all six subnormal corpus values at optimization levels 0/1; strict host controls preserved them. Axis 5 of rank 6 passed and axis 6 correctly rejected. Independent `APPROVE-BLOCKER`. |
| 0014 | [MPSGraph FLOAT32 affine layout composition](tasks/0014-mpsgraph-float32-affine-layout-composition.md) | Complete | 0005/0008; current Model layout semantics and Compiler inference; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016; not 0006–0007/0009–0013 or Model 0026 | Every concurrent Metal capability/preparation/native schema/candidate/materialization/Engine Metal scope | None | Probe -> capability/topology -> schema/state validation -> candidates/lifecycle/authentication -> Engine proof -> docs/review | Raw-bit selector-chain/direct-target probe; ABI-v4 exact exports; focused Metal/conformance/Compiler/Engine/architecture/Javadoc/docs checks | Commit `01e81be2` delivered exact local affine/`CONTIGUOUS` composition, schema/candidate version 4, and real Engine proof; independent Class C review returned `APPROVE` with zero findings. |
| 0015 | [ACCELERATOR FLOAT32 tensor binary arithmetic](tasks/0015-accelerator-float32-tensor-binary-arithmetic.md) | Complete | 0014; Model 0027; Config 0006; Engine 0018 | Every Metal capability/preparation/native schema/candidate/materialization scope | `numerical-profile-backends` (complete) | Integrated after CPU 0017; shared documents last | Native/export, focused Metal/conformance/public Engine, architecture, full-build, and documentation checkpoint | Delivered accelerator-only tensor `ADD`/`SUB`/`MUL`/`DIV` under the bounded DAZ/FTZ oracle; strict retains NEG/affine/`CONTIGUOUS`; schema 5 and version-six identities fail closed. |
| 0016 | [Profile-qualified FLOAT32 ABS, EXP, and SIGMOID](tasks/0016-profile-qualified-float32-abs-exp-sigmoid.md) | Blocked | 0015; Model 0027; Config 0006; Engine 0018 | 0017–0020 and every Metal capability/preparation/native schema/candidate/materialization/Engine Metal scope | None | Exact replacement gate only | Reproducible real-device oracle; no production change | Exact `ABS` passed, but `EXP`/`SIGMOID` flushed representable subnormal results and `SIGMOID` also exceeded its one-ULP gate; all controls passed and the probe was removed. |
| 0017 | [ACCELERATOR FLOAT32 SUM/MEAN/SUM_TO_SHAPE reductions](tasks/0017-accelerator-float32-sum-mean-sum-to-shape-reductions.md) | Blocked | 0019; Model 0027; Config 0006; Engine 0018 | 0018/0020 and any 0016 restart; every Metal capability/preparation/native schema/candidate/materialization/Engine Metal scope | None | Model decision or exact replacement gate only | Corrected 300-executable/2,400-run oracle; no production change | After probe-only NSString/pool ownership correction and a passing rank-16 axis-8 smoke, the full valid probe returned `+0` for `SUM([-0,-0])`; the old binary-tree FLOAT32 plus DAZ/FTZ contract permits only `-0`. |
| 0018 | [ACCELERATOR FLOAT32 rank-two MATMUL seeded-gradient checkpoint](tasks/0018-accelerator-float32-rank2-matmul-seeded-gradient-checkpoint.md) | Blocked | 0020; Model 0027; Config 0006; Engine 0018; not 0009/0016/0017 | Any 0016 restart; every Metal capability/preparation/native schema/candidate/materialization/Engine Metal scope | None | Model decision or exact replacement gate only | Fresh real-M3 oracle; no production change | Smoke passed, then the first full direct context-zero/optimization-zero `K=1` cell returned `+0` for `+0.0f * -1.0f`, outside the unchanged set containing only `-0`; reduced-precision-none set/read and controls passed, probe removed, clean tree. |
| 0019 | [Exact profile-qualified FLOAT32 ABS](tasks/0019-exact-profile-qualified-float32-abs.md) | Complete | 0015; Model 0027; Config 0006; Engine 0018; not 0016 | Any 0016 restart; 0017–0018/0020; every Metal capability/preparation/native schema/candidate/materialization/Engine Metal scope | None | Serial after 0015, before reduction/MATMUL successors | Exact both-profile Metal ABS checkpoint | Implementation `a6d1796d` plus remediation `bcb717a6`; ABI/export, Metal/conformance/Engine/architecture/full-build/docs passed and independent Class C final review approved with zero findings. |
| 0020 | [ACCELERATOR FLOAT32 reductions after exact-zero sign refinement](tasks/0020-accelerator-float32-reductions-after-zero-sign-refinement.md) | Complete | Model 0028; 0019; Model 0027; Config 0006; Engine 0018 | 0018 and any 0016 restart; every Metal capability/preparation/native schema/candidate/materialization/Engine Metal scope | None | Serial after Model 0028, before 0018 | Full fresh corrected 301-executable/2,401-run oracle, native/export, Metal/conformance/Engine/architecture/full-build/docs and independent Class C review | Implementation `9ddb75f6` plus documentation remediation `5b77c742`; bounded accelerator reductions, schema 7, version-eight identities, local scalar materialization, validation, and independent `APPROVE` with zero findings are complete. |
| 0021 | [ACCELERATOR FLOAT32 rank-two MATMUL after exact-zero sign refinement](tasks/0021-accelerator-float32-rank2-matmul-after-zero-sign-refinement.md) | Complete | Model 0029 after Complete; 0020; Model 0027; Config 0006; Engine 0018; not 0009/0016/0017/0018 | Any 0016 restart; every Metal capability/preparation/native schema/candidate/materialization/Engine Metal scope | None | Serial after Model 0029 | Refined full-contraction real-M3 oracle, native/export, Metal/conformance/Compiler/real Engine/architecture/full-build/docs, and independent Class C review | Implementation `ef2c6a1a`, evidence remediations `be5543f9`/`5631d51f`, and final independent `APPROVE` with zero findings delivered accelerator rank-two MATMUL, exact local transposes, seeded gradients, schema 8/wire 15, and version-nine identities. |
| 0022 | [ACCELERATOR profile capability monotonicity](tasks/0022-accelerator-profile-capability-monotonicity.md) | Complete | 0021; 0014–0015/0019–0020; Model 0027; Config 0006; Engine 0018 | Any 0016 restart; every Metal capability/preparation/native-preflight/candidate/codec/public Engine/architecture-status scope; 0023 | None | Serial after 0021 | Contract-minimal capability/Java-native preflight/identity/real Engine smoke, native build/export, architecture/full-build/docs, then independent Class C review | Implementation `41517594`, documentation remediation `90cd5fd9`, and final independent `APPROVE` with zero findings delivered exact NEG/ABS/affine/CONTIGUOUS under both profiles, retained accelerator-only families and strict negatives, edge-local MATMUL authentication, and unchanged ABI v4/schema 8/version-nine identities. |
| 0023 | [Exact INT32 GATHER and ONE_HOT](tasks/0023-exact-int32-gather-and-one-hot.md) | Complete | 0022 Complete; 0021; Model 0018C–0018D/0019A2; Compiler 0005C; CPU 0006A2; Config 0006; Engine 0018 | Any 0016 restart and every Metal capability/preparation/native schema/candidate/codec/ingress/materialization/public Engine scope | None | Implemented serially after 0022 from exact base `854c2e8e`; reviewed after evidence checkpoint `80e6cedd` | Documented selectors; one disposable one-run implementation smoke; focused OOB prevalidation, typed ingress/publication, conformance, Compiler guard, real Engine, native/export, architecture/full-build/docs evidence; one review-only production-ABI SNaN probe | Canonical positive-rank INT32 GATHER/ONE_HOT, exact invalid-index parity, typed ingress/local BOOL publication, schema 9/wires 16–17/DEPTH 5, identities 10, unchanged FLOAT32-only transfer, and final independent `APPROVE` with zero findings. |
| 0024 | [Exact INT32 SCATTER_ELEMENTS replacement](tasks/0024-exact-int32-scatter-elements-replacement.md) | Complete | 0023 Complete; Model 0018G–0018H/0018O/0025C; Compiler 0005C; CPU 0006B1; Config 0006; Engine 0018 | Any 0016–0018 restart; arithmetic scatter/Scatter-ND work; every Metal capability/preparation/native schema/candidate/codec/ingress/materialization/public Engine scope | None | Implemented serially from exact clean base `ad07e7e5d029f0472f1d05e82f2e869899d7dbf7`; documentation remediation followed implementation | One disposable one-run raw-bit Set smoke; focused bounds/duplicate, schema/identity, conformance, real Engine, native/export, architecture/Javadoc/full-build/docs proof; independent Class C review | Implementation `a947e574732273bee4469d42afe8935082d53109`, docs `aa191ca469010d081150e97dcddd504ec626dd9e`, final independent lean Class C `APPROVE` with zero findings. |
| 0025 | [Exact FLOAT32 UNFOLD_AXIS materialization](tasks/0025-exact-float32-unfold-axis-materialization.md) | Complete | 0024 Complete; Model 0017M–0017N/0018R/0023D; Compiler 0005C; CPU 0006A1; Config 0006; Engine 0018 | Any 0016–0018 restart; every concurrent Metal capability/schema/candidate/materialization scope; fold, 2D/3D window, pad, convolution, pooling, Model 0026, or shared production work | None | Implemented serially from exact clean planning base `c54ccbde0dc8defd4def1e25c3fd2898a1d66f4e`; two review remediations preceded final approval | One one-run raw-bit slice/expand/concat smoke; focused reused Metal/conformance/Engine/native/export/architecture/full-build/docs proof; independent lean Class C review | Implementation `44edd86092509348e1e72cf7f0f4c3b13d141fa8`, selector-cap/capability remediation `d0a947fa953bddb2714c1917c7114ac345c530b7`, wrong-kind remediation and final reviewed revision `f88066e3ad0547987bb03b2d18ed2813f97de223`; final `APPROVE` with zero findings. |
| 0026 | [Profile-common canonical FLOAT32 SOFTMAX](tasks/0026-profile-common-float32-softmax.md) | Blocked | 0025 Complete; Model 0016I–0016J; Compiler 0005B/0005C; CPU 0007E precedent; Config 0006; Engine 0018 | Any 0016–0018 restart; every concurrent Metal capability/schema/candidate/materialization scope; LOG_SOFTMAX, other normalization/loss/attention/fusion, profile-semantic, or shared production work | None | Mandatory one-run direct-selector numerical gate only | One `[4,8]` production-settings execution; no production change | Flattened index 18 returned positive zero `0x00000000` for the `-90.0f` input in the zero-max slice; StrictMath reference `0x0008ec28` differs by 584,744 ULPs. Probe removed; schema/identity/capability/gradient implementation did not begin. |
| 0027 | [Profile-common FLOAT32 batch-normalization inference](tasks/0027-profile-common-float32-batch-normalization-inference.md) | Blocked | 0025 Complete; Model 0021B; Compiler 0005B/0005C; CPU 0007F1 precedent; Config 0006; Engine 0018; 0026 is not a functional dependency | Every concurrent Metal capability/schema/native-preflight/candidate/codec/materialization/public Engine scope; training, multi-output, backward closure, and any 0016–0018 restart | None | Mandatory one-run direct-selector numerical gate only | One `[2,8,2]` production-settings execution; no production change | Channel 6 returned `0x00000000`/`0x80000000` where exact positive/negative minimum subnormals `0x00000001`/`0x80000001` were required. Other 30 cells and all five unchanged feeds passed; probe removed; implementation did not begin. |
| 0030 | [Profile-common exact canonical FLOAT32 MAX_POOL2D forward](tasks/0030-profile-common-exact-canonical-float32-max-pool2d-forward.md) | Blocked | 0025 Complete; Model 0020A; Compiler 0005D/0006B11; CPU 0008G precedent; Config 0006; Engine 0018 | Every concurrent Metal capability/schema/native-preflight/candidate/codec/materialization/public Engine scope; convolution, average/1D/3D pooling, backward, multi-output, and any 0016–0018 restart | None | Mandatory one-execution raw-winner gate only | One `[1,1,1,18]` disjoint-window execution; no production change | `[+0,-0]` returned negative zero, and first/later qNaN/sNaN windows returned finite `3.0f`; ordinary/subnormal/shape/input/guard controls passed, artifacts were removed, and implementation did not begin. |
| 0031 | [ACCELERATOR canonical FLOAT32 unbiased CONV2D forward](tasks/0031-accelerator-canonical-float32-unbiased-conv2d-forward.md) | Blocked | 0025 Complete; Model 0020; Compiler 0005D/0006B11; Config 0006; Engine 0018; 0030 blocker evidence is independent | Every concurrent Metal capability/schema/native-preflight/candidate/codec/materialization/public Engine scope; biased/gradient/Conv1d/Conv3d/pooling work; any 0016–0018 restart | None | Documentation-first structural review only; no device gate | Header/API mapping and exact candidate/schema research; no production change | Direct MPSGraph maps the full grouped NCHW/OIHW geometry, but its shape-dependent contraction algorithm is undocumented. One allowed-set execution cannot authorize all geometry, a matrix is prohibited, and narrowing to one probe shape is unsound. Unblocking requires an exact custom kernel or preceding Model/architecture broadening. |
| 0032 | [Profile-common canonical FLOAT32 no-grad MSE NONE](tasks/0032-profile-common-canonical-float32-no-grad-mse-none.md) | Blocked | 0025 Complete; Model 0022; Compiler 0005D/0006B11; CPU 0008I/0008M precedent; Config 0006; Engine 0018; prior blockers remain independent | Every concurrent Metal capability/schema/native-preflight/candidate/codec/materialization/public Engine scope; MSE SUM/MEAN, categorical/index loss, backward/training, and any 0016–0018 restart | None | Documentation-first selector and numerical review only; no device gate | Model/profile, installed-header, Compiler/CPU, current-schema, and prior-FTZ evidence; no production change | No direct MSE selector exists. Arithmetic decomposition cannot borrow operation-scoped accelerator binary relaxations, one run cannot close independent FTZ/special-class/rounding gaps, and the prohibited matrix is not a route. |
| 0033 | [Profile-common canonical FLOAT32 no-grad default SDPA forward](tasks/0033-profile-common-canonical-float32-no-grad-unmasked-noncausal-default-scale-sdpa-forward.md) | Blocked | 0025 Complete; Model 0019E/0023F; Compiler 0005D/0006B11; CPU 0008H precedent; Config 0006; Engine 0018; earlier blockers remain independent | Every concurrent Metal capability/schema/native-preflight/candidate/codec/materialization/public Engine scope; mask/causal/weights/backward work; any 0016–0018 restart | None | Documentation-first selector, runtime, mask, and numerical review only; no device gate | Model/profile, installed-header/runtime, Compiler/CPU, schema, and prior-FTZ evidence; no production change | Direct unmasked output-only SDPA is available on the runtime, but attention is unrelaxed and its documentation leaves independent FTZ, accumulation, stable-softmax, special-class, and shape-algorithm gaps. Additive mask semantics mismatch Model, one run cannot close the gaps, and schema 12/wire 20/ATTENTION 7/identity 13 remain conditional and unreserved. |

## Dependency DAG and authorized frontiers

Historical implementation branch:

`0001 -> 0002 -> 0003 -> 0004 -> 0005 -> 0008 -> {0009, 0010, 0011, 0012, 0013, 0014}`

Completed profile spine and serial successors:

`Model 0027 -> Config 0006 -> Engine 0018`

`0014 + Model 0027 + Config 0006 + Engine 0018 -> 0015`

`0015 -> {0016 (Blocked), 0019 (Complete) -> 0017 (Blocked)}`

`Model 0028 (Complete) -> 0020 (Complete) -> 0018 (Blocked)`

`0018 blocker evidence -> Model 0029 (Complete) -> 0021 (Complete) -> 0022 (Complete) -> 0023 (Complete) -> 0024 (Complete) -> 0025 (Complete) -> {0026 (Blocked), 0027 (Blocked), 0030 (Blocked), 0031 (Blocked), 0032 (Blocked), 0033 (Blocked)}`

Historical 0006–0007 and 0009–0013 keep their recorded `Blocked` status and evidence. Blocked
[0016](tasks/0016-profile-qualified-float32-abs-exp-sigmoid.md) keeps its failed three-operation
contract and is not a dependency of any successor. Complete
[0019](tasks/0019-exact-profile-qualified-float32-abs.md) remains the earlier serial Metal
successor. [0017](tasks/0017-accelerator-float32-sum-mean-sum-to-shape-reductions.md) remains
Blocked under its old exact-zero sign contract; Model 0028 and Metal 0020 are Complete.
[0018](tasks/0018-accelerator-float32-rank2-matmul-seeded-gradient-checkpoint.md) remains Blocked
under its unchanged contract after its fresh probe found the one-term zero-sign mismatch before
production edits. Model 0029 is Complete at `30826783` after its bounded proof, validation, and
independent Class C approval. Complete
[0021](tasks/0021-accelerator-float32-rank2-matmul-after-zero-sign-refinement.md) delivered
accelerator rank-two MATMUL plus seeded gradients at `ef2c6a1a`, corrected evidence at
`be5543f9`/`5631d51f`, and passed final independent Class C review with zero findings. Complete
[0022](tasks/0022-accelerator-profile-capability-monotonicity.md) corrected the accelerator
capability subset defect at `41517594`, remediated documentation at `90cd5fd9`, and passed final
independent Class C review with zero findings. Complete
[0023](tasks/0023-exact-int32-gather-and-one-hot.md) delivered at `9a7911c9` plus evidence
`80e6cedd` and passed final independent lean Class C review with zero findings. Exact replacement
[0024](tasks/0024-exact-int32-scatter-elements-replacement.md) completed at
`a947e574732273bee4469d42afe8935082d53109` plus documentation remediation
`aa191ca469010d081150e97dcddd504ec626dd9e` and final independent lean Class C approval.
[0025](tasks/0025-exact-float32-unfold-axis-materialization.md) is Complete at final reviewed
revision `f88066e3ad0547987bb03b2d18ed2813f97de223`.
[0026](tasks/0026-profile-common-float32-softmax.md),
[0027](tasks/0027-profile-common-float32-batch-normalization-inference.md),
[0030](tasks/0030-profile-common-exact-canonical-float32-max-pool2d-forward.md),
[0031](tasks/0031-accelerator-canonical-float32-unbiased-conv2d-forward.md),
[0032](tasks/0032-profile-common-canonical-float32-no-grad-mse-none.md), and
[0033](tasks/0033-profile-common-canonical-float32-no-grad-unmasked-noncausal-default-scale-sdpa-forward.md)
are Blocked without production changes. Tasks 0031–0033 additionally ran no probe. There is no
Ready Metal serial frontier.
These edges serialize shared Metal mutation; they do not claim that one operation family requires
another.

## Integration ownership and shared documents

- Integration owner: each serial task implementer, with mandatory independent Class C review.
- Shared documents: the next separately authorized task owns synchronized planning,
  architecture-status, API/user status, capability/preparer guides, and glossary updates after
  executable behavior stabilizes.
- Blocked Metal 0016–0018 and 0026–0027/0030–0033 have no active write or review scope. Complete
  Model 0028 owns the reduction semantic contract, Complete Model 0029 owns the MATMUL
  final-publication semantic contract, and Complete Metal 0021–0025 retain their reviewed
  implementations.

## Milestones and current frontier

Metal 0001–0005, 0008, 0014–0015, and 0019–0025 are Complete. Task 0015 landed at `42c4cfbf` plus
evidence-wording remediation `fb102a46`. Task 0019 landed at implementation `a6d1796d` plus
mixed-owner test remediation `bcb717a6`; its complete evidence and independent Class C final
`APPROVE` with zero findings passed. Metal 0020 landed at implementation `9ddb75f6` plus
documentation remediation `5b77c742`; its corrected fresh gate totaled 301 executables/2,401 runs,
and its ABI/export, 93 Metal, four conformance, nine real-dylib Engine, nine architecture, full
87-actionable-task build, documentation/diff checks, and independent Class C final `APPROVE` with
zero findings passed. Metal 0021 landed at implementation `ef2c6a1a`, worker-evidence remediation
`be5543f9`, and final evidence correction `5631d51f`; its fresh 384-executable/3,072-run real-M3
oracle, ABI/export, focused suites, real Engine forward/seeded-gradient proof, full build,
documentation/diff evidence, and independent Class C final `APPROVE` with zero findings passed.

Current ABI v4 retains exactly thirteen exports, points to node schema 11, and uses version-twelve
workload/exact-policy/candidate/compatibility/route/codec identities. Complete Model 0028 owns the
root-only exact-zero reduction rule. Complete Model 0029 owns the MATMUL-only final-publication
exact-zero sign rule. Metal 0018 remains Blocked without production changes. Complete Metal 0022
makes the exact NEG/ABS/affine/`CONTIGUOUS` domain common to both profiles with identical semantics
and valid accelerator composition. Its implementation `41517594`, documentation remediation
`90cd5fd9`, and final independent Class C `APPROVE` with zero findings were the reviewed base.
Complete Metal 0023 at `9a7911c9` adds exact common-profile FLOAT32+INT32 GATHER and
INT32-to-BOOL ONE_HOT at wires `16..17`/DEPTH `5`; evidence `80e6cedd` and final independent lean
Class C `APPROVE` with zero findings close it. Complete Metal 0024 adds exact common-profile
FLOAT32/INT32/FLOAT32 `SCATTER_ELEMENTS/NONE`, schema 10/wire 18, typed auxiliary third input,
bounds-before-uniqueness validation, and version-eleven identities while retaining ABI v4,
thirteen exports, and FLOAT32-only transfer. Implementation
`a947e574732273bee4469d42afe8935082d53109`, documentation remediation
`aa191ca469010d081150e97dcddd504ec626dd9e`, and final independent lean Class C approval close it.
Complete Metal 0025 adds bounded common-profile exact FLOAT32 UNFOLD_AXIS at schema 11, wire 19,
WINDOW_AXIS 6, and version-twelve identities. It landed at implementation
`44edd86092509348e1e72cf7f0f4c3b13d141fa8`, remediations
`d0a947fa953bddb2714c1917c7114ac345c530b7` and
`f88066e3ad0547987bb03b2d18ed2813f97de223`, and final independent lean Class C approval. Metal
0026 is Blocked: its mandatory single `[4,8]` production-settings direct-selector execution
returned `0x00000000` at flattened index 18 where the frozen four-ULP StrictMath oracle required
`0x0008ec28`, a 584,744-ULP failure. The probe was removed and no production capability changed.
Metal 0027 is also Blocked: its only `[2,8,2]` direct BatchNorm inference execution returned signed
zero for both signed minimum-subnormal channel-6 outputs. Metal 0030 is independently Blocked: its
sole `[1,1,1,18]` direct `MAX_POOL2D` execution returned negative zero for `[+0,-0]` and finite
`3.0f` for each first/later qNaN/sNaN window. Its ordinary/subnormal/shape/input/guard controls
passed and artifacts were removed. Metal 0031 is Blocked without a probe: MPSGraph's descriptor
structurally maps full grouped NCHW/OIHW Conv2d geometry, but the selector gives no contraction
algorithm contract, so one allowed-set execution cannot authorize every shape and lean policy
prohibits the needed matrix. Metal 0032 is also Blocked without a probe: the smallest
profile-common MSE `NONE` candidate has no direct MPSGraph selector, while a subtraction/square
decomposition cannot borrow accelerator binary DAZ/FTZ and leaves independent FTZ, special-class,
and rounding gaps that one run cannot close. Metal 0033 is Blocked without a probe: the
macOS-15 direct unmasked output-only SDPA selector is structurally available, but attention is
unrelaxed, additive mask semantics mismatch Model, and independent FTZ, accumulation,
stable-softmax, special-class, and shape-algorithm gaps cannot be closed by one run. Exact
custom-kernel proof, authoritative complete operation-scoped documentation, or preceding
operation-specific Model/architecture broadening is required. Schema 12/wire 20/attribute 7 and
version-thirteen identities remain conditional, unimplemented, and unreserved; no Metal task is
Ready.

Metal 0006 remains `Blocked` after exact RECIPROCAL/LOG/SQRT/RSQRT/RELU/TANH probe failures.
Metal 0007 remains independently `Blocked` after eight direct-output executions returned positive
zero where the current exact Model SUM/MEAN results are `2.0f`/`0.5f`.

Task 0005's original strict binary delivery remains historically withdrawn by its 12,096
subnormal-domain mismatches. Complete 0015 does not reopen that strict claim: it restores only
tensor `FLOAT32` `ADD`, `SUB`, `MUL`, and `DIV` under `ACCELERATOR` after a fresh bounded DAZ/FTZ
oracle passed every operation and control gate. Strict Metal still rejects binary arithmetic; an
explicit CPU may own it.

Metal 0009 is independently `Blocked`: at optimization levels `0` and `1`, three executables per
level and eight runs per case, `K=1` MATMUL by one flushed positive/negative minimum and ordinary
subnormals to zero, although minimum normals survived and every permitted host association/FMA
oracle preserved the minimum subnormal. Selector, direct-target, Shape, permutation, and transpose
gates passed; signed zero is not a blocker. Independent review returned `APPROVE-BLOCKER`. Tasks
0006, 0007, and 0009 retain no production, test, or probe changes.

Metal 0010 is independently `Blocked`: on Apple M3 Max, optimization levels `0` and `1`, three
independent executables per level, and eight runs per case, positive and negative minimum
subnormal compared equal to `±0` instead of remaining distinct. Raw identity preserved the input
bits; ordinary, infinity, NaN, canonical BOOL target, canary, and permutation controls passed.
No production, test, or probe changes remain.

Metal 0011 is independently `Blocked`: at optimization levels `0` and `1`, three independent
executables per level and eight runs per case, its exact embedded-scalar matrix found 17,328
mismatches in 77,824 checks. Reproducible failures include scalar `ADD +0` retaining input `-0`,
minimum-subnormal flushing, minimum-subnormal `DIV +0` producing NaN, maximum-finite self-division
producing zero, and infinity/maximum-finite division producing NaN. Exact constants, direct
targets, canaries, permutations, repetition, and identity controls passed. Independent review
confirmed no Model relaxation admits the results. No production, test, native, or probe changes
remain.

Metal 0012 is independently `Blocked`: at optimization levels `0` and `1`, reduced-precision fast
math disabled, three executables per length/level, and eight runs per case, the propagating MIN/MAX
selectors were order-dependent for opposite signed zeros and zero/minimum-subnormal pairs. The
43,824 executions recorded 2,016 MIN plus 2,016 MAX mismatches and 2,976 subnormal failures.
NaN classification, minimum normals, identity, canaries, and bindings passed. Independent review
returned `APPROVE-BLOCKER`; no production, native, test, or probe changes remain.

Metal 0013 is independently `Blocked`: at optimization levels `0` and `1`, reduced-precision fast
math disabled, twelve executables covering three per level/length pair for lengths `1` and `2`,
and eight direct runs for all eight kind/mode combinations, inclusive length-one SUM and PRODUCT
flushed all six subnormal corpus values to zero. Strict host `+0 + minimum-subnormal` and
`+1 * minimum-subnormal` controls preserved them; exclusive identities, raw identity, canaries,
and bindings passed. Corrected review evidence proves rank-six axis `5` valid for both kinds and
axis `6` correctly rejected, so axis is not a blocker. Independent review returned
`APPROVE-BLOCKER`; no production, native, test, or probe changes remain.

Metal 0015 is Complete. [0016](tasks/0016-profile-qualified-float32-abs-exp-sigmoid.md) is Blocked
without production changes: its 324-executable/2,592-run Apple M3 Max gate proved exact `ABS`, but
`EXP` and `SIGMOID` reproducibly flushed representable subnormal results to positive zero across
levels, Shapes, contexts, and forms; `SIGMOID` also produced two-ULP ordinary finite results. All
controls passed and the disposable probe was removed. The unchanged broad contract is not weakened.

Complete [0019](tasks/0019-exact-profile-qualified-float32-abs.md) adds only exact canonical `ABS`
under both profiles. `EXP`, `SIGMOID`, `RECIPROCAL`, `LOG`, `SQRT`, `RSQRT`, `RELU`, and `TANH`
remain closed.

[0017](tasks/0017-accelerator-float32-sum-mean-sum-to-shape-reductions.md) remains Blocked under its
old contract. Its corrected 300-executable/2,400-run gate passed the valid rank-16 axis-8 smoke and
then found `SUM([-0,-0]) -> +0` outside the old allowed set; no production change or probe artifact
remains. Model 0028 completed the final-result-only exact-zero sign refinement at `fc003ab8`.
Detailed [0020](tasks/0020-accelerator-float32-reductions-after-zero-sign-refinement.md) reran the
full fresh corrected gate successfully, removed the disposable probe, implemented the bounded
accelerator reduction domain at `9ddb75f6`, completed documentation remediation at `5b77c742`, and
passed independent Class C review with zero findings.

[0018](tasks/0018-accelerator-float32-rank2-matmul-seeded-gradient-checkpoint.md) remains Blocked
under its unchanged contract. Its fresh real-M3 non-square direct smoke passed, then its first full
direct context-zero/optimization-zero `K=1` cell published `+0` for `+0.0f * -1.0f`, where only
`-0` was then permitted. Reduced-precision-fast-math-none set/read, selector, direct-target, and
controls were valid; zero is not subnormal and K=1 has no reassociation/FMA alternative. The probe
was removed and no production change remains. Model 0029 completed the separate MATMUL-only
final-publication decision at `30826783`: the underlying product remains `-0`, and only publication
may choose `+0`. Complete Metal 0021 then passed its complete fresh oracle, removed the disposable
probe, delivered the scoped accelerator route through `ef2c6a1a` plus evidence corrections
`be5543f9`/`5631d51f`, and passed independent Class C review with zero findings. Complete Metal
0022 corrected accelerator profile capability monotonicity through `41517594` plus documentation
remediation `90cd5fd9` and final independent approval. Complete Metal 0023 delivered its focused
exact indexing scope through `9a7911c9` plus evidence `80e6cedd` and final independent approval.
Complete Metal 0024 delivered exact functional Scatter Elements replacement through
`a947e574732273bee4469d42afe8935082d53109` plus documentation remediation
`aa191ca469010d081150e97dcddd504ec626dd9e` and final independent approval. Complete Metal 0025
delivered only exact bounded FLOAT32 UNFOLD_AXIS materialization at implementation `44edd860`,
remediation `d0a947fa`, and final reviewed remediation `f88066e3`; final independent review
approved with zero findings. Metal 0026 then ran its one permitted production-settings direct
SOFTMAX gate and is Blocked by subnormal probability flushing: actual `0x00000000`, reference
`0x0008ec28`, distance 584,744 ULPs. Metal 0027 is independently Blocked after its sole direct
BatchNorm gate flushed positive/negative minimum-subnormal outputs to same-signed zero. Neither
task changed production or retained a probe artifact. Comparison, scalar, extrema, scan,
LOG_SOFTMAX, every normalization, broader window/fold, BatchNorm training, backward closure, and
Model 0026 remain unauthorized.

## Delivered lifecycle and ABI boundary

- Task 0001 is deliberately non-executing and fail closed. Task 0002 established the strict
  positive-rank canonical FLOAT32 NEG domain, task 0014 added strict affine/`CONTIGUOUS`
  composition, task 0015 added the accelerator tensor-binary domain, task 0019 added exact
  canonical ABS to both profile matrices, task 0020 added the accelerator canonical
  `SUM`/`MEAN`/`SUM_TO_SHAPE` domain, and task 0021 adds accelerator positive static rank-two
  MATMUL with exact local transposes and explicitly seeded gradients. Task 0003 changes
  singleton-NEG route choice, not occurrence capability or partitioning.
- The caller-supplied macOS arm64 native library uses an Objective-C C ABI reached through JDK 26
  Foreign Function and Memory (FFM). It is not packaged or discovered by the backend. ABI version
  1 established seven foundation functions and statuses `0..7`; version 2 retained them, added
  three typed MPSGraph functions and statuses `8..11`; version 3 retained all ten, added three
  custom-NEG functions and status `12`; version 4 replaced only the NEG-specific MPSGraph create
  operation with the typed whole-partition create operation. Its pointed-to node table currently
  uses schema version 11. Opaque resource kinds are never reinterpreted. ABI v4 exports exactly:

  ```text
  synaptik_metal_foundation_abi_version
  synaptik_metal_context_create
  synaptik_metal_context_release
  synaptik_metal_buffer_create
  synaptik_metal_buffer_release
  synaptik_metal_buffer_upload
  synaptik_metal_buffer_download
  synaptik_metal_mpsgraph_executable_create
  synaptik_metal_mpsgraph_executable_release
  synaptik_metal_mpsgraph_executable_run
  synaptik_metal_neg_kernel_pipeline_create
  synaptik_metal_neg_kernel_pipeline_release
  synaptik_metal_neg_kernel_pipeline_run
  ```

  The old `synaptik_metal_mpsgraph_neg_executable_create` symbol is absent.
- Tasks 0008, 0014, 0015, 0019, 0020, 0021, 0022, 0023, 0024, and 0025 retain the exact
  thirteen-symbol ABI. Schema version 11 admits NEG/affine/`CONTIGUOUS` under both profiles on
  wires `1` and `6..11`, exact ABS under both profiles on wire `12`, accelerator tensor binary on
  wires `2..5`, accelerator `SUM=13`/`MEAN=14` with typed reduction forms, accelerator
  `MATMUL=15`, exact common-profile `GATHER=16`/`ONE_HOT=17` with `DEPTH=5`, replacement
  `SCATTER_ELEMENTS=18`, and bounded common-profile `UNFOLD_AXIS=19` with `WINDOW_AXIS=6`. Java
  profile/type preflight rejects every incompatible node set before downcall. Explicit
  unavailable/canonical/affine-view states enforce graph-local view provenance, including exact
  local rank-two transpose authentication for MATMUL.
  Affine outputs use authenticated full-logical-size represented-order targets. Scalar FLOAT32
  reductions materialize as four bytes; local BOOL materializes as one byte per element; exact
  canonical FLOAT32/INT32 ingress is current; canonical-only positive-rank FLOAT32 transfer remains
  unchanged.
- Analysis validates the complete maximal Metal partition, selects the route, and declares exact
  buffers/workspaces. Finalization cannot change that route or add undeclared shared requirements;
  it creates route-specific persistent resources only after slot assignment.
- Finalizer, shared Prepare, and `PreparedExecution` transfer persistent resources transactionally.
  The prepared execution is the sole long-lived Runtime owner; repeated executable or schedule
  occurrences add no ownership. Physical cleanup stays backend-private and deterministic.
- Caller buffers are borrowed per run; splat/output buffers and binding workspaces are run-owned.
  Cold binding performs checked type/address work. The hot route uses direct typed handles and one
  synchronous native submission/downcall into assigned destinations, without route selection.
- `MTLBuffer` remains the resident representation across adjacent Metal work. A prepared
  shape-specialized `MPSGraphExecutable` is not per-run workspace; transfers and waits occur only
  at explicit boundaries required by the synchronous lifecycle.
- Public `MetalBackendConfiguration` snapshots one explicit absolute native-library path.
  `MetalBackendIntegration.open(...)` validates ABI, opens one default-device context, supplies
  capability/availability/Prepare/ingress/materialization roles, and rolls partial opening back.
  Ownership transfers only through Engine's explicit builder; Metal performs no discovery.

## Task 0004 delivered constraints and remaining risks

- 0004 derives complete typed candidates from canonical workload facts, target capability,
  exact policy, and budget; operation family chooses a generator but is not a universal cache key.
- Compatibility includes explicit schema/version and target/workload identity. An optional
  encoded decision selects only a compatible candidate and still leads to fresh authenticated
  preparation; 0004 adds no outer cache hit or measurement path.
- The delivered boundary contains no `Map<String,Object>`, reflection, string dispatch, central
  knob registry, generic parameter bag, Planning route choice, Runtime cache access, or hidden
  global resource.
- The completed 0004 brief fixes route-specific typed shapes and conservatively session-scoped
  target compatibility without changing cache-file or native ABI ownership. Its package-private
  codec remains unconsumed by current `tools/tuning` and Engine tuning composition. Explicit
  single-owner Metal Engine execution is current without consuming that codec. No broader
  operation/type, mixed-owner Engine path, async execution, packaging/discovery, or performance
  claim is implied.
- Main risks are moving lowering into shared layers, leaking Metal fields through opaque seams,
  confusing capability with availability, and extending native/prepared lifetimes beyond their
  explicit owners.

## History and update policy

Detailed ABI inventories, native/test evidence, context identifiers, and route implementation
chronology remain in tasks 0001–0004 and Git history. The strategy note is planning guidance, not
a capability, platform, ABI, or performance promise.

Update this map only for task order/status/result, dependencies, package direction, a future Metal
gate, or a live lifecycle/ABI risk. Keep detailed evidence in task briefs. If a planning change
conflicts with `ARCHITECTURE.md` or an accepted ADR, stop and use the architecture-decision
process.
