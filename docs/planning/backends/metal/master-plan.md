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
  capability. Accelerator may add only results inside Model's total recursive FLOAT32 set; it may
  always execute the strict behavior and must never remove a strict operation or descriptor
  domain.
- Candidate generators return complete valid typed route configurations, are version-controlled,
  tested, and colocated with their routes. Shared orchestration treats them opaquely.
- Before any future Metal operation becomes capability-true, every selectable candidate needs a
  complete-domain subset proof from an authoritative selector contract or auditable custom/
  composition algorithm. A bounded device corpus is regression/rejection evidence only and cannot
  authorize unsampled ranks, extents, broadcasts, axes, modes, or shape-dependent opaque behavior.
  A single domain-proved numerical-pass survivor needs no comparative cost gate. Multiple survivors
  with identical proven semantics/domain may close without timing only by strict structural
  dominance: no more compute dispatches, no more route-owned temporary bytes, and at least one
  strict improvement, using exact source-bound facts for closed custom routes or supported actual
  observation for opaque routes. Multiple nondominated survivors remain pending an authorized
  controlled environment. Local-device timing is diagnostic only and never qualification,
  route-selection, or tuning-identity authority. Route choice is cold and fixed; no runtime
  benchmark, retry, fallback, or matrix is permitted.
- Safe heuristics remain correct without tuning. Model 0026 owns IEEE FLOAT16 semantics; the
  completed low-precision closure maps every admitted homogeneous FLOAT32 occurrence to BFLOAT16
  and FLOAT16 without treating shared two-byte storage as capability evidence.
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
  `f88066e3ad0547987bb03b2d18ed2813f97de223`. Historical profile-common FLOAT32 SOFTMAX Task
  0026 and BatchNorm inference Task 0027 remain Blocked on their frozen strict-side/no-FTZ gates.
  With Complete Model 0030, their observed FTZ values are recursively reachable for
  `ACCELERATOR`, but neither one run nor an opaque direct selector proves the complete recursive
  output subset or supplies a strict route. Task 0030 remains independently Blocked because exact
  max-pool signed-zero/NaN winner selection is outside the recursive floating floors. Task 0031 is
  Blocked before a probe: direct MPSGraph structurally maps full Conv2d geometry, but its
  shape-dependent contraction algorithm is undocumented and one lean execution cannot prove a
  complete recursive subset. Task 0032 remains Blocked before a probe under its frozen
  profile-common premise. MSE arithmetic is now recursively reachable under `ACCELERATOR`, but
  MPSGraph has no direct selector and no complete decomposition/custom route, schema, capability,
  full-formula, or gradient-route proof exists. Task 0033 remains Blocked before a probe: direct
  unmasked SDPA exists and attention arithmetic is recursively reachable under `ACCELERATOR`, but
  additive mask semantics mismatch Model and the opaque selector lacks complete-domain proof for
  contractions, scale, guards, softmax, output, specials, and shape-dependent algorithms. Task
  0034 remains Blocked before a probe: direct one-output SORT exists, but exact ordering
  documentation omits independent stability, NaN, signed-zero, subnormal, exact-bit, and
  shape-algorithm properties. Task 0035 remains Blocked before a probe: direct dropout hides
  randomness and returns one output, while opaque Philox state cannot represent Model's exact
  zero-input/two-input/three-output INT64 state transition. Task 0036 remains Blocked before a
  probe: direct Conv3d inherits 0031's opaque contraction boundary; average Pool3d arithmetic is
  recursively reachable, but mapping/divisor/route proof is absent, and max Pool3d retains exact
  winner-selection obligations exposed by 0030. Task 0037 remains Blocked before a probe:
  recurrent arithmetic is recursively reachable under `ACCELERATOR`, but direct RNN lacks runtime
  INT64 valid lengths, atomic validation, skipped padded work, and `finalHidden`, and the opaque
  selector lacks complete recurrence/state proof. Documentation/audit-only Task 0038 is Complete.
  Task 0040 failed its one BFLOAT16 raw-bit Gather gate. Task 0041 is Complete at implementation
  `ba16d942` plus remediation `386705ca` after final Class C approval with zero findings. Task 0042
  is Complete at `9feb2505705263b6efb417d606678c606c2b9598` after final independent Class C
  review returned `APPROVE` with zero findings. Task 0043 is Complete at remediation
  `77e6091b2a452faa04fa2b674bc295ddc7be88b7` after same-reviewer Class C approval with zero
  remaining findings. Task 0044 is Complete as a documentation/audit no-change evaluation from
  planning revision `3d458b7a1d356f32fcd2c04de408ae9aa22355b0`. Task 0045 is Complete at
  same-reviewer-approved P1 remediation `26c6c911`: it owns the exact ad-hoc-signed local macOS
  arm64 package contract. Task 0046 is Complete at independently approved implementation
  `4aad1ab6ced318107e65bb9beef0013f8a7ff6e5`. Task 0047 is Blocked on a real GPU-capable hosted
  runner. Task 0048 is Complete at independently approved implementation `89f9fbb9`. Documentation-
  only Task 0049 is Complete after approved P1 remediation `6d4246f7`. Evidence-only Task 0050
  completed the sole final packaged/extracted-Metal full build. Task 0051 remains Blocked as the
  consumed historical EXP/SIGMOID oracle record: its sampled direct EXP and inherited composed
  SIGMOID routes are `DOMAIN-BLOCKED`, its other candidates failed numerically, and its one-shot
  oracle is never rerun. Task 0053 is now Blocked on the unavailable pinned no-axiom
  constructive-real exponential bridge; its retained source/evidence remain fail-closed and earned
  no `DOMAIN-PASS`. Task 0057 is Complete for exact BOOL classification/logic/WHERE. Task 0052 is
  Complete. Its Gate 1A/API, Gate 1B custom-route proof, sole Apple M3 Max numerical oracle,
  structural survivor adjudication, route freeze, and atomic production cutover all passed.
  Every opaque direct candidate remains `DOMAIN-BLOCKED`; all sixteen custom candidates passed.
  Fourteen operation rows had one survivor; fused CLAMP is fixed
  solely because its one dispatch/zero route bytes strictly dominate composed CLAMP's two
  dispatches/4,194,304 bytes. Retained local timing is diagnostic history only. The immutable
  Gate-1/Gate-2 and Gate-3 packages remain under
  [`evidence/0052/`](evidence/0052/README.md) and
  [`evidence/0052-gate3/`](evidence/0052-gate3/README.md). Documentation-only Task 0054 remains the
  exact historical pre-cutover `19+2+15+79=115` inventory and names its supersession. Task 0055 is
  Complete at `ddeff1b2`: its historical ABI 5/schema 13, six-carrier typed transfer, identity 14,
  and zero-new-capability foundation are landed. Task 0056 is Complete at implementation
  `4f35576c`, conformance correction `4a5cbdef`, and approved review remediation `1718b28a`: the
  exhaustive dual-route catalog and immutable prepared-route identity landed with zero capability,
  device, oracle, timing, benchmark, public-API, or autotuning change. Task 0057 is Complete at
  implementation `e5d9d5c7` plus approved validation remediation `7a3bb072`.
  Task 0058's original bounded implementation completed at `13a2e540` plus evidence/contracts
  `ac261e88`. Its authorized no-gradient scalar/reciprocal extension is Complete at implementation
  `5ab9c44c` plus contracts/evidence `63c070cc` and independent cumulative Class C approval with
  zero P0/P1/P2. It admits only ACCELERATOR canonical positive-rank FLOAT32 no-gradient scalar
  `ADD/SUB/MUL/DIV` and `RECIPROCAL`.
- Task 0059 is Complete at schema/layout cutover plan `6478d249`, implementation/focused proof
  `783eabe1`, package/schema documentation remediation `ac0db5c4`, BOOL atomic-publication proof
  `536e52b8`, and cumulative review remediation `02a097eb`; independent cumulative Class C review
  of `6478d249..02a097eb` returned `APPROVE` with zero P0/P1/P2. It owns only the remaining cast,
  indexing, and layout wires `39` and `69..84`; previously approved wires `6..19` are regression
  scope. The cutover is exactly `61/54` capability and `79/36` structural execution. SELECT and
  positive-step SLICE support all six carriers over exact resolved positive-stride non-overlapping
  storage layouts; unresolved, zero/negative-stride, overlapping, out-of-span, gradient-bearing,
  empty, and reduction/update/fold domains remain false. Schema 15 and identity 16 replace schema
  14/identity 15; ABI 5 and thirteen exports remain.
- Task 0060 is Complete from clean base `f3ad5e12`: planning `dd94e492`, implementation
  `d06db07e`, native/public Engine proof `eda09533`, backend/native documentation `62f18cd8`, and
  final fold-domain/count remediation `a4fe4754`. Independent cumulative re-review of
  `55cdebc3..a4fe4754` returned `APPROVE` with zero P0/P1/P2. It admits only replacement
  SCATTER_ND, signed/crop SLICE_UPDATE, and non-overlapping
  `FLOAT64`/`FLOAT32`/`BFLOAT16` exact-copy/zero-fill folds, modular INT32/INT64 PROD, and BOOL
  ALL/ANY. Capability is exactly `69/46`, structural execution `87/28`, the MPSGraph catalog
  `75/35/5`, and custom catalog
  `46/69/0`. SCATTER_ADD, scatter reductions, overlapping folds, ARG extrema, and advanced floating
  aggregates remain false. Schema 15, identity 16, ABI 5, and thirteen exports remain fixed.
- Task 0061 is Complete at corrected approved plan `d81ec940` and final reviewed implementation
  head `3913bac1`; independent cumulative code, evidence, and security reviews returned zero
  P0/P1/P2. It admits no-gradient promoted INT32/INT64 MATMUL under both profiles and, under
  accelerator, every positive-static FLOAT32 vector/matrix/batched/broadcast geometry with
  generated gradients plus no-gradient mixed BFLOAT16/FLOAT32 pairs. Existing all-FLOAT32
  rank-two matrix products, including authenticated transposes nested in custom programs, retain
  MPSGraph; new forms use seven fixed custom signatures with exact physical transpose addressing.
  Schema 15, ABI 5, thirteen exports, `69/46` capability, `87/28` structural execution, and
  `75/35/5` MPSGraph catalog remain fixed. The custom catalog is `47/68/0`, and all backend-local
  identities are version 17 with version 16 rejected.
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
- Metal 0026 and 0027 remain historically Blocked without production changes under their frozen
  profile-common/no-FTZ gates. Complete Model 0030 makes the observed SOFTMAX/BatchNorm FTZ
  values recursively reachable only for `ACCELERATOR`; neither sample proves a complete
  opaque-selector subset or supplies a strict route.
- Historical Metal 0028's FLOAT16 reservation is recorded by planning-only Task 0039 as Blocked on
  Draft Model 0026. Historical Metal 0029 is now detailed by Blocked Task 0040, whose direct
  BFLOAT16 Gather selector canonicalized the first selected signaling NaN.
- Complete Task 0066 supersedes only Task 0040's future-route conclusion with an exact custom raw-
  movement GATHER route. Task 0040's direct-selector `0xffa6 -> 0x7fc0` failure and every historical
  evidence record remain unchanged.
- Metal 0030 is Blocked without production changes. Its one raw-winner execution passed ordinary
  and signed-subnormal preservation plus shape/input/guard controls, but failed signed-zero order
  and every NaN-class cell; exact winner selection remains outside the recursive floating floors.
- Metal 0031 is Blocked without a device probe or production change. Its accelerator-only
  unbiased no-grad canonical FLOAT32 Conv2d candidate has an exact documented NCHW/OIHW mapping,
  but its opaque shape-dependent contraction lacks complete recursive-subset proof over the full
  geometry. One sample or prohibited matrix cannot supply that proof.
- Metal 0032 remains Blocked without a device probe or production change under its frozen
  profile-common premise. MSE arithmetic is now recursively reachable under `ACCELERATOR`, but
  MPSGraph has no direct selector and no complete decomposition/custom implementation, formula,
  generated-gradient route, or complete-domain proof exists. Schema 13 only carries the structure.
- Metal 0033 remains Blocked without a device probe or production change. Complete Model 0030
  recursively reaches `ACCELERATOR` attention arithmetic, but the current direct selector's
  additive mask contract mismatches Model and its opaque contractions, scale, guards, softmax,
  output, special classes, gradients, and shape algorithms lack complete-domain subset proof.
- Metal 0034 is Blocked without a device probe or production change. Its smallest ordering
  candidate is profile-common canonical rank-one FLOAT32 no-grad ascending SORT. ARGSORT's direct
  INT32 result mismatches Model INT64. Schema 13 now carries INT64 and two-output TOP_K, but no
  proved cast composition, exact TOP_K output-order route, or exact custom stable-sort kernel
  exists.
- Metal 0035 is Blocked without a device probe or production change. Its smallest nondegenerate
  candidate is profile-common exact canonical rank-one FLOAT32 no-grad INITIAL_STATE plus explicit-
  state DROPOUT over the complete probability domain. MPSGraph's hidden-random one-output dropout
  and opaque Philox state cannot implement exact key/counter-plus-N, mask, next-state, binary64
  probability, or special/scaling semantics. Schema 13 now carries zero-input, three-output, and
  INT64 structure, but no exact custom random route/lifecycle/proof exists; `p=0` is not a
  substitute.
- Metal 0036 remains Blocked without a device probe or production change. Direct Conv3d inherits
  0031's opaque shape-dependent contraction boundary. Average Pool3d arithmetic is recursively
  reachable for `ACCELERATOR`, but mapping, ceil-tail, divisor, and route proof are absent; max
  Pool3d retains exact winner selection exposed by 0030. UNFOLD3D/FOLD3D remain separate custom
  movement/overlap work.
- Metal 0037 remains Blocked without a device probe or production change. Complete Model 0030
  recursively reaches `ACCELERATOR` recurrent arithmetic, but direct RNN lacks runtime INT64 valid
  lengths, atomic validation, skipped padded work, zero-padding, and final-hidden output. Schema 13
  carries the required cardinality/type structure, but no complete recurrent implementation,
  state-publication, GRU/LSTM, or gradient-route proof exists.
- At this pre-P12 checkpoint, Model 0026 remained an independent FLOAT16 Draft. Model 0027–0029,
  Config 0006, Engine 0018, CPU 0017, and Metal 0015/0019/0020/0021/0022/0023/0024/0025 were
  Complete.
- Task 0038 is Complete from base `ec4499bd9e18ed863c9091baa9cfe82dbcf06a9b`. It froze the exact
  implemented common/accelerator occurrence matrices and synchronized stale current summaries
  without changing production, native, schema, identity, or test code.
- Task 0039 remains historically Blocked at that Draft Model-0026 checkpoint without a probe or
  production change. P12 subsequently completed public FLOAT16 type, oracle, promotion,
  conversion, family contracts, schema 19, and identity 28; it did not retroactively change the
  task's evidence or decision.
- Task 0040 is Blocked from exact clean planning revision
  `c300582727ec568a7482dd974f9d1b2e2e13f82c`. Its sole direct BFLOAT16 Gather execution returned
  `0x7fc0` for required selected raw word `0xffa6`; inputs were unchanged, the later output/guard
  checks made no success claim, artifacts were removed, and no production/schema change exists.
- Task 0041 completed its bounded prepared-route/invocation trace at implementation `ba16d942`
  plus remediation `386705ca`, from exact clean common base
  `0e796a82f18fbaaa1d5210638f41e0785e22094c` after Trace 0003's final independent
  public-API/documentation approval. Final independent Class C rereview returned `APPROVE` with
  zero findings.
- Task 0042 is Complete at `9feb2505705263b6efb417d606678c606c2b9598` after final independent
  Class C review returned `APPROVE` with zero findings. It adds retained public local/complete-plan
  Metal tuning collaborations and the minimum private Engine adapter/fallback generalization for
  exactly one eligible singleton-`NEG` partition, with session-only reuse and no native/capability
  change.
- Task 0043 is Complete at remediation `77e6091b2a452faa04fa2b674bc295ddc7be88b7`
  after same-reviewer Class C approval with zero remaining findings.
- Task 0044 is Complete from planning revision `3d458b7a1d356f32fcd2c04de408ae9aa22355b0`.
  It records the bounded report-only facts and retains both singleton-`NEG` routes plus the existing
  no-selected-decision safe heuristic without a performance endorsement or production change.
- Task 0045 is Complete at remediation `26c6c911` after the same independent Class C reviewer
  returned `APPROVE` with zero remaining findings. Its exact schema-1 ad-hoc local package and
  fail-closed verifier are the only native inputs Task 0046 consumes.
- Task 0046 is Complete at `4aad1ab6ced318107e65bb9beef0013f8a7ff6e5` after independent Class C
  review returned `APPROVE` with zero findings. It adds one opt-in verifier-backed reproducible
  local ZIP without build, signing, install, discovery, publication, versioning, or loader behavior.
- Task 0047 is planning-only and Blocked: standard `macos-26` has no documented usable Metal GPU,
  while required M2 GPU runner `macos-26-xlarge` is unavailable to the current personal public
  repository without an eligible organization/enterprise plan, billing, and positive spending.
- Task 0048 is Complete at independently approved implementation `89f9fbb9`. It persists immutable
  source-owned splats per prepared execution and closes the optimization frontier without a general
  mutable output/workspace pool, whose result/concurrency lifetime remains unsafe under the current
  contract.
- Task 0049 is Complete after documenting the existing synchronous completion barrier and one
  system-default Metal device context per integration. Its P1 status-drift remediation `6d4246f7`
  passed final cumulative independent Class C review with `APPROVE` and zero findings.
- Task 0050 is Complete after final program verification from exact Task 0049 completion revision
  `2c6326a9`: the current native package/archive/extraction verifier chain passed and the sole final
  full build executed all 87 actionable tasks against the extracted dylib.
- Production dependencies may point to Model, Config, Planning, Runtime, Prepare,
  Backend Contract, and Trace, never Engine or Training. Task 0002's Compiler edge is test-only.

## Package map

```text
io.github.pho001.synaptik.backend.metal
  public capability, explicit native-library configuration, typed trace observer, retained local
  and complete-plan tuning collaborations, and closeable Engine integration; package-private
  native ABI, device/queue, storage/workspace, preparation, trace producer, routes, binding,
  generator/batch/decision/codec, and resources
io.github.pho001.synaptik.backend.metal.prepare
  deferred extraction only after multiple preparation families prove the seam
io.github.pho001.synaptik.backend.metal.route.mpsgraph
  deferred broader MPSGraph extraction; never a custom-kernel or CPU home
```

Task 0042 adds only the two cohesive public collaboration types beside the existing public facades.
Both wrap the existing private version-twelve route values; they do not widen the visibility of
routes, compatibility facts, native state, or the trace producer. A later task must update this map
before extracting a package or widening another type.

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
| 0031 | [ACCELERATOR canonical FLOAT32 unbiased CONV2D forward](tasks/0031-accelerator-canonical-float32-unbiased-conv2d-forward.md) | Blocked | 0025 Complete; Model 0020; Compiler 0005D/0006B11; Config 0006; Engine 0018; 0030 blocker evidence is independent | Every concurrent Metal capability/schema/native-preflight/candidate/codec/materialization/public Engine scope; biased/gradient/Conv1d/Conv3d/pooling work; any 0016–0018 restart | None | Documentation-first structural review only; no device gate | Header/API mapping and exact candidate/schema research; no production change | Direct MPSGraph maps the full grouped NCHW/OIHW geometry, but its opaque shape-dependent contraction lacks complete recursive-subset proof. One execution or prohibited matrix cannot authorize the full domain; a conforming custom or completely proved selector route is required. |
| 0032 | [Profile-common canonical FLOAT32 no-grad MSE NONE](tasks/0032-profile-common-canonical-float32-no-grad-mse-none.md) | Blocked | 0025 Complete; Model 0022; Compiler 0005D/0006B11; CPU 0008I/0008M precedent; Config 0006; Engine 0018; prior blockers remain independent | Every concurrent Metal capability/schema/native-preflight/candidate/codec/materialization/public Engine scope; MSE SUM/MEAN, categorical/index loss, backward/training, and any 0016–0018 restart | None | Documentation-first selector and numerical review only; no device gate | Model/profile, installed-header, Compiler/CPU, current-schema, and prior-FTZ evidence; no production change | Historical profile-common scope remains Blocked. Complete Model 0030 recursively reaches `ACCELERATOR` MSE arithmetic and schema 13 carries MSE structure, but no direct selector or complete decomposition/custom implementation, formula, gradient route, and complete-domain proof exists. |
| 0033 | [Profile-common canonical FLOAT32 no-grad default SDPA forward](tasks/0033-profile-common-canonical-float32-no-grad-unmasked-noncausal-default-scale-sdpa-forward.md) | Blocked | 0025 Complete; Model 0019E/0023F; Compiler 0005D/0006B11; CPU 0008H precedent; Config 0006; Engine 0018; earlier blockers remain independent | Every concurrent Metal capability/schema/native-preflight/candidate/codec/materialization/public Engine scope; mask/causal/weights/backward work; any 0016–0018 restart | None | Documentation-first selector, runtime, mask, and numerical review only; no device gate | Model/profile, installed-header/runtime, Compiler/CPU, schema, and prior-FTZ evidence; no production change | Review-needed Model 0030 recursively reaches `ACCELERATOR` attention arithmetic, but additive mask semantics mismatch Model and the opaque selector lacks complete-domain proof for contractions, scale, guards, softmax, specials, output, gradients, and shape-dependent algorithms. |
| 0034 | [Profile-common canonical FLOAT32 no-grad ascending SORT](tasks/0034-profile-common-canonical-float32-no-grad-ascending-sort.md) | Blocked | 0025 Complete; Model 0019C/0019C1; Compiler 0005C/0006B11; CPU 0006C precedent; Config 0006; Engine 0018; earlier blockers remain independent | Every concurrent Metal capability/schema/native-preflight/candidate/codec/materialization/public Engine scope; ordering/top-K/INT64/gradient work; any 0016–0018 restart | None | Documentation-first selector, ordering, type, and arity review only; no device gate | Model/profile, installed-header/runtime, Compiler/CPU, current schema/type/publication evidence; no production change | Direct one-output SORT aligns structurally, but ordering is unrelaxed and the header omits independent stability/NaN/signed-zero/subnormal/exact-bit/shape-algorithm properties. ARGSORT returns INT32 rather than Model INT64. Schema 13 now carries INT64 and two-output TOP_K, but no proved cast/TOP_K output-order route exists and one run cannot close the gaps. |
| 0035 | [Profile-common canonical FLOAT32 no-grad explicit-state dropout](tasks/0035-profile-common-canonical-float32-no-grad-explicit-state-dropout.md) | Blocked | 0025 Complete; Model 0019B/0019B1; Compiler 0003/0005C/0006B11; CPU 0006D precedent; Config 0006; Engine 0018; earlier blockers remain independent | Every concurrent Metal capability/schema/native-ABI/preflight/candidate/codec/local-type/materialization/public Engine scope; multi-output/INT64/random/gradient work; any 0016–0018 restart | None | Documentation-first state/API/schema review only; no device gate | Model/profile, installed-header, Compiler/CPU, current schema/type/lifecycle evidence; no production change | Direct dropout hides randomness and returns one output; opaque Philox state cannot implement exact Model INT64[2] initialization/transition or mandatory mask/three-output dropout. Schema 13 carries the required structure, but no exact custom random route, lifecycle proof, or route approval exists. |
| 0036 | [Extended 3D inference](tasks/0036-extended-3d-inference.md) | Blocked | 0025 Complete; Model 0025H/0025J/0025K; Compiler 0006B/0006B1/0006B2; CPU 0008A/0008G1 precedent; Config 0006; Engine 0018; 0030/0031 evidence is independent | Every concurrent Metal capability/schema/native-preflight/candidate/codec/materialization/public Engine scope; biased/gradient convolution, pooling, 3D-window/custom-kernel work; any 0016–0018 restart | None | Documentation-first 3D selector, mapping, and numerical review only; no device gate | Model/profile, installed-header/runtime, Compiler/CPU, current schema, and 0030/0031 evidence; no production change | Direct Conv3d inherits the opaque contraction boundary. Average Pool3d arithmetic is recursively reachable for `ACCELERATOR` but lacks mapping/divisor/route proof; max Pool3d retains exact winner selection exposed by 0030. |
| 0037 | [Profile-common canonical FLOAT32 no-grad FORWARD RNN_TANH](tasks/0037-profile-common-canonical-float32-no-grad-forward-rnn-tanh.md) | Blocked | 0025 Complete; Model 0025E/0025F; Compiler 0006A; Config 0006; Engine 0018; current CPU has no recurrent precedent; earlier blockers remain independent | Every concurrent Metal capability/schema/native-ABI/preflight/candidate/codec/input-type/materialization/public Engine scope; recurrent, multi-input, multi-output, INT64, custom-kernel, or gradient work; any 0016–0018 restart | None | Documentation-first recurrent API, arity, type, and numerical review only; no device gate | Model/profile, installed-header/runtime, Compiler/CPU, current schema/type/lifecycle, and prior tanh evidence; no production change | Complete Model 0030 recursively reaches `ACCELERATOR` recurrent arithmetic, but direct RNN lacks runtime valid lengths, atomic validation, skipped padding, and final hidden. Schema 13 carries the required signature; no complete recurrence/state/gradient route or route approval exists. |
| 0038 | [Current profile capability manifest closure](tasks/0038-current-profile-capability-manifest-closure.md) | Complete | 0022–0025 Complete; finalized Metal capability evidence through 0037; Model 0027–0029; Config 0006; Engine 0018 | Every concurrent Metal capability/schema/native/candidate/codec documentation scope and shared README/architecture/API/backend/user/master/roadmap edits | None | Planning authorization `1bb2a0e5`, then one documentation/audit closure commit from exact base `ec4499bd9e18ed863c9091baa9cfe82dbcf06a9b` | Focused existing capability/conformance tests and targeted architecture checks passed; Markdown/link/terminology, diff, and documentation-only scope passed; no device probe or full build | Exact current common and accelerator-only occurrence matrices, strict-subset monotonicity, fail-closed remainder, and ABI/schema/identity boundary are synchronized; every Blocked row remains unsupported and no manifest API was added. |
| 0039 | [Profile-common IEEE FLOAT16 prerequisite gate](tasks/0039-profile-common-ieee-float16-prerequisite-gate.md) | Blocked | Model 0026 Complete; reviewed public binary16 representation, promotion, conversion, family semantics, profile result sets, and downstream fail-closed enum adoption; Metal 0038 Complete | Every concurrent Model dtype/profile task and Metal capability/schema/native/type/ingress/publication/identity/transfer scope | None | Documentation-only prerequisite record; post-dependency implementation requires separate authorization and schema revalidation | Planning links/anchors/fences/final newlines, status consistency, documentation-only path scope, and `git diff --check`; no build, test, native compile, or device probe | No public FLOAT16 oracle exists. `MPSDataTypeFloat16` is representation evidence only; current Metal plumbing is closed. Task 0040 consumed no schema/type/identity values, so this candidate remains conditional and unreserved. |
| 0040 | [Profile-common exact canonical BFLOAT16 Gather](tasks/0040-profile-common-exact-canonical-bfloat16-gather.md) | Blocked | 0023, 0025, and 0038 Complete; current Model BFLOAT16/Gather semantics; Compiler indexing capture; CPU 0006A2; Config 0006; Engine 0018 | Every concurrent Metal capability/schema/native/type/ingress/publication/identity/transfer scope and any restart consuming the same next values | None | Sole mandatory direct-selector gate only; no implementation authorized after failure | Exactly one graph/compile/execution over the 20-word reverse-permutation corpus; planning/status/Markdown/diff validation only | First output returned canonical quiet-NaN `0x7fc0` instead of selected negative signaling-NaN `0xffa6`. Inputs were unchanged; later output/guard checks made no success claim. Artifacts were removed, schema 12/wire 20/type 4/identities 13 remain unimplemented and unreserved, and production is unchanged. |
| 0041 | [Prepared route and invocation trace](tasks/0041-prepared-route-and-invocation-trace.md) | Complete | Trace 0003 Complete and independently approved after remediation at `0e796a82f18fbaaa1d5210638f41e0785e22094c`; Metal 0001–0005/0008/0014–0015/0019–0025; Engine 0017–0018 | Every concurrent Metal integration/preparation/executable/native-failure/hot-path/API scope and shared Trace/Metal/API/roadmap documents | None | Implemented at `ba16d942`; remediated at `386705ca`; final independent Class C rereview approved with zero findings | Focused Metal trace tests; actual public Engine session collector scenario; architecture/Javadoc/docs/diff checks; independent Class C review | Completed the optional Metal observer overload, disabled fast path, stream-local producer IDs/events, typed PREPARE finalization and RUN native-invocation outcomes, truthful `NOT_QUERIED`, containment/redaction, failure precedence, and no native ABI change. |
| 0042 | [Metal route tuning workflow](tasks/0042-metal-route-tuning-workflow.md) | Complete | Metal 0004/0041; Prepare 0004/0008; tools/tuning 0001–0002/0004; Config 0006A–0006B; CPU 0010I–0010J precedent; Engine 0006A–0009/0011/0015–0018; Runtime 0016 | Every concurrent Metal integration/preparation/candidate/codec/trace/API scope; Engine tuning/composition/lifecycle/fallback; CPU tuning identities; tools/tuning/API/architecture/master/roadmap documents | None | Implemented at `9feb2505705263b6efb417d606678c606c2b9598` from exact base `31ab01a65ff1e4842a43471a54e74544fb603760`; final independent Class C review approved with zero findings | Lean fake-native Metal collaboration tests; focused Engine and CPU identity/behavior regressions; one actual public `prepareTuned`→run singleton-NEG scenario; architecture/Javadocs/docs/diff; one full build; independent review | Completed retained opaque Metal local and fixed complete-plan collaborations, session-only two-route tuning, and the minimum private Engine adapter/backend-neutral fallback while preserving public Config/result, CPU identity bytes, native ABI/schema/capability, and all excluded ownership domains. |
| 0043 | [Reproducible Metal route benchmark](tasks/0043-reproducible-metal-route-benchmark.md) | Complete | 0042 Complete and independently approved; benchmark 0001/CPU 0010M protocol precedent; current public Compiler/Prepare/Runtime/Trace/Metal collaborations | Every concurrent benchmark build/report/schema/documentation scope and Metal candidate/trace/API/master/roadmap documents | None | Implemented from planning base `1d8f8cb03cb631ab59c25bbfa025369f4d2547e9`; P2 remediation at `77e6091b2a452faa04fa2b674bc295ddc7be88b7`; same-reviewer Class C rereview approved with zero remaining findings | Focused benchmark build/Javadocs; updated actual smoke and metadata-complete baseline; parsed JSON/failure/docs/diff checks; no full build or matrix | Completed schema-1 report-only `[1_048_576]` singleton FLOAT32 NEG benchmark over both opaque candidates with thread-safe snapshot attestation, reconstructible generator schema 2, exact raw-bit checks, paired alternating samples, and no winner/cache/threshold or retained result artifact. |
| 0044 | [Custom singleton-NEG benchmark evaluation and no-change closure](tasks/0044-custom-singleton-neg-benchmark-evaluation.md) | Complete | 0043 Complete and same-reviewer approved at `77e6091b2a452faa04fa2b674bc295ddc7be88b7`; updated reviewed Task 0043 baseline facts | Every concurrent Metal route/default/candidate/benchmark-evaluation scope and the same task/master/roadmap planning documents | None | Executed from exact planning revision `3d458b7a1d356f32fcd2c04de408ae9aa22355b0`; direct Class A completion after documentation validation | Exact report-fact audit and changed-Markdown/link/fence/newline/trailing-whitespace/forbidden-claim/path/diff checks passed; no build, test, native action, or measurement | Completed the no-change closure: both routes are exact only for the frozen case; both and the existing CUSTOM_SINGLE_NEG no-selected-decision safe heuristic remain; no winner, performance endorsement, default/order/policy/cache/removal/threshold/Shape-special-case change is authorized. |
| 0045 | [Verified local macOS arm64 native package](tasks/0045-verified-local-native-package.md) | Complete | 0025 Complete for ABI v4/schema 11; 0044 Complete as the serialized frontier | Native Metal build/package/README and master/roadmap edits; every Gradle distribution, native discovery, release-signing, or publication scope | None | Implemented at `07174014`; three P1 findings remediated at `26c6c911`; same-reviewer Class C rereview approved with zero remaining findings | Atomic build/package, exact Mach-O/dependency/export/signature/manifest/checksum, repeat/copy/original-and-P1 regression proofs, 126 packaged-dylib Metal tests, and 15 Engine integration tests passed | Completed the exact ad-hoc-signed verified local package only; no Developer ID, notarization, authentication, archive, publication, Gradle, Java, ABI, schema, or loader change |
| 0046 | [Explicit verified Metal native local archive](tasks/0046-explicit-verified-native-local-archive.md) | Complete | 0045 Complete at same-reviewer-approved remediation `26c6c911` | Every concurrent Metal Gradle/native distribution, native README, release-process, master, or roadmap edit; native discovery, release-signing, notarization, publication, or install work | None | Implemented from planning commit `35b222be`; independent Class C review approved implementation `4aad1ab6` with zero findings | Ordinary-build isolation; missing/invalid/symlink path failures; direct verifier reuse; two byte-identical ZIPs; exact entries/modes/timestamps/path/version checks; extracted verification and focused real-device load passed | Added one explicit, opt-in, unversioned local Gradle ZIP over only the reviewed Task 0045 package; manual extraction and absolute-path loading remain |
| 0047 | [Apple Silicon Metal CI](tasks/0047-apple-silicon-metal-ci.md) | Blocked | 0046 Complete at independently approved implementation `4aad1ab6`; enabled real GPU-capable hosted runner | Every concurrent CI/workflow, Gradle wrapper/toolchain, Metal native package/archive, benchmark protocol, native README, release-process, master, or roadmap edit | None | No implementation until an eligible organization/enterprise plan, billing/spending, repository access to `macos-26-xlarge`, and one real Metal provider run resolve the gate | Planning links/Markdown/status/diff only; no workflow, Gradle, native build, device probe, or artifact action while Blocked | Specifies an exact future portable plus M2-GPU functional workflow with immutable pins, real no-skip Metal/Engine execution, one report-only smoke, and no matrix/cache/upload/release path; current personal public repository cannot schedule the required runner |
| 0048 | [Persistent immutable Metal splats and no-general-pool closure](tasks/0048-persistent-immutable-splats-and-no-general-pool-closure.md) | Complete | 0046 Complete at independently approved implementation `4aad1ab6`; Runtime 0016; Prepare 0006; Engine 0010; current Metal finalization/schedule/trace/tuning spine | Every concurrent Metal preparation/finalization/executable/schedule/buffer/transfer, trace/tuning identity, backend/runtime/API/decision documentation, Metal test, master, or roadmap edit | None | Implemented from sole-Ready planning commit `08b9e7ea`; independent Class C review approved implementation `89f9fbb9` with zero findings while 0047 remains Blocked | 49 focused execution tests plus nine identity tests passed; exactly two explicit real-device tests passed with zero skips/failures/errors; Javadocs, focused Runtime architecture, docs, diff, and clean checks passed; no full build by instruction | Persists each immutable source-owned splat once per prepared execution with fresh read-only run bindings and exact deferred release; explicitly rejects a general output/workspace pool under current RunResult/concurrency ownership. |
| 0049 | [Synchronous single-default-device Metal contract](tasks/0049-synchronous-single-default-device-contract.md) | Complete | 0048 Complete at independently approved implementation `89f9fbb9` plus documentation finalization `a4ff40c2`; Runtime 0016; Prepare 0006; Engine 0010; Planning device eligibility; Trace 0003; Metal 0041–0042 | Every concurrent edit to the original nine documentation paths or review-remediation Trace master, or any async execution, result/resource lifetime, Metal device/context, planner device-selection, native ABI, trace, cache, tuning, route, capability, or pooling decision | None | Executed at `6e95d523`; P1 status-drift remediation `6d4246f7`; final cumulative independent Class C review `APPROVE` with zero findings | Original nine paths plus cumulative tenth Trace-master path; links/anchors, fences, newlines, whitespace, planning status, diff, and clean checks passed; no code/tests/Javadocs/builds/benchmarks/probes | Documents synchronous completed-state result semantics and one system-default context per integration; preserves all identities, context-bound resources, no cross-device semantics, and no general output/workspace pool. |
| 0050 | [Final packaged Metal repository verification](tasks/0050-final-packaged-metal-repository-verification.md) | Complete | 0045–0046 Complete package/archive contract; 0048 Complete implementation; 0049 Complete at finalization `2c6326a9` | Every concurrent source, test, Gradle, native, package, archive, Metal planning, or roadmap edit during verification | None | Final verification from exact clean revision `2c6326a9`; no implementation or second full build | Native build; fixed ad-hoc sign; package/verifier; explicit archive; permission-preserving extraction/reverification; exactly one full build with 87/87 actionable tasks; 19/130 Metal suites/tests and 1/15 explicit Metal integration suite/tests passed with zero failures/errors/skips | Proves the final-approved repository against the freshly built, packaged, archived, extracted Metal dylib; records artifact identities and remaining external blockers without changing behavior. |
| 0051 | [ACCELERATOR FLOAT32 EXP/SIGMOID recursive-floor realization](tasks/0051-accelerator-float32-exp-sigmoid-recursive-floor-realization.md) | Blocked | Model 0030/0031 Complete and independently approved; 0050 current packaged baseline; retained 0016 device evidence | Every concurrent Metal capability/schema/native-preflight/candidate/codec/tuning/public Engine scope and shared numerical-profile documentation | None | Planned at `3a818745`; one capability-false 83-word regression oracle consumed; Gate-1B evidence correction; superseded by 0053 for all future candidate/proof/oracle/cost/production work | Retained oracle hashes/verdicts; Gate-1B per-candidate ledger; successor obligations transferred to 0053; no new device, native, timing, build, test, schema, identity, capability, or production action | Historical record only: direct EXP and its composed SIGMOID are `DOMAIN-BLOCKED` despite bounded numerical passes; original custom EXP/custom SIGMOID/direct SIGMOID retain failures; its oracle is never rerun. |
| 0052 | [ACCELERATOR FLOAT32 comparisons, extrema, and scans](tasks/0052-accelerator-float32-comparisons-extrema-scans.md) | Complete | Model 0030/0031 Complete and independently approved; 0050 packaged baseline; retained 0010/0012/0013 evidence; current Compiler capture; ADR 0022 | Every concurrent Metal capability/schema/native-preflight/candidate/codec/tuning/public Engine scope and shared numerical-profile documentation | None | Gate 1/2 evidence → structural survivor adjudication with diagnostic-only local timing → fixed routes → atomic schema/identity/native/capability/preparation/execution/test/docs cutover | Focused Metal, schema/native, conformance, Compiler-contract, and no-skip real-dylib Engine proof passed; final full repository build deliberately deferred | Adds all fifteen ACCELERATOR comparison/extrema/scan kinds through fixed custom routes; sole survivors needed no comparison and fused CLAMP won solely by strict dispatch/temporary-byte dominance. Task 0055 later migrated production to ABI 5/schema 13 and identity 14. |
| 0053 | [Certified ACCELERATOR FLOAT32 custom EXP and stable SIGMOID](tasks/0053-certified-accelerator-float32-custom-exp-stable-sigmoid.md) | Blocked | Model 0030/0031 Complete and independently approved; 0051 preserved consumed evidence; 0055 schema-13 foundation; 0056 canonical route identity/catalog; pinned Lean 4.34.1 + Sollya/MPFI/MPFR/GMP toolchain | Every concurrent Metal capability/schema/native-preflight/candidate/codec/tuning/public Engine scope and shared numerical-profile documentation; Task-0052 task/evidence | None | Resume only after a pinned no-axiom constructive-real exponential bridge exists | Retained proof/source/header/site audit and all-word corroboration; no Task-0051 rerun, local timing, benchmark, or final full build | Finite certificate and checkpoint review remain fail-closed; no `DOMAIN-PASS`, capability, compiled-MSL audit, or device execution. |
| 0054 | [Current ACCELERATOR 115-kind completeness audit](tasks/0054-current-accelerator-115-kind-completeness-audit.md) | Complete | 0051–0053 current records and reviewed 0052 evidence; current Model enums, Metal capability/private schema, and Compiler gradient capture at `93b3d379` | Every production, capability, schema, identity, native, test, evidence, proof, device, timing, trace, or build scope | None | Documentation/read-only audit only; no successor task made Ready | Enum/ledger recomputation; exact `19+2+15+79=115` and 79-kind blocker-partition sums; Markdown links/fences/newlines/status and diff checks | Records every current Model operation-kind constant exactly once, corrects stale Task-0052/0053 claims, documents bounded true rows plus false dtype/shape/layout/gradient rows, private schema/type/cardinality blockers, and the serial dependency partition. |
| 0055 | [Private Metal schema, type, and cardinality foundation](tasks/0055-private-schema-type-cardinality-foundation.md) | Complete | 0052 Complete; 0054 historical audit; current 115-kind Model signatures and approved planning base `b7b9bab8d099539977c7fefc4c69b9f53db7592b` | Every concurrent Metal capability/schema/native ABI/type/transfer/identity/package scope during implementation | None | Implemented atomically at `ddeff1b2` with no dual decoder and zero new capability kinds | Focused Java/CPU/Metal/native/package/typed-transfer/public Engine checks; no timing and no second full repository build | Replaced the 160-byte record with one bounded schema-13 image; landed ABI 5, type wires `1..6`, operation wires `1..115`, attributes `0..41`, cardinality, BOOL/INT/rank-zero structure, all-six transfer, identity 14, and exact thirteen exports. |
| 0056 | [Deterministic dual-route catalog and fixed route identity](tasks/0056-deterministic-dual-route-catalog-and-fixed-route-identity.md) | Complete | 0055 Complete at `ddeff1b2`; signed-32 image remediation `ce569b65`; ABI/schema documentation remediations `486ff493` and `5b80d37c`; current 115-kind Model and installed MPSGraph SDK headers | Every concurrent Metal capability/schema/native/candidate/codec/preparation/finalization/execution/test/documentation scope; 0053 proof/oracle work | None | Implemented at `4f35576c`; conformance correction `4a5cbdef`; P2 empty-handoff remediation `1718b28a`; final independent Class C `APPROVE` with zero remaining findings | Exhaustive 115-kind/count, fixed identity/wire/codec, forcing/result-set/no-fallback, `34/81` capability, native export/schema, public Engine, Javadoc, architecture, and documentation checks passed; explicit direction required no final full build | Landed deterministic internal dual-route metadata and fixed plan identity while preserving every current production route and only the existing future controlled-autotune seam. |
| 0057 | [Exact BOOL classification, logic, and WHERE](tasks/0057-exact-bool-classification-logic-where.md) | Complete | 0052, 0055, and 0056 Complete; exact Model classification/logical/WHERE semantics; Compiler/Prepare/Runtime/Engine contracts; ADR 0022 | Every concurrent Metal capability/schema/native/generated-kernel/candidate/codec/route/preparation/finalization/buffer/publication/trace/test/package/documentation scope; any resumed 0053 work | None | Planning `8927a0f2` → implementation `e5d9d5c7` → approved validation remediation `7a3bb072` | Native/export/package; focused Metal/schema/malformed controls; `41/74` capability; exact raw-bit/truth-table/broadcast/WHERE; public no-skip Engine route/liveness proof; final Class C approval; docs/diff; no timing or full build | Adds wires `40..45` and `51` under both profiles through fixed `CUSTOM_PROGRAM`, retains forceable singleton MPSGraph structure, excludes wire `73`, advances schema/identities to 14/15, and preserves ABI 5 with thirteen exports. |
| 0058 | [Remaining elementwise arithmetic](tasks/0058-remaining-elementwise-arithmetic.md) | Complete | 0015, 0052, 0055, 0056, and 0057 Complete; Task-0011 blocker evidence; current Model recursive result-set contracts; Compiler/Prepare/Runtime/Engine contracts | Every concurrent Metal capability/schema/native/custom-source/catalog/candidate/route/preparation/resource/publication/test/package/documentation scope; any resumed 0053 work | None | Original plan `6181299a` → original implementation/evidence through `5760901b` → extension plan `0dba3035` → implementation `5ab9c44c` → contracts/evidence `63c070cc` → cumulative Class C approval with zero P0/P1/P2 | Retained all-`2^32` exact-unary proof; native/package/export; exact `50/65` capability; `62/53` structural registry; scalar raw-word/operand-order/gradient rejection; public no-skip Engine; Javadoc/Markdown/diff; no timing or full build | Keeps exact raw `FLOOR`/`CEIL`/`SIGN`/`RELU`; admits only ACCELERATOR canonical positive-rank FLOAT32 no-gradient scalar `ADD/SUB/MUL/DIV` and `RECIPROCAL` using one exact rank-one raw constant and one binary primitive; keeps scalar/tensor power, elementary/recursive blockers, and Task-0053 `EXP`/`SIGMOID` false. |
| 0059 | [Casts, layout, and indexing](tasks/0059-casts-layout-indexing.md) | Complete | 0055–0058 Complete; current Model cast/layout/indexing contracts; Compiler/Prepare/Runtime/Engine contracts; reviewer inventory from clean `67d68071` | Every concurrent Metal capability/schema/native/custom-source/catalog/candidate/route/preparation/resource/publication/test/package/documentation scope; any resumed 0053 work | None | Cutover plan `6478d249` → implementation/proof `783eabe1` → package/docs `ac0db5c4` → BOOL proof `536e52b8` → cumulative review remediation `02a097eb` → independent Class C `APPROVE` with zero P0/P1/P2 | Complete Metal JVM and focused native/schema/indexing/SELECT-SLICE suites; CPU/public Engine transfer proof; package/Gradle verification; changed-module Javadocs; no final full repository build | Landed exact `61/54` capability and `79/36` structural coverage; schema 15, identity 16, ABI 5, thirteen exports; all-carrier resolved positive-stride SELECT/SLICE; exact physical-span transfer and publication; every excluded domain remains false |
| 0060 | [Exact replacement, non-overlap fold, and aggregate reductions](tasks/0060-exact-replacement-fold-and-aggregate-reductions.md) | Complete | 0059 Complete at `f3ad5e12`; current Model scatter/slice/fold/aggregate contracts; Compiler/Prepare/Runtime/Engine contracts | Every concurrent Metal capability/schema/native/custom-source/catalog/candidate/route/preparation/resource/publication/test/package/documentation scope; pooling and ordering successors; any resumed 0053 work | None | Plan `dd94e492` → implementation `d06db07e` → proof `eda09533` → docs `62f18cd8` → fold-domain/count remediation `a4fe4754` → independent cumulative re-review `APPROVE` with zero P0/P1/P2 | Complete Metal tests; packaged-dylib public positive/negative Engine proof, repetition/concurrency/failure recovery; package verifier; Javadocs; no final full build by request | Exact `69/46` capability and `87/28` structural execution; unchanged schema 15/identity 16/ABI 5/13 exports; replacement-only Scatter-ND/slice-update, FLOAT64/FLOAT32/BFLOAT16 non-overlap folds, modular integer PROD, BOOL ALL/ANY; integral/BOOL folds and other blockers remain false |
| 0061 | [General static MATMUL domain](tasks/0061-general-static-matmul-domain.md) | Complete | 0060 Complete at `a4fe4754`; current Model MATMUL/promotion and Compiler gradient contracts | Every concurrent Metal capability/schema/native/custom/route/package/shared-document scope and resumed 0053 production | None | Corrected plan `d81ec940` → implementation/proof from `3cc49d94` through final remediation `3913bac1` → independent cumulative code/evidence/security `APPROVE` with zero P0/P1/P2 | Native build/sign/package; 187 Metal, 278 Compiler, 8 conformance, and 25 public Engine tests; certificate, Javadoc, architecture, docs/diff | Adds general promoted integral and accelerator FLOAT32/mixed MATMUL without new wire/schema/ABI/export; retains rank-two FLOAT32 MPSGraph, including authenticated nested transposes, and routes only new forms through seven fixed custom signatures |
| 0062 | [ACCELERATOR MSE and normalization/loss proof boundary](tasks/0062-accelerator-mse-and-normalization-loss-boundary.md) | Complete | 0061 Complete at reviewed head `3913bac1`; Model 0030/0031; current loss/normalization, Compiler-gradient, FLOAT32 primitive/reduction, catalog/schema/lifecycle contracts | Every concurrent Metal capability/schema/native/custom/route/package/shared-document scope and resumed 0053 production | None | Implementation/proof `1cf8a6ef` → documentation `3e0e4ea4` → nested-input remediation `b12dbe73` → evidence checkpoint `e80a03f6` → documentation remediation `06c57844` → independent cumulative code/evidence/security `APPROVE` with zero remaining P0/P1/P2 | Native build/sign/package/verification; 193 Metal, 9 conformance, 26 public Engine, and 9 architecture tests; Javadoc/docs/diff; identities/counts/exports | Implements only same-type FLOAT32 ACCELERATOR MSE `NONE`/`SUM`/`MEAN` through one fixed proved MPSGraph composition; keeps the other eight normalization/loss wires and generated backward fail-closed. |
| 0063 | [Exact ordering and arg extrema](tasks/0063-exact-ordering-and-arg-extrema.md) | Complete | 0062 Complete through `06c57844`; current Model ordering/top-K/arg-extrema and Compiler-gradient contracts; completed six-carrier, INT64, schema-15, custom-program, catalog, lifecycle, and public Engine foundations | Every concurrent Metal capability/schema/native/custom-source/multi-output/route/package/shared-document scope and resumed 0053 production | None | Approved plan and external P1 unsigned-32-bit cap correction -> implementation/proof `9931d5f8` -> documentation `f7800a26` -> remediation `8a74b499` -> final source/test correction `86399d53` -> active-document reconciliation and external final cumulative `APPROVE` at `c80d79c0` with zero P0/P1/P2 | Native build/sign/package/verification; complete packaged Metal suite; Metal conformance; four-case Engine integration; Javadoc; architecture tests; documentation/link/diff checks; tracked full-width INT32 XOR key and clean worktree confirmed | Implements exact stable SORT/ARGSORT, exact TOP_K with sorted and coordinate-order multi-output publication, and first/last ARG_MAX/ARG_MIN across the specified six- or five-carrier domains. Custom-only routing, integer-only comparator paths, canonical dense positive-rank limits, and pre-resource `UINT32_MAX` rejection are landed. Current production is `75/40`, structural execution `93/22`, catalogs `75/35/5` and `52/63/0`, schema 15, identity 19, ABI 5. |
| 0064 | [Convolution, pooling, and attention boundary](tasks/0064-convolution-pooling-and-attention-boundary.md) | Complete | 0063 Complete through `c80d79c0`; approved plan `994cd69199cdb5e2524e9b7f48bbdd0fac4a9094`; current Model convolution/pooling/attention and Compiler-gradient contracts; retained 0030/0031/0033/0036 and 0053 blockers | Every concurrent Metal capability/schema/native/custom-source/catalog/candidate/route/preparation/package/shared-document scope and resumed 0053 production | None | Approved plan → implementation/proof `58d1da7f` → documentation and seven remediation revisions → completion reconciliation `3df362e1` | Native/package, complete Metal and Compiler, conformance, CPU-free Engine, Javadoc/architecture, full test/build, documentation/diff, and independent cumulative Class C review all passed | Completed six fixed custom convolution/pooling rows; attention and absent ConvTranspose remain fail-closed; capability `81/34`, structural `99/16`, catalogs `75/35/5` and `58/57/0`, schema 15, identity 20, ABI 5, thirteen exports. |
| 0065 | [Explicit-state RNG, dropout, and recurrent boundary](tasks/0065-explicit-state-rng-dropout-and-recurrent-boundary.md) | Complete | 0064 Complete through `3df362e1`; approved plan `9ceee2b53c88a8c87ca4b8a42cdd2c92682074c9`; Model explicit-state RNG/dropout and recurrent contracts; Compiler saved-mask gradients and BPTT boundary | Every concurrent Metal random/recurrent/schema/native/custom-route/package/shared-document scope and resumed 0053 production | None | Approved plan → exact custom implementation/proof → package/docs → cumulative code/evidence/security review | Native build/sign/package; complete Metal, focused Compiler, conformance, CPU-free Engine, Javadoc/architecture, full test/build, documentation/diff | Adds both-profile raw INITIAL_STATE and accelerator FLOAT32 DROPOUT at `83/32` capability and `101/14` structural execution with identity 21; retains eager distributions and all recurrent rows fail-closed |
| 0066 | [Dtype, layout, and gradient gap closure](tasks/0066-dtype-layout-gradient-gap-closure.md) | Complete | 0065 Complete through documentation `c48b94d7fb2dfa6391901ede580cf886d74cc889`; approved plan `e0ec3d2360b0b7ab1119ce0613a612bf3e18b218`; current exact six-carrier Model cast/layout/indexing/window contracts; Compiler generated-gradient, saved-role, and logical-layout closure contracts | Every concurrent Metal capability/schema/native/custom-source/catalog/candidate/route/preparation/resource/publication/transfer/package/shared-document scope; concurrent Compiler layout/gradient or transfer-contract edits; resumed 0053 production | None | Approved plan with zero P0/P1/P2 → implementation `0b88f897` → package/documentation reconciliation → cumulative independent Class C review | Native build/sign/package; complete Metal, focused Compiler direct/generated-graph, conformance, CPU-free public Engine forward/backward/saved-role evidence, Javadoc/architecture, one full test/build, documentation/diff, and cumulative independent Class C review | Broadens exactly existing wires `6..11,16..19,39..45,51,69,71..84`; adds only the bounded Compiler static-crop/layout accommodations; keeps capability `83/32`, structural `101/14`, MPSGraph `75/35/5`, schema 15, ABI 5, and thirteen exports; completes custom `70/45/0` and identity 22 while preserving every semantic blocker. |
| 0067 | [Current 115-kind evidence audit and reconciliation](tasks/0067-current-115-kind-evidence-audit.md) | Complete | 0066 Complete through reviewed documentation `209ba28a`; current 40-family/115-kind Model registry; current Compiler inference/autograd/saved-role contracts; schema-15/identity-22 Metal boundary | Every concurrent Metal or Compiler capability/inference/autograd/schema/native/catalog/route/preparation/test/package/shared-document scope; resumed 0053 production | None | Canonical plans `dbdf8b06`/`4b583efe`/`6b8be016` → implementation/evidence `7f9601e1` → public Engine reconciliation `2541c6f3` → findings `1d52989a` → evidence correction `4e604605` → code-review remediation and cumulative approval `453ecf22` | Permanent 115-wire capability/route/catalog audit; real native alias and public Engine proof; native build; complete Metal/Compiler; Metal conformance/integration; architecture/Javadoc; full test/build; docs/diff | Reconciles all 115 rows at exact `83/32`, `101/14`, MPSGraph `75/35/5`, custom `70/45/0`, and Compiler `38/111/133 + 4 = 40/115/137`; resolves A-001..A-006 and R-001..R-005 with independent code/security/evidence `APPROVE` and no Task-0067 blocker. |
| 0068 | [Release-grade native integration verification](tasks/0068-release-grade-native-integration-verification.md) | Complete | 0067 Complete and independently approved at `cb830587`; current Tasks 0045–0046 package/archive contract; ABI-5/schema-15/identity-22 native boundary | Every concurrent Metal, Compiler, native, package, Gradle, test, planning, or shared-document edit during verification | None | Registered matrix `6820dd63` → fresh warnings-as-errors native build → fixed signing/package/Gradle ZIP/extracted-dylib verification → complete native/JVM/Engine/Compiler/build matrix → evidence correction `1775081a` → independent code/security/evidence `APPROVE` | Package/ZIP verification; 247 Metal backend + 10 Metal conformance + 47 tests across the five explicitly named Metal integration classes = 304 precisely scoped Metal tests; 282 Compiler and 22 complete conformance tests; 3712 repository tests; architecture/Javadoc; full `76`-task test and `87`-task build; JUnit inventory; artifact hashes; docs/diff/clean; three independent reviews | All gates passed without production-source remediation, relevant skip, environmental blocker, benchmark entry point, production fallback/retry/host repair, or local timing/tuning result used as evidence or decision. Mandatory full-suite tests retained their public tuning/fallback assertions. Independent code/security/evidence review at `1775081a` returned `APPROVE` with zero P0/P1/P2. |
| 0069 | [Source-owned binary32 aggregate-floor slices](tasks/0069-source-owned-binary32-aggregate-slices.md) | Review needed — Slices 1–2 approved; Slice 3 implemented under fresh checkpoint authorization | 0068 Complete at approved `1775081a`; current Model/Compiler contracts; bounded 0053/0060/0061 evidence | Every concurrent Metal capability/native/custom/catalog/route/identity/package/shared-document edit; Compiler indexing/reduction/autograd edit; resumed 0053 production | None | Proof substrate and L1/ScatterAdd approved; singleton VARIANCE implemented with exact primitive proof, fixed custom route, fresh package, and full validation for independent review | Slice-3 proof/certificate/compiled-MSL audit, raw malformed matrix, native/package/Metal/Compiler/Engine/architecture/Javadoc/full validation; fixed route only, no timing or fallback | Adds only exact accelerator FLOAT32 no-gradient singleton VARIANCE at wire 112 while preserving the existing non-domain direct structural VARIANCE recipe. |
| 0073 | [Custom-only low-precision cutover](tasks/0073-custom-only-low-precision-cutover.md) | Complete | 0071 Complete; Trace 0003 Complete; active Model low-precision/schema-19 contracts; ADR 0025 | Concurrent Metal/native ABI, Trace low-precision DTO, conformance identity, or shared-contract/docs edits; resumed 0053 production | None | Existing user-authorized dirty cutover → integrated validation → final independent Class C `APPROVE` | Fresh native/package/exports; focused Metal, Trace, conformance, Engine, architecture, Javadoc/docs; one full build with CPU Test heap override | Removes machine-bound certificate/vendor path and fixes every BFLOAT16/FLOAT16 partition to one custom candidate; final rereview `APPROVE` with zero unresolved findings. |

## Dependency DAG and authorized frontiers

Historical implementation branch:

`0001 -> 0002 -> 0003 -> 0004 -> 0005 -> 0008 -> {0009, 0010, 0011, 0012, 0013, 0014}`

Completed profile spine and serial successors:

`Model 0027 -> Config 0006 -> Engine 0018`

`0014 + Model 0027 + Config 0006 + Engine 0018 -> 0015`

`0015 -> {0016 (Blocked), 0019 (Complete) -> 0017 (Blocked)}`

`Model 0028 (Complete) -> 0020 (Complete) -> 0018 (Blocked)`

`0018 blocker evidence -> Model 0029 (Complete) -> 0021 (Complete) -> 0022 (Complete) -> 0023 (Complete) -> 0024 (Complete) -> 0025 (Complete) -> {0026 (Blocked), 0027 (Blocked), 0030 (Blocked), 0031 (Blocked), 0032 (Blocked), 0033 (Blocked), 0034 (Blocked), 0035 (Blocked), 0036 (Blocked), 0037 (Blocked)}`

`0022 (Complete) + 0023 (Complete) + 0024 (Complete) + 0025 (Complete) + finalized evidence through 0037 -> 0038 (Complete)`

`Model 0026 (Draft) -> 0039 (Blocked)`

`0023 (Complete) + 0025 (Complete) + 0038 (Complete) + Model BFLOAT16/Gather + Config 0006 + Engine 0018 -> 0040 (Blocked)`

`Trace 0003 (Complete) -> 0041 (Complete)`

`0004 (Complete) + 0041 (Complete) + Engine 0009/0015–0018 (Complete) + tools/tuning 0001–0002 (Complete) -> 0042 (Complete) -> 0043 (Complete) -> 0044 (Complete) -> 0045 (Complete) -> 0046 (Complete) -> 0047 (Blocked)`

`0046 (Complete) + Runtime 0016 (Complete) + Prepare 0006 (Complete) + Engine 0010 (Complete) -> 0048 (Complete)`

`0048 (Complete) + Runtime 0016 (Complete) + Prepare 0006 (Complete) + Engine 0010 (Complete) + Planning device eligibility + Trace 0003 + 0041–0042 (Complete) -> 0049 (Complete)`

`0049 (Complete) -> 0050 (Complete final program verification)`

`Model 0031 (Complete) -> Model 0030 (Complete) -> {0051 (historical Blocked) -> 0053 (Blocked on external constructive-real bridge), 0052 (Complete)}`

`0051 (historical Blocked) + reviewed 0052 evidence + retained 0053 evidence + historical source -> 0054 (Complete pre-cutover audit); 0052 production cutover (Complete) supersedes that inventory`

`0052 (Complete) + 0054 (historical inventory) + current source -> 0055 (Complete foundation) -> 0056 (Complete deterministic dual-route catalog) -> {0053 (Blocked), 0057 (Complete)}`

`0057 (Complete) + approved tensor arithmetic/result-set contracts -> 0058 (Complete remaining elementwise arithmetic plus bounded no-gradient scalar/reciprocal extension)`

`0058 (Complete) + current cast/layout/indexing contracts -> 0059 (Complete)`

`0059 (Complete) + current replacement/fold/aggregate contracts -> 0060 (Complete)`

`0060 (Complete) + current MATMUL/promotion/gradient contracts -> 0061 (Complete)`
`0061 (Complete) + current loss/normalization and Compiler gradient contracts + proved FLOAT32 primitive/reduction routes -> 0062 (Complete)`
`0062 (Complete) + current ordering/top-K/arg-extrema and Compiler gradient contracts + schema-15 custom-program/multi-output foundations -> 0063 (Complete)`
`0063 (Complete) + current convolution/pooling/attention, gradient, and retained selector/proof evidence -> 0064 (Complete through 3df362e1)`
`0064 (Complete) + current explicit-state RNG/dropout/recurrent contracts + schema-15/custom-program/multi-output foundations + retained 0035/0037/0053 evidence -> 0065 (Complete)`
`0065 (Complete) + current dtype/layout/indexing/window and Compiler generated-gradient/saved-role/logical-layout contracts + schema-15 custom-program foundations -> 0066 (Complete)`
`0066 (Complete) + current 115-kind Model/Compiler/Metal source and behavioral evidence -> 0067 (Complete)`
`0067 (Complete) + current native/package/archive contracts -> 0068 (Complete)`
`0068 (Complete) + current Model/Compiler contracts + retained 0053/0060/0061 evidence -> 0069 (Review needed; proof substrate + L1_NORM + SCATTER_ADD approved; singleton VARIANCE implemented under fresh authorization)`

`0071 (Complete) + Trace 0003 (Complete) + active low-precision/schema-19 contracts + ADR 0025 -> 0073 (Complete; integrated validation and independent Class C rereview passed)`

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
[0032](tasks/0032-profile-common-canonical-float32-no-grad-mse-none.md),
[0033](tasks/0033-profile-common-canonical-float32-no-grad-unmasked-noncausal-default-scale-sdpa-forward.md),
[0034](tasks/0034-profile-common-canonical-float32-no-grad-ascending-sort.md),
[0035](tasks/0035-profile-common-canonical-float32-no-grad-explicit-state-dropout.md),
[0036](tasks/0036-extended-3d-inference.md),
[0037](tasks/0037-profile-common-canonical-float32-no-grad-forward-rnn-tanh.md),
[0039](tasks/0039-profile-common-ieee-float16-prerequisite-gate.md) and
[0040](tasks/0040-profile-common-exact-canonical-bfloat16-gather.md) are Blocked without production
changes. Tasks 0031–0037 and 0039 ran no probe; Task 0040 ran exactly its one failed gate.
[0041](tasks/0041-prepared-route-and-invocation-trace.md) is Complete at implementation `ba16d942`
plus remediation `386705ca` after final independent Class C approval with zero findings.
[0042](tasks/0042-metal-route-tuning-workflow.md) is Complete at
`9feb2505705263b6efb417d606678c606c2b9598` after final independent Class C review returned
`APPROVE` with zero findings. [0043](tasks/0043-reproducible-metal-route-benchmark.md) is Complete
at remediation `77e6091b2a452faa04fa2b674bc295ddc7be88b7` after same-reviewer Class C approval with zero
remaining findings. [0044](tasks/0044-custom-singleton-neg-benchmark-evaluation.md) is Complete.
[0045](tasks/0045-verified-local-native-package.md) is Complete at same-reviewer-approved
remediation `26c6c911`. [0046](tasks/0046-explicit-verified-native-local-archive.md) is Complete at
independently approved implementation `4aad1ab6ced318107e65bb9beef0013f8a7ff6e5`.
[0047](tasks/0047-apple-silicon-metal-ci.md) is Blocked on a real GPU-capable hosted runner.
[0048](tasks/0048-persistent-immutable-splats-and-no-general-pool-closure.md) is Complete at
independently approved implementation `89f9fbb9093db3fb86629189cb98ae521a631b27`.
[0049](tasks/0049-synchronous-single-default-device-contract.md) is Complete after final cumulative
independent Class C `APPROVE` with zero findings on remediation `6d4246f7`.
[0050](tasks/0050-final-packaged-metal-repository-verification.md) is Complete after the sole final
packaged/extracted-Metal repository build from exact revision `2c6326a9`. Detailed
[0051](tasks/0051-accelerator-float32-exp-sigmoid-recursive-floor-realization.md) remains Blocked
after its consumed regression oracle: direct EXP and its composed SIGMOID are domain-blocked, the
remaining candidates failed, and its invocation is never rerun.
[0053](tasks/0053-certified-accelerator-float32-custom-exp-stable-sigmoid.md) supersedes 0051 for
new EXP/SIGMOID work but is now Blocked. It retains an integer-only raw-word candidate, one shared
MSL/C core, exact Lean Nat/Int definitions and structural/algebraic certificates, source audit, and
an independent partitioned check of every `2^32` word with zero unresolved/failures. The retained
finite certificate still lacks a pinned no-axiom constructive-real exponential bridge for Taylor
enclosures, exponential identities, range reduction, reconstruction error, and binary32 rounding;
source-to-Lean equivalence and recursive stable-SIGMOID site membership therefore remain open.
Independent checkpoint review reported no findings and explicitly accepted fail-closed evidence,
not `DOMAIN-PASS`. Native integration still returns unsupported operation, capability/catalog/route
are unchanged, compiled-MSL audit and device smoke remain blocked, and local timing is never run.
[0057](tasks/0057-exact-bool-classification-logic-where.md) is Complete at planning
`8927a0f2`, implementation `e5d9d5c7`, and approved validation remediation `7a3bb072`: exact wires
`40..45` and `51` execute under both profiles; wire `73 SELECT` remains excluded.
[0052](tasks/0052-accelerator-float32-comparisons-extrema-scans.md) is Complete: all retained
semantic, numerical, and structural facts passed; routes were frozen; and the atomic production
cutover landed. Its timings remain diagnostics only. Opaque routes remain domain-blocked and
received no inferred facts. Its exact Gate-1/Gate-2 and Gate-3 packages remain retained under
[`evidence/0052/`](evidence/0052/README.md) and
[`evidence/0052-gate3/`](evidence/0052-gate3/README.md).
[0054](tasks/0054-current-accelerator-115-kind-completeness-audit.md) remains Complete as the exact
historical pre-cutover `19 + 2 + 15 + 79 = 115` inventory and explicitly records its supersession.
[0055](tasks/0055-private-schema-type-cardinality-foundation.md) is Complete at `ddeff1b2`.
[0056](tasks/0056-deterministic-dual-route-catalog-and-fixed-route-identity.md) is Complete at
implementation `4f35576c`, conformance correction `4a5cbdef`, and final approved remediation
`1718b28a`. Its normative
[per-wire route-evidence audit](tasks/0056-route-evidence-audit.md) cites an exact installed-header
selector or finite composition and current Model semantic/signature source for every one of the
115 rows. These current cold, per-operation structural catalogs describe implementation
availability, not admitted capability or a selected partition route: MPSGraph
`75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE` and custom
`73 AVAILABLE / 42 PENDING / 0 UNAVAILABLE-WITH-PROOF`. The closed route identity retains wires
`1..3`; schema 19 embeds the fixed route, exact numerical profile, custom execution extension, and
type wires `1..7`; identities are version 29, and capability is `86 true / 29 false`.
Package-private forcing remains result-set-only after fresh authentication; every partition
containing BFLOAT16/FLOAT16 values, every selected Task-0066 occurrence, and every exact Task-0069
occurrence rejects MPSGraph as a partition route, while non-domain VARIANCE retains its prior direct
structural recipe. Eligible FLOAT32-only steps inside a low-containing custom partition may still
use an internal MPSGraph boundary under the existing FLOAT32 policy.
There is no device, oracle, timing, benchmark, public API, hot fallback, retry, cache, or autotune
behavior.
Every other Metal
task retains its recorded status.
These edges serialize shared Metal mutation; they do not claim that one operation family requires
another.

## Integration ownership and shared documents

- Complete Tasks 0046, 0048, 0049, and 0050 retain their reviewed local-archive, persistent-
  immutable-splat, synchronous/single-default-device, and final-verification boundaries. Blocked
  Task 0047 owns no workflow, provider, runner, Gradle, or native-distribution implementation scope
  until its external GPU-hosted-runner gate is resolved. Blocked Task 0051 is historical and owns no
  future source, oracle, or production work. Task 0053 is Blocked on the unavailable constructive-
  real proof bridge; its retained candidate and evidence remain fail-closed. Complete Task 0057
  owns the exact BOOL classification/logic/WHERE production cutover. Complete Task 0058 owns the
  remaining elementwise-arithmetic catalog, exact raw `FLOOR`/`CEIL`/`SIGN`/`RELU`, and the bounded
  ACCELERATOR no-gradient scalar `ADD/SUB/MUL/DIV` plus `RECIPROCAL` cutover. Scalar/tensor power,
  elementary/recursive blockers, and Task-0053 `EXP`/`SIGMOID` remain unchanged.
  Local timing remains unauthorized. Complete Task 0052 owns its landed accelerator-only
  capability/preparation/
  execution/test/documentation cutover and immutable evidence. Task 0057 generalizes its native
  whole-program route to the shared exact custom-program identity without changing Task-0052
  semantics. Task-0052 opaque candidates remain domain-blocked.
- Complete Task 0054 owns only the historical pre-cutover 115-kind audit and serial blocker
  partition. It grants no production scope and explicitly defers current post-cutover counts to a
  fresh audit.
- Complete Task 0055 owns the historical ABI-5/schema-13 six-carrier foundation. Task 0057
  superseded it with route-bearing schema 14; Task 0059 cleanly supersedes schema 14 with
  storage-layout schema 15 while retaining ABI 5 and its thirteen exports. No older schema has a
  compatibility reader.
- Complete Task 0056 owns the internal 115-kind structural route catalog and one immutable
  prepared-route identity. Task 0057 advanced its candidate/codec identities to 15 and fixed the
  shared route name to `CUSTOM_PROGRAM`; Task 0059 advanced all backend-local identities to 16,
  Complete Tasks 0061–0066 advanced them through 17–22, Tasks 0067–0068 retained 22, Task-0069
  Slices 1 through 3 advanced them through 23–25, Task 0070 to 26, and Task 0071 to 27. Identity 26
  and every older identity fail closed. Catalog states still are not correctness approval; local timing is never
  authority, and only the existing cold authenticated handoff remains a seam for separately
  approved controlled autotuning.
- Complete Metal 0044 owns its documentation/audit no-change record; no active owner may reinterpret
  its bounded report as a winner or production decision.
- Complete Metal 0043 retains its reviewed benchmark implementation and evidence; Complete Metal
  0042 retains its reviewed tuning collaborations and Engine integration. Task 0044 changed neither
  production surface.
- Blocked Metal 0016–0018, 0026–0027, 0030–0037, 0039–0040, 0051, and 0053 have no active
  production write scope. Task 0061 is Complete through reviewed implementation head `3913bac1`.
  Task 0062 is Complete through final documentation remediation `06c57844` after independent
  cumulative code, evidence, and security approval. Task 0063 is Complete through implementation
  `9931d5f8`, documentation `f7800a26`, remediation `8a74b499`, final source/test correction
  `86399d53`, and active-document reconciliation `c80d79c0`; external final cumulative review at
  `c80d79c0` returned `APPROVE` with zero P0/P1/P2. Task 0064 is Complete through final
  reconciliation `3df362e1` after package/full validation and independent cumulative Class C
  review passed. Task 0065 is Complete through approved plan
  `9ceee2b53c88a8c87ca4b8a42cdd2c92682074c9`, implementation `05d83074`, and correctness
  remediation `ed7a3369`; independent cumulative code/evidence/documentation and
  security/determinism review returned `APPROVE` with zero unresolved P0/P1/P2. Task 0066 is
  Complete through approved plan `e0ec3d2360b0b7ab1119ce0613a612bf3e18b218`, implementation
  `0b88f897`, documentation `946fff39`, alias-safety remediations `b7d2616f`/`07641c51`, and
  scalar/window contract remediations `31eb77c5`/`4b426898`; independent cumulative quality and
  security review at `4b426898` returned `APPROVE` with zero P0/P1/P2. It owns only the approved
  backend-neutral static-crop/layout accommodations plus the exact Metal occurrence cutover. Task
  0067 is Complete through cumulative remediation/review head `453ecf22`; independent code,
  security, and evidence review returned `APPROVE` with zero remaining P0/P1/P2 and no Task-0067
  blocker. Task 0068 is Complete after its fresh native/package/extracted-dylib matrix and evidence
  correction `1775081a`; independent code, security, and evidence review returned `APPROVE` with
  zero remaining P0/P1/P2. Task 0069 Slices 1 through 3 are Complete after cumulative proof,
  code/evidence/security review, raw-native boundary remediation, and active-document
  reconciliation. Task 0070's authenticated pointwise generation and Task 0071's anchor-epilogue
  and structural-trace cutovers are Complete with retained task-local source, proof, AIR, and
  runtime evidence. Model 0028 owns the reduction semantic contract, Complete
  Model 0029 owns the MATMUL final-publication semantic
  contract, and Complete Metal 0021–0025 retain reviewed implementations.

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

Current ABI 7 exposes exactly thirteen exports and accepts one bounded schema-19 route-bearing
program image with the exact numerical-profile wire. Type wires are `1..7`, operation wires are
`1..115`, attribute wires are `0..41`, route wires are `1..3`, and
workload/exact-policy/candidate/compatibility/route/codec identities are version twenty-nine.
The historical schema/identity cutovers remain prerequisites, but no older identity or image is
accepted. The current catalog has exactly 86 production kinds and 101 structurally executable
kinds. Every partition containing BFLOAT16 or FLOAT16 values uses the authenticated custom program,
including all six no-gradient raw-preserving families. MPSGraph, classic MPS, MPP, CPU, retry, and
fallback are not low-precision route alternatives. The certificate/environment/vendor-probe layer
is removed. Identity 29 is monotonic because workload compatibility binds profile, schema, complete
encoded image and extension, route policy, and exact program semantics.
The custom-only cutover is tracked by [Task 0073](tasks/0073-custom-only-low-precision-cutover.md)
as Complete. Fresh native/package verification, focused module/identity tests, real Apple-GPU
execution, and a full build passed; the full build required a one-off CPU Test heap override after
the stock worker ran out of memory in an unrelated test. Independent Class C review resolved route,
Javadoc, and trace wording; final narrow rereview returned `APPROVE` with zero unresolved findings,
including ADR 0023 provider-evidence and glossary observer-`Error` checks against source and tests.
Developer ID signing, notarization, publication, and public release remain outside scope.
Complete Model 0028 owns the root-only exact-zero reduction rule. Complete Model 0029 owns the
MATMUL-only final-publication exact-zero sign rule. Metal 0018 remains Blocked without production
changes. Complete Metal 0022
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
0026 remains historically Blocked: its mandatory single `[4,8]` production-settings direct-selector
execution returned `0x00000000` at flattened index 18 where its frozen four-ULP StrictMath oracle
required `0x0008ec28`. Metal 0027 likewise remains historically Blocked after its only `[2,8,2]`
direct BatchNorm inference execution flushed both signed minimum-subnormal channel-6 outputs.
With Complete Model 0030, those FTZ observations are recursively reachable only for
`ACCELERATOR`; strict capability remains false, and an opaque selector still requires complete
domain recursive-subset proof. Metal 0030 is independently Blocked because its sole
`[1,1,1,18]` direct `MAX_POOL2D` execution returned negative zero for `[+0,-0]` and finite `3.0f`
for each first/later qNaN/sNaN window, violating exact winner selection. Its other controls passed
and artifacts were removed. Metal 0031 is Blocked without a probe: MPSGraph's descriptor
structurally maps full grouped NCHW/OIHW Conv2d geometry, but the selector gives no contraction
algorithm contract, so one execution cannot prove the complete recursive output subset over every
shape. Metal 0032 remains Blocked without a probe under its frozen profile-common premise. Its MSE
formula is recursively reachable for `ACCELERATOR` through subtraction, `delta*delta`, reduction,
and the exact divisor. Complete Task 0062 supersedes that premise only with an implemented same-type
positive-rank FLOAT32 ACCELERATOR domain using the already-qualified SUB/MUL/full-SUM/full-MEAN
routes. Production reflects that bounded cutover after its cumulative Class C review gate passed
with zero remaining P0/P1/P2 findings.
Metal 0033 remains Blocked without a probe: attention arithmetic is recursively reachable for
`ACCELERATOR`, but the macOS-15 direct unmasked output-only SDPA selector is opaque, additive mask
semantics mismatch Model, and no
complete-domain proof covers contractions, scale, guards/masks, softmax, special classes, output
contraction, gradients, or shape-dependent algorithms. Metal 0034 remains Blocked without a probe:
direct ascending one-output SORT aligns structurally, but the exact ordering contract for stable
ties, NaNs, signed zero, subnormals, represented-bit movement, and shape-dependent algorithms is
undocumented. ARGSORT returns INT32 rather than Model INT64. Schema 13 carries INT64 and two-output
TOP_K, but the required cast/output-order routes remain unproved. Metal 0035 remains Blocked without
a probe: direct dropout hides randomness and returns one output, and opaque Philox state cannot
establish Model INT64[2] key/counter-plus-N. Schema 13 carries zero-input, three-output, and INT64
structure, but only an exact custom random route plus lifecycle and proof can unblock it; `p=0`
narrowing is prohibited. Metal 0036 remains Blocked without a probe: direct
Conv3d inherits the opaque shape-dependent contraction boundary. Average Pool3d arithmetic is
recursively reachable for `ACCELERATOR`, but Pooling4D/stencil do not prove its complete NCDHW
mapping, ceil-tail, fixed divisor, or route; max Pool3d additionally retains exact
winner-selection obligations that 0030's evidence disproves for a generic direct assumption.
UNFOLD3D/FOLD3D remain separate custom movement/overlap work. Metal 0037 remains Blocked without a
probe: recurrent arithmetic is recursively reachable for `ACCELERATOR`, but direct RNN lacks
runtime INT64 valid lengths, atomic validation, skipped padded work, positive-zero padding, and
`finalHidden`; no complete recurrence, state-publication, GRU/LSTM, or gradient route proof exists.
Before the FP16 cutover, schema 18, type wires `1..6`, version-twenty-seven identities, ABI 5,
thirteen exports, and supported all-six rank-`0..16` cross-owner transfer were the landed boundary.
P12 supersedes that snapshot with schema 19, type wires `1..7`, version-twenty-eight identities,
ABI 6, fourteen exports, and all-seven transfer; operation wires `1..115`, attribute wires
`0..41`, variable cardinality, and custom execution plans remain fixed. Every other identity fails
closed. The schema registry is not capability: BOOL consumption includes the exact positive-rank
ordering/top-K and logic/WHERE domains, integral MATMUL remains the only general INT arithmetic
consumer, loss ownership is the bounded accelerator MSE forward domain only, and the six Task-0064
convolution/pooling rows use their fixed custom program rather than the historically blocked opaque
direct/composed assumptions. Planning-only Task 0039 historically reserved no FLOAT16 value while
Draft Model 0026 provided no public type or oracle; P12 no longer shares that limitation.
Task 0040 is Blocked after its one direct BFLOAT16 Gather run canonicalized required `0xffa6` to
`0x7fc0`; its artifacts
were removed and production remains unchanged. Task 0041 is Complete at
implementation `ba16d942` plus remediation `386705ca` after final independent Class C approval
with zero findings. Documentation/audit-only Task 0038 is Complete. Task 0042 is Complete at
`9feb2505705263b6efb417d606678c606c2b9598` after final independent Class C approval with zero
findings. Tasks 0043–0044 are Complete. Task 0045 is Complete at same-reviewer-approved remediation
`26c6c911`. Task 0046 is Complete at independently approved implementation `4aad1ab6`; its artifact
remains unauthenticated and local. Task 0047 is Blocked on the unavailable required M2 GPU hosted
runner. Task 0048 is Complete at independently approved implementation `89f9fbb9`. Documentation-
only Task 0049 is Complete after approved P1 remediation `6d4246f7`. Evidence-only Task 0050
completed the final packaged/extracted-Metal 87-task repository build; Tasks 0051 and 0053 are
Blocked, with 0053 retaining fail-closed proof evidence; Task 0052 is Complete through its
custom-cost/production cutover; Task 0054 is the Complete historical pre-cutover read-only
`19+2+15+79=115` inventory; Task 0055 is Complete at `ddeff1b2` with zero capability widening;
Task 0056 is Complete at implementation `4f35576c` plus approved remediation `1718b28a`; and Task
0057 is Complete at planning `8927a0f2`, implementation `e5d9d5c7`, and approved validation
remediation `7a3bb072`.
Task 0058 is Complete through `63c070cc`. Task 0059 is Complete through cumulative review
remediation `02a097eb`; independent Class C review of `6478d249..02a097eb` returned `APPROVE` with
zero P0/P1/P2. Task 0060 is Complete through implementation `d06db07e`, proof `eda09533`,
documentation `62f18cd8`, and fold-domain/count remediation `a4fe4754`; independent cumulative
re-review of `55cdebc3..a4fe4754` returned `APPROVE` with zero P0/P1/P2. Task 0061 is Complete at
corrected approved plan `d81ec940` and reviewed implementation head
`3913bac1`; independent cumulative code, evidence, and security reviews each returned `APPROVE`
with zero P0/P1/P2. It retained `69/46` capability, `87/28` structural execution, `75/35/5`
MPSGraph catalog, `47/68/0` custom catalog, and backend-local identity 17. Task 0062 is Complete
through documentation remediation `06c57844`; independent cumulative code, evidence, and security
review returned `APPROVE` with zero remaining P0/P1/P2. Its cutover advanced capability to `70/45`,
structural execution to `88/27`, and backend-local identity to 18. Task 0063 is Complete through
implementation `9931d5f8`, documentation `f7800a26`, remediation `8a74b499`, final source/test
correction `86399d53`, and active-document reconciliation `c80d79c0`. Its reviewed
unsigned-32-bit-bounded custom cutover implements SORT/ARGSORT/TOP_K/ARG_MAX/ARG_MIN with no
MPSGraph fallback. That Task-0063 cutover advanced capability to `75/40`, structural execution to
`93/22`, catalogs to `75/35/5` and `52/63/0`, and backend-local identity to 19. Schema 15, ABI 5,
and thirteen exports remain fixed. External final cumulative review at `c80d79c0` returned
`APPROVE` with zero P0/P1/P2; Task 0053 remains Blocked on its external constructive-real bridge.

Task 0064 is Complete through final reconciliation `3df362e1`. Its six fixed custom rows implement
accelerator Conv2d/Conv3d and average Pool2d/Pool3d plus both-profile exact maximum
Pool2d/Pool3d, while attention and absent ConvTranspose contracts remain fail-closed.

Task 0065 is Complete through approved plan `9ceee2b53c88a8c87ca4b8a42cdd2c92682074c9`,
implementation `05d83074`, and correctness remediation `ed7a3369`. Its fixed custom rows implement
exact both-profile zero-input `INITIAL_STATE` and accelerator FLOAT32 `DROPOUT`, including replay,
branching/chaining, saved-mask backward, evaluation bypass, prepared reuse, and
session/concurrency isolation. Generic eager distributions add no graph row, and all recurrent
rows remain blocked.

Task 0066 is Complete through approved plan `e0ec3d2360b0b7ab1119ce0613a612bf3e18b218`,
implementation `0b88f897`, documentation `946fff39`, alias-safety remediations
`b7d2616f`/`07641c51`, and scalar/window contract remediations `31eb77c5`/`4b426898`.
Independent cumulative quality and security review returned `APPROVE` at `4b426898` with zero
P0/P1/P2. Its exact existing-wire cutover implements all 36 casts, all-carrier movement,
FLOAT64/FLOAT32/BFLOAT16 classification and promoted WHERE, scalar/affine logical versus physical
layout handling, INT64 indexing parity, and bounded generated gradients with saved roles.
Every selected occurrence uses `CUSTOM_PROGRAM`; dynamic/empty/signed-stride/overlap,
additive/reduction, transcendental, attention, and recurrent blockers remain fail-closed. Current
capability was `83/32`, structural execution `101/14`, catalogs `75/35/5` MPSGraph and
`70/45/0` custom, schema 15, backend-local identity 22, ABI 5, and thirteen exports at the
Task-0066 checkpoint. Tasks 0067 and 0068 preserved and verified that ledger.

Task 0069 Slices 1 through 3 are Complete and independently approved. Their shared proof substrate
and exact rank-one accelerator L1/ScatterAdd plus singleton VARIANCE source-owned custom routes
retain current capability `86/29`, custom catalog `73/42/0`, and the exact Task-0069 compiled
source/AIR evidence. Java and native creation reject direct MPSGraph only for those exact custom
occurrences; the existing non-domain VARIANCE structural fixture remains direct. ScatterAdd
completely validates INT32/INT64 indices before encoding or mutation, retains duplicate updates in
source order, raw-copies unaddressed cells, and closes the compiler-generated rank-one Gather data
cotangent. Singleton VARIANCE requires input `[1]`, axis `[0]`, correction zero, scalar or retained
`[1]` output, and uses the literal DIV-SUB-MUL-DIV custom kernel.

Task 0070 added authenticated execution records, compact materialized slots, and bounded
deterministic generated units for eligible linear canonical FLOAT32
`FLOOR`/`CEIL`/`SIGN`/`RELU` chains. Four universal raw-word Lean certificates, a source
certificate, real Metal reflection, an immediately-before-dispatch test observer, and the pinned
compiled generated-MSL/AIR audit cover that cutover.

[Task 0071](tasks/0071-anchor-epilogue-fusion.md) adds schema-18 typed anchor instructions for exact
ACCELERATOR MATMUL/Conv2d epilogues.
Its source and compiled-AIR certificates, Lean order/store/broadcast/DAZ-FTZ model, real-device
observer suite, and public Engine smoke cover one physical dispatch and one final store with no
materialized suffix intermediate. Structural PREPARE and planned RUN payloads expose only bounded
typed facts and aggregates; lifecycle-close tracing remains deferred because logical prepared
units share/refcount native resources. At the Task-0071 checkpoint, structural `101/14`,
capability `86/29`, MPSGraph `75/35/5`, custom `73/42/0`, ABI 5, thirteen exports, schema 18, and
identity 27 were fixed. P12 subsequently superseded the active boundary with ABI 6, fourteen
exports, schema 19, and identity 28.

Metal 0006 remains historically `Blocked` after its frozen direct-selector
RECIPROCAL/LOG/SQRT/RSQRT/RELU/TANH special-class and sign gate failures. Complete Model 0030
does not authorize gross class/sign errors, but a future conforming custom or composed
`ACCELERATOR` route may recurse through the new floors after complete formula, schema, capability,
and route proof. Metal 0007 remains independently `Blocked` after eight direct-output executions
returned positive zero where that task's current exact Model SUM/MEAN results are `2.0f`/`0.5f`.

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

Metal 0015 is Complete. [0016](tasks/0016-profile-qualified-float32-abs-exp-sigmoid.md) remains
historically Blocked without production changes: under its frozen no-FTZ/one-step oracle, the
324-executable/2,592-run Apple M3 Max gate proved exact `ABS`, while `EXP` and `SIGMOID` flushed
representable subnormal results and sigmoid had two-step ordinary finite results. All controls
passed and the disposable probe was removed.

Complete [0019](tasks/0019-exact-profile-qualified-float32-abs.md) adds only exact canonical `ABS`
under both profiles. Complete Model 0030 makes recursive `ACCELERATOR` `EXP`/`SIGMOID`
results semantically reachable. Task 0051's sole 83-word oracle gave bounded numerical passes to
direct EXP and recursively composed SIGMOID, while direct SIGMOID and both original custom
candidates failed. The two sampled passes are regression evidence only: opaque direct EXP lacks an
authoritative all-binary32/rank-`1..16` contract, and the composition inherits that
`DOMAIN-BLOCKED` dependency. Task 0051 is now historical and its oracle is never rerun.
[0053](tasks/0053-certified-accelerator-float32-custom-exp-stable-sigmoid.md) supersedes it and is
Blocked on an unavailable pinned no-axiom constructive-real exponential bridge. Its retained
integer-only raw-word source would still require machine-checkable proofs of the complete raw
partition, DAZ, range reduction, table/index and approximation bounds, reconstruction/FTZ,
inclusive EXP distance `<=5`, launch/index coverage, and recursive stable-SIGMOID membership,
plus an independent all-word checker and review before production eligibility. Opaque MPSGraph
candidates remain domain-blocked. Local timing is not run.

[0052](tasks/0052-accelerator-float32-comparisons-extrema-scans.md) owns the next bounded
inventory for six comparisons, tensor/scalar/clamp/reduction extrema, and cumulative SUM/PRODUCT
under Model 0030's exact/discrete and recursive aggregate floors. Its Gate 1 audit found every
opaque direct candidate API-present but `DOMAIN-BLOCKED`; auditable custom predicates, extrema,
CLAMP, reductions, and ordered scans all proved `DOMAIN-PASS`. The exact custom source, validator,
audit, wrapper, and proof were recovered byte-for-byte and their recorded hashes revalidated.
Exactly one 98-output Apple M3 Max numerical invocation passed all 16 custom candidates; it is not
rerun. The custom-only Gate-3 process retained four alternating warmup rounds, eight retained
rounds, and all 128 raw samples as diagnostic history. Exact source facts record one dispatch/zero
route-owned temporary bytes for every fused route and two dispatches/4,194,304 bytes for composed
CLAMP. Fourteen operation rows had one survivor and needed no comparative gate. Fused CLAMP is
fixed solely because its structural facts strictly dominate composed CLAMP under identical proven
semantics/domain; the recorded medians did not authorize it. Opaque routes remain domain-blocked
and received no inferred facts. The atomic production cutover is complete: all fifteen kinds are
capability-true only for their exact ACCELERATOR domain, use fixed custom whole-program routes, and
retain direct assigned targets, declared run-owned logical-value buffers, and one synchronous
Java/native invocation. The original schema 12/identity 13 cutover is historical; Task 0055 later
migrated then-current production to ABI 5/schema 13 and identity 14.

[0054](tasks/0054-current-accelerator-115-kind-completeness-audit.md) remains the exact historical
pre-cutover audit at base `93b3d379`: 40 enums, 115 constants, and
`19+2+15+79=115`. Its schema-11 and capability-false Task-0052 counts are historical facts, not
claims about the post-cutover tree.
Gross class/sign-failing direct selectors for `RECIPROCAL`, `LOG`, `SQRT`, `RSQRT`, `RELU`, and
`TANH` remain unusable; a conforming custom or
composed route would still require complete formula, schema, capability, and route proof.

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
approved with zero findings. Metal 0026 and 0027 remain historically Blocked under their frozen
profile-common/no-FTZ gates: direct SOFTMAX returned `0x00000000` where its reference was
`0x0008ec28`, and direct BatchNorm flushed both signed minimum-subnormal outputs to same-signed
zero. Their probes were removed. Under Complete Model 0030 those observations are recursively
reachable only for `ACCELERATOR`; neither sample proves a complete opaque-selector subset, no
strict route exists, and current capabilities remain false. Task 0052 completed its independent
proof, sole regression oracle, structural survivor adjudication with diagnostic-only local timing,
and production cutover. Complete Task 0055 supplies the zero-new-kind private
schema/type/cardinality foundation required before
the remaining operation workstreams. LOG_SOFTMAX, every normalization, broader window/fold,
BatchNorm
training, backward closure, and Model 0026 remain unauthorized pending complete-domain proofs.

## Delivered lifecycle and ABI boundary

- Task 0001 is deliberately non-executing and fail closed. Task 0002 established the strict
  positive-rank canonical FLOAT32 NEG domain, task 0014 added strict affine/`CONTIGUOUS`
  composition, task 0015 added the accelerator tensor-binary domain, task 0019 added exact
  canonical ABS to both profile matrices, task 0020 added the accelerator canonical
  `SUM`/`MEAN`/`SUM_TO_SHAPE` domain, task 0021 added accelerator rank-two FLOAT32 MATMUL, and task
  0061 generalizes promoted integral and accelerator FLOAT32/mixed MATMUL. Task 0003 changes
  singleton-NEG route choice, not occurrence capability or partitioning.
- The caller-supplied macOS arm64 native library uses an Objective-C C ABI reached through JDK 26
  Foreign Function and Memory (FFM). It is not discovered or published by the backend. ABI
  versions 1 through 6 are historical; ABI 7 removes the former certification-environment query
  and retains the existing resource ownership. Opaque resource kinds are never reinterpreted.
  ABI 7 exports exactly:

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

- Historical tasks retain their recorded wire assignments. The current bounded schema-19 image
  covers stable type wires `1..7`, operation wires `1..115`, attribute wires `0..41`, and route
  wires `1..3`; Java and native validation independently authenticate bounds, topology,
  cardinality, attributes, types, layout, route, and current support. Explicit unavailable,
  canonical, and affine-view states enforce graph-local provenance, including exact local
  identity-prefix, last-two-axis transpose authentication for MATMUL. Affine outputs use
  authenticated full-logical-size represented-order targets. Canonical host ingress, publication,
  and CPU/Metal transfer support all seven current data types at ranks `0..16` with exact byte
  widths and strict BOOL-byte validation. Transfer coverage changes no operation capability.
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
  target compatibility without changing cache-file or native ABI ownership. Task 0042's public
  wrappers and private Engine adapter consume that package-private codec. Task 0055 later migrated
  the current codec and all related identities to version fourteen without a dual decoder. Task
  0042 does not authorize cross-session reuse, cache-format work, a device fingerprint, or broader
  operation/type, mixed-owner, async, packaging/discovery, or performance claims.
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
