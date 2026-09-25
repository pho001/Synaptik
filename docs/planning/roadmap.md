# Implementation Roadmap

## Purpose and authority

This roadmap is the active index for project priority, area status, blockers, and authorized
implementation frontiers. It is not an architecture contract.
[`ARCHITECTURE.md`](../../ARCHITECTURE.md) is authoritative and wins if a planning document
conflicts with it.

Read an area's linked master plan for its dependency DAG and the authorized task briefs for
executable scope. Completed-task narratives and validation evidence do not belong in this index.

## Execution policy

- Each independent workstream may expose one authorized frontier when its DAG dependencies are
  satisfied.
- Before launch, the planner verifies that the task is `Ready`, is at an authorized frontier, and
  has current dependency, conflict, parallel-group, base, integration-order, and
  integration-validation metadata.
- Task-table and project-area order express priority, not an undeclared dependency. Concurrent
  writes are authorized only under the worktree, ownership, contract-first, and integration rules
  in the [Planning Guide](planning-guide.md).
- Record active frontiers and parallel groups in the owning master plans; summarize them here only
  when they affect cross-area priority, dependencies, or integration.
- Allowed and forbidden dependencies remain defined only by `ARCHITECTURE.md`.

## Ordered project areas

The order is the default delivery priority. “Next gate” identifies an unfinished decision or task
boundary; it does not promote Draft work to `Ready` or create a DAG edge.

| Order | Project area | Current status | Entry or next gate |
|---:|---|---|---|
| 1 | [`modules/model`](modules/model/master-plan.md) | Complete through reviewed 0029; 0026 Draft | Model 0029 completed at `30826783` after proof, validation, synchronized documentation, and independent Class C approval; no Model task is Ready. |
| 2 | [`modules/trace`](modules/trace/master-plan.md) | In progress, deliberately interleaved; 0001–0002 Complete, 0003–0008 Draft | Resume 0003 only after its producer vocabulary is stable; no Trace task is Ready. |
| 3 | [`modules/backend-contract`](modules/backend-contract/master-plan.md) | Complete through 0004 | Reopen only for a concrete shared-contract need. |
| 4 | [`modules/config`](modules/config/master-plan.md) | In progress, interleaved; 0001–0003, 0006, and 0006A–0006B Complete; 0004–0005 and 0007–0008 Draft | 0006 completed at `314e049` plus `37e9e9db`; no Config task is Ready. |
| 5 | [`modules/planning`](modules/planning/master-plan.md) | Complete through documentation-only 0007; profile-qualified query cutover Complete through Engine 0018 | No Planning task is Ready. |
| 6 | [`modules/runtime`](modules/runtime/master-plan.md) | Complete through 0016; profile-free boundary preserved by Engine 0018 | No Runtime task is Ready. |
| 7 | [`modules/compiler`](modules/compiler/master-plan.md) | Complete through documentation-only 0006B10; 0006C and 0007 Draft; profile artifact cutover Complete through Engine 0018 | No Compiler task is Ready. |
| 8 | [`modules/prepare`](modules/prepare/master-plan.md) | Complete through documentation-only 0008; profile projection cutover Complete through Engine 0018 | No Prepare task is Ready. |
| 9 | [`backends/openblas-provider`](backends/openblas-provider/master-plan.md) | Required baseline Complete; optional 0004 Blocked and deferred | Resume 0004 only when both direct-BFLOAT16 application binary interface (ABI) and one-final-narrowing proofs exist. |
| 10 | [`backends/cpu`](backends/cpu/master-plan.md) | Complete through profile realization 0017; 0007A1D Review needed; 0010D1 and 0011 Blocked | CPU 0017 completed at `372a8b98`; both profiles use identical exact CPU capability, routes, execution, and profile-separated identities. No CPU task is Ready. |
| 11 | [`modules/engine`](modules/engine/master-plan.md) | Complete through numerical-profile spine 0018 | 0018 completed at `ce7a7dfa` plus `07a01b9c`; no Engine task is Ready. |
| 12 | [`backends/metal`](backends/metal/master-plan.md) | Complete through documentation/audit-only 0038; recorded blockers through 0040 | Metal 0040's sole direct BFLOAT16 Gather gate returned `0x7fc0` for selected raw word `0xffa6`; artifacts were removed, production/schema stayed unchanged, and no Metal task is Ready. |

| 13 | [`backends/cuda`](backends/cuda/master-plan.md) | Draft | Create a detailed 0001 brief only when CUDA becomes the authorized frontier. |
| 14 | [`extensions/onnx`](extensions/onnx/master-plan.md) | Draft | Define the first bounded mapping task only at an authorized frontier. |
| 15 | [`extensions/data`](extensions/data/master-plan.md) | Draft; architecture decision required | 0001 must authorize the Data/Text/Vision modules, build edges, decision record, and architecture tests first. |
| 16 | [`extensions/nn`](extensions/nn/master-plan.md) | In progress under recorded interleaves; 0001–0020C, 0021A, and 0025–0025A Complete; 0021B–0024 Draft | 0021B needs a detailed brief and truthful concrete-backend recurrent coverage; no NN task is Ready. |
| 17 | [`extensions/text`](extensions/text/master-plan.md) | Draft; architecture decision required | Wait for Data 0001, then define the first tokenizer task. |
| 18 | [`extensions/vision`](extensions/vision/master-plan.md) | Draft; architecture decision required | Join the coordinated Data 0001 decision before decoder or image APIs. |
| 19 | [`extensions/training`](extensions/training/master-plan.md) | Complete through 0001 | No later Training task is Ready; parameter groups and broader optimizers remain Draft. |
| 20 | [`extensions/checkpoint`](extensions/checkpoint/master-plan.md) | Draft; architecture decision required | Authorize the model-only and optional Training adapter boundaries before 0001. |
| 21 | [`tools/benchmarks`](tools/benchmarks/master-plan.md) | Complete through 0001; 0002–0003 Draft | Report-only smoke, baseline, and evidence profiles are current; no benchmark task is Ready. |
| 22 | [`tools/tuning`](tools/tuning/master-plan.md) | Complete through 0004 | No Tuning task is Ready. |
| 23 | [`tools/cli`](tools/cli/master-plan.md) | Draft | Define commands only after their Engine and diagnostic contracts are stable. |

## Authorized frontiers

Numerical profiles

[CPU 0017](backends/cpu/tasks/0017-explicit-accelerator-numerical-profile-realization.md) is
Complete at implementation `372a8b98`. For every backend, capability/behavior under
`STRICT_IEEE` is a subset of capability/behavior under `ACCELERATOR` for the same occurrence
domain: accelerator may always execute the strict result and may add only named Model-authorized
results. CPU already realizes this invariant through one identical capability matrix and unchanged
portable/OpenBLAS routes; requested profiles remain distinct in plans, generated artifacts,
workloads, tuning, and cache compatibility without Runtime/Trace state.

[Metal 0015](backends/metal/tasks/0015-accelerator-float32-tensor-binary-arithmetic.md) is Complete
at implementation `42c4cfbf` plus evidence-wording remediation `fb102a46`. Complete
[Metal 0019](backends/metal/tasks/0019-exact-profile-qualified-float32-abs.md) adds exact canonical
`ABS` to both profile matrices. Complete Metal 0020 adds accelerator reductions, and Complete Metal
0021 adds positive static rank-two MATMUL with authenticated local transposes and seeded gradients.
Complete Metal 0022 implements the required monotonic matrix: exact canonical NEG/ABS and
affine/`CONTIGUOUS` are common to both profiles, while tensor binary/reduction/MATMUL remains
accelerator-only and strict-false. Implementation `41517594`, documentation remediation
`90cd5fd9`, and final independent Class C `APPROVE` with zero findings close it. Complete Metal 0023
landed at `9a7911c9` plus evidence `80e6cedd`; final independent lean Class C review approved with
zero findings after one concrete-risk production-ABI SNaN probe. Complete Metal 0024 landed at
implementation `a947e574732273bee4469d42afe8935082d53109` plus documentation remediation
`aa191ca469010d081150e97dcddd504ec626dd9e`; final independent lean Class C review approved with
zero findings. Complete Metal 0025 delivered bounded common-profile exact FLOAT32 UNFOLD_AXIS at
implementation `44edd860`, remediation `d0a947fa`, and final reviewed remediation `f88066e3`;
final independent lean Class C review approved with zero findings. Metal 0026 is Blocked by
subnormal SOFTMAX flushing, Metal 0027 by signed minimum-subnormal BatchNorm flushing, and Metal
0030 by signed-zero/NaN max-pool selection failures. Metal 0031 is independently Blocked without a
probe because one execution cannot authorize an undocumented shape-dependent Conv2d contraction.
Metal 0032 is likewise Blocked without a probe: MPSGraph has no direct MSE selector, arithmetic
decomposition cannot import accelerator binary relaxations into exact loss semantics, and one run
cannot close independent FTZ, special-class, and rounding gaps. Metal 0033 is Blocked without a
probe: direct unmasked output-only SDPA exists, but attention is unrelaxed and its documentation
leaves independent FTZ, accumulation, stable-softmax, special-class, and shape-algorithm gaps.
Metal 0034 is Blocked without a probe: direct one-output SORT exists, but unrelaxed ordering's
stability, NaN, signed-zero, subnormal, exact-bit, and shape-algorithm contracts are undocumented.
Metal 0035 is Blocked without a probe: hidden-random one-output dropout and opaque Philox state
cannot implement Model's exact INT64[2] initialization/transition or mandatory three-output
dropout, and current Metal lacks zero-input, multi-output, and INT64 representation. Metal 0036 is
Blocked without a probe: direct Conv3d structurally maps NCDHW/OIDHW but inherits 0031's opaque
shape-dependent contraction, while Pooling4D/stencil do not establish exact unrelaxed Pool3d
mapping and semantics. Metal 0037 is Blocked without a probe: direct RNN lacks runtime INT64 valid
lengths, atomic validation, skipped padded work, and `finalHidden`; multiple recurrent numeric gaps
remain unrelaxed. None changed production. Documentation/audit-only Metal 0038 is Complete. Metal
0040 is Blocked after its one BFLOAT16 Gather gate canonicalized the first selected signaling NaN;
no Metal task is Ready.

Task 0019 landed at implementation `a6d1796d` plus mixed-owner test remediation `bcb717a6`. Its
native ABI/export, Metal, conformance, real Engine, architecture, full-build, documentation, and
diff checks passed; independent Class C final review returned `APPROVE` with zero findings.

The completed cross-area profile DAG is:

`Model 0027 -> Config 0006 -> Engine 0018 -> {CPU 0017, Metal 0015}`

The active semantic and Metal serial DAG is:

`Metal 0015 -> {Metal 0016 (Blocked), Metal 0019 (Complete) -> Metal 0017 (Blocked)}`

`Model 0028 (Complete) -> Metal 0020 (Complete) -> Metal 0018 (Blocked)`

`Metal 0018 blocker evidence -> Model 0029 (Complete) -> Metal 0021 (Complete) -> Metal 0022 (Complete) -> Metal 0023 (Complete) -> Metal 0024 (Complete) -> Metal 0025 (Complete) -> {Metal 0026 (Blocked), Metal 0027 (Blocked), Metal 0030 (Blocked), Metal 0031 (Blocked), Metal 0032 (Blocked), Metal 0033 (Blocked), Metal 0034 (Blocked), Metal 0035 (Blocked), Metal 0036 (Blocked), Metal 0037 (Blocked)}`

`Metal 0022/0023/0024/0025 (Complete) + finalized evidence through Metal 0037 -> Metal 0038 (Complete)`

[Metal 0016](backends/metal/tasks/0016-profile-qualified-float32-abs-exp-sigmoid.md) is Blocked
without production changes. Its Apple M3 Max gate proved exact `ABS`, but `EXP` and `SIGMOID`
reproducibly flushed representable subnormal results to positive zero across every level, Shape,
context, and form; `SIGMOID` also exceeded its one-ULP gate on ordinary finite inputs. All controls
passed and the probe was removed. Historical 0006 remains Blocked under its own contract.

[Metal 0019](backends/metal/tasks/0019-exact-profile-qualified-float32-abs.md) is Complete. It adds
only canonical exact `ABS` under both profiles, with no relaxed unary semantics. `EXP`, `SIGMOID`,
and the six historically failing 0006 operations remain closed, including the forbidden
`RELU(NaN) -> +0` and `TANH(NaN) -> +1` observations.

[Metal 0017](backends/metal/tasks/0017-accelerator-float32-sum-mean-sum-to-shape-reductions.md)
remains Blocked under its unchanged old contract. After probe-only dangling autorelease-string and
per-cell pool fixes, a minimal valid rank-16 axis-8 cell passed and the corrected full gate completed
300 executables/2,400 runs. Accelerator `SUM([-0,-0])` returned positive zero, but the old
binary-tree/per-step-FLOAT32 plus DAZ/FTZ set permitted only negative zero because zero is not
subnormal. No production change or probe artifact remains.

[Model 0028](modules/model/tasks/0028-accelerator-reduction-exact-zero-sign-freedom.md) is
Complete at `fc003ab8`. Metal 0020 completed at implementation `9ddb75f6` plus documentation
remediation `5b77c742` after its full fresh corrected gate, validation, and independent Class C
`APPROVE` with zero findings.

[Metal 0018](backends/metal/tasks/0018-accelerator-float32-rank2-matmul-seeded-gradient-checkpoint.md)
is Blocked under its unchanged historical contract. Its fresh real-M3 smoke passed, then the first
full direct context-zero/optimization-zero `K=1` cell returned positive zero for
`+0.0f * -1.0f`, outside that task's set containing only negative zero. Reduced-precision-none,
selector, direct-target, binding, and probe controls were valid; the probe was removed and no
production change exists.
[Model 0029](modules/model/tasks/0029-accelerator-matmul-exact-zero-sign-freedom.md) is Complete at
`30826783` after its proof, 23-task validation, documentation/diff scope check, and independent
Class C `APPROVE` with zero findings. Metal 0021 passed its fresh full oracle and completed at
implementation `ef2c6a1a` plus evidence corrections `be5543f9`/`5631d51f`; final independent
Class C review returned `APPROVE` with zero findings. Metal 0022 completed at `41517594` plus
`90cd5fd9` and final independent approval; Metal 0023 completed at `9a7911c9` plus evidence
`80e6cedd` and final independent approval. Metal 0024 completed at
`a947e574732273bee4469d42afe8935082d53109` plus documentation remediation
`aa191ca469010d081150e97dcddd504ec626dd9e` and final independent approval. Metal 0025 completed at
implementation `44edd86092509348e1e72cf7f0f4c3b13d141fa8`, first remediation
`d0a947fa953bddb2714c1917c7114ac345c530b7`, and final remediation/reviewed revision
`f88066e3ad0547987bb03b2d18ed2813f97de223`; final approval had zero findings. Metal 0026,
Metal 0027, Metal 0030, Metal 0031, Metal 0032, Metal 0033, Metal 0034, Metal 0035, Metal 0036, and
Metal 0037 are Blocked without production changes; 0031–0037 ran no probe. Documentation/audit-only
Metal 0038 is Complete. Planning-only Metal 0039 remains Blocked. Metal 0040 ran exactly one failed
direct-selector gate and is Blocked without a production change. No Metal task is Ready.

Engine
[0017](modules/engine/tasks/0017-reusable-inference-session-api.md) is Complete from exact base
`4fc4fd3d1e44613cdbdc44051e0bc637992d0de4`; 0018 is Complete at `ce7a7dfa` plus `07a01b9c`.
No Engine task is Ready.

Training
[0001](extensions/training/tasks/0001-public-training-session-and-sgd-lifecycle.md) is Complete
from exact base `2e110c7688ead480faa4215ff67f9814fa81f4ad`. Mandatory external Class C review
first returned `BLOCK` with six P1 and one P2 findings; after remediation and validation, external
re-review returned `APPROVE` with findings `0`. No later Training task is authorized.
Metal
[0008](backends/metal/tasks/0008-mpsgraph-float32-affine-transforms.md) is `Complete` after
implementation `7e39f705`, remediation `9713e528` and `aa42ed711`, and independent Class C
`APPROVE` with zero findings. It delivers the five bounded forward affine transforms with exact
logical views, authenticated dense represented-order targets, ABI v4's exact thirteen exports,
and unchanged canonical-only transfer.

[0005](backends/metal/tasks/0005-mpsgraph-mixed-binary-whole-partition.md) remains a completed
historical delivery whose strict binary claim was withdrawn after 12,096 subnormal-domain failures.
Complete 0015 does not rewrite that history: it restored wires `2..5` only for `ACCELERATOR` after
the fresh bounded oracle passed. Complete 0019 appends exact ABS wire `12` under both profiles.
Task 0020 adds accelerator `SUM=13` and `MEAN=14`; Complete task 0021 appends accelerator
`MATMUL=15`. Complete task 0023 retains ABI v4's exactly thirteen exports, advances node schema to
9, appends `GATHER=16`, `ONE_HOT=17`, and `DEPTH=5`, and advances route, candidate, compatibility,
workload, exact-policy, and codec identities to version ten. Evidence `80e6cedd` and final
independent lean Class C approval close it. Complete 0024 advances schema to 10, appends
`SCATTER_ELEMENTS=18`, and advances those identities to version eleven. Complete 0025 implements
schema 11/wire 19/typed window-axis state and version-twelve identities for exact bounded canonical
FLOAT32 UNFOLD_AXIS; implementation `44edd860`, remediations `d0a947fa`/`f88066e3`, and final
independent approval close it. Blocked 0026 ran exactly one `[4,8]` production-settings direct
SOFTMAX gate, returned positive zero instead of reference subnormal `0x0008ec28` at flattened index
18, removed the probe, and did not begin schema 12/wire 20/version-thirteen implementation. Blocked
0027 independently used those planned next numbers but did not implement them: its sole `[2,8,2]`
BatchNorm gate flushed both signed minimum-subnormal outputs to same-signed zero.
Blocked 0030 independently ran exactly one raw-winner MAX_POOL2D execution, preserved its exact
failure evidence, removed its artifacts, and implemented none of the planned next schema numbers.
Blocked Task 0040 consumes none of schema 12, wire 20, local value type 4, or version-thirteen
identities. Those values, attribute 7, INT64, ABI/export changes, and later wires remain
unimplemented and unreserved; every restart must revalidate from the then-current frontier.
Blocked 0031 ran no probe: its direct Conv2d descriptor mapping is structurally complete, but the
selector's shape-dependent contraction algorithm is undocumented and lean policy prohibits the
matrix required to authorize full geometry.
Blocked 0032 also ran no probe. Its smallest first-class loss candidate is profile-common
canonical no-grad same-typed FLOAT32 MSE `NONE`, but no direct MPSGraph selector exists and a
subtraction/square decomposition cannot inherit operation-scoped accelerator binary relaxations.
Independent FTZ, special-class, and rounding gaps require more than one lean run, while the matrix
is prohibited.
Blocked [0033](backends/metal/tasks/0033-profile-common-canonical-float32-no-grad-unmasked-noncausal-default-scale-sdpa-forward.md)
also ran no probe. Its smallest attention candidate is profile-common canonical FLOAT32 no-grad
unmasked noncausal default-scale one-output SDPA. The macOS-15 direct selector is available on the
current runtime, but attention is unrelaxed and the headers do not close independent FTZ,
accumulation, stable-softmax, special-class, or shape-algorithm gaps. The additive mask surface
mismatches Model's BOOL eligibility semantics and provides no adjacent route.
Blocked [0034](backends/metal/tasks/0034-profile-common-canonical-float32-no-grad-ascending-sort.md)
also ran no probe. Its smallest candidate is profile-common canonical rank-one FLOAT32 no-grad
ascending SORT. The direct selector is runtime-available but omits the complete exact ordering
contract. ARGSORT returns INT32 rather than Model INT64, and mandatory two-output TOP_K cannot fit
the current Metal one-output schema.
Blocked [0035](backends/metal/tasks/0035-profile-common-canonical-float32-no-grad-explicit-state-dropout.md)
also ran no probe. Its smallest nondegenerate candidate is profile-common exact canonical rank-one
FLOAT32 no-grad INITIAL_STATE plus explicit-state DROPOUT over the complete probability domain.
Direct dropout hides randomness and returns one output; opaque Philox state cannot establish exact
Model key/counter-plus-N, mask, or next-state semantics. Exact custom random execution requires a
complete zero-input/multi-output/local-INT64 schema and lifecycle.

Blocked [0036](backends/metal/tasks/0036-extended-3d-inference.md) also ran no probe. Its smallest
candidate is accelerator-only unbiased no-grad canonical positive static rank-five FLOAT32 Conv3d.
The direct selector maps NCDHW/OIDHW and full geometry but inherits 0031's undocumented
shape-dependent contraction. Pooling4D/stencil do not authorize exact MAX/AVERAGE_POOL3D mapping
and semantics, while 0030 forbids generic max-pool assumptions. UNFOLD3D/FOLD3D remain separate
custom movement/overlap work; conditional schema is unimplemented and unreserved.

Blocked [0037](backends/metal/tasks/0037-profile-common-canonical-float32-no-grad-forward-rnn-tanh.md)
also ran no probe. Its smallest candidate is profile-common canonical positive static FLOAT32
no-grad bias-free FORWARD RNN_TANH over every runtime valid length. Direct RNN has no valid-length
input, atomic validation, skipped padded work, positive-zero padding contract, or final-hidden
output. Its independent contraction/add/tanh/state-order gaps are unrelaxed. Exact custom recurrent
execution requires complete five-input/two-output/caller-INT64 schema and lifecycle.

[0006](backends/metal/tasks/0006-mpsgraph-float32-unary-algebra.md) remains `Blocked` by exact
RECIPROCAL/LOG/SQRT/RSQRT/RELU/TANH probe failures. [0007](backends/metal/tasks/0007-mpsgraph-float32-reductions.md)
remains independently `Blocked` because repeated direct-output probes returned positive zero where
the exact Model SUM/MEAN results are `2.0f`/`0.5f`.

[0009](backends/metal/tasks/0009-mpsgraph-float32-rank2-matmul-training-checkpoint.md) is
independently `Blocked`. Its mandatory Apple M3 Max probe passed selector, direct-target, Shape,
permutation, transpose, repetition, and independent-executable gates, but `K=1` multiplication by
one flushed positive/negative minimum and ordinary subnormals to zero. Strict host multiplication,
`+0.0f` accumulation, and `fmaf` preserve the minimum subnormal, so zero is outside every
Model-permitted association/FMA result. Signed zero is not a blocker. Independent review returned
`APPROVE-BLOCKER`. Tasks 0006, 0007, and 0009 retain no production, test, or probe changes.

[0010](backends/metal/tasks/0010-mpsgraph-float32-bool-predicates-and-selection.md) is independently
`Blocked`. On an Apple M3 Max at optimization levels `0` and `1`, three independent executables per
level and eight runs per case treated positive and negative minimum subnormal as equal to `±0`.
Raw identity preserved the input bits; ordinary, infinity, NaN, canonical BOOL target, canary, and
permutation controls passed. No production, test, or probe changes remain.

[0011](backends/metal/tasks/0011-mpsgraph-float32-scalar-pointwise-arithmetic.md) is independently
`Blocked`. At optimization levels `0` and `1`, three independently compiled executables per level
and eight runs per case produced 17,328 mismatches in 77,824 scalar checks. Failures include scalar
`ADD +0` retaining input `-0`, minimum-subnormal flushing, minimum-subnormal `DIV +0` producing
NaN, maximum-finite self-division producing zero, and infinity/maximum-finite division producing
NaN. Exact constants, direct targets, canaries, permutations, repetition, and identity controls
passed. Independent review confirmed no current Model relaxation admits the results. No
production, native, test, or probe changes remain.

[0012](backends/metal/tasks/0012-mpsgraph-float32-extrema-reductions.md) is independently
`Blocked`. At optimization levels `0` and `1`, fast math disabled, three executables per
length/level, and eight runs per case, its 43,824-execution gate found order-dependent opposite
signed zeros and zero/minimum-subnormal results: 2,016 MIN plus 2,016 MAX mismatches and 2,976
subnormal failures. NaN classification, minimum normals, identity, canaries, and bindings passed.
Independent review returned `APPROVE-BLOCKER`; no production, native, test, or probe changes remain.

[0013](backends/metal/tasks/0013-mpsgraph-float32-cumulative-scans.md) is independently
`Blocked`. At optimization levels `0` and `1`, fast math disabled, twelve executables covering
three per level/length pair for lengths `1` and `2`, and eight direct runs for all eight kind/mode
combinations, inclusive length-one SUM and PRODUCT flushed all six subnormal corpus values to
zero. Strict host operations preserved them; exclusive identities, raw identity, canaries, and
bindings passed. Corrected review evidence shows rank-six axis `5` valid for both kinds and axis
`6` correctly rejected, so axis is not a blocker. Independent review returned `APPROVE-BLOCKER`;
no production, native, test, or probe changes remain.

[0014](backends/metal/tasks/0014-mpsgraph-float32-affine-layout-composition.md) is `Complete` at
`01e81be2` with exact strict local affine/`CONTIGUOUS` composition. [0015](backends/metal/tasks/0015-accelerator-float32-tensor-binary-arithmetic.md)
is `Complete` at `42c4cfbf` plus `fb102a46` after the operation-by-operation DAZ/FTZ oracle, focused
validation, combined serial checkpoint, and independent Class C approval all passed.

Current Metal uses ABI v4 with exactly thirteen exports and node schema 11. Both profile matrices
admit exact canonical `FLOAT32` NEG/ABS/affine/`CONTIGUOUS`, bounded canonical FLOAT32
`UNFOLD_AXIS`, canonical positive-rank FLOAT32 data GATHER with canonical INT32 indices,
positive-rank INT32-to-BOOL ONE_HOT, and canonical FLOAT32/INT32/FLOAT32
`SCATTER_ELEMENTS/NONE`; accelerator additionally admits tensor `FLOAT32`
`ADD`/`SUB`/`MUL`/`DIV`, canonical `SUM`/`MEAN`/`SUM_TO_SHAPE`, and positive static rank-two
MATMUL with authenticated local transposes. Java/native validation authenticates an affine MATMUL
operand only on its consuming edge; general affine publication and valid local consumers remain
available. Version-twelve identities separate profile/type/topology compatibility, scalar
reduction and BOOL materialization remain local-only, FLOAT32-only transfer is unchanged, and
Runtime/Trace remain profile-free. Metal 0025 is Complete at final reviewed revision `f88066e3`.
Metal 0026–0027/0030–0037/0039–0040 are Blocked and changed no executable capability; 0031–0037
and 0039 ran no device probe, while 0040 ran exactly its one failed BFLOAT16 Gather gate. Schema
12/wires beginning at 20/attribute 7, local type 4, INT64 type/ingress, ABI/export changes, and
version-thirteen identities remain unimplemented and unreserved. Documentation/audit-only Metal
0038 is Complete, and no Metal task is Ready.

Strategic gate: historical blocker evidence is preserved, and no backend task may define Model
semantics. Complete Model 0028 owns bounded reduction exact-zero sign freedom; Complete Metal 0020
implements that rule under accelerator only. Blocked Metal 0018 retains its unchanged historical
MATMUL contract and no production changes. Complete Model 0029 split the shared contraction row,
and Complete Metal 0021 implemented the bounded successor at `ef2c6a1a`; evidence corrections
`be5543f9`/`5631d51f` and final independent Class C `APPROVE` with zero findings close that task.
Complete Metal 0022 reuses exact NEG/affine/`CONTIGUOUS` lowering under accelerator, supports valid
whole-partition composition, and retained its ABI/schema/identity versions. Implementation
`41517594`, documentation remediation `90cd5fd9`, and final independent Class C `APPROVE` with zero
findings close it. No numerical probe ran because no selector, wire, lowering, or result freedom
changed. Complete Metal 0023 passed its documented-selector/minimal-smoke evidence and final
independent review at implementation `9a7911c9` plus evidence `80e6cedd`. Complete Metal 0024
passed its one-run Set smoke, implementation at `a947e574732273bee4469d42afe8935082d53109`,
documentation remediation at `aa191ca469010d081150e97dcddd504ec626dd9e`, and final independent
lean Class C approval with zero findings. Complete Metal 0025 passed its one-run movement smoke,
implementation `44edd860`, selector-cap/capability remediation `d0a947fa`, wrong-kind remediation
`f88066e3`, and final independent approval. Metal 0026 is Blocked by its exact one-run numerical
gate: actual positive zero `0x00000000`, StrictMath reference `0x0008ec28`, distance 584,744 ULPs.
Metal 0027 is independently Blocked by prohibited no-FTZ failures: required
`0x00000001`/`0x80000001`, actual `0x00000000`/`0x80000000`. Metal 0030 is independently Blocked:
its sole direct max-pool run returned negative zero for `[+0,-0]` and finite `3.0f` for every
first/later qNaN/sNaN window. All three probes were removed and none changed production. Metal 0031
is Blocked before a probe because one allowed-set run cannot authorize full Conv2d geometry through
an undocumented shape-dependent contraction algorithm. Metal 0032 is Blocked before a probe
because no direct MSE selector exists, binary accelerator freedoms are operation-scoped, and one
run cannot close its independent exact-loss gaps. Metal 0033 is Blocked before a probe because
attention is unrelaxed and the direct selector leaves independent exact-attention gaps across
opaque shape-dependent algorithms. Metal 0034 is Blocked before a probe because ordering is
unrelaxed and direct SORT leaves independent stability, special-order, exact-bit, and
shape-algorithm gaps; ARGSORT/TOP_K also conflict with current type/arity boundaries. Metal 0035
is Blocked before a probe because direct dropout hides randomness, opaque Philox state cannot
represent exact Model INT64[2] state, and current Metal lacks zero-input/multi-output/INT64
support. It requires an exact custom kernel plus the complete schema/lifecycle; `p=0` narrowing is
prohibited. Metal 0036 is Blocked before a probe because direct Conv3d inherits the undocumented
shape-dependent contraction boundary, while neither Pooling4D nor stencil establishes the
unrelaxed exact Pool3d contract. Metal 0037 is Blocked before a probe because direct RNN lacks
runtime INT64 lengths, atomic validation, skipped padding semantics, and `finalHidden`, while
contraction/add/tanh/state ordering has multiple unrelaxed gaps. No task may infer generic fast
math, transfer row relaxations, replace a prohibited matrix with narrowing, or reserve conditional
schema.

## Blocked, review-needed, and deferred work

- Metal 0006 is `Blocked` by the exact special-value/underflow failures recorded in its brief.
  Unblocking requires custom kernels or an explicitly accepted relaxed numerical contract.
- Metal 0016 is independently `Blocked` by reproducible `EXP`/`SIGMOID` result flushing and
  `SIGMOID` ordinary finite results beyond its one-ULP gate. Its exact three-operation contract is
  unchanged; an exact replacement for both failed operations is required to restart it.
- Metal 0017 is `Blocked` under its old accelerator reduction contract. Its corrected full probe
  completed 300 executables/2,400 runs and found `SUM([-0,-0]) -> +0`, while the old permitted set
  contained only `-0`; zero is not subnormal, so DAZ/FTZ did not admit the result. No production
  change or retained probe artifact remains. Complete Model 0028 owns the separate semantic
  decision. Metal 0020 is the dependent Complete successor at `9ddb75f6` plus `5b77c742`;
  historical 0017 remains Blocked rather than being reopened.
- Metal 0007 is independently `Blocked` by its repeated exact reduction counterexample. Unblocking
  requires an exact replacement route or an explicit Model contract change.
- Metal 0009 is independently `Blocked` by exact subnormal flushing in `K=1` MATMUL. Unblocking
  requires an exact replacement route or an explicit Model numerical-contract change; signed-zero
  behavior does not block it.
- Metal 0018 is `Blocked` under its unchanged historical full-contraction contract. Its mandatory
  fresh M3 smoke passed, but the first full direct `K=1` cell produced `+0` for
  `+0.0f * -1.0f`, where that task permitted only `-0`. The exact reduced-precision-none and probe
  controls passed; the probe was removed and no production edit exists. Model 0029 completed the
  separate final-publication semantic decision at `30826783`; the product remains `-0`, and only
  publication may choose `+0`. Complete Metal 0021, not 0018, is the implementation successor;
  its fresh full oracle, implementation/evidence chain, and independent Class C review passed.
- Metal 0010 is independently `Blocked` by exact subnormal comparison collapse. Unblocking requires
  an exact replacement route or an explicit Model numerical-contract change; canonical BOOL and
  ordinary/special-value controls do not remove the blocker.
- Metal 0011 is independently `Blocked` by 17,328 exact scalar mismatches in 77,824 checks.
  Unblocking requires an exact replacement route or explicit Model numerical-contract change;
  successful constant, target, canary, permutation, repetition, and identity controls do not
  remove the arithmetic blocker.
- Metal 0012 is independently `Blocked` by order-dependent signed-zero and subnormal extrema
  results. Unblocking requires an exact replacement route or explicit Model numerical-contract
  change; passing NaN classification, minimum-normal, identity, canary, and binding controls do
  not remove the blocker.
- Metal 0013 is independently `Blocked` by exact subnormal flushing in cumulative SUM/PRODUCT.
  Unblocking requires an exact replacement route or explicit Model numerical-contract change;
  passing exclusive identities, raw identity, canaries, bindings, and valid axis `5` do not remove
  the blocker. Axis `6` correctly rejects and is not a blocker.
- [OpenBLAS provider 0004](backends/openblas-provider/tasks/0004-optional-direct-bfloat16-output-gemm-capability.md)
  and dependent CPU 0010D1 remain an optional blocked branch. Pinned evidence proves neither the
  required exported direct BFLOAT16-output ABI nor full-contraction FLOAT32 accumulation followed
  by one final BFLOAT16 narrowing. Existing FLOAT32/FLOAT64 and portable BFLOAT16 work remain
  unaffected.
- CPU 0011 remains `Blocked` in the [CPU master plan](backends/cpu/master-plan.md) until a concrete
  Intel workload and supported oneMKL interface evidence both exist.
- CPU 0007A1D remains `Review needed`, not an active frontier; its stable semantic and structural
  evidence does not satisfy its failed performance gate.
- CPU 0010M is `Complete` with an evidence-backed no-change conclusion: production retains scalar
  compute, configured/available parallelism `1`/`1`, minimum elements per worker `1`, no worker
  group, and all existing fallbacks/thresholds. The report-only protocol is hardened; a future
  comparison still requires a separately reviewed, fully sealed matrix before measurement.
- Model 0026 remains an independent FLOAT16 Draft. Model 0027–0029, Config 0006, Engine 0018, CPU
  0017, and Metal 0015/0019/0020/0021/0022/0023/0024/0025/0038 are Complete. Metal 0016–0018,
  0026–0027, 0030–0037, planning-only 0039, and 0040 remain Blocked under their recorded contracts.
  Task 0040's one direct BFLOAT16 Gather execution canonicalized required selected `0xffa6` to
  `0x7fc0`; it removed all artifacts and made no production/schema change. No Metal task is Ready;
  every unfinished or blocked operation family remains unauthorized.
- Planning 0007 review found a stale glossary `Compile` status sentence and stale
  `GraphCompilationPort` Javadoc about the Engine facade. Compiler 0006B10 corrected and
  independently reviewed both without reopening Planning capability work.
- Prepare 0007 review found a stale CPU-finalizer status paragraph in the glossary's `Backend
  partition finalization` entry. Prepare 0008 corrected and independently reviewed it without
  reopening executable Prepare or CPU work.
- Data, Text, and Vision require the coordinated decision owned by
  [Data 0001](extensions/data/master-plan.md); Checkpoint requires its separate decision in the
  [Checkpoint plan](extensions/checkpoint/master-plan.md). Planning authorization alone does not
  create these Gradle projects or permit implementation.

## Nearest next step

Metal 0040 is Blocked from exact clean planning revision
`c300582727ec568a7482dd974f9d1b2e2e13f82c`. Its sole disposable direct BFLOAT16 Gather program
created one graph, compiled once, and executed once over the exact 20-word corpus and reverse INT32
permutation. Both inputs were unchanged, but output ordinal zero returned canonical quiet-NaN
`0x7fc0` instead of selected negative signaling-NaN `0xffa6`; later output and guard checks make no
success claim. The source and binary were removed. There was no retry, matrix, exhaustive corpus,
production/test/schema/identity change, native build, or full build. Unblocking requires a
separately planned exact replacement route, not a weaker direct-selector claim.

Metal 0030 remains Blocked after its only permitted direct `MAX_POOL2D` execution. The frozen
`[1,1,1,18] -> [1,1,1,9]` run returned negative zero `0x80000000` for `[+0,-0]`, where positive
zero was required, and finite `3.0f` for all first/later qNaN/sNaN windows; its other documented
controls passed and its artifacts were removed.

Metal 0031 is now Blocked without a device probe. The accelerator-only canonical FLOAT32 unbiased
no-grad Conv2d candidate has a complete documented NCHW/OIHW descriptor mapping, but MPSGraph does
not document or pin its shape-dependent contraction algorithm. One allowed-set execution cannot
authorize full Model geometry, lean policy prohibits a geometry/algorithm matrix, and narrowing to
one probe geometry is unsound. Unblocking requires an exact custom kernel or preceding
Model/architecture broadening. Average Pool2d, composed Pool1d/Conv1d, larger Pool3d/Conv3d, and
backward topology retain separate or inherited blockers and provide no smaller sound route.

Metal 0032 is now Blocked without a device probe. The smallest first-class loss candidate is
profile-common canonical same-typed FLOAT32 no-grad MSE `NONE`, which avoids complete-domain
accumulation. MPSGraph has no direct MSE selector; subtraction plus square cannot import
accelerator binary DAZ/FTZ into unrelaxed loss, and one run cannot close independent FTZ,
special-class, and rounding gaps. Dense categorical `NONE` still reduces over classes, index loss
adds selection/bounds/ignore obligations without a direct selector, and `SUM`/`MEAN` add unrelaxed
accumulation and denominator/scalar publication, so none is a smaller fallback. Unblocking requires
an exact custom-kernel proof, an authoritative complete operation-scoped MPSGraph MSE contract, or
preceding MSE-specific Model/architecture broadening.
Metal 0033 is now Blocked without a device probe. Its smallest candidate is profile-common
canonical FLOAT32 no-grad unmasked noncausal default-scale one-output SDPA. The legacy direct
selector is runtime-available, while the descriptor route requires macOS 27 and is unavailable on
the current macOS 26.6.2 runtime. The direct/additive mask contract does not implement Model's BOOL
selection or BOOL-plus-causal AND, and standalone binary/reduction/MATMUL/SOFTMAX relaxations cannot
transfer into unrelaxed attention. The selector documentation leaves independent FTZ,
accumulation, stable-softmax, special-class, and shape-algorithm gaps that one run cannot close.
Unblocking requires authoritative complete selector documentation, an exact custom-kernel proof,
or preceding attention-specific Model/architecture broadening.
Metal 0034 is now Blocked without a device probe. Its smallest candidate is profile-common
canonical rank-one FLOAT32 no-grad ascending SORT. Direct one-output SORT is runtime-available, but
ordering has no accelerator relaxation and Apple does not document stable ties, NaNs-last,
signed-zero order, ordinary subnormal comparison, exact represented-bit movement, or one behavior
across shape-dependent algorithms. ARGSORT's direct INT32 result mismatches Model's mandatory INT64
and the current Metal type/publication path; mandatory two-output TOP_K cannot fit the current
one-output schema and lacks Model's output-order guarantees. One run cannot close these independent
gaps. Unblocking requires an authoritative complete direct-selector contract or an exact custom
stable-sort kernel.
Metal 0035 is now Blocked without a device probe. Its smallest nondegenerate candidate is
profile-common exact canonical rank-one FLOAT32 no-grad INITIAL_STATE plus explicit-state DROPOUT
over every valid binary64 probability. Direct MPSGraph dropout hides randomness and returns only
the value; its opaque Philox state does not establish Model's exact INT64[2] key/counter input,
counter-plus-N output, mandatory mask/nextState, threshold, or special/scaling rules. Current Metal
also lacks zero-input nodes, multi-output nodes, and local INT64. A `p=0` identity route is not a
sound slice. Unblocking requires an exact Metal-private custom random kernel and complete
multi-output/local-INT64 schema, native lifecycle, and proof.
Metal 0036 is now Blocked without a device probe. Its smallest candidate is accelerator-only
unbiased no-grad canonical positive static rank-five FLOAT32 Conv3d. The macOS-13.2 direct selector
has a complete NCDHW/OIDHW groups/stride/dilation/explicit-padding mapping, but its contraction
algorithm and shape-selection boundary are undocumented; it therefore inherits Task 0031's
documentation-first blocker. Pooling4D exposes four unnamed spatial axes and stencil operates on
the last four dimensions, but neither surface establishes the full NCDHW Pool3d mapping, ceil-tail,
max winner, fixed-divisor average, exceptional-value, or zero-sign contract. Task 0030's signed-zero
and NaN failures forbid generic MPSGraph max assumptions. UNFOLD3D/FOLD3D require separate custom
movement/overlap planning. Unblocking requires an exact custom route, authoritative complete
operation-scoped documentation, or preceding Model/architecture broadening.
Metal 0037 is now Blocked without a device probe. Its smallest candidate is profile-common
canonical positive static FLOAT32 no-grad bias-free FORWARD RNN_TANH over the complete runtime
valid-length domain. The macOS-12.3 direct selector structurally maps source, weights, and initial
state, but it has no INT64 valid-length input, atomic validation, skipped padded work,
positive-zero padding contract, or `finalHidden`; its optional training output has the wrong role.
Recurrent scans are unrelaxed, and the header leaves independent contraction, add, tanh, and
per-step state-order gaps. GRU and LSTM have the same length/final-state mismatch plus their own
output-role/order gaps. Unblocking requires an exact custom recurrent kernel and complete
five-input/two-output/caller-INT64 schema, native lifecycle, and proof.

Schema 12, wires beginning at 20, attribute 7, local value type 4, INT64 type/ingress, ABI/export
changes, and identity 13 remain unimplemented and unreserved. Metal 0026/0027 remain separately
finalized Blocked. Documentation/audit-only Metal 0038 is Complete. Planning-only Metal 0039 is
Blocked on Draft Model 0026. Metal 0040 is Blocked by its failed one-execution BFLOAT16 raw-bit
gate; no Metal task is Ready.

## History policy

Completed task narratives, prior validation evidence, agent context identifiers, retrospectives,
and old reorder stories remain historical evidence in Git and task files. They are not default
executor input. Read historical evidence only when a current brief explicitly identifies an
unresolved question that requires it. Do not reconstruct global history before executing a
verified current brief.

## Update policy

Update this roadmap only when project priority, project-area status, authorized frontiers, a
blocker, or a cross-area parallel group changes. Keep detailed scope and result evidence in task
files; keep master plans and this roadmap to links and one-line status or gate summaries. If a
planning change conflicts with `ARCHITECTURE.md`, stop and use the architecture-decision process
instead of changing this roadmap alone.
